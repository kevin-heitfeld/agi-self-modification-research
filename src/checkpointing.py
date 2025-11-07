"""
Checkpointing System - Safe State Management

This module provides comprehensive checkpointing capabilities for the AGI
self-modification research project. It enables saving, restoring, and tracking
model states throughout the modification process.

Key Features:
- Save full model states with metadata
- Restore previous states (rollback capability)
- Track modification history
- Efficient storage using safetensors
- Checkpoint comparison and validation
- Automated checkpoint management

Safety Philosophy:
Every modification must be reversible. This checkpointing system ensures
that the system can always return to a known-good state if something goes wrong.

Author: AGI Self-Modification Research Team
Date: November 6, 2025
"""

import os
import json
import uuid
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from pathlib import Path
import shutil
from safetensors.torch import save_file, load_file


class Checkpoint:
    """
    Represents a single checkpoint of the model state.
    
    Contains:
    - Model state dict
    - Configuration
    - Metadata (timestamp, benchmarks, description, etc.)
    - Modification history
    """
    
    def __init__(
        self,
        checkpoint_id: str,
        timestamp: str,
        description: str,
        model_state: Optional[Dict] = None,
        config: Optional[Dict] = None,
        metadata: Optional[Dict] = None
    ):
        self.checkpoint_id = checkpoint_id
        self.timestamp = timestamp
        self.description = description
        self.model_state = model_state
        self.config = config or {}
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert checkpoint info to dictionary (excluding model state)."""
        return {
            'checkpoint_id': self.checkpoint_id,
            'timestamp': self.timestamp,
            'description': self.description,
            'config': self.config,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Checkpoint':
        """Create checkpoint from dictionary."""
        return cls(
            checkpoint_id=data['checkpoint_id'],
            timestamp=data['timestamp'],
            description=data['description'],
            config=data.get('config', {}),
            metadata=data.get('metadata', {})
        )


class CheckpointManager:
    """
    Manages model checkpoints for safe experimentation and rollback.
    
    This class handles:
    - Creating checkpoints at key moments
    - Restoring previous states
    - Tracking modification history
    - Managing checkpoint storage
    - Comparing checkpoints
    - Cleaning up old checkpoints
    
    Usage:
        >>> manager = CheckpointManager(checkpoint_dir='checkpoints')
        >>> 
        >>> # Save baseline
        >>> checkpoint_id = manager.save_checkpoint(
        ...     model=model,
        ...     description="Baseline before first modification",
        ...     benchmarks={'perplexity': 11.27}
        ... )
        >>> 
        >>> # Make modifications...
        >>> 
        >>> # Restore if needed
        >>> manager.restore_checkpoint(model, checkpoint_id)
    """
    
    def __init__(self, checkpoint_dir: str = 'checkpoints'):
        """
        Initialize the checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to store checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Metadata file
        self.metadata_file = self.checkpoint_dir / 'checkpoint_metadata.json'
        
        # Load existing metadata
        self.checkpoints: Dict[str, Checkpoint] = {}
        self._load_metadata()
    
    def save_checkpoint(
        self,
        model: nn.Module,
        description: str,
        benchmarks: Optional[Dict[str, float]] = None,
        modification_details: Optional[Dict] = None,
        auto_id: bool = True
    ) -> str:
        """
        Save a checkpoint of the current model state.
        
        Args:
            model: The model to checkpoint
            description: Human-readable description of this checkpoint
            benchmarks: Optional benchmark results at this state
            modification_details: Optional details about modifications made
            auto_id: Whether to auto-generate checkpoint ID
        
        Returns:
            checkpoint_id: Unique identifier for this checkpoint
        
        Example:
            >>> checkpoint_id = manager.save_checkpoint(
            ...     model=model,
            ...     description="After attention head pruning",
            ...     benchmarks={'perplexity': 11.5, 'mmlu': 0.45},
            ...     modification_details={'pruned_heads': [2, 5, 8]}
            ... )
        """
        # Generate checkpoint ID
        if auto_id:
            # Use UUID for guaranteed uniqueness, add short timestamp for readability
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            unique_id = uuid.uuid4().hex[:8]
            checkpoint_id = f"checkpoint_{timestamp}_{unique_id}"
        else:
            checkpoint_id = description.lower().replace(' ', '_')
        
        # Create checkpoint directory
        checkpoint_path = self.checkpoint_dir / checkpoint_id
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        
        # Prepare metadata
        metadata = {
            'timestamp': timestamp,
            'description': description,
            'benchmarks': benchmarks or {},
            'modification_details': modification_details or {},
            'model_type': type(model).__name__,
            'num_parameters': sum(p.numel() for p in model.parameters())
        }
        
        # Save model state using safetensors (efficient and safe)
        state_dict = model.state_dict()
        state_dict_path = checkpoint_path / 'model.safetensors'
        
        try:
            # Save with safetensors
            save_file(state_dict, str(state_dict_path))
        except Exception as e:
            # Fallback to torch save
            print(f"Warning: safetensors failed, using torch.save: {e}")
            torch_path = checkpoint_path / 'model.pt'
            torch.save(state_dict, torch_path)
            metadata['format'] = 'torch'
        else:
            metadata['format'] = 'safetensors'
        
        # Save model config if available
        if hasattr(model, 'config'):
            config = model.config.to_dict() if hasattr(model.config, 'to_dict') else {}
            config_path = checkpoint_path / 'config.json'
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
        else:
            config = {}
        
        # Save metadata
        metadata_path = checkpoint_path / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Create checkpoint object
        checkpoint = Checkpoint(
            checkpoint_id=checkpoint_id,
            timestamp=timestamp,
            description=description,
            config=config,
            metadata=metadata
        )
        
        # Add to registry
        self.checkpoints[checkpoint_id] = checkpoint
        self._save_metadata()
        
        print(f"✓ Checkpoint saved: {checkpoint_id}")
        print(f"  Location: {checkpoint_path}")
        print(f"  Description: {description}")
        if benchmarks:
            print(f"  Benchmarks: {benchmarks}")
        
        return checkpoint_id
    
    def restore_checkpoint(
        self,
        model: nn.Module,
        checkpoint_id: str,
        strict: bool = True
    ) -> Checkpoint:
        """
        Restore a model from a checkpoint.
        
        Args:
            model: The model to restore into
            checkpoint_id: ID of the checkpoint to restore
            strict: Whether to strictly enforce state dict matching
        
        Returns:
            The checkpoint object
        
        Raises:
            ValueError: If checkpoint doesn't exist
        
        Example:
            >>> checkpoint = manager.restore_checkpoint(model, 'checkpoint_20251106_120000')
            >>> print(f"Restored to: {checkpoint.description}")
        """
        if checkpoint_id not in self.checkpoints:
            raise ValueError(f"Checkpoint '{checkpoint_id}' not found")
        
        checkpoint = self.checkpoints[checkpoint_id]
        checkpoint_path = self.checkpoint_dir / checkpoint_id
        
        # Load state dict
        safetensors_path = checkpoint_path / 'model.safetensors'
        torch_path = checkpoint_path / 'model.pt'
        
        if safetensors_path.exists():
            state_dict = load_file(str(safetensors_path))
        elif torch_path.exists():
            state_dict = torch.load(torch_path, map_location='cpu')
        else:
            raise FileNotFoundError(f"No model file found in {checkpoint_path}")
        
        # Restore state
        model.load_state_dict(state_dict, strict=strict)
        
        print(f"✓ Checkpoint restored: {checkpoint_id}")
        print(f"  Description: {checkpoint.description}")
        print(f"  Timestamp: {checkpoint.timestamp}")
        
        return checkpoint
    
    def list_checkpoints(self, sort_by: str = 'timestamp') -> List[Checkpoint]:
        """
        List all available checkpoints.
        
        Args:
            sort_by: Sort by 'timestamp' or 'checkpoint_id'
        
        Returns:
            List of checkpoint objects
        """
        checkpoints = list(self.checkpoints.values())
        
        if sort_by == 'timestamp':
            checkpoints.sort(key=lambda c: c.timestamp, reverse=True)
        elif sort_by == 'checkpoint_id':
            checkpoints.sort(key=lambda c: c.checkpoint_id)
        
        return checkpoints
    
    def get_checkpoint_info(self, checkpoint_id: str) -> Dict[str, Any]:
        """
        Get detailed information about a checkpoint.
        
        Args:
            checkpoint_id: ID of the checkpoint
        
        Returns:
            Dictionary with checkpoint information
        """
        if checkpoint_id not in self.checkpoints:
            raise ValueError(f"Checkpoint '{checkpoint_id}' not found")
        
        checkpoint = self.checkpoints[checkpoint_id]
        checkpoint_path = self.checkpoint_dir / checkpoint_id
        
        info = checkpoint.to_dict()
        
        # Add file size information
        safetensors_path = checkpoint_path / 'model.safetensors'
        torch_path = checkpoint_path / 'model.pt'
        
        if safetensors_path.exists():
            info['file_size_mb'] = safetensors_path.stat().st_size / (1024 * 1024)
            info['format'] = 'safetensors'
        elif torch_path.exists():
            info['file_size_mb'] = torch_path.stat().st_size / (1024 * 1024)
            info['format'] = 'torch'
        
        return info
    
    def compare_checkpoints(
        self,
        checkpoint_id1: str,
        checkpoint_id2: str
    ) -> Dict[str, Any]:
        """
        Compare two checkpoints.
        
        Args:
            checkpoint_id1: First checkpoint ID
            checkpoint_id2: Second checkpoint ID
        
        Returns:
            Dictionary with comparison results
        """
        if checkpoint_id1 not in self.checkpoints:
            raise ValueError(f"Checkpoint '{checkpoint_id1}' not found")
        if checkpoint_id2 not in self.checkpoints:
            raise ValueError(f"Checkpoint '{checkpoint_id2}' not found")
        
        cp1 = self.checkpoints[checkpoint_id1]
        cp2 = self.checkpoints[checkpoint_id2]
        
        comparison = {
            'checkpoint1': {
                'id': checkpoint_id1,
                'description': cp1.description,
                'timestamp': cp1.timestamp,
                'benchmarks': cp1.metadata.get('benchmarks', {})
            },
            'checkpoint2': {
                'id': checkpoint_id2,
                'description': cp2.description,
                'timestamp': cp2.timestamp,
                'benchmarks': cp2.metadata.get('benchmarks', {})
            }
        }
        
        # Compare benchmarks if available
        bench1 = cp1.metadata.get('benchmarks', {})
        bench2 = cp2.metadata.get('benchmarks', {})
        
        if bench1 and bench2:
            benchmark_diffs = {}
            for key in set(bench1.keys()) | set(bench2.keys()):
                val1 = bench1.get(key, None)
                val2 = bench2.get(key, None)
                if val1 is not None and val2 is not None:
                    benchmark_diffs[key] = {
                        'checkpoint1': val1,
                        'checkpoint2': val2,
                        'difference': val2 - val1,
                        'percent_change': ((val2 - val1) / val1 * 100) if val1 != 0 else None
                    }
            comparison['benchmark_differences'] = benchmark_diffs
        
        return comparison
    
    def delete_checkpoint(self, checkpoint_id: str) -> None:
        """
        Delete a checkpoint.
        
        Args:
            checkpoint_id: ID of the checkpoint to delete
        
        Warning:
            This permanently deletes the checkpoint!
        """
        if checkpoint_id not in self.checkpoints:
            raise ValueError(f"Checkpoint '{checkpoint_id}' not found")
        
        checkpoint_path = self.checkpoint_dir / checkpoint_id
        
        # Remove directory
        if checkpoint_path.exists():
            shutil.rmtree(checkpoint_path)
        
        # Remove from registry
        del self.checkpoints[checkpoint_id]
        self._save_metadata()
        
        print(f"✓ Checkpoint deleted: {checkpoint_id}")
    
    def cleanup_old_checkpoints(
        self,
        keep_last: int = 10,
        keep_tagged: bool = True
    ) -> List[str]:
        """
        Clean up old checkpoints to save space.
        
        Args:
            keep_last: Number of most recent checkpoints to keep
            keep_tagged: Whether to keep checkpoints with special tags
        
        Returns:
            List of deleted checkpoint IDs
        """
        checkpoints = self.list_checkpoints(sort_by='timestamp')
        
        deleted = []
        for i, checkpoint in enumerate(checkpoints):
            # Keep recent checkpoints
            if i < keep_last:
                continue
            
            # Keep tagged checkpoints if requested
            if keep_tagged and checkpoint.metadata.get('important', False):
                continue
            
            # Delete
            try:
                self.delete_checkpoint(checkpoint.checkpoint_id)
                deleted.append(checkpoint.checkpoint_id)
            except Exception as e:
                print(f"Warning: Failed to delete {checkpoint.checkpoint_id}: {e}")
        
        return deleted
    
    def tag_checkpoint(
        self,
        checkpoint_id: str,
        tag: str,
        important: bool = False
    ) -> None:
        """
        Add a tag to a checkpoint.
        
        Args:
            checkpoint_id: ID of the checkpoint
            tag: Tag to add
            important: Mark as important (prevents auto-cleanup)
        """
        if checkpoint_id not in self.checkpoints:
            raise ValueError(f"Checkpoint '{checkpoint_id}' not found")
        
        checkpoint = self.checkpoints[checkpoint_id]
        
        if 'tags' not in checkpoint.metadata:
            checkpoint.metadata['tags'] = []
        
        if tag not in checkpoint.metadata['tags']:
            checkpoint.metadata['tags'].append(tag)
        
        if important:
            checkpoint.metadata['important'] = True
        
        self._save_metadata()
        
        print(f"✓ Tagged checkpoint '{checkpoint_id}' with '{tag}'")
    
    def get_latest_checkpoint(self) -> Optional[Checkpoint]:
        """Get the most recent checkpoint."""
        checkpoints = self.list_checkpoints(sort_by='timestamp')
        return checkpoints[0] if checkpoints else None
    
    def _load_metadata(self) -> None:
        """Load checkpoint metadata from disk."""
        if not self.metadata_file.exists():
            return
        
        try:
            with open(self.metadata_file, 'r') as f:
                data = json.load(f)
            
            for checkpoint_data in data.get('checkpoints', []):
                checkpoint = Checkpoint.from_dict(checkpoint_data)
                self.checkpoints[checkpoint.checkpoint_id] = checkpoint
        
        except Exception as e:
            print(f"Warning: Failed to load checkpoint metadata: {e}")
    
    def _save_metadata(self) -> None:
        """Save checkpoint metadata to disk."""
        data = {
            'checkpoints': [cp.to_dict() for cp in self.checkpoints.values()],
            'last_updated': datetime.now().isoformat()
        }
        
        with open(self.metadata_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def export_history(self, output_path: str) -> None:
        """
        Export checkpoint history to a file.
        
        Args:
            output_path: Path to save the history
        """
        history = []
        
        for checkpoint in self.list_checkpoints(sort_by='timestamp'):
            history.append({
                'checkpoint_id': checkpoint.checkpoint_id,
                'timestamp': checkpoint.timestamp,
                'description': checkpoint.description,
                'benchmarks': checkpoint.metadata.get('benchmarks', {}),
                'modification_details': checkpoint.metadata.get('modification_details', {}),
                'tags': checkpoint.metadata.get('tags', [])
            })
        
        with open(output_path, 'w') as f:
            json.dump({'history': history}, f, indent=2)
        
        print(f"✓ History exported to: {output_path}")
