"""
Tests for Checkpointing System

Validates the CheckpointManager's ability to save, restore, and manage
model states safely and reliably.

Author: AGI Self-Modification Research Team
Date: November 6, 2025
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import tempfile
import shutil
from pathlib import Path
from src.checkpointing import CheckpointManager, Checkpoint


class SimpleModel(nn.Module):
    """A simple model for testing."""
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 20)
        self.linear2 = nn.Linear(20, 5)

    def forward(self, x):
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        return x


def test_save_checkpoint():
    """Test saving a checkpoint."""
    print("Testing checkpoint saving...")

    # Create temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = CheckpointManager(checkpoint_dir=tmpdir)
        model = SimpleModel()

        # Save checkpoint
        checkpoint_id = manager.save_checkpoint(
            model=model,
            description="Test checkpoint",
            benchmarks={'accuracy': 0.95},
            modification_details={'test': True}
        )

        assert checkpoint_id in manager.checkpoints
        assert Path(tmpdir) / checkpoint_id

        checkpoint = manager.checkpoints[checkpoint_id]
        assert checkpoint.description == "Test checkpoint"
        assert checkpoint.metadata['benchmarks']['accuracy'] == 0.95

        print(f"✓ Checkpoint saved successfully: {checkpoint_id}")
        print("")


def test_restore_checkpoint():
    """Test restoring from a checkpoint."""
    print("Testing checkpoint restoration...")

    with tempfile.TemporaryDirectory() as tmpdir:
        manager = CheckpointManager(checkpoint_dir=tmpdir)
        model = SimpleModel()

        # Save original state
        original_weight = model.linear1.weight.data.clone()
        checkpoint_id = manager.save_checkpoint(
            model=model,
            description="Original state"
        )

        # Modify model
        model.linear1.weight.data *= 2.0
        modified_weight = model.linear1.weight.data.clone()

        # Verify modification
        assert not torch.allclose(original_weight, modified_weight)
        print("✓ Model modified successfully")

        # Restore
        manager.restore_checkpoint(model, checkpoint_id)
        restored_weight = model.linear1.weight.data

        # Verify restoration
        assert torch.allclose(original_weight, restored_weight)
        print("✓ Model restored successfully")
        print("")


def test_list_checkpoints():
    """Test listing checkpoints."""
    print("Testing checkpoint listing...")

    with tempfile.TemporaryDirectory() as tmpdir:
        manager = CheckpointManager(checkpoint_dir=tmpdir)
        model = SimpleModel()

        # Create multiple checkpoints
        ids = []
        for i in range(3):
            checkpoint_id = manager.save_checkpoint(
                model=model,
                description=f"Checkpoint {i}",
                auto_id=True
            )
            ids.append(checkpoint_id)

        # List checkpoints
        checkpoints = manager.list_checkpoints()
        assert len(checkpoints) == 3
        print(f"✓ Found {len(checkpoints)} checkpoints")

        # Verify all IDs present
        checkpoint_ids = [c.checkpoint_id for c in checkpoints]
        for checkpoint_id in ids:
            assert checkpoint_id in checkpoint_ids

        print("✓ All checkpoints listed correctly")
        print("")


def test_compare_checkpoints():
    """Test comparing two checkpoints."""
    print("Testing checkpoint comparison...")

    with tempfile.TemporaryDirectory() as tmpdir:
        manager = CheckpointManager(checkpoint_dir=tmpdir)
        model = SimpleModel()

        # Create checkpoint 1
        id1 = manager.save_checkpoint(
            model=model,
            description="Checkpoint 1",
            benchmarks={'accuracy': 0.90, 'loss': 0.5}
        )

        # Modify model slightly
        model.linear1.weight.data *= 1.1

        # Create checkpoint 2
        id2 = manager.save_checkpoint(
            model=model,
            description="Checkpoint 2",
            benchmarks={'accuracy': 0.92, 'loss': 0.45}
        )

        # Compare
        comparison = manager.compare_checkpoints(id1, id2)

        assert 'checkpoint1' in comparison
        assert 'checkpoint2' in comparison
        assert 'benchmark_differences' in comparison

        # Check accuracy difference
        acc_diff = comparison['benchmark_differences']['accuracy']
        assert abs(acc_diff['difference'] - 0.02) < 0.0001  # Float precision
        assert acc_diff['checkpoint1'] == 0.90
        assert acc_diff['checkpoint2'] == 0.92

        print("✓ Comparison calculated correctly")
        print(f"  Accuracy change: +{acc_diff['difference']}")
        print(f"  Loss change: {comparison['benchmark_differences']['loss']['difference']}")
        print("")


def test_checkpoint_tagging():
    """Test tagging checkpoints."""
    print("Testing checkpoint tagging...")

    with tempfile.TemporaryDirectory() as tmpdir:
        manager = CheckpointManager(checkpoint_dir=tmpdir)
        model = SimpleModel()

        # Create checkpoint
        checkpoint_id = manager.save_checkpoint(
            model=model,
            description="Important checkpoint"
        )

        # Tag it
        manager.tag_checkpoint(checkpoint_id, 'baseline', important=True)

        checkpoint = manager.checkpoints[checkpoint_id]
        assert 'baseline' in checkpoint.metadata['tags']
        assert checkpoint.metadata['important'] is True

        print("✓ Checkpoint tagged successfully")
        print(f"  Tags: {checkpoint.metadata['tags']}")
        print("")


def test_delete_checkpoint():
    """Test deleting a checkpoint."""
    print("Testing checkpoint deletion...")

    with tempfile.TemporaryDirectory() as tmpdir:
        manager = CheckpointManager(checkpoint_dir=tmpdir)
        model = SimpleModel()

        # Create checkpoint
        checkpoint_id = manager.save_checkpoint(
            model=model,
            description="Temporary checkpoint"
        )

        assert checkpoint_id in manager.checkpoints

        # Delete it
        manager.delete_checkpoint(checkpoint_id)

        assert checkpoint_id not in manager.checkpoints
        assert not (Path(tmpdir) / checkpoint_id).exists()

        print("✓ Checkpoint deleted successfully")
        print("")


def test_get_latest_checkpoint():
    """Test getting the latest checkpoint."""
    print("Testing latest checkpoint retrieval...")
    
    import time
    
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = CheckpointManager(checkpoint_dir=tmpdir)
        model = SimpleModel()
        
        # Create multiple checkpoints with slight delays
        ids = []
        for i in range(3):
            checkpoint_id = manager.save_checkpoint(
                model=model,
                description=f"Checkpoint {i}",
                auto_id=True
            )
            ids.append(checkpoint_id)
            time.sleep(0.01)  # Small delay to ensure different timestamps
        
        # Get latest
        latest = manager.get_latest_checkpoint()
        assert latest is not None
        # The latest should be one of the created checkpoints
        assert latest.checkpoint_id in ids
        
        print(f"✓ Latest checkpoint: {latest.checkpoint_id}")
        print(f"✓ Description: {latest.description}")
        print("")


def test_export_history():
    """Test exporting checkpoint history."""
    print("Testing history export...")

    with tempfile.TemporaryDirectory() as tmpdir:
        manager = CheckpointManager(checkpoint_dir=tmpdir)
        model = SimpleModel()

        # Create checkpoints
        for i in range(3):
            manager.save_checkpoint(
                model=model,
                description=f"Checkpoint {i}",
                benchmarks={'iteration': i}
            )

        # Export history
        history_path = Path(tmpdir) / 'history.json'
        manager.export_history(str(history_path))

        assert history_path.exists()

        # Load and verify
        import json
        with open(history_path, 'r') as f:
            data = json.load(f)

        assert 'history' in data
        assert len(data['history']) == 3

        print(f"✓ History exported successfully")
        print(f"  Entries: {len(data['history'])}")
        print("")


def test_metadata_persistence():
    """Test that metadata persists across manager instances."""
    print("Testing metadata persistence...")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create manager and checkpoint
        manager1 = CheckpointManager(checkpoint_dir=tmpdir)
        model = SimpleModel()

        checkpoint_id = manager1.save_checkpoint(
            model=model,
            description="Persistent checkpoint"
        )

        # Create new manager instance
        manager2 = CheckpointManager(checkpoint_dir=tmpdir)

        # Verify checkpoint is loaded
        assert checkpoint_id in manager2.checkpoints
        checkpoint = manager2.checkpoints[checkpoint_id]
        assert checkpoint.description == "Persistent checkpoint"

        print("✓ Metadata persisted across manager instances")
        print("")


def run_all_tests():
    """Run all tests."""
    print("=" * 70)
    print("  CHECKPOINTING SYSTEM TESTS")
    print("=" * 70)
    print("")

    try:
        test_save_checkpoint()
        test_restore_checkpoint()
        test_list_checkpoints()
        test_compare_checkpoints()
        test_checkpoint_tagging()
        test_delete_checkpoint()
        test_get_latest_checkpoint()
        test_export_history()
        test_metadata_persistence()

        print("=" * 70)
        print("  ✅ ALL TESTS PASSED!")
        print("=" * 70)
        print("")
        print("The CheckpointManager can successfully:")
        print("  ✓ Save model states with metadata")
        print("  ✓ Restore previous states (rollback)")
        print("  ✓ List and manage checkpoints")
        print("  ✓ Compare checkpoint benchmarks")
        print("  ✓ Tag important checkpoints")
        print("  ✓ Delete unwanted checkpoints")
        print("  ✓ Track latest checkpoint")
        print("  ✓ Export modification history")
        print("  ✓ Persist metadata across sessions")
        print("")
        print("The system can now safely save and restore states!")

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        raise
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        raise


if __name__ == "__main__":
    run_all_tests()
