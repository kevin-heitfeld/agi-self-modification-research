"""
WeightInspector - Read-only access to model weights

Allows the system to examine its own weights, understand weight distributions,
compare layers, and track weight changes over time.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class WeightInspector:
    """
    Provides read-only introspective access to model weights.
    
    The system can use this to:
    - Examine weights of any layer
    - Compute weight statistics (mean, std, distribution)
    - Compare weights across layers
    - Find similar weight patterns
    - Track weight changes over time (via checkpoints)
    """
    
    def __init__(self, model: torch.nn.Module, model_name: str = "model"):
        """
        Initialize WeightInspector with a model to examine.
        
        Args:
            model: The PyTorch model to inspect
            model_name: Name for this model (for logging/tracking)
        """
        self.model = model
        self.model_name = model_name
        
        # Build layer registry
        self.layers = self._build_layer_registry()
        
        # Cache for weight statistics (computed on-demand)
        self._stats_cache = {}
        
        # Detect shared weights on initialization
        self._shared_weights = self._detect_shared_weights()
        
        logger.info(f"WeightInspector initialized for {model_name}")
        logger.info(f"Found {len(self.layers)} named parameters")
        if self._shared_weights:
            logger.warning(f"Detected {len(self._shared_weights)} groups of shared weights")
    
    def _build_layer_registry(self) -> Dict[str, torch.nn.Parameter]:
        """Build a registry of all named parameters in the model"""
        registry = {}
        for name, param in self.model.named_parameters():
            registry[name] = param
        return registry
    
    def _detect_shared_weights(self) -> Dict[int, List[str]]:
        """
        Detect weight tensors that share memory.
        
        This checks ALL module attributes (not just named_parameters) to find
        weight sharing, since PyTorch deduplicates shared parameters in named_parameters().
        
        Returns:
            Dict mapping data_ptr -> list of layer names sharing that memory
            Only includes groups with multiple layers (actual sharing)
        """
        ptr_to_names = {}
        
        # Check all module attributes to find weight sharing
        for module_name, module in self.model.named_modules():
            for attr_name in dir(module):
                try:
                    attr = getattr(module, attr_name)
                    if isinstance(attr, torch.nn.Parameter):
                        # Build full parameter name
                        if module_name:
                            full_name = f"{module_name}.{attr_name}"
                        else:
                            full_name = attr_name
                        
                        ptr = attr.data_ptr()
                        if ptr not in ptr_to_names:
                            ptr_to_names[ptr] = []
                        
                        # Avoid duplicates
                        if full_name not in ptr_to_names[ptr]:
                            ptr_to_names[ptr].append(full_name)
                except (AttributeError, TypeError):
                    # Skip non-parameter attributes
                    continue
        
        # Return only groups with multiple layers (actual sharing)
        return {
            ptr: names for ptr, names in ptr_to_names.items()
            if len(names) > 1
        }
    
    def get_shared_weights(self) -> Dict[str, List[str]]:
        """
        Get information about weight sharing in the model.
        
        Returns:
            Dictionary where each key is a representative layer name and
            value is a list of all layers sharing that tensor.
            
        Example:
            >>> inspector.get_shared_weights()
            {
                'lm_head.weight': ['lm_head.weight', 'model.embed_tokens.weight'],
            }
        """
        if not self._shared_weights:
            return {}
        
        # Convert from ptr-based to name-based for easier use
        result = {}
        for ptr, names in self._shared_weights.items():
            # Use first name as the key
            result[names[0]] = names
        
        return result
    
    def get_shared_layers(self, layer_name: str) -> List[str]:
        """
        Get list of layers that share memory with the given layer.
        
        Args:
            layer_name: Name of the layer to check
            
        Returns:
            List of layer names sharing the same tensor (excluding the input layer)
            Empty list if the layer doesn't share memory with any other layers
            
        Example:
            >>> inspector.get_shared_layers('lm_head.weight')
            ['model.embed_tokens.weight']
        """
        # First check if it's in the standard layer registry
        if layer_name in self.layers:
            param = self.layers[layer_name]
            ptr = param.data_ptr()
        else:
            # Maybe it's a name we detected but isn't in named_parameters
            # Try to find it in the shared weights
            ptr = None
            for p, names in self._shared_weights.items():
                if layer_name in names:
                    ptr = p
                    break
            
            if ptr is None:
                raise KeyError(f"Layer '{layer_name}' not found")
        
        if ptr not in self._shared_weights:
            return []
        
        # Return all layers except the one we're checking
        return [name for name in self._shared_weights[ptr] if name != layer_name]
    
    def get_layer_names(self, filter_pattern: Optional[str] = None) -> List[str]:
        """
        Get list of all layer names in the model.
        
        Args:
            filter_pattern: Optional string to filter layer names (case-insensitive)
            
        Returns:
            List of layer names
            
        Example:
            >>> inspector.get_layer_names(filter_pattern="attention")
            ['layer.0.attention.q.weight', 'layer.0.attention.k.weight', ...]
        """
        names = list(self.layers.keys())
        
        if filter_pattern:
            pattern_lower = filter_pattern.lower()
            names = [n for n in names if pattern_lower in n.lower()]
        
        return sorted(names)
    
    def get_layer_weights(self, layer_name: str) -> Dict[str, Any]:
        """
        Get weights for a specific layer.
        
        Args:
            layer_name: Name of the layer to inspect
            
        Returns:
            Dictionary containing:
                - name: Layer name
                - shape: Weight tensor shape
                - dtype: Data type
                - device: Device (cpu/cuda)
                - requires_grad: Whether gradients are enabled
                - data: The actual weight tensor (detached, no gradients)
                
        Raises:
            KeyError: If layer_name doesn't exist
        """
        if layer_name not in self.layers:
            raise KeyError(f"Layer '{layer_name}' not found. Use get_layer_names() to see available layers.")
        
        param = self.layers[layer_name]
        
        # Return detached copy (read-only, no gradients)
        return {
            "name": layer_name,
            "shape": tuple(param.shape),
            "dtype": str(param.dtype),
            "device": str(param.device),
            "requires_grad": param.requires_grad,
            "num_parameters": param.numel(),
            "data": param.detach().clone()  # Safe copy
        }
    
    def get_weight_statistics(self, layer_name: str, use_cache: bool = True) -> Dict[str, Any]:
        """
        Compute statistical properties of a layer's weights.
        
        Args:
            layer_name: Name of the layer to analyze
            use_cache: Whether to use cached statistics (default: True)
            
        Returns:
            Dictionary containing:
                - mean: Mean of all weights
                - std: Standard deviation
                - min: Minimum weight value
                - max: Maximum weight value
                - median: Median weight value
                - abs_mean: Mean of absolute values
                - zeros_percentage: Percentage of weights that are zero
                - near_zero_percentage: Percentage of weights near zero (< 0.001)
                - l1_norm: L1 norm of weights
                - l2_norm: L2 norm of weights
                - histogram: Histogram of weight values (bins and counts)
        """
        # Check cache first
        if use_cache and layer_name in self._stats_cache:
            return self._stats_cache[layer_name]
        
        if layer_name not in self.layers:
            raise KeyError(f"Layer '{layer_name}' not found.")
        
        param = self.layers[layer_name]
        weights = param.detach().cpu().float()
        weights_flat = weights.flatten()
        
        # Compute statistics
        stats = {
            "name": layer_name,
            "shape": tuple(weights.shape),
            "num_parameters": weights.numel(),
            
            # Basic statistics
            "mean": float(weights.mean()),
            "std": float(weights.std()),
            "min": float(weights.min()),
            "max": float(weights.max()),
            "median": float(weights.median()),
            "abs_mean": float(weights.abs().mean()),
            
            # Sparsity measures
            "zeros_percentage": float((weights == 0).sum() / weights.numel() * 100),
            "near_zero_percentage": float((weights.abs() < 0.001).sum() / weights.numel() * 100),
            
            # Norms
            "l1_norm": float(weights.abs().sum()),
            "l2_norm": float(torch.norm(weights, p=2)),
            "frobenius_norm": float(torch.norm(weights, p='fro')),
            
            # Distribution
            "histogram": self._compute_histogram(weights_flat),
            
            # Percentiles (computed efficiently)
            "percentiles": self._compute_percentiles(weights_flat)
        }
        
        # Add shared weight warning if applicable
        shared_layers = self.get_shared_layers(layer_name)
        if shared_layers:
            stats['shared_with'] = shared_layers
            stats['warning'] = (
                f"⚠️  WEIGHT SHARING DETECTED: This layer shares memory with "
                f"{', '.join(shared_layers)}. Modifying this layer will also "
                f"modify the shared layers!"
            )
            logger.warning(f"Layer '{layer_name}' shares weights with: {shared_layers}")
        
        # Cache results
        self._stats_cache[layer_name] = stats
        
        return stats
    
    def _compute_histogram(self, weights_flat: torch.Tensor, bins: int = 50) -> Dict[str, List]:
        """Compute histogram of weight values"""
        hist, bin_edges = torch.histogram(weights_flat, bins=bins)
        
        return {
            "bins": bins,
            "counts": hist.tolist(),
            "edges": bin_edges.tolist(),
            "bin_width": float(bin_edges[1] - bin_edges[0])
        }
    
    def _compute_percentiles(self, weights_flat: torch.Tensor) -> Dict[str, float]:
        """Compute percentiles efficiently (handles large tensors)"""
        # For very large tensors, sample to avoid memory issues
        if weights_flat.numel() > 10_000_000:
            # Sample 10M values randomly
            indices = torch.randperm(weights_flat.numel())[:10_000_000]
            weights_sample = weights_flat[indices]
        else:
            weights_sample = weights_flat
        
        # Sort for percentile calculation
        sorted_weights = torch.sort(weights_sample)[0]
        n = len(sorted_weights)
        
        return {
            "5th": float(sorted_weights[int(n * 0.05)]),
            "25th": float(sorted_weights[int(n * 0.25)]),
            "75th": float(sorted_weights[int(n * 0.75)]),
            "95th": float(sorted_weights[int(n * 0.95)]),
        }
    
    def compare_weights(self, layer1: str, layer2: str) -> Dict[str, Any]:
        """
        Compare weights between two layers.
        
        Args:
            layer1: Name of first layer
            layer2: Name of second layer
            
        Returns:
            Dictionary containing comparison metrics:
                - correlation: Correlation coefficient (if shapes match)
                - cosine_similarity: Cosine similarity (if shapes match)
                - mean_difference: Difference in means
                - std_difference: Difference in standard deviations
                - distribution_comparison: Statistical comparison
        """
        stats1 = self.get_weight_statistics(layer1)
        stats2 = self.get_weight_statistics(layer2)
        
        weights1 = self.layers[layer1].detach().cpu().float().flatten()
        weights2 = self.layers[layer2].detach().cpu().float().flatten()
        
        comparison = {
            "layer1": layer1,
            "layer2": layer2,
            "shape1": stats1["shape"],
            "shape2": stats2["shape"],
            "mean_difference": stats1["mean"] - stats2["mean"],
            "std_difference": stats1["std"] - stats2["std"],
            "l2_norm_ratio": stats1["l2_norm"] / stats2["l2_norm"] if stats2["l2_norm"] > 0 else float('inf'),
        }
        
        # Only compute these if shapes match
        if weights1.shape == weights2.shape:
            comparison["correlation"] = float(torch.corrcoef(torch.stack([weights1, weights2]))[0, 1])
            comparison["cosine_similarity"] = float(torch.nn.functional.cosine_similarity(
                weights1.unsqueeze(0), weights2.unsqueeze(0)
            ))
            comparison["euclidean_distance"] = float(torch.norm(weights1 - weights2, p=2))
            comparison["shapes_match"] = True
        else:
            comparison["shapes_match"] = False
            comparison["note"] = "Shapes don't match - only basic statistics compared"
        
        return comparison
    
    def find_similar_weights(
        self, 
        reference_layer: str, 
        top_k: int = 5,
        metric: str = "correlation"
    ) -> List[Tuple[str, float]]:
        """
        Find layers with similar weight patterns to a reference layer.
        
        Args:
            reference_layer: Name of the reference layer
            top_k: Number of similar layers to return
            metric: Similarity metric ('correlation', 'cosine', or 'l2_distance')
            
        Returns:
            List of (layer_name, similarity_score) tuples, sorted by similarity
        """
        if reference_layer not in self.layers:
            raise KeyError(f"Layer '{reference_layer}' not found.")
        
        ref_stats = self.get_weight_statistics(reference_layer)
        ref_weights = self.layers[reference_layer].detach().cpu().float().flatten()
        
        similarities = []
        
        for layer_name in self.layers.keys():
            if layer_name == reference_layer:
                continue
            
            try:
                stats = self.get_weight_statistics(layer_name)
                weights = self.layers[layer_name].detach().cpu().float().flatten()
                
                # Only compare if shapes match
                if weights.shape != ref_weights.shape:
                    continue
                
                if metric == "correlation":
                    score = float(torch.corrcoef(torch.stack([ref_weights, weights]))[0, 1])
                elif metric == "cosine":
                    score = float(torch.nn.functional.cosine_similarity(
                        ref_weights.unsqueeze(0), weights.unsqueeze(0)
                    ))
                elif metric == "l2_distance":
                    score = -float(torch.norm(ref_weights - weights, p=2))  # Negative so higher is better
                else:
                    raise ValueError(f"Unknown metric: {metric}")
                
                similarities.append((layer_name, score))
            
            except Exception as e:
                logger.warning(f"Could not compare {layer_name}: {e}")
                continue
        
        # Sort by similarity score (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def get_weight_summary(self) -> Dict[str, Any]:
        """
        Get a high-level summary of all weights in the model.
        
        Returns:
            Dictionary containing:
                - total_parameters: Total number of parameters
                - total_layers: Number of named parameters
                - overall_statistics: Statistics across all weights (computed efficiently)
                - layer_groups: Statistics grouped by layer type
        """
        total_params = 0
        layer_groups = defaultdict(list)
        
        # Compute statistics incrementally to avoid memory issues
        running_sum = 0.0
        running_sq_sum = 0.0
        running_min = float('inf')
        running_max = float('-inf')
        total_zeros = 0
        
        for name, param in self.layers.items():
            total_params += param.numel()
            
            # Group by layer type (first part of name)
            layer_type = name.split('.')[0]
            layer_groups[layer_type].append(name)
            
            # Compute statistics incrementally
            weights = param.detach().cpu().float().flatten()
            running_sum += float(weights.sum())
            running_sq_sum += float((weights ** 2).sum())
            running_min = min(running_min, float(weights.min()))
            running_max = max(running_max, float(weights.max()))
            total_zeros += int((weights == 0).sum())
        
        # Compute overall statistics
        mean = running_sum / total_params
        variance = (running_sq_sum / total_params) - (mean ** 2)
        std = variance ** 0.5
        
        summary = {
            "model_name": self.model_name,
            "total_parameters": total_params,
            "total_layers": len(self.layers),
            
            "overall_statistics": {
                "mean": mean,
                "std": std,
                "min": running_min,
                "max": running_max,
                "l2_norm_squared": running_sq_sum,  # Avoid computing full L2 norm
                "zeros_percentage": (total_zeros / total_params) * 100,
            },
            
            "layer_groups": {
                group: len(layers) 
                for group, layers in layer_groups.items()
            },
            
            "layer_groups_detail": dict(layer_groups)
        }
        
        return summary
    
    def query_weights(self, query: str) -> List[str]:
        """
        Natural language-style query for layer names.
        
        Args:
            query: Query string (e.g., "attention", "layer 5", "query weights")
            
        Returns:
            List of matching layer names
            
        Examples:
            >>> inspector.query_weights("attention")
            >>> inspector.query_weights("layer 0 self_attn")
            >>> inspector.query_weights("mlp")
        """
        query_lower = query.lower()
        matches = []
        
        for name in self.layers.keys():
            name_lower = name.lower()
            
            # Simple substring matching
            if query_lower in name_lower:
                matches.append(name)
            # Handle "layer N" queries
            elif "layer" in query_lower:
                parts = query_lower.split()
                if "layer" in parts:
                    idx = parts.index("layer")
                    if idx + 1 < len(parts) and parts[idx + 1].isdigit():
                        layer_num = parts[idx + 1]
                        if f"layers.{layer_num}" in name_lower:
                            matches.append(name)
        
        return sorted(matches)
    
    def clear_cache(self):
        """Clear the statistics cache"""
        self._stats_cache.clear()
        logger.info("WeightInspector cache cleared")
    
    def export_weights(self, layer_name: str, output_path: Path) -> None:
        """
        Export weights to a file for external analysis.
        
        Args:
            layer_name: Name of the layer to export
            output_path: Path to save the weights (will use torch.save)
        """
        if layer_name not in self.layers:
            raise KeyError(f"Layer '{layer_name}' not found.")
        
        weights_data = self.get_layer_weights(layer_name)
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save(weights_data, output_path)
        logger.info(f"Exported weights for '{layer_name}' to {output_path}")
    
    def __repr__(self) -> str:
        return f"WeightInspector(model={self.model_name}, layers={len(self.layers)})"
