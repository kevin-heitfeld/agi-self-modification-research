"""
WeightInspector - Read-only access to model weights

Allows the system to examine its own weights, understand weight distributions,
compare layers, and track weight changes over time.
"""

import torch
import numpy as np
import json
from typing import Dict, List, Optional, Any, Tuple, Union
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
    
    def __init__(self, model: torch.nn.Module, model_name: str = "model") -> None:
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
    
    def list_parameters(self, filter_pattern: Optional[str] = None) -> Dict[str, Any]:
        """
        Get summary of parameter names in the model with patterns.
        
        Returns a structured summary showing parameter patterns and sample names rather than
        a flat list of all parameter names. This makes it easier to understand the model
        structure at a glance and discover what parameters exist without scrolling through
        hundreds of individual names.
        
        Parameters are the actual weight/bias tensors (leaf nodes with data), not container modules.
        
        Args:
            filter_pattern: Optional string to filter parameter names (case-insensitive)
            
        Returns:
            Dictionary containing:
                - total_parameters: Total count of matching parameters
                - patterns: Dict mapping parameter patterns to counts
                - sample_names: List of ~20 example parameter names
                - note: Instructions for accessing specific parameters
            
        Example:
            >>> inspector.list_parameters(filter_pattern="attention")
            {'total_parameters': 144, 
             'patterns': {'model.layers.{N}.self_attn.q_proj.weight': 36, ...},
             'sample_names': ['model.layers.0.self_attn.q_proj.weight', ...]}
        """
        names = list(self.layers.keys())
        
        if filter_pattern:
            pattern_lower = filter_pattern.lower()
            names = [n for n in names if pattern_lower in n.lower()]
        
        names = sorted(names)
        
        # Extract patterns by replacing numbers with {N}
        import re
        pattern_counts = {}
        for name in names:
            # Replace all numbers with {N} to create a pattern
            pattern = re.sub(r'\b\d+\b', '{N}', name)
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
        
        # Return summary for better usability
        return {
            'total_parameters': len(names),
            'patterns': pattern_counts,
            'sample_names': names[:20] if len(names) > 20 else names,
            'note': 'To examine specific parameters, use get_weight_statistics() with the full parameter name. Pattern {N} represents layer indices (0-35).'
        }
    
    def get_layer_parameters(self, layer_prefix: str) -> List[str]:
        """
        Get all parameter names that belong to a specific layer/module.
        
        This is a helper function for working with container modules that have multiple parameters.
        For example, "model.layers.0" contains multiple parameters like:
        - model.layers.0.self_attn.q_proj.weight
        - model.layers.0.self_attn.q_proj.bias
        - model.layers.0.self_attn.k_proj.weight
        - etc.
        
        Args:
            layer_prefix: The layer/module name prefix (e.g., "model.layers.0")
            
        Returns:
            List of parameter names under this layer/module
            
        Example:
            >>> params = inspector.get_layer_parameters("model.layers.0")
            >>> print(f"Layer 0 has {len(params)} parameters")
            >>> # Get statistics for all of them
            >>> stats_list = inspector.get_weight_statistics(params)
        """
        # Find all parameters that start with this prefix
        matching_params = [
            name for name in self.layers.keys() 
            if name.startswith(layer_prefix + '.')
        ]
        return sorted(matching_params)
    
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
            raise KeyError(f"Parameter '{layer_name}' not found. Use introspection.weights.list_parameters() to see all available parameters.")
        
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
    
    def get_weight_statistics(self, parameter_names: Union[str, List[str]], use_cache: bool = True) -> List[Dict[str, Any]]:
        """
        Compute statistical properties of one or more parameters' weights.
        
        ALWAYS returns a list of dictionaries for consistent handling, regardless of whether
        a single parameter name (str) or multiple names (List[str]) are provided.
        
        Args:
            parameter_names: Either a single parameter name (str) or list of parameter names (List[str])
            use_cache: Whether to use cached statistics (default: True)
            
        Returns:
            List of dictionaries (one per parameter), each containing:
                - name: Parameter name
                - shape: Weight tensor shape
                - num_parameters: Number of parameters
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
                - percentiles: [5th, 25th, 50th, 75th, 95th]
                - shared_with: List of parameters sharing weights (if applicable)
                - error: Error message (only present if parameter retrieval failed)
                
        Examples:
            >>> # Single parameter - returns list with one dict
            >>> result = get_weight_statistics(parameter_names="model.layers.0.mlp.gate_proj.weight")
            >>> print(result[0]['mean'])
            0.0012
            
            >>> # Multiple parameters - returns list with multiple dicts
            >>> result = get_weight_statistics(parameter_names=[
            ...     "model.layers.0.mlp.gate_proj.weight",
            ...     "model.layers.0.mlp.up_proj.weight",
            ...     "model.layers.1.mlp.gate_proj.weight"
            ... ])
            >>> for stats in result:
            ...     print(f"{stats['name']}: mean={stats['mean']:.4f}")
        """
        # Normalize input to list
        param_list = [parameter_names] if isinstance(parameter_names, str) else parameter_names
        
        results = []
        for single_param in param_list:
            try:
                # Check cache first
                if use_cache and single_param in self._stats_cache:
                    results.append(self._stats_cache[single_param])
                    continue
                
                # Compute statistics for this parameter
                stats = self._compute_single_parameter_statistics(single_param)
                
                # Cache results
                self._stats_cache[single_param] = stats
                results.append(stats)
                
            except Exception as e:
                logger.error(f"Error getting statistics for parameter '{single_param}': {e}")
                # Get parameter names summary (now returns dict, not list)
                param_names_info = self.list_parameters()
                results.append({
                    "parameter_name": single_param,
                    "error": str(e),
                    "available_parameters_sample": param_names_info.get("sample_names", [])
                })
        
        return results
    
    def _compute_single_parameter_statistics(self, parameter_name: str) -> Dict[str, Any]:
        """
        Internal helper to compute statistics for a single parameter.
        
        Args:
            parameter_name: Name of the parameter to analyze
            
        Returns:
            Dictionary containing parameter statistics
            
        Raises:
            KeyError: If parameter_name doesn't exist
        """
        
        if parameter_name not in self.layers:
            # Check for comma-separated string error
            comma_separated_hint = ""
            if ',' in parameter_name:
                suggested_params = [name.strip() for name in parameter_name.split(',')]
                # Check if these are valid parameter names
                matching_params = [name for name in suggested_params if name in self.layers]
                
                if matching_params:
                    comma_separated_hint = (
                        f"\n\n❌ SYNTAX ERROR: You passed a comma-separated STRING, but this function requires a JSON LIST!"
                        f"\n\n🔧 WRONG (what you did):"
                        f"\n   \"parameter_names\": \"{parameter_name}\""
                        f"\n\n✅ CORRECT (what you should do):"
                        f"\n   \"parameter_names\": {json.dumps(suggested_params)}"
                        f"\n\nThe function accepts Union[str, List[str]] - that means EITHER:"
                        f"\n  - A single string: \"model.layers.0.mlp.gate_proj.weight\""
                        f"\n  - A JSON list: [\"model.layers.0.mlp.gate_proj.weight\", \"model.layers.1.mlp.gate_proj.weight\"]"
                        f"\n\nDo NOT concatenate parameter names with commas into a single string!"
                    )
            
            # Check if this is a container module (exists in model but has no weights itself)
            container_hint = ""
            try:
                # Check if this looks like a module path (could be a container)
                if parameter_name and not parameter_name.endswith('.weight') and not parameter_name.endswith('.bias'):
                    # Try to find parameters that start with this prefix
                    matching_params = [name for name in self.layers.keys() if name.startswith(parameter_name + '.')]
                    if matching_params:
                        container_hint = (
                            f"\n\n💡 HINT: '{parameter_name}' is a CONTAINER MODULE (has no weights itself)."
                            f"\n\nThis function requires PARAMETER names (leaf tensors with actual weight values)."
                            f"\n\nParameters under '{parameter_name}':"
                            f"\n{chr(10).join(f'  - {name}' for name in matching_params[:10])}"
                            f"{f'{chr(10)}  ... and {len(matching_params) - 10} more' if len(matching_params) > 10 else ''}"
                            f"\n\nTo get statistics for all weights in '{parameter_name}', use the helper function:"
                            f"\n```python"
                            f"\n# Get all parameters in this layer"
                            f"\nparams = introspection.weights.get_layer_parameters('{parameter_name}')\n"
                            f"print(f'Found {{len(params)}} parameters')\n"
                            f"\n# Get statistics for all of them"
                            f"\nstats_list = introspection.weights.get_weight_statistics(params)"
                            f"\n\n# Iterate over results"
                            f"\nfor stats in stats_list:"
                            f"\n    print(f\"{{stats['name']}}: mean={{stats['mean']:.4f}}\")"
                            f"\n```"
                        )
            except Exception:
                pass  # If detection fails, just show the basic error
            
            raise KeyError(
                f"Parameter '{parameter_name}' not found. "
                f"Use introspection.weights.list_parameters() to see all available parameters."
                f"{comma_separated_hint}"
                f"{container_hint}"
            )
        
        param = self.layers[parameter_name]
        weights = param.detach().cpu().float()
        weights_flat = weights.flatten()
        
        # Compute statistics
        stats = {
            "name": parameter_name,
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
        shared_layers = self.get_shared_layers(parameter_name)
        if shared_layers:
            stats['shared_with'] = shared_layers
            stats['warning'] = (
                f"⚠️  WEIGHT SHARING DETECTED: This parameter shares memory with "
                f"{', '.join(shared_layers)}. Modifying this parameter will also "
                f"modify the shared parameters!"
            )
            logger.warning(f"Parameter '{parameter_name}' shares weights with: {shared_layers}")
        
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
        stats1_list = self.get_weight_statistics(layer1)
        stats2_list = self.get_weight_statistics(layer2)
        
        # Extract first dict from list (since get_weight_statistics now always returns list)
        stats1 = stats1_list[0]
        stats2 = stats2_list[0]
        
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
        
        ref_stats_list = self.get_weight_statistics(reference_layer)
        ref_stats = ref_stats_list[0]  # Extract from list
        ref_weights = self.layers[reference_layer].detach().cpu().float().flatten()
        
        similarities = []
        
        for layer_name in self.layers.keys():
            if layer_name == reference_layer:
                continue
            
            try:
                stats_list = self.get_weight_statistics(layer_name)
                stats = stats_list[0]  # Extract from list
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
            
            # layer_groups_detail removed - it contains all 434 layer names
            # which creates unnecessarily verbose output
            # Use list_parameters() or get_layer_parameters() if you need specific names
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
