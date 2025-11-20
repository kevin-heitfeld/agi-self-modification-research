"""
Weights Module - Function-based interface to WeightInspector

Provides simplified function-based access to model weight inspection
for use in code execution sandbox.

Functions:
    get_weight_statistics(model, param_name) - Get weight stats for a parameter
    list_parameters(model) - List all parameter names
    get_layer_parameters(model, layer_prefix) - Get all parameters under a layer
    compare_parameters(model, param1, param2) - Compare weights between parameters
    get_shared_weights(model) - Find shared weight groups
    find_similar_weights(model, param_name, top_k) - Find parameters with similar weights

Author: AGI Self-Modification Research Team
Date: November 14, 2025
"""

from typing import Dict, List, Any, Union, Optional
import torch.nn as nn

# Import the actual WeightInspector class
from ..introspection.weight_inspector import WeightInspector

# Cache inspectors to avoid recreating them for each call
_inspector_cache: Dict[int, WeightInspector] = {}


def _get_inspector(model: nn.Module) -> WeightInspector:
    """Get or create a cached WeightInspector for a model."""
    model_id = id(model)
    if model_id not in _inspector_cache:
        _inspector_cache[model_id] = WeightInspector(model)
    return _inspector_cache[model_id]


def get_weight_statistics(model: nn.Module, parameter_names: Union[str, List[str]]) -> List[Dict[str, Any]]:
    """
    Get statistical information about weights in one or more parameters.
    
    ALWAYS returns a list of dictionaries for consistent handling, regardless of whether
    a single parameter name (str) or multiple names (List[str]) are provided.
    
    Args:
        model: PyTorch model to inspect
        parameter_names: Either a single parameter name (str) or list of parameter names (List[str])
        
    Returns:
        List of dictionaries (one per parameter), each containing:
            - name: Parameter name
            - shape: Weight tensor shape
            - dtype: Data type
            - device: Device (cpu/cuda)
            - requires_grad: Whether trainable
            - num_parameters: Total number of elements
            - mean: Mean value
            - std: Standard deviation
            - min: Minimum value
            - max: Maximum value
            - l1_norm, l2_norm, frobenius_norm: Various norms
            - sparsity: Percentage of near-zero values
            - error: Error message (only present if retrieval failed)
    
    Example:
        >>> # Single parameter - returns list with one dict
        >>> result = get_weight_statistics(model, 'model.layers.0.self_attn.q_proj.weight')
        >>> print(f"Shape: {result[0]['shape']}")
        >>> print(f"Mean: {result[0]['mean']:.6f}")
        
        >>> # Multiple parameters - returns list with multiple dicts
        >>> params = ['model.layers.0.self_attn.q_proj.weight',
        ...           'model.layers.0.self_attn.k_proj.weight']
        >>> result = get_weight_statistics(model, params)
        >>> for stats in result:
        ...     print(f"{stats['name']}: mean={stats['mean']:.4f}")
    """
    inspector = _get_inspector(model)
    result = inspector.get_weight_statistics(parameter_names)
    
    # Check for errors and raise exceptions instead of returning error dicts
    for item in result:
        if isinstance(item, dict) and 'error' in item:
            raise ValueError(
                f"Error getting statistics for '{item.get('parameter_name', 'unknown')}': {item['error']}"
            )
    
    return result


def list_parameters(model: nn.Module, filter_pattern: Optional[str] = None) -> Dict[str, Any]:
    """
    Get summary of parameter names in the model with patterns.
    
    Returns a structured summary showing parameter patterns and sample names rather than
    a flat list of all 434+ parameter names. This makes it easier to understand the model
    structure at a glance and discover what parameters exist without scrolling through
    hundreds of individual names.
    
    Parameters are the actual weight/bias tensors (leaf nodes with data), not container modules.
    
    Args:
        model: PyTorch model to inspect
        filter_pattern: Optional string to filter parameter names (case-insensitive)
        
    Returns:
        Dictionary containing:
            - total_parameters: Total count of matching parameters
            - patterns: Dict mapping parameter patterns to counts
            - sample_names: List of ~20 example parameter names
            - note: Instructions for accessing specific parameters
        
    Example:
        >>> summary = list_parameters(model)
        >>> print(f"Total parameters: {summary['total_parameters']}")
        >>> print(f"\\nParameter patterns:")
        >>> for pattern, count in summary['patterns'].items():
        ...     print(f"  {pattern}: {count} parameters")
        >>> print(f"\\nSample parameter names:")
        >>> for name in summary['sample_names'][:5]:
        ...     print(f"  - {name}")
        >>>
        >>> # Filter by pattern
        >>> attn_params = list_parameters(model, filter_pattern="attention")
        >>> print(f"Found {attn_params['total_parameters']} attention parameters")
    """
    inspector = _get_inspector(model)
    return inspector.list_parameters(filter_pattern=filter_pattern)


def get_layer_parameters(model: nn.Module, layer_prefix: str) -> List[str]:
    """
    Get all parameter names under a specific layer or module.
    
    This is a convenience function for dealing with container modules
    (like 'model.layers.0') which don't have weights themselves but
    contain multiple parameters.
    
    Args:
        model: PyTorch model to inspect
        layer_prefix: Layer prefix to search for (e.g., 'model.layers.0')
        
    Returns:
        List of parameter names that start with the given prefix
    
    Example:
        >>> # Get all parameters in layer 0
        >>> params = get_layer_parameters(model, 'model.layers.0')
        >>> print(f"Found {len(params)} parameters in layer 0")
        >>> for param in params:
        ...     print(param)
        
        >>> # Then get statistics for all of them
        >>> stats = get_weight_statistics(model, params)
    """
    all_params = list_parameters(model)
    # Add trailing dot to ensure we match 'model.layers.0.xxx' not 'model.layers.01.xxx'
    prefix_with_dot = layer_prefix if layer_prefix.endswith('.') else layer_prefix + '.'
    return [p for p in all_params if p.startswith(prefix_with_dot)]


def compare_parameters(model: nn.Module, param1: str, param2: str) -> Dict[str, Any]:
    """
    Compare weight statistics between two parameters.
    
    Args:
        model: PyTorch model to inspect
        param1: First parameter name
        param2: Second parameter name
        
    Returns:
        Dictionary containing:
            - param1_stats: Statistics for first parameter
            - param2_stats: Statistics for second parameter
            - comparison: Comparison metrics
                - mean_diff: Difference in means
                - std_ratio: Ratio of standard deviations
                - shape_match: Whether shapes match
                - correlation: Correlation coefficient (if shapes match)
    
    Example:
        >>> comp = compare_parameters(model, 
        ...     'model.layers.0.self_attn.q_proj.weight',
        ...     'model.layers.1.self_attn.q_proj.weight')
        >>> print(f"Mean difference: {comp['comparison']['mean_diff']:.6f}")
        >>> print(f"Correlation: {comp['comparison']['correlation']:.4f}")
    """
    inspector = _get_inspector(model)
    return inspector.compare_weights(param1, param2)


def get_shared_weights(model: nn.Module) -> Dict[str, List[str]]:
    """
    Find groups of parameters that share memory (tied weights).
    
    Args:
        model: PyTorch model to inspect
        
    Returns:
        Dictionary mapping group ID to list of layer names sharing weights.
        Only returns groups with 2+ members (actual weight sharing).
    
    Example:
        >>> shared = get_shared_weights(model)
        >>> for group_id, layers in shared.items():
        ...     print(f"Weight sharing group {group_id}:")
        ...     for layer in layers:
        ...         print(f"  - {layer}")
    """
    inspector = _get_inspector(model)
    # Convert ptr-based dict to string keys
    shared_dict = {}
    for ptr, layers in inspector._shared_weights.items():
        shared_dict[f"group_{ptr}"] = layers
    return shared_dict


def find_similar_weights(model: nn.Module, layer_name: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Find layers with statistically similar weights to the specified layer.
    
    Args:
        model: PyTorch model to inspect
        layer_name: Reference layer name
        top_k: Number of most similar layers to return
        
    Returns:
        List of dictionaries with similar layers, each containing:
            - name: Layer name
            - similarity_score: Similarity metric (0-1)
            - metrics: Statistical comparison metrics
    
    Example:
        >>> similar = find_similar_weights(model, 'model.layers.0.mlp.fc1.weight')
        >>> for layer in similar:
        ...     print(f"{layer['name']}: similarity={layer['similarity_score']:.4f}")
    """
    inspector = _get_inspector(model)
    
    # Get reference stats (now always returns list)
    try:
        result = inspector.get_weight_statistics(layer_name)
        ref_stats = result[0]  # Extract first dict from list
    except:
        return []
    
    # Compare with all other layers
    similar = []
    for other_name in inspector.layers.keys():
        if other_name == layer_name:
            continue
        
        try:
            result = inspector.get_weight_statistics(other_name)
            other_stats = result[0]  # Extract first dict from list
            
            # Calculate similarity based on statistical properties
            # Only compare if shapes match
            if ref_stats['shape'] != other_stats['shape']:
                continue
            
            mean_diff = abs(ref_stats['mean'] - other_stats['mean'])
            std_diff = abs(ref_stats['std'] - other_stats['std'])
            
            # Normalize differences
            mean_sim = 1.0 / (1.0 + mean_diff)
            std_sim = 1.0 / (1.0 + std_diff)
            
            similarity_score = (mean_sim + std_sim) / 2.0
            
            similar.append({
                'name': other_name,
                'similarity_score': similarity_score,
                'metrics': {
                    'mean_diff': mean_diff,
                    'std_diff': std_diff,
                    'mean_sim': mean_sim,
                    'std_sim': std_sim
                }
            })
        except:
            continue
    
    # Sort by similarity and return top_k
    similar.sort(key=lambda x: x['similarity_score'], reverse=True)
    return similar[:top_k]


def clear_cache():
    """Clear the inspector cache. Useful for memory cleanup."""
    global _inspector_cache
    _inspector_cache.clear()
