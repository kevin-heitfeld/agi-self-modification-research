"""
Weights Module - Function-based interface to WeightInspector

Provides simplified function-based access to model weight inspection
for use in code execution sandbox.

Functions:
    get_weight_statistics(model, layer_name) - Get weight stats for a layer
    list_layers(model) - List all layers with weights
    compare_layers(model, layer1, layer2) - Compare weights between layers
    get_shared_weights(model) - Find shared weight groups
    find_similar_weights(model, layer_name, top_k) - Find layers with similar weights

Author: AGI Self-Modification Research Team
Date: November 14, 2025
"""

from typing import Dict, List, Any, Union
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


def get_weight_statistics(model: nn.Module, layer_name: Union[str, List[str]]) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Get statistical information about weights in one or more parameters.
    
    Args:
        model: PyTorch model to inspect
        layer_name: Either:
                   - A single parameter name (str) - returns dict for that parameter
                   - A list of parameter names (List[str]) - returns list of dicts
        
    Returns:
        If layer_name is a string:
            Dictionary containing:
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
        
        If layer_name is a list:
            List of dicts (one per parameter) with the same structure as above.
    
    Example:
        >>> # Single parameter
        >>> stats = get_weight_statistics(model, 'model.layers.0.self_attn.q_proj.weight')
        >>> print(f"Shape: {stats['shape']}")
        >>> print(f"Mean: {stats['mean']:.6f}, Std: {stats['std']:.6f}")
        
        >>> # Multiple parameters at once
        >>> params = ['model.layers.0.self_attn.q_proj.weight',
        ...           'model.layers.0.self_attn.k_proj.weight']
        >>> stats_list = get_weight_statistics(model, params)
        >>> for stats in stats_list:
        ...     print(f"{stats['name']}: mean={stats['mean']:.4f}")
    """
    inspector = _get_inspector(model)
    result = inspector.get_weight_statistics(layer_name)
    
    # Check for errors and raise exceptions instead of returning error dicts
    if isinstance(result, list):
        # Check each result for errors
        for item in result:
            if isinstance(item, dict) and 'error' in item:
                raise ValueError(
                    f"Error getting statistics for '{item.get('layer_name', 'unknown')}': {item['error']}"
                )
        return result
    elif isinstance(result, dict) and 'error' in result:
        raise ValueError(
            f"Error getting statistics for '{result.get('layer_name', layer_name)}': {result['error']}"
        )
    
    return result


def list_layers(model: nn.Module) -> List[str]:
    """
    List all parameter names in the model.
    
    Args:
        model: PyTorch model to inspect
        
    Returns:
        List of parameter names (sorted)
    
    Example:
        >>> params = list_layers(model)
        >>> print(f"Total parameters: {len(params)}")
        >>> for param in params[:5]:
        ...     print(param)
    """
    inspector = _get_inspector(model)
    return sorted(inspector.layers.keys())


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
    all_params = list_layers(model)
    # Add trailing dot to ensure we match 'model.layers.0.xxx' not 'model.layers.01.xxx'
    prefix_with_dot = layer_prefix if layer_prefix.endswith('.') else layer_prefix + '.'
    return [p for p in all_params if p.startswith(prefix_with_dot)]


def compare_layers(model: nn.Module, layer1: str, layer2: str) -> Dict[str, Any]:
    """
    Compare weight statistics between two layers.
    
    Args:
        model: PyTorch model to inspect
        layer1: First layer name
        layer2: Second layer name
        
    Returns:
        Dictionary containing:
            - layer1_stats: Statistics for first layer
            - layer2_stats: Statistics for second layer
            - comparison: Comparison metrics
                - mean_diff: Difference in means
                - std_ratio: Ratio of standard deviations
                - shape_match: Whether shapes match
                - correlation: Correlation coefficient (if shapes match)
    
    Example:
        >>> comp = compare_layers(model, 
        ...     'model.layers.0.self_attn.q_proj.weight',
        ...     'model.layers.1.self_attn.q_proj.weight')
        >>> print(f"Mean difference: {comp['comparison']['mean_diff']:.6f}")
        >>> print(f"Correlation: {comp['comparison']['correlation']:.4f}")
    """
    inspector = _get_inspector(model)
    return inspector.compare_weights(layer1, layer2)


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
    
    # Get reference stats
    try:
        result = inspector.get_weight_statistics(layer_name)
        ref_stats = result if isinstance(result, dict) else result[0]
    except:
        return []
    
    # Compare with all other layers
    similar = []
    for other_name in inspector.layers.keys():
        if other_name == layer_name:
            continue
        
        try:
            result = inspector.get_weight_statistics(other_name)
            other_stats = result if isinstance(result, dict) else result[0]
            
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
