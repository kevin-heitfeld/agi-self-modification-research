"""
Architecture Module - Function-based interface to ArchitectureNavigator

Provides simplified function-based access to model architecture information
for use in code execution sandbox.

Functions:
    get_architecture_summary(model) - High-level architecture overview
    describe_layer(model, layer_name) - Detailed layer description
    list_layers(model, filter_pattern) - List available layers
    get_layer_info(model, layer_name) - Layer metadata and stats
    find_similar_layers(model, layer_name) - Find architecturally similar layers

Author: AGI Self-Modification Research Team
Date: November 14, 2025
"""

from typing import Dict, List, Any, Optional
import torch.nn as nn

# Import the actual ArchitectureNavigator class
from ..introspection.architecture_navigator import ArchitectureNavigator

# Cache navigators to avoid recreating them for each call
_navigator_cache: Dict[int, ArchitectureNavigator] = {}


def _get_navigator(model: nn.Module) -> ArchitectureNavigator:
    """Get or create a cached ArchitectureNavigator for a model."""
    model_id = id(model)
    if model_id not in _navigator_cache:
        # Try to get model config if available
        model_config = None
        if hasattr(model, 'config'):
            if hasattr(model.config, 'to_dict'):
                model_config = model.config.to_dict()
        
        _navigator_cache[model_id] = ArchitectureNavigator(model, model_config)
    
    return _navigator_cache[model_id]


def get_architecture_summary(model: nn.Module) -> Dict[str, Any]:
    """
    Get a high-level summary of the model architecture.
    
    Args:
        model: PyTorch model to inspect
        
    Returns:
        Dictionary containing:
            - model_type: Type of model (e.g., 'Transformer')
            - description: Natural language description
            - total_parameters: Total parameter count
            - trainable_parameters: Trainable parameter count
            - num_layers: Number of transformer layers (if applicable)
            - hidden_size: Hidden dimension size (if applicable)
            - vocab_size: Vocabulary size (if applicable)
            - attention_heads: Number of attention heads (if applicable)
            - weight_sharing: List of shared weight groups (if detected)
    
    Example:
        >>> summary = get_architecture_summary(model)
        >>> print(f"Model type: {summary['model_type']}")
        >>> print(f"Total parameters: {summary['total_parameters']:,}")
        >>> print(f"Description: {summary['description']}")
    """
    navigator = _get_navigator(model)
    return navigator.get_architecture_summary()


def describe_layer(model: nn.Module, layer_name: str) -> Dict[str, Any]:
    """
    Get detailed description of a specific layer.
    
    Args:
        model: PyTorch model to inspect
        layer_name: Name of the layer (e.g., 'model.layers.0.self_attn')
        
    Returns:
        Dictionary containing:
            - name: Layer name
            - type: Layer type (e.g., 'Attention', 'MLP')
            - explanation: Natural language description
            - parameters: Parameter count
            - input_dim: Input dimension (if applicable)
            - output_dim: Output dimension (if applicable)
            - sub_layers: List of sub-components (if applicable)
    
    Example:
        >>> info = describe_layer(model, 'model.layers.0.self_attn')
        >>> print(info['explanation'])
        >>> print(f"Parameters: {info['parameters']:,}")
    """
    navigator = _get_navigator(model)
    result = navigator.describe_layer(layer_name)
    # Handle both single layer and list of layers
    if isinstance(result, list):
        return result[0] if result else {}
    return result


def list_layers(model: nn.Module, filter_pattern: Optional[str] = None) -> Dict[str, Any]:
    """
    Get summary of layer names in the model with patterns.
    
    Returns a structured summary showing layer patterns and sample names rather than
    a flat list of all 500+ layer/module names. This makes it easier to understand the
    model structure at a glance and discover what layers exist without scrolling through
    hundreds of individual names.
    
    Args:
        model: PyTorch model to inspect
        filter_pattern: Optional string to filter layer names (case-insensitive)
        
    Returns:
        Dictionary containing:
            - total_layers: Total count of matching layers
            - patterns: Dict mapping layer patterns to counts
            - sample_names: List of ~20 example layer names
            - note: Instructions for accessing specific layers
    
    Example:
        >>> summary = list_layers(model)
        >>> print(f"Total layers: {summary['total_layers']}")
        >>> print(f"\\nLayer patterns:")
        >>> for pattern, count in summary['patterns'].items():
        ...     print(f"  {pattern}: {count} layers")
        >>> print(f"\\nSample layer names:")
        >>> for name in summary['sample_names'][:5]:
        ...     print(f"  - {name}")
        >>>
        >>> # Filter by pattern
        >>> attn_layers = list_layers(model, filter_pattern="attn")
        >>> print(f"Found {attn_layers['total_layers']} attention layers")
    """
    # Get all named modules
    names = [name for name, _ in model.named_modules() if name]
    
    # Apply filter if provided
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
        'total_layers': len(names),
        'patterns': pattern_counts,
        'sample_names': names[:20] if len(names) > 20 else names,
        'note': 'To examine specific layers, use get_layer_info() or describe_layer() with the full layer name. Pattern {N} represents layer indices (0-35).'
    }


def get_layer_info(model: nn.Module, layer_name: str) -> Dict[str, Any]:
    """
    Get metadata and statistics for a specific layer.
    
    Args:
        model: PyTorch model to inspect
        layer_name: Name of the layer
        
    Returns:
        Dictionary containing:
            - name: Layer name
            - type: Module class name
            - parameters: Total parameters in layer
            - trainable_parameters: Trainable parameters
            - parameter_names: List of parameter names in layer
            - shape_info: Shape information for each parameter
    
    Example:
        >>> info = get_layer_info(model, 'model.layers.0')
        >>> print(f"Type: {info['type']}")
        >>> print(f"Parameters: {info['parameters']:,}")
        >>> for param_name, shape in info['shape_info'].items():
        ...     print(f"  {param_name}: {shape}")
    """
    # Get the module
    module = dict(model.named_modules()).get(layer_name)
    if module is None:
        return {'error': f"Layer '{layer_name}' not found"}
    
    # Count parameters
    total_params = sum(p.numel() for p in module.parameters())
    trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
    
    # Get parameter names and shapes
    param_names = []
    shape_info = {}
    for name, param in module.named_parameters():
        param_names.append(name)
        shape_info[name] = list(param.shape)
    
    return {
        'name': layer_name,
        'type': module.__class__.__name__,
        'parameters': total_params,
        'trainable_parameters': trainable_params,
        'parameter_names': param_names,
        'shape_info': shape_info
    }


def find_similar_layers(model: nn.Module, layer_name: str) -> List[Dict[str, Any]]:
    """
    Find layers with similar architecture to the specified layer.
    
    Args:
        model: PyTorch model to inspect
        layer_name: Name of the reference layer
        
    Returns:
        List of dictionaries with similar layers, each containing:
            - name: Layer name
            - similarity_score: Similarity metric (0-1)
            - reason: Why this layer is similar
    
    Example:
        >>> similar = find_similar_layers(model, 'model.layers.0.mlp')
        >>> for layer in similar[:5]:  # Top 5
        ...     print(f"{layer['name']}: {layer['reason']}")
    """
    # Get reference layer info
    ref_module = dict(model.named_modules()).get(layer_name)
    if ref_module is None:
        return []
    
    ref_type = ref_module.__class__.__name__
    ref_params = sum(p.numel() for p in ref_module.parameters())
    
    # Find similar layers
    similar = []
    for name, module in model.named_modules():
        if not name or name == layer_name:
            continue
        
        module_type = module.__class__.__name__
        module_params = sum(p.numel() for p in module.parameters())
        
        # Check for similarity
        reasons = []
        if module_type == ref_type:
            reasons.append(f"Same type ({module_type})")
        
        if module_params > 0 and ref_params > 0:
            param_ratio = module_params / ref_params
            if 0.9 <= param_ratio <= 1.1:
                reasons.append(f"Similar parameter count (~{module_params:,})")
        
        if reasons:
            similarity_score = len(reasons) / 2.0  # Normalize
            similar.append({
                'name': name,
                'similarity_score': min(similarity_score, 1.0),
                'reason': ', '.join(reasons)
            })
    
    # Sort by similarity score
    similar.sort(key=lambda x: x['similarity_score'], reverse=True)
    
    return similar[:20]  # Top 20


def clear_cache():
    """Clear the navigator cache. Useful for memory cleanup."""
    global _navigator_cache
    _navigator_cache.clear()
