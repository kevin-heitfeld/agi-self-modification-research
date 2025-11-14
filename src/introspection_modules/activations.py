"""
Activations Module - Function-based interface to ActivationMonitor

Provides simplified function-based access to activation monitoring
for use in code execution sandbox.

Functions:
    capture_activations(model, tokenizer, text, layer_names) - Capture activations for text
    get_activation_statistics(model, tokenizer, layer_name) - Get activation stats
    list_layers(model, filter_pattern) - List available layers
    clear_cache() - Clear activation cache

Author: AGI Self-Modification Research Team
Date: November 14, 2025
"""

from typing import Dict, List, Any, Optional
import torch.nn as nn

# Import the actual ActivationMonitor class
from ..introspection.activation_monitor import ActivationMonitor

# Cache monitors to avoid recreating them for each call
_monitor_cache: Dict[int, ActivationMonitor] = {}


def _get_monitor(model: nn.Module, tokenizer: Any) -> ActivationMonitor:
    """Get or create a cached ActivationMonitor for a model."""
    model_id = id(model)
    if model_id not in _monitor_cache:
        _monitor_cache[model_id] = ActivationMonitor(model, tokenizer)
    return _monitor_cache[model_id]


def capture_activations(
    model: nn.Module,
    tokenizer: Any,
    text: str,
    layer_names: List[str]
) -> Dict[str, Dict[str, Any]]:
    """
    Capture activations from specified layers while processing text.
    
    Args:
        model: PyTorch model to monitor
        tokenizer: Tokenizer for the model
        text: Input text to process
        layer_names: List of layer names to capture (e.g., ['model.layers.0', 'model.layers.1'])
    
    Returns:
        Dictionary mapping layer names to activation info:
            - shape: Activation tensor shape [batch, seq_len, hidden_dim]
            - mean: Mean activation value
            - std: Standard deviation
            - max: Maximum activation
            - min: Minimum activation
            - sparsity: Percentage of near-zero activations
    
    Example:
        >>> activations = capture_activations(
        ...     model, tokenizer, 
        ...     "Hello world",
        ...     ['model.layers.0', 'model.layers.5']
        ... )
        >>> for layer, stats in activations.items():
        ...     print(f"{layer}: shape={stats['shape']}, mean={stats['mean']:.4f}")
    """
    monitor = _get_monitor(model, tokenizer)
    # First, capture activations for the specified layers
    monitor.capture_activations(text, layer_names)
    
    # Then get statistics for each layer
    result = {}
    for layer_name in layer_names:
        stats = monitor.get_activation_statistics(layer_name)
        # Handle both single and list returns
        if isinstance(stats, list):
            result[layer_name] = stats[0] if stats else {}
        else:
            result[layer_name] = stats
    
    return result


def get_activation_statistics(
    model: nn.Module,
    tokenizer: Any,
    layer_name: str
) -> Dict[str, Any]:
    """
    Get activation statistics for a single layer (uses cached activations if available).
    
    Args:
        model: PyTorch model to monitor
        tokenizer: Tokenizer for the model
        layer_name: Name of the layer
        
    Returns:
        Dictionary containing:
            - shape: Activation tensor shape
            - mean: Mean activation value
            - std: Standard deviation
            - max: Maximum activation
            - min: Minimum activation
            - sparsity: Percentage of near-zero activations
            - from_cache: Whether this came from cached activations
    
    Example:
        >>> stats = get_activation_statistics(model, tokenizer, 'model.layers.0')
        >>> print(f"Mean: {stats['mean']:.4f}")
        >>> print(f"From cache: {stats['from_cache']}")
    """
    monitor = _get_monitor(model, tokenizer)
    result = monitor.get_activation_statistics(layer_name)
    # Handle both single and list returns
    if isinstance(result, list):
        return result[0] if result else {}
    return result


def list_layers(model: nn.Module, filter_pattern: Optional[str] = None) -> List[str]:
    """
    List all layer names available for activation monitoring.
    
    Args:
        model: PyTorch model to inspect
        filter_pattern: Optional string to filter layer names
                       (case-insensitive substring match)
    
    Returns:
        List of layer names (sorted)
    
    Example:
        >>> # Get all layers
        >>> layers = list_layers(model)
        >>> print(f"Total layers: {len(layers)}")
        >>>
        >>> # Get only attention layers
        >>> attn_layers = list_layers(model, 'attn')
        >>> print(f"Attention layers: {attn_layers}")
    """
    # Get all named modules
    names = [name for name, _ in model.named_modules() if name]
    
    # Apply filter if provided
    if filter_pattern:
        pattern_lower = filter_pattern.lower()
        names = [n for n in names if pattern_lower in n.lower()]
    
    return sorted(names)


def clear_cache():
    """
    Clear activation cache and monitors. 
    
    Useful for freeing memory after activation analysis.
    
    Example:
        >>> clear_cache()  # Free activation memory
    """
    global _monitor_cache
    # Clear internal caches in each monitor
    for monitor in _monitor_cache.values():
        monitor.clear_activations(force=True)
        monitor.clear_hooks()
    # Clear monitor cache
    _monitor_cache.clear()
