"""
Activations Module - Function-based interface to ActivationMonitor

Provides simplified function-based access to activation monitoring
for use in code execution sandbox.

Functions:
    capture_activations(model, tokenizer, text, layer_names) - Capture activations for text
    capture_attention_weights(model, tokenizer, text, layer_names) - Capture WITH attention weights
    get_activation_statistics(model, tokenizer, layer_name) - Get activation stats
    get_input_shape(model, tokenizer, sample_text) - Get input shape and tokenization info
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


def capture_attention_weights(
    model: nn.Module,
    tokenizer: Any,
    text: str,
    layer_names: List[str]
) -> Dict[str, Dict[str, Any]]:
    """
    Capture activations WITH attention weights by temporarily disabling Flash Attention.
    
    **WARNING**: This is SLOWER than capture_activations() because it temporarily
    switches from Flash Attention 2 (fast, fused) to eager attention (standard, 
    materializes full attention matrices).
    
    Flash Attention 2 uses kernel fusion and never materializes the full attention
    matrix, which is why it's fast. This function temporarily disables it to capture
    attention weights for analysis.
    
    Args:
        model: PyTorch model to monitor
        tokenizer: Tokenizer for the model
        text: Input text to process
        layer_names: List of layer names to capture
    
    Returns:
        Dictionary mapping layer names to activation/attention info:
            - shape: Activation tensor shape [batch, seq_len, hidden_dim]
            - mean: Mean activation value
            - std: Standard deviation
            - max: Maximum activation
            - min: Minimum activation
            - sparsity: Percentage of near-zero activations
            - attention_shape: Shape of attention weights [batch, heads, seq_len, seq_len] (if available)
            - attention_mean: Mean attention weight (if available)
            - attention_std: Standard deviation of attention (if available)
    
    Example:
        >>> result = capture_attention_weights(
        ...     model, tokenizer,
        ...     "Hello world",
        ...     ['model.layers.0.self_attn', 'model.layers.5.self_attn']
        ... )
        >>> for layer, stats in result.items():
        ...     print(f"{layer}: shape={stats['shape']}, attn_shape={stats.get('attention_shape', 'N/A')}")
    """
    monitor = _get_monitor(model, tokenizer)
    raw_result = monitor.capture_attention_weights(text, layer_names)
    
    # Convert to same format as capture_activations for consistency
    result = {}
    for layer_name in layer_names:
        stats = monitor.get_activation_statistics(layer_name)
        # Handle both single and list returns
        if isinstance(stats, list):
            layer_stats = stats[0] if stats else {}
        else:
            layer_stats = stats
        
        # Add attention weight info if available
        if layer_name in raw_result.get('attention_weights', {}):
            import torch
            attn_tensor = raw_result['attention_weights'][layer_name]
            layer_stats['attention_shape'] = list(attn_tensor.shape)
            layer_stats['attention_mean'] = float(attn_tensor.mean().item())
            layer_stats['attention_std'] = float(attn_tensor.std().item())
            layer_stats['attention_max'] = float(attn_tensor.max().item())
            layer_stats['attention_min'] = float(attn_tensor.min().item())
        
        result[layer_name] = layer_stats
    
    return result


def get_input_shape(
    model: nn.Module,
    tokenizer: Any,
    sample_text: str = "test"
) -> Dict[str, Any]:
    """
    Get information about input shape and tokenization.
    
    Useful for understanding how inputs are processed and what dimensions
    activations will have.
    
    Args:
        model: PyTorch model
        tokenizer: Tokenizer for the model
        sample_text: Sample text to tokenize (default: "test")
    
    Returns:
        Dictionary containing:
            - sample_text: The input text used
            - num_tokens: Number of tokens after tokenization
            - token_ids: The actual token IDs
            - token_strings: Human-readable token strings
            - input_shape: Shape of input tensors [batch_size, sequence_length]
            - hidden_size: Size of hidden states (from model config)
            - note: Explanation of dimensions
    
    Example:
        >>> shape_info = get_input_shape(model, tokenizer, "Hello world")
        >>> print(f"Tokens: {shape_info['num_tokens']}")
        >>> print(f"Hidden size: {shape_info['hidden_size']}")
        >>> print(f"Expected activation shape: [1, {shape_info['num_tokens']}, {shape_info['hidden_size']}]")
    """
    monitor = _get_monitor(model, tokenizer)
    return monitor.get_input_shape(sample_text)


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
