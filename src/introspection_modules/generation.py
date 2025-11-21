"""
Generation Introspection Module - Access to generation-time data

Provides access to data captured during generation that the model
cannot obtain by re-processing text (attention weights, KV cache stats, etc.)

Functions:
    get_last_generation_attention() - Get attention weights from last generation
    get_cache_statistics() - Get H2O cache statistics
    get_token_importance_scores() - Get cumulative attention scores per token

Author: AGI Self-Modification Research Team
Date: November 21, 2025
"""

from typing import Dict, List, Any, Optional, Tuple
import torch
import logging

logger = logging.getLogger(__name__)


def get_last_generation_attention(
    manual_generator: Any,
    layer_indices: Optional[List[int]] = None,
    aggregate: str = "none"
) -> Dict[str, Any]:
    """
    Get attention weights captured during the last generation.
    
    This provides access to attention patterns from the actual generation
    process, without requiring the model to re-process text.
    
    Args:
        manual_generator: The ManualGenerator instance (injected by sandbox)
        layer_indices: Which layers to return (None = all layers)
        aggregate: How to aggregate attention weights:
                  - "none": Return full tensors
                  - "mean": Average across layers/heads
                  - "sum": Sum across layers/heads
    
    Returns:
        Dictionary containing:
            - available: bool - Whether attention weights are available
            - num_layers: int - Number of layers
            - num_heads: int - Number of attention heads per layer
            - query_length: int - Number of query tokens
            - key_length: int - Number of key tokens
            - attention_weights: List of tensors (one per layer) or aggregated tensor
            - shape_info: Description of tensor shapes
    
    Example:
        >>> # Get attention from last generation
        >>> attn = introspection.generation.get_last_generation_attention()
        >>> if attn['available']:
        ...     print(f"Attention shape: {attn['shape_info']}")
        ...     # Analyze which tokens got attention
        ...     for layer_idx, layer_attn in enumerate(attn['attention_weights']):
        ...         print(f"Layer {layer_idx} attention pattern: {layer_attn.shape}")
        
        >>> # Get aggregated attention across all layers
        >>> attn = introspection.generation.get_last_generation_attention(aggregate='mean')
        >>> # attn['attention_weights'] is now [query_len, key_len]
    """
    # Check if H2O cache manager exists and has attention weights
    if not hasattr(manual_generator, 'h2o_cache'):
        return {
            "available": False,
            "error": "H2O cache manager not initialized. Set enable_h2o_eviction=True."
        }
    
    h2o_cache = manual_generator.h2o_cache
    attention_weights = h2o_cache.get_last_attention_weights()
    
    if attention_weights is None:
        return {
            "available": False,
            "error": "No attention weights captured. Make sure generation happened with output_attentions=True."
        }
    
    # Extract metadata
    num_layers = len(attention_weights)
    if num_layers == 0:
        return {"available": False, "error": "Empty attention weights"}
    
    first_layer = attention_weights[0]
    # Shape: [batch, num_heads, query_len, key_len]
    batch_size, num_heads, query_len, key_len = first_layer.shape
    
    # Filter layers if requested
    if layer_indices is not None:
        attention_weights = tuple(attention_weights[i] for i in layer_indices if i < num_layers)
        num_layers = len(attention_weights)
    
    # Aggregate if requested
    if aggregate == "mean":
        # Stack and average across layers and heads: [query_len, key_len]
        stacked = torch.stack([attn for attn in attention_weights])
        aggregated = stacked.mean(dim=(0, 1, 2))  # Average batch, layers, heads
        result_weights = aggregated.cpu().numpy().tolist()
        shape_info = f"Aggregated (mean): [{query_len}, {key_len}]"
    elif aggregate == "sum":
        # Stack and sum across layers and heads: [query_len, key_len]
        stacked = torch.stack([attn for attn in attention_weights])
        aggregated = stacked.sum(dim=(0, 1, 2))  # Sum batch, layers, heads
        result_weights = aggregated.cpu().numpy().tolist()
        shape_info = f"Aggregated (sum): [{query_len}, {key_len}]"
    else:
        # Return per-layer tensors
        result_weights = [attn[0].cpu().numpy().tolist() for attn in attention_weights]
        shape_info = f"Per-layer: {num_layers} layers × [{num_heads}, {query_len}, {key_len}]"
    
    return {
        "available": True,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "query_length": query_len,
        "key_length": key_len,
        "batch_size": batch_size,
        "attention_weights": result_weights,
        "shape_info": shape_info,
        "aggregate_method": aggregate
    }


def get_cache_statistics(manual_generator: Any) -> Dict[str, Any]:
    """
    Get H2O cache statistics (memory usage, eviction info, etc.).
    
    Args:
        manual_generator: The ManualGenerator instance (injected by sandbox)
    
    Returns:
        Dictionary containing:
            - available: bool - Whether H2O cache is active
            - total_tokens: int - Total tokens in conversation
            - cached_tokens: int - Tokens currently in cache
            - evicted_tokens: int - Tokens evicted from cache
            - system_prompt_tokens: int - System prompt tokens (always cached)
            - recent_window_tokens: int - Recent window tokens (always cached)
            - heavy_hitter_tokens: int - Middle tokens kept due to high attention
            - avg_attention_score: float - Average attention score
            - max_attention_score: float - Maximum attention score
            - min_attention_score: float - Minimum attention score
            - eviction_count: int - Number of times cache was evicted
            - cache_utilization: float - Percentage of cache slots used
    
    Example:
        >>> stats = introspection.generation.get_cache_statistics()
        >>> if stats['available']:
        ...     print(f"Cache: {stats['cached_tokens']}/{stats['total_tokens']} tokens")
        ...     print(f"Evicted: {stats['evicted_tokens']} tokens")
        ...     print(f"Heavy hitters: {stats['heavy_hitter_tokens']}")
    """
    if not hasattr(manual_generator, 'h2o_cache'):
        return {
            "available": False,
            "error": "H2O cache manager not initialized"
        }
    
    h2o_cache = manual_generator.h2o_cache
    stats = h2o_cache.get_statistics()
    
    cache_utilization = (stats.cached_tokens / h2o_cache.max_cache_tokens * 100) if h2o_cache.max_cache_tokens > 0 else 0.0
    
    return {
        "available": True,
        "total_tokens": stats.total_tokens,
        "cached_tokens": stats.cached_tokens,
        "evicted_tokens": stats.evicted_tokens,
        "system_prompt_tokens": stats.system_prompt_tokens,
        "recent_window_tokens": stats.recent_window_tokens,
        "heavy_hitter_tokens": stats.heavy_hitter_tokens,
        "avg_attention_score": stats.avg_attention_score,
        "max_attention_score": stats.max_attention_score,
        "min_attention_score": stats.min_attention_score,
        "eviction_count": h2o_cache.eviction_count,
        "cache_utilization": cache_utilization,
        "max_cache_tokens": h2o_cache.max_cache_tokens
    }


def get_token_importance_scores(
    manual_generator: Any,
    top_k: Optional[int] = None,
    normalize: bool = False
) -> Dict[str, Any]:
    """
    Get cumulative attention scores for all tokens in conversation.
    
    This shows which tokens the model has paid the most attention to
    across all generations.
    
    Args:
        manual_generator: The ManualGenerator instance (injected by sandbox)
        top_k: Return only top K most important tokens (None = all)
        normalize: Normalize scores to [0, 1] range
    
    Returns:
        Dictionary containing:
            - available: bool - Whether importance scores are available
            - num_tokens: int - Number of tokens with scores
            - token_scores: Dict[int, float] - Token position -> importance score
            - top_tokens: List[Tuple[int, float]] - Top K (position, score) pairs
    
    Example:
        >>> scores = introspection.generation.get_token_importance_scores(top_k=10)
        >>> if scores['available']:
        ...     print("Most important token positions:")
        ...     for pos, score in scores['top_tokens']:
        ...         print(f"  Token {pos}: importance {score:.4f}")
    """
    if not hasattr(manual_generator, 'h2o_cache'):
        return {
            "available": False,
            "error": "H2O cache manager not initialized"
        }
    
    h2o_cache = manual_generator.h2o_cache
    scores = h2o_cache.get_token_importance_scores()
    
    if not scores:
        return {
            "available": False,
            "error": "No token importance scores available yet"
        }
    
    # Normalize if requested
    if normalize:
        max_score = max(scores.values())
        if max_score > 0:
            scores = {pos: score / max_score for pos, score in scores.items()}
    
    # Get top K
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    if top_k is not None:
        top_tokens = sorted_scores[:top_k]
    else:
        top_tokens = sorted_scores
    
    return {
        "available": True,
        "num_tokens": len(scores),
        "token_scores": scores,
        "top_tokens": top_tokens,
        "normalized": normalize
    }


# Module metadata for introspection API
__all__ = [
    'get_last_generation_attention',
    'get_cache_statistics',
    'get_token_importance_scores'
]

__doc_summary__ = """
Access generation-time data:
- get_last_generation_attention() - Attention weights from last generation
- get_cache_statistics() - H2O cache memory usage and eviction stats  
- get_token_importance_scores() - Cumulative attention importance per token
"""
