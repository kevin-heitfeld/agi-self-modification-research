"""
H2O (Heavy-Hitter Oracle) KV Cache Manager

Implements intelligent KV cache eviction based on cumulative attention scores.
Keeps full conversation text visible while capping GPU memory usage.

Based on: "H2O: Heavy-Hitter Oracle for Efficient Generative Inference 
          of Large Language Models" (Zhang et al., 2023)

Author: AGI Self-Modification Research Team
Date: November 21, 2025
"""

import torch
import logging
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CacheStatistics:
    """Statistics about the current cache state"""
    total_tokens: int
    cached_tokens: int
    evicted_tokens: int
    system_prompt_tokens: int
    recent_window_tokens: int
    heavy_hitter_tokens: int
    avg_attention_score: float
    max_attention_score: float
    min_attention_score: float


class H2OCacheManager:
    """
    Heavy-Hitter Oracle KV Cache Eviction Manager
    
    Tracks cumulative attention scores and intelligently evicts
    low-attention tokens when cache reaches capacity.
    
    Policy:
        1. Always keep system prompt (attention sink)
        2. Always keep recent window
        3. Keep high-attention tokens from conversation middle (heavy hitters)
        4. Evict low-attention tokens when cache is full
    
    Example:
        >>> cache_mgr = H2OCacheManager(max_cache_tokens=7000, system_prompt_tokens=6000)
        >>> # After each generation:
        >>> cache_mgr.update_attention_scores(attention_weights)
        >>> cache_mgr.total_tokens = current_token_count
        >>> # Evict if needed:
        >>> if cache_mgr.should_evict():
        ...     keep_positions = cache_mgr.select_tokens_to_keep()
        ...     new_cache = cache_mgr.evict_cache(past_key_values, keep_positions)
    """
    
    def __init__(
        self, 
        max_cache_tokens: int,
        system_prompt_tokens: int,
        recent_window: int,
        min_attention_threshold: float = 0.0
    ):
        """
        Initialize H2O Cache Manager.
        
        Args:
            max_cache_tokens: Maximum KV cache size (in tokens) - get from ModelManager.get_optimal_limits()
            system_prompt_tokens: Number of system prompt tokens (always keep) - set after caching system prompt
            recent_window: Number of recent tokens to always keep - get from ModelManager.get_optimal_limits()
            min_attention_threshold: Minimum attention score to consider a token (default: 0.0)
        """
        self.max_cache_tokens = max_cache_tokens
        self.system_prompt_tokens = system_prompt_tokens
        self.recent_window = recent_window
        self.min_attention_threshold = min_attention_threshold
        
        # Track cumulative attention per token position
        self.attention_scores: Dict[int, float] = {}
        
        # Current state
        self.total_tokens: int = 0
        self.eviction_count: int = 0
        
        # Store last attention weights for introspection
        self.last_attention_weights: Optional[Tuple[torch.Tensor, ...]] = None
        
        logger.info(f"H2O Cache Manager initialized: max={max_cache_tokens}, "
                   f"system={system_prompt_tokens}, recent={recent_window}")
    
    def update_attention_scores(self, attention_weights: Tuple[torch.Tensor, ...]) -> None:
        """
        Update cumulative attention scores from latest generation.
        
        Args:
            attention_weights: Tuple of attention tensors, one per layer
                             Each tensor: [batch, num_heads, query_len, key_len]
        """
        if not attention_weights:
            logger.warning("No attention weights provided to H2O cache manager")
            return
        
        # Store for introspection API
        self.last_attention_weights = attention_weights
        
        # Stack all layers: [num_layers, batch, num_heads, query_len, key_len]
        try:
            stacked_attention = torch.stack([attn for attn in attention_weights])
        except Exception as e:
            logger.error(f"Failed to stack attention weights: {e}")
            return
        
        # Sum across layers, batch, and heads: [query_len, key_len]
        # This gives us total attention each key position received
        attention_sum = stacked_attention.sum(dim=(0, 1, 2))
        
        # Accumulate attention RECEIVED by each key position
        for key_pos in range(attention_sum.shape[1]):
            attention_received = attention_sum[:, key_pos].sum().item()
            if attention_received > self.min_attention_threshold:
                self.attention_scores[key_pos] = self.attention_scores.get(key_pos, 0.0) + attention_received
        
        logger.debug(f"Updated attention scores for {attention_sum.shape[1]} tokens. "
                    f"Total tracked: {len(self.attention_scores)}")
    
    def should_evict(self) -> bool:
        """Check if cache eviction is needed."""
        return self.total_tokens > self.max_cache_tokens
    
    def select_tokens_to_keep(self) -> List[int]:
        """
        Select which token positions to keep in cache based on H2O policy.
        
        Returns:
            Sorted list of token positions to keep in cache
        """
        if self.total_tokens <= self.max_cache_tokens:
            return list(range(self.total_tokens))  # Keep all
        
        keep_tokens: Set[int] = set()
        
        # 1. System prompt (attention sink) - always keep
        system_end = min(self.system_prompt_tokens, self.total_tokens)
        keep_tokens.update(range(system_end))
        logger.debug(f"Keeping system prompt: 0-{system_end}")
        
        # 2. Recent window - always keep
        recent_start = max(0, self.total_tokens - self.recent_window)
        keep_tokens.update(range(recent_start, self.total_tokens))
        logger.debug(f"Keeping recent window: {recent_start}-{self.total_tokens}")
        
        # 3. High-attention tokens from conversation middle (heavy hitters)
        remaining_slots = self.max_cache_tokens - len(keep_tokens)
        conversation_start = system_end
        conversation_end = recent_start
        
        if conversation_start < conversation_end and remaining_slots > 0:
            # Get attention scores for conversation middle
            middle_tokens = range(conversation_start, conversation_end)
            token_scores = [(t, self.attention_scores.get(t, 0.0)) for t in middle_tokens]
            
            # Sort by attention (descending) and take top K
            token_scores.sort(key=lambda x: x[1], reverse=True)
            top_tokens = [t for t, score in token_scores[:remaining_slots]]
            keep_tokens.update(top_tokens)
            
            if top_tokens:
                avg_score = sum(score for _, score in token_scores[:remaining_slots]) / len(top_tokens)
                logger.debug(f"Keeping {len(top_tokens)} heavy hitters from middle "
                           f"(avg attention: {avg_score:.4f})")
        
        kept_positions = sorted(keep_tokens)
        evicted_count = self.total_tokens - len(kept_positions)
        
        logger.info(f"H2O Eviction: keeping {len(kept_positions)}/{self.total_tokens} tokens "
                   f"(evicting {evicted_count})")
        
        return kept_positions
    
    def evict_cache(
        self, 
        past_key_values: Tuple[Tuple[torch.Tensor, torch.Tensor], ...],
        keep_positions: Optional[List[int]] = None
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
        """
        Evict low-attention tokens from KV cache.
        
        Args:
            past_key_values: Current KV cache tuple of (key, value) pairs per layer
                           Each key/value: [batch_size, num_heads, seq_len, head_dim]
            keep_positions: Token positions to keep (if None, calls select_tokens_to_keep)
            
        Returns:
            Evicted KV cache with same structure but only selected tokens
        """
        if keep_positions is None:
            keep_positions = self.select_tokens_to_keep()
        
        if len(keep_positions) == self.total_tokens:
            logger.debug("No eviction needed - keeping all tokens")
            return past_key_values
        
        evicted_cache = []
        
        for layer_idx, (key, value) in enumerate(past_key_values):
            # Index along seq_len dimension (dim=2)
            # key/value shape: [batch_size, num_heads, seq_len, head_dim]
            evicted_key = key[:, :, keep_positions, :]
            evicted_value = value[:, :, keep_positions, :]
            evicted_cache.append((evicted_key, evicted_value))
        
        self.eviction_count += 1
        evicted_count = self.total_tokens - len(keep_positions)
        
        logger.info(f"Evicted {evicted_count} tokens from KV cache "
                   f"(eviction #{self.eviction_count})")
        
        return tuple(evicted_cache)
    
    def get_statistics(self) -> CacheStatistics:
        """Get current cache statistics."""
        if not self.should_evict():
            cached = self.total_tokens
            evicted = 0
        else:
            keep_positions = self.select_tokens_to_keep()
            cached = len(keep_positions)
            evicted = self.total_tokens - cached
        
        system_kept = min(self.system_prompt_tokens, self.total_tokens)
        recent_kept = min(self.recent_window, self.total_tokens)
        heavy_hitter_kept = cached - system_kept - recent_kept
        
        scores = list(self.attention_scores.values())
        avg_score = sum(scores) / len(scores) if scores else 0.0
        max_score = max(scores) if scores else 0.0
        min_score = min(scores) if scores else 0.0
        
        return CacheStatistics(
            total_tokens=self.total_tokens,
            cached_tokens=cached,
            evicted_tokens=evicted,
            system_prompt_tokens=system_kept,
            recent_window_tokens=recent_kept,
            heavy_hitter_tokens=max(0, heavy_hitter_kept),
            avg_attention_score=avg_score,
            max_attention_score=max_score,
            min_attention_score=min_score
        )
    
    def reset(self) -> None:
        """Reset cache manager state (useful for starting new conversations)."""
        self.attention_scores.clear()
        self.total_tokens = 0
        self.eviction_count = 0
        self.last_attention_weights = None
        logger.info("H2O Cache Manager reset")
    
    def get_last_attention_weights(self) -> Optional[Tuple[torch.Tensor, ...]]:
        """
        Get the last captured attention weights (for introspection).
        
        Returns:
            Tuple of attention tensors from last generation, or None
        """
        return self.last_attention_weights
    
    def get_token_importance_scores(self) -> Dict[int, float]:
        """
        Get cumulative attention scores for all tracked tokens.
        
        Returns:
            Dictionary mapping token position -> cumulative attention score
        """
        return dict(self.attention_scores)
    
    def __repr__(self) -> str:
        stats = self.get_statistics()
        return (f"H2OCacheManager(total={stats.total_tokens}, "
                f"cached={stats.cached_tokens}, evicted={stats.evicted_tokens}, "
                f"evictions={self.eviction_count})")
