"""
Generation Introspection Module - Access to generation-time data

This module previously provided access to attention weights and H2O cache statistics.
With the switch to Flash Attention 2 and self-summarization:
- Attention weights are not captured (Flash Attention 2 doesn't materialize them)
- H2O cache eviction has been replaced by model-generated self-summarization

This module is retained for potential future generation-time introspection features.

Author: AGI Self-Modification Research Team
Date: November 21, 2025
"""

from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


def get_generation_info(manual_generator: Any) -> Dict[str, Any]:
    """
    Get basic information about the generation system configuration.
    
    Args:
        manual_generator: The ManualGenerator instance (injected by sandbox)
    
    Returns:
        Dictionary containing:
            - quantize_kv_cache: Whether KV cache quantization is enabled
            - self_summarization_enabled: Whether self-summarization is active
            - system_prompt_length: Length of cached system prompt in tokens
            - conversation_cache_length: Current conversation KV cache length
    
    Example:
        >>> info = introspection.generation.get_generation_info()
        >>> print(f"System prompt: {info['system_prompt_length']} tokens")
        >>> print(f"Self-summarization: {info['self_summarization_enabled']}")
    """
    # Get KV cache length
    conv_cache_length = 0
    if manual_generator.conversation_kv_cache is not None:
        conv_cache_length = manual_generator._get_cache_length(manual_generator.conversation_kv_cache)
    
    return {
        "quantize_kv_cache": manual_generator.quantize_kv_cache,
        "self_summarization_enabled": manual_generator.enable_self_summarization,
        "system_prompt_length": manual_generator.system_prompt_length,
        "conversation_cache_length": conv_cache_length,
        "device": manual_generator.device
    }


# Module metadata for introspection API
__all__ = [
    'get_generation_info'
]

__doc_summary__ = """
Access generation system information:
- get_generation_info() - Configuration and cache statistics
"""
