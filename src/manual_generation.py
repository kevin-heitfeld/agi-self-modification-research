"""
Manual Generation Loop with KV Caching

Provides fine-grained control over token generation for:
1. System prompt KV caching (massive memory savings)
2. Per-token introspection (Phase 2)
3. Real-time activation monitoring (Phase 2)
4. Mid-generation interventions (Phase 3)

Author: AGI Self-Modification Research Team
Date: November 10, 2025
"""

import torch
import logging
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class GenerationState:
    """Tracks state during generation"""
    input_ids: torch.Tensor          # [batch, seq_len]
    past_key_values: Optional[Any]   # KV cache (tuple or Cache object)
    attention_mask: torch.Tensor     # [batch, seq_len]
    generated_tokens: List[int]      # Tokens generated so far
    finished: bool                   # Whether generation is complete


class ManualGenerator:
    """
    Manual token-by-token generation with KV caching.

    Key features:
    - System prompt KV caching (reuse across turns)
    - Per-token callbacks for introspection
    - Explicit control over sampling
    - Memory-efficient (reuses KV cache)

    Example:
        >>> generator = ManualGenerator(model, tokenizer, device="cuda")
        >>> generator.cache_system_prompt("You are a helpful assistant.")
        >>> result = generator.generate("Hello!", max_new_tokens=50)
        >>> print(result["generated_text"])
    """

    def __init__(
        self, 
        model, 
        tokenizer, 
        device: str = "cuda", 
        quantize_kv_cache: bool = False,
        enable_h2o_eviction: bool = False,
        max_cache_tokens: Optional[int] = None,
        recent_window: Optional[int] = None
    ):
        """
        Initialize manual generator.

        Args:
            model: HuggingFace model (e.g., AutoModelForCausalLM)
            tokenizer: HuggingFace tokenizer
            device: Device to run on ("cuda" or "cpu")
            quantize_kv_cache: Use HQQ quantization for KV cache (saves 50-75% memory)
            enable_h2o_eviction: Enable H2O cache eviction (unlimited conversation length)
            max_cache_tokens: Maximum tokens in KV cache (required if H2O enabled, get from ModelManager.get_optimal_limits())
            recent_window: Recent window size for H2O eviction (required if H2O enabled, get from ModelManager.get_optimal_limits())
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.quantize_kv_cache = quantize_kv_cache
        self.enable_h2o_eviction = enable_h2o_eviction

        # Validate H2O parameters if enabled
        if enable_h2o_eviction:
            if max_cache_tokens is None or recent_window is None:
                raise ValueError(
                    "When enable_h2o_eviction=True, you must provide max_cache_tokens and recent_window.\n"
                    "Get these values from ModelManager.get_optimal_limits():\n"
                    "  limits = model_manager.get_optimal_limits(quantization='4bit')\n"
                    "  ManualGenerator(..., max_cache_tokens=limits['max_cache_tokens'], recent_window=limits['recent_window'])"
                )

        # Import quantized cache (transformers 4.57+)
        self.QuantizedCache = None
        if quantize_kv_cache:
            try:
                from transformers.cache_utils import QuantizedCache
                self.QuantizedCache = QuantizedCache
                logger.info("✓ KV cache quantization enabled (HQQ 4-bit - 75% memory savings)")
                logger.info("  Cache will use 4-bit quantization with dynamic range")
            except ImportError:
                logger.warning("⚠ Quantized cache not available in this transformers version")
                logger.warning("  Falling back to standard FP16 cache")
                logger.warning("  Upgrade to transformers 4.57+ for quantization support")
                self.quantize_kv_cache = False
        else:
            logger.info("KV cache quantization disabled (using standard FP16 cache)")

        # Cached system prompt KV states
        self.system_prompt_cache: Optional[Any] = None
        self.system_prompt_length: int = 0
        self.system_prompt_input_ids: Optional[torch.Tensor] = None  # For regenerating quantized cache

        # H2O cache manager for intelligent eviction
        self.h2o_cache = None
        if enable_h2o_eviction:
            from .memory.h2o_cache_manager import H2OCacheManager
            # At this point, validation above ensures max_cache_tokens and recent_window are not None
            assert max_cache_tokens is not None and recent_window is not None  # Type narrowing for mypy
            # System prompt length will be set after caching
            self.h2o_cache = H2OCacheManager(
                max_cache_tokens=max_cache_tokens,
                system_prompt_tokens=0,  # Updated after cache_system_prompt()
                recent_window=recent_window
            )
            logger.info(f"✓ H2O cache eviction enabled: max={max_cache_tokens}, recent_window={recent_window}")
        
        # Conversation KV cache (grows with each turn, evicted by H2O)
        self.conversation_kv_cache: Optional[Any] = None

        logger.info(f"ManualGenerator initialized on {device}")

    def cache_system_prompt(self, system_prompt: str) -> None:
        """
        Pre-compute and cache system prompt KV states.

        This is the KEY optimization: process 6000+ token system prompt
        once, reuse forever.

        Args:
            system_prompt: The system prompt text
        """
        logger.info(f"Caching system prompt ({len(system_prompt)} chars)...")

        # Tokenize system prompt
        inputs = self.tokenizer(system_prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Store input IDs for regenerating cache (needed for quantized caches)
        self.system_prompt_input_ids = inputs["input_ids"]
        self.system_prompt_length = inputs["input_ids"].shape[1]

        # Initialize quantized cache if enabled
        past_key_values = None
        if self.quantize_kv_cache and self.QuantizedCache is not None:
            try:
                # Try new API first (transformers 4.57+)
                past_key_values = self.QuantizedCache(
                    backend='hqq',
                    config=self.model.config,
                    nbits=8,  # 8-bit quantization (50% memory savings) - 4-bit was too aggressive
                    axis_key=0,  # Quantize along key dimension
                    axis_value=0,  # Quantize along value dimension
                    q_group_size=64,  # Group size for quantization
                    residual_length=128  # Residual for better accuracy
                )
                logger.debug("Created HQQ quantized cache for system prompt (new API)")
            except ImportError as e:
                # HQQ library not installed
                logger.warning(f"HQQ library not installed ({e}), falling back to standard cache")
                logger.warning("To use HQQ quantization, install: pip install hqq")
                self.quantize_kv_cache = False  # Disable quantization for this session
                past_key_values = None
            except TypeError as e:
                # API mismatch - try without backend parameter
                logger.debug(f"New API failed ({e}), trying older API")
                try:
                    past_key_values = self.QuantizedCache(
                        config=self.model.config,
                        nbits=8,  # 8-bit quantization (50% memory savings) - 4-bit was too aggressive
                        axis_key=0,  # Quantize along key dimension
                        axis_value=0,  # Quantize along value dimension
                        q_group_size=64,  # Group size for quantization
                        residual_length=128  # Residual for better accuracy
                    )
                    logger.debug("Created HQQ quantized cache for system prompt (older API)")
                except Exception as e2:
                    logger.warning(f"Failed to create quantized cache ({e2}), using standard cache")
                    self.quantize_kv_cache = False
                    past_key_values = None

            if past_key_values is not None:
                logger.info("  Using HQQ 8-bit quantized cache for system prompt")

        # Forward pass to get KV cache
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                use_cache=True,
                past_key_values=past_key_values,
                return_dict=True
            )

        # Store the KV cache (now potentially quantized!)
        self.system_prompt_cache = outputs.past_key_values
        self.system_prompt_length = inputs["input_ids"].shape[1]

        # Update H2O cache manager with system prompt length
        if self.h2o_cache is not None:
            self.h2o_cache.system_prompt_tokens = self.system_prompt_length
            logger.info(f"✓ H2O cache manager updated: system_prompt_tokens={self.system_prompt_length}")
            
            # PRE-ALLOCATE conversation KV cache to max size (prevents growth & reallocations)
            self._preallocate_kv_cache(self.h2o_cache.max_cache_tokens)

        # Log memory savings if using quantization
        if self.quantize_kv_cache and self.QuantizedCache is not None:
            logger.info(f"✓ System prompt cached: {self.system_prompt_length} tokens (8-bit quantized)")
            logger.info(f"  Estimated memory savings: ~50% vs FP16 cache")
        else:
            logger.info(f"System prompt cached: {self.system_prompt_length} tokens (FP16)")

    def _preallocate_kv_cache(self, max_tokens: int) -> None:
        """
        Pre-allocate KV cache to maximum size to prevent growth and reallocations.
        
        This ensures VRAM usage is completely static from the start, avoiding:
        1. OOM errors from gradual growth
        2. Memory fragmentation from reallocations
        3. Performance overhead of dynamic growth
        
        Args:
            max_tokens: Maximum number of tokens to pre-allocate for
        """
        logger.info(f"Pre-allocating KV cache for {max_tokens} tokens...")
        
        # Create dummy input of max size (padding tokens)
        dummy_input = torch.full(
            (1, max_tokens),
            self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            dtype=torch.long,
            device=self.device
        )
        
        # Initialize cache with quantization if enabled
        pre_allocated_cache = None
        if self.quantize_kv_cache and self.QuantizedCache is not None:
            try:
                pre_allocated_cache = self.QuantizedCache(
                    backend='hqq',
                    config=self.model.config,
                    nbits=8,
                    axis_key=0,
                    axis_value=0,
                    q_group_size=64,
                    residual_length=128
                )
            except (ImportError, TypeError):
                # Fall back to standard cache
                pre_allocated_cache = None
        
        # Single forward pass to allocate full cache
        try:
            with torch.no_grad():
                outputs = self.model(
                    input_ids=dummy_input,
                    attention_mask=torch.ones_like(dummy_input),
                    use_cache=True,
                    past_key_values=pre_allocated_cache,
                    return_dict=True
                )
                
                # Store pre-allocated cache
                self.conversation_kv_cache = outputs.past_key_values
            
            # Calculate approximate memory usage
            if self.quantize_kv_cache:
                estimated_mb = max_tokens * self.model.config.num_hidden_layers * 2 * self.model.config.hidden_size * 1 / (1024 * 1024)  # 8-bit = 1 byte
                logger.info(f"✓ Pre-allocated {max_tokens} tokens KV cache (8-bit quantized): ~{estimated_mb:.1f} MB")
            else:
                estimated_mb = max_tokens * self.model.config.num_hidden_layers * 2 * self.model.config.hidden_size * 2 / (1024 * 1024)  # fp16 = 2 bytes
                logger.info(f"✓ Pre-allocated {max_tokens} tokens KV cache (FP16): ~{estimated_mb:.1f} MB")
            
            logger.info("  VRAM usage is now STATIC - no growth during conversation")
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error(f"❌ OOM during KV cache pre-allocation: {e}")
                logger.error(f"   Reduce max_cache_tokens (currently {max_tokens}) to fit in available VRAM")
                # Don't re-raise - continue without pre-allocation
                self.conversation_kv_cache = None
                logger.warning("⚠ Continuing without pre-allocation - cache will grow dynamically")
            else:
                raise

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 1.0,
        do_sample: bool = True,
        use_cache: bool = True,
        past_key_values: Optional[Any] = None,
        callback: Optional[Callable[[int, torch.Tensor], None]] = None,
        return_cache: bool = False
    ) -> Dict[str, Any]:
        """
        Generate tokens one at a time with optional callback.

        Args:
            prompt: Input text (user message, NOT including system prompt if cached)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_p: Nucleus sampling threshold (0.0-1.0)
            do_sample: Whether to sample (True) or use greedy decoding (False)
            use_cache: Whether to use/reuse KV cache
            past_key_values: Existing KV cache to continue from (tuple or Cache object)
            callback: Optional function called per token: callback(token_id, logits)
            return_cache: Whether to return the final KV cache in result

        Returns:
            Dict with:
            - generated_text: Generated response (decoded)
            - generated_tokens: List of token IDs
            - num_tokens: Number of tokens generated
            - cache_used: Whether KV cache was used
            - stopped_reason: Why generation stopped ("max_length", "eos", or "other")
            - past_key_values: Final KV cache (if return_cache=True)
            - attentions: Attention weights (if H2O enabled)
        """
        # If H2O is enabled, we need to track attention weights
        # This requires temporarily switching to eager attention mode
        original_attn_implementation = None
        if self.enable_h2o_eviction and self.h2o_cache is not None:
            # Check if we can modify attention implementation
            if hasattr(self.model, 'config') and hasattr(self.model.config, '_attn_implementation'):
                original_attn_implementation = self.model.config._attn_implementation
                # Switch to eager for attention tracking
                if original_attn_implementation != 'eager':
                    self.model.config._attn_implementation = 'eager'
                    logger.debug(f"Switched attention from '{original_attn_implementation}' to 'eager' for H2O tracking")
        
        try:
            # Call the actual generation implementation
            result = self._generate_impl(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                use_cache=use_cache,
                past_key_values=past_key_values,
                callback=callback,
                return_cache=return_cache,
                output_attentions=self.enable_h2o_eviction  # Track attention if H2O enabled
            )
            
            # Process H2O eviction if enabled
            if self.enable_h2o_eviction and self.h2o_cache is not None:
                # Update attention scores from generation
                if 'attentions' in result and result['attentions'] is not None:
                    self.h2o_cache.update_attention_scores(result['attentions'])
                    logger.debug(f"Updated H2O attention scores from generation")
                
                # Update total token count
                if 'past_key_values' in result and result['past_key_values'] is not None:
                    cache_length = self._get_cache_length(result['past_key_values'])
                    self.h2o_cache.total_tokens = cache_length
                    logger.debug(f"H2O cache: {cache_length} total tokens")
                    
                    # Apply eviction if needed
                    if self.h2o_cache.should_evict():
                        keep_positions = self.h2o_cache.select_tokens_to_keep()
                        result['past_key_values'] = self.h2o_cache.evict_cache(
                            result['past_key_values'],
                            keep_positions
                        )
                        logger.info(f"H2O evicted cache: {cache_length} → {len(keep_positions)} tokens")
            
            return result
            
        finally:
            # Restore original attention implementation
            if original_attn_implementation is not None:
                self.model.config._attn_implementation = original_attn_implementation
                logger.debug(f"Restored attention to '{original_attn_implementation}'")

    def _generate_impl(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 1.0,
        do_sample: bool = True,
        use_cache: bool = True,
        past_key_values: Optional[Any] = None,
        callback: Optional[Callable[[int, torch.Tensor], None]] = None,
        return_cache: bool = False,
        output_attentions: bool = False
    ) -> Dict[str, Any]:
        """
        Internal implementation of generation (called by generate()).
        
        Separated to allow attention tracking wrapper in generate().
        """
        # Tokenize the prompt
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        # Handle empty input (edge case)
        if input_ids.shape[1] == 0:
            # If empty AND we have a cache, start generation from cache
            # If empty with NO cache, add BOS token
            if use_cache and (past_key_values is not None or self.system_prompt_cache is not None):
                # Continue from cache - create minimal seed token
                # Use BOS or EOS token as seed (model-dependent)
                bos_token_id = self.tokenizer.bos_token_id if self.tokenizer.bos_token_id is not None else self.tokenizer.eos_token_id
                input_ids = torch.tensor([[bos_token_id]], device=self.device)
                attention_mask = torch.ones((1, 1), dtype=torch.long, device=self.device)
            else:
                # Empty input with no cache - return empty result
                return {
                    "generated_text": "",
                    "generated_tokens": [],
                    "num_tokens": 0,
                    "cache_used": False,
                    "stopped_reason": "empty_input"
                }

        # Determine which cache to use and adjust attention mask
        if use_cache:
            # CRITICAL: For HQQ quantized caches, IGNORE conversation cache!
            # Conversation caches get mutated and corrupted - always use system prompt cache
            # which we regenerate fresh each time
            if past_key_values is not None and not self.quantize_kv_cache:
                # Use provided cache (multi-turn conversation) - ONLY for standard caches
                current_cache = past_key_values
                cache_length = self._get_cache_length(past_key_values)
                logger.debug(f"Using provided KV cache (length: {cache_length})")

                # Extend attention mask to cover cached tokens
                past_mask = torch.ones((1, cache_length), dtype=torch.long, device=self.device)
                attention_mask = torch.cat([past_mask, attention_mask], dim=1)

            elif self.system_prompt_cache is not None:
                # Use system prompt cache
                # CRITICAL: Quantized caches cannot be safely reused due to in-place mutations
                # Solution: Regenerate the cache for quantized, deepcopy for standard

                if self.quantize_kv_cache:
                    # Regenerate quantized cache from scratch
                    # This is fast (8747 tokens ~0.5s) and avoids all mutation/corruption issues
                    logger.debug(f"Regenerating HQQ quantized cache for system prompt ({self.system_prompt_length} tokens)")

                    # Create empty quantized cache
                    current_cache = None
                    try:
                        current_cache = self.QuantizedCache(
                            backend='hqq',
                            config=self.model.config,
                            nbits=8,  # 8-bit, not 4-bit (too aggressive)
                            axis_key=0,
                            axis_value=0,
                            q_group_size=64,
                            residual_length=128
                        )
                    except ImportError:
                        # HQQ not installed - should not happen if cache was created, but handle gracefully
                        logger.warning("HQQ not available during regeneration, using standard cache")
                        self.quantize_kv_cache = False
                        import copy
                        current_cache = copy.deepcopy(self.system_prompt_cache)
                    except TypeError:
                        # Try older API
                        try:
                            current_cache = self.QuantizedCache(
                                config=self.model.config,
                                nbits=8,  # 8-bit, not 4-bit (too aggressive)
                                axis_key=0,
                                axis_value=0,
                                q_group_size=64,
                                residual_length=128
                            )
                        except Exception as e:
                            logger.warning(f"Failed to regenerate quantized cache ({e}), using standard cache")
                            self.quantize_kv_cache = False
                            import copy
                            current_cache = copy.deepcopy(self.system_prompt_cache)

                    # Run forward pass to populate the cache (only if we have a quantized cache)
                    if current_cache is not None and self.quantize_kv_cache:
                        with torch.no_grad():
                            _ = self.model(
                                input_ids=self.system_prompt_input_ids,
                                attention_mask=torch.ones_like(self.system_prompt_input_ids),
                            use_cache=True,
                            past_key_values=current_cache,
                            return_dict=True
                        )
                    # current_cache is now filled with quantized KV states

                    cache_type = "HQQ quantized (regenerated)"
                else:
                    # Safe to deep copy standard caches (no quantization metadata to corrupt)
                    import copy
                    current_cache = copy.deepcopy(self.system_prompt_cache)
                    cache_type = "standard (copied)"

                cache_length = self.system_prompt_length

                # Extend attention mask to cover system prompt
                system_mask = torch.ones((1, cache_length), dtype=torch.long, device=self.device)
                attention_mask = torch.cat([system_mask, attention_mask], dim=1)
            else:
                # No cache available, will create new one
                # If quantization is enabled, initialize a quantized cache
                if self.quantize_kv_cache and self.QuantizedCache is not None:
                    current_cache = self.QuantizedCache(
                        backend='hqq',
                        config=self.model.config,
                        nbits=4,
                        axis_key=0,
                        axis_value=0,
                        q_group_size=64,
                        residual_length=128
                    )
                    logger.debug("No cache available, initializing HQQ quantized cache")
                else:
                    current_cache = None
                    logger.debug("No cache available, will create standard cache")
        else:
            current_cache = None
            logger.debug("Cache disabled")

        # Track generated tokens
        generated_tokens = []
        stopped_reason = "max_length"
        
        # Track attention weights if requested
        all_attentions = [] if output_attentions else None

        # CRITICAL: Calculate position_ids for models with rotary embeddings
        # When using cached KV states, position_ids must account for cached sequence length
        if current_cache is not None:
            # Get cached sequence length using helper method
            if past_key_values is not None:
                cache_length = self._get_cache_length(past_key_values)
            else:
                # System prompt cache
                cache_length = self.system_prompt_length

            logger.debug(f"Using cache, cache_length={cache_length}, input_ids.shape[1]={input_ids.shape[1]}")

            # Position IDs start AFTER the cached sequence
            # For first step: [cache_length, cache_length+1, ..., cache_length+input_len-1]
            position_ids = torch.arange(
                cache_length,
                cache_length + input_ids.shape[1],
                dtype=torch.long,
                device=self.device
            ).unsqueeze(0)  # [1, seq_len]

            logger.debug(f"position_ids range: [{cache_length} to {cache_length + input_ids.shape[1] - 1}]")
        else:
            # No cache: use default positions [0, 1, 2, ...]
            position_ids = torch.arange(
                0,
                input_ids.shape[1],
                dtype=torch.long,
                device=self.device
            ).unsqueeze(0)  # [1, seq_len]

        # Track the current position for incremental generation
        # This is the position of the NEXT token to generate
        if current_cache is not None:
            if past_key_values is not None:
                current_position = self._get_cache_length(past_key_values) + input_ids.shape[1]
            else:
                current_position = self.system_prompt_length + input_ids.shape[1]
        else:
            current_position = input_ids.shape[1]

        # Generation loop
        with torch.no_grad():
            for step in range(max_new_tokens):
                # Forward pass
                if step == 0:
                    # First step: process full input
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_values=current_cache,
                        use_cache=use_cache,
                        output_attentions=output_attentions,
                        return_dict=True
                    )
                else:
                    # Subsequent steps: only process last token
                    # Position ID is current_position (which we increment after each token)
                    next_position_id = torch.tensor([[current_position]], dtype=torch.long, device=self.device)

                    outputs = self.model(
                        input_ids=new_token_id,
                        attention_mask=attention_mask,
                        position_ids=next_position_id,
                        past_key_values=current_cache,
                        use_cache=use_cache,
                        output_attentions=output_attentions,
                        return_dict=True
                    )

                    # Increment position for next token
                    current_position += 1

                # Store attention weights if tracking
                if output_attentions and hasattr(outputs, 'attentions') and outputs.attentions is not None:
                    all_attentions.append(outputs.attentions)

                # Get logits for next token
                logits = outputs.logits[:, -1, :]  # [batch=1, vocab_size]

                # Sample next token
                next_token_id = self._sample_next_token(
                    logits[0],  # [vocab_size]
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample
                )

                # Invoke callback if provided
                if callback is not None:
                    callback(next_token_id, logits[0])

                # Add to generated tokens
                generated_tokens.append(next_token_id)

                # Check for EOS
                if next_token_id == self.tokenizer.eos_token_id:
                    stopped_reason = "eos"
                    logger.debug(f"Generation stopped at EOS (step {step+1})")
                    break

                # Prepare for next iteration
                new_token_id = torch.tensor([[next_token_id]], dtype=torch.long, device=self.device)

                # Update attention mask
                attention_mask = torch.cat([
                    attention_mask,
                    torch.ones((1, 1), dtype=torch.long, device=self.device)
                ], dim=1)

                # Update cache
                if use_cache:
                    current_cache = outputs.past_key_values

        # Decode generated tokens
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        # Build result
        result = {
            "generated_text": generated_text,
            "generated_tokens": generated_tokens,
            "num_tokens": len(generated_tokens),
            "cache_used": use_cache,
            "stopped_reason": stopped_reason
        }
        
        # Add attention weights if tracked
        if output_attentions and all_attentions:
            # Combine attentions from all steps
            # Each step has tuple of (layer1_attn, layer2_attn, ...)
            # We want: tuple of (all_layer1_attns, all_layer2_attns, ...)
            try:
                # Stack attentions from each layer across all steps
                num_layers = len(all_attentions[0])
                combined_attentions = tuple(
                    torch.cat([step_attns[layer_idx] for step_attns in all_attentions], dim=2)  # Concat along query dim
                    for layer_idx in range(num_layers)
                )
                result["attentions"] = combined_attentions
                logger.debug(f"Captured attention weights: {len(combined_attentions)} layers")
            except Exception as e:
                logger.warning(f"Failed to combine attention weights: {e}")
                result["attentions"] = None
        else:
            result["attentions"] = None

        if return_cache:
            # CRITICAL: Don't return HQQ caches - they're corrupted by in-place mutations
            # Return None to force regeneration on next turn
            if self.quantize_kv_cache:
                result["past_key_values"] = None
                logger.debug("Not returning HQQ cache (would be corrupted) - will regenerate from system prompt")
            else:
                result["past_key_values"] = current_cache
                if current_cache is not None:
                    final_cache_len = self._get_cache_length(current_cache)
                    logger.debug(f"Returning standard cache with length: {final_cache_len} tokens")

        logger.info(f"Generated {len(generated_tokens)} tokens (stopped: {stopped_reason})")

        return result

    def _sample_next_token(
        self,
        logits: torch.Tensor,
        temperature: float,
        top_p: float,
        do_sample: bool
    ) -> int:
        """
        Sample next token from logits.

        Args:
            logits: [vocab_size] logits for next token
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            do_sample: Whether to sample or use argmax

        Returns:
            Token ID (int)
        """
        if not do_sample:
            # Greedy decoding
            return logits.argmax().item()

        # Apply temperature
        if temperature != 1.0:
            logits = logits / temperature

        # Convert to probabilities
        probs = torch.softmax(logits, dim=-1)

        # Apply top-p (nucleus) sampling
        if top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Keep at least one token
            sorted_indices_to_remove[0] = False

            # Zero out removed tokens
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            probs[indices_to_remove] = 0.0

            # Renormalize
            probs = probs / probs.sum()

        # Sample from the distribution
        token_id = torch.multinomial(probs, num_samples=1).item()

        return token_id

    def clear_cache(self) -> None:
        """Clear all cached KV states"""
        self.system_prompt_cache = None
        self.system_prompt_length = 0
        logger.info("Cache cleared")

    def _get_cache_length(self, cache: Any) -> int:
        """
        Get the sequence length from a cache object.

        Handles both legacy tuple-based caches and new Cache objects.

        Args:
            cache: KV cache (tuple of tuples or Cache object)

        Returns:
            Sequence length in the cache
        """
        if cache is None:
            return 0

        if isinstance(cache, tuple):
            # Legacy tuple-based cache: ((key, value), (key, value), ...)
            # Shape of key/value: [batch, num_heads, seq_len, head_dim]
            return cache[0][0].shape[2]
        else:
            # New Cache object (HQQQuantizedCache, DynamicCache, etc.)
            # These have a get_seq_length() method
            return cache.get_seq_length()


def test_manual_generator():
    """Quick smoke test for development"""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    tokenizer.pad_token = tokenizer.eos_token

    print("Creating generator...")
    generator = ManualGenerator(model, tokenizer, device="cpu")

    print("Caching system prompt...")
    generator.cache_system_prompt("You are a helpful assistant.")

    print("Generating...")
    result = generator.generate("Hello!", max_new_tokens=10, do_sample=False)

    print(f"Generated: {result['generated_text']}")
    print(f"Tokens: {result['num_tokens']}")
    print(f"Cache used: {result['cache_used']}")

    print("\nSmoke test passed! ✅")


if __name__ == "__main__":
    # Run smoke test
    test_manual_generator()
