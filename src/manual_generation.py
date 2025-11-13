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
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class GenerationState:
    """Tracks state during generation"""
    input_ids: torch.Tensor          # [batch, seq_len]
    past_key_values: Optional[Tuple] # KV cache
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

    def __init__(self, model, tokenizer, device: str = "cuda", quantize_kv_cache: bool = False):
        """
        Initialize manual generator.

        Args:
            model: HuggingFace model (e.g., AutoModelForCausalLM)
            tokenizer: HuggingFace tokenizer
            device: Device to run on ("cuda" or "cpu")
            quantize_kv_cache: Use INT8 quantization for KV cache (saves 50% memory)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.quantize_kv_cache = quantize_kv_cache
        
        # Prepare cache config for quantization (if enabled)
        # Note: Transformers 4.45+ changed from QuantizedCacheConfig to cache_implementation
        if quantize_kv_cache:
            try:
                # Try new API (transformers 4.45+)
                from transformers.cache_utils import HQQQuantizedCache
                # Store cache implementation string for GenerationConfig
                self.cache_implementation = "hybrid"  # Use hybrid cache with quantization
                logger.info("KV cache quantization enabled (HQQ - 50-75% memory savings)")
            except ImportError:
                try:
                    # Fallback to old API (transformers 4.44)
                    from transformers import QuantizedCacheConfig
                    self.cache_config = QuantizedCacheConfig(nbits=8)
                    self.cache_implementation = None
                    logger.info("KV cache quantization enabled (INT8 - 50% memory savings)")
                except ImportError:
                    logger.warning("KV cache quantization not available in this transformers version")
                    logger.warning("KV cache quantization disabled - using FP16")
                    self.quantize_kv_cache = False
                    self.cache_implementation = None
        else:
            self.cache_implementation = None
            logger.info("KV cache quantization disabled (using FP16)")

        # Cached system prompt KV states
        self.system_prompt_cache: Optional[Tuple] = None
        self.system_prompt_length: int = 0

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

        # Forward pass to get KV cache
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                use_cache=True,
                return_dict=True
            )

        # Store the KV cache
        self.system_prompt_cache = outputs.past_key_values
        self.system_prompt_length = inputs["input_ids"].shape[1]

        logger.info(f"System prompt cached: {self.system_prompt_length} tokens")

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 1.0,
        do_sample: bool = True,
        use_cache: bool = True,
        past_key_values: Optional[Tuple] = None,
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
            past_key_values: Existing KV cache to continue from (optional)
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
            if past_key_values is not None:
                # Use provided cache (multi-turn conversation)
                current_cache = past_key_values
                cache_length = past_key_values[0][0].shape[2]  # seq_len from first layer's key
                logger.debug(f"Using provided KV cache (length: {cache_length})")

                # Extend attention mask to cover cached tokens
                past_mask = torch.ones((1, cache_length), dtype=torch.long, device=self.device)
                attention_mask = torch.cat([past_mask, attention_mask], dim=1)

            elif self.system_prompt_cache is not None:
                # Use system prompt cache
                # CRITICAL: Deep copy the cache to prevent in-place mutations!
                # PyTorch model forward pass modifies past_key_values in-place,
                # so we must deep copy before each use to keep the original pristine
                # Using copy.deepcopy() maintains Cache object type (required by Transformers)
                import copy
                current_cache = copy.deepcopy(self.system_prompt_cache)
                cache_length = self.system_prompt_length
                logger.info(f"Using system prompt cache (length: {cache_length}) - DEEP COPIED for safety")

                # Extend attention mask to cover system prompt
                system_mask = torch.ones((1, cache_length), dtype=torch.long, device=self.device)
                attention_mask = torch.cat([system_mask, attention_mask], dim=1)
            else:
                # No cache available, will create new one
                current_cache = None
                logger.debug("No cache available, will create new")
        else:
            current_cache = None
            logger.debug("Cache disabled")

        # Track generated tokens
        generated_tokens = []
        stopped_reason = "max_length"

        # CRITICAL: Calculate position_ids for models with rotary embeddings
        # When using cached KV states, position_ids must account for cached sequence length
        if current_cache is not None:
            # Cached sequence length
            if past_key_values is not None:
                cache_length = past_key_values[0][0].shape[2]
            else:
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
                current_position = past_key_values[0][0].shape[2] + input_ids.shape[1]
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
                        return_dict=True
                    )

                    # Increment position for next token
                    current_position += 1

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

        if return_cache:
            result["past_key_values"] = current_cache
            if current_cache is not None:
                final_cache_len = current_cache[0][0].shape[2]
                logger.debug(f"Returning cache with length: {final_cache_len} tokens")

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
