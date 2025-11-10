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
    
    def __init__(self, model, tokenizer, device: str = "cuda"):
        """
        Initialize manual generator.
        
        Args:
            model: HuggingFace model (e.g., AutoModelForCausalLM)
            tokenizer: HuggingFace tokenizer
            device: Device to run on ("cuda" or "cpu")
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
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
                current_cache = self.system_prompt_cache
                cache_length = self.system_prompt_length
                logger.debug(f"Using system prompt cache (length: {cache_length})")
                
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
        
        # Generation loop
        with torch.no_grad():
            for step in range(max_new_tokens):
                # Forward pass
                if step == 0:
                    # First step: process full input
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        past_key_values=current_cache,
                        use_cache=use_cache,
                        return_dict=True
                    )
                else:
                    # Subsequent steps: only process last token
                    outputs = self.model(
                        input_ids=new_token_id,
                        attention_mask=attention_mask,
                        past_key_values=current_cache,
                        use_cache=use_cache,
                        return_dict=True
                    )
                
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
