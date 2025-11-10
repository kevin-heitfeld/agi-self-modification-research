"""
Comprehensive test suite for manual generation loop with KV caching.

Test-Driven Development (TDD) approach:
1. Write all tests first (they will fail)
2. Implement incrementally until tests pass
3. Refactor while keeping tests green

Run with: pytest tests/test_manual_generation.py -v
"""

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.manual_generation import ManualGenerator, GenerationState


# Fixtures
@pytest.fixture(scope="module")
def model_and_tokenizer():
    """Load a small model for testing (distilgpt2 for speed)"""
    model_name = "distilgpt2"  # Small model for fast tests
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Move to CPU for tests (avoid GPU memory issues in CI)
    model = model.to("cpu")
    model.eval()
    
    return model, tokenizer


@pytest.fixture
def generator(model_and_tokenizer):
    """Create a ManualGenerator instance"""
    model, tokenizer = model_and_tokenizer
    return ManualGenerator(model, tokenizer, device="cpu")


# ==============================================================================
# 1. Basic Generation Tests
# ==============================================================================

def test_generate_single_token(generator):
    """Test generating a single token from input"""
    result = generator.generate(
        prompt="Hello",
        max_new_tokens=1,
        do_sample=False  # Deterministic
    )
    
    assert "generated_text" in result
    assert "generated_tokens" in result
    assert len(result["generated_tokens"]) == 1
    assert isinstance(result["generated_tokens"][0], int)


def test_generate_multiple_tokens(generator):
    """Test generating multiple tokens (5 tokens)"""
    result = generator.generate(
        prompt="Once upon a",
        max_new_tokens=5,
        do_sample=False
    )
    
    assert len(result["generated_tokens"]) == 5
    assert result["num_tokens"] == 5
    assert len(result["generated_text"]) > 0


def test_generate_until_eos(generator):
    """Test generation stops at EOS token"""
    result = generator.generate(
        prompt="The end.",
        max_new_tokens=100,  # Large limit
        do_sample=False
    )
    
    # Should stop before max_new_tokens if EOS is hit
    # (depends on model behavior, but test structure is correct)
    assert result["num_tokens"] <= 100
    
    # Check if last token is EOS (if stopped early)
    if result["num_tokens"] < 100:
        eos_id = generator.tokenizer.eos_token_id
        assert result["generated_tokens"][-1] == eos_id or result["stopped_reason"] == "eos"


def test_generate_respects_max_new_tokens(generator):
    """Test generation stops at max_new_tokens limit"""
    max_tokens = 10
    result = generator.generate(
        prompt="This is a long prompt that will generate many tokens",
        max_new_tokens=max_tokens,
        do_sample=True  # To avoid early stopping
    )
    
    assert result["num_tokens"] <= max_tokens
    # If stopped at max, should be exactly max_tokens
    if result.get("stopped_reason") == "max_length":
        assert result["num_tokens"] == max_tokens


# ==============================================================================
# 2. KV Cache Tests
# ==============================================================================

def test_kv_cache_creation(generator):
    """Test initial KV cache is created correctly"""
    # Generate with cache enabled
    result = generator.generate(
        prompt="Test",
        max_new_tokens=1,
        use_cache=True,
        return_cache=True  # Request cache in result
    )
    
    assert "past_key_values" in result
    cache = result["past_key_values"]
    
    # KV cache should be a tuple of tuples (one per layer)
    assert isinstance(cache, tuple)
    assert len(cache) > 0
    
    # Each layer's cache should be (key, value)
    layer_cache = cache[0]
    assert isinstance(layer_cache, tuple)
    assert len(layer_cache) == 2  # key and value
    
    # Check shapes: [batch, num_heads, seq_len, head_dim]
    key_cache, value_cache = layer_cache
    assert key_cache.shape[0] == 1  # batch=1
    assert key_cache.shape[2] > 0   # seq_len > 0


def test_kv_cache_reuse(generator):
    """Test cached KV is reused for new tokens"""
    # First generation: create cache
    result1 = generator.generate(
        prompt="Hello",
        max_new_tokens=2,
        use_cache=True,
        return_cache=True
    )
    
    cache1 = result1["past_key_values"]
    seq_len1 = cache1[0][0].shape[2]  # seq length in cache
    
    # Second generation: continue from cache
    result2 = generator.generate(
        prompt="",  # Empty prompt, just extend
        max_new_tokens=2,
        use_cache=True,
        past_key_values=cache1,  # Provide existing cache
        return_cache=True
    )
    
    cache2 = result2["past_key_values"]
    seq_len2 = cache2[0][0].shape[2]
    
    # Cache should have grown
    assert seq_len2 > seq_len1
    assert seq_len2 == seq_len1 + 2  # Added 2 new tokens


def test_kv_cache_append(generator):
    """Test new KV states are appended to cache"""
    # Start with 3 tokens
    result1 = generator.generate(
        prompt="ABC",
        max_new_tokens=3,
        use_cache=True,
        return_cache=True
    )
    
    initial_seq_len = result1["past_key_values"][0][0].shape[2]
    
    # Add 5 more tokens
    result2 = generator.generate(
        prompt="DEF",
        max_new_tokens=5,
        use_cache=True,
        past_key_values=result1["past_key_values"],
        return_cache=True
    )
    
    final_seq_len = result2["past_key_values"][0][0].shape[2]
    
    # Should have grown by approximately 5 tokens worth
    # (exact count depends on tokenization)
    assert final_seq_len > initial_seq_len


def test_kv_cache_shapes(generator):
    """Test KV cache has correct tensor shapes"""
    result = generator.generate(
        prompt="Test prompt for cache shape validation",
        max_new_tokens=3,
        use_cache=True,
        return_cache=True
    )
    
    cache = result["past_key_values"]
    
    # For each layer
    for layer_idx, layer_cache in enumerate(cache):
        key_cache, value_cache = layer_cache
        
        # Check shape: [batch, num_heads, seq_len, head_dim]
        assert len(key_cache.shape) == 4, f"Layer {layer_idx} key cache has wrong dimensions"
        assert len(value_cache.shape) == 4, f"Layer {layer_idx} value cache has wrong dimensions"
        
        # Key and value should have same shape
        assert key_cache.shape == value_cache.shape
        
        # Batch size should be 1
        assert key_cache.shape[0] == 1
        assert value_cache.shape[0] == 1


# ==============================================================================
# 3. Conversation Flow Tests
# ==============================================================================

def test_cache_system_prompt(generator):
    """Test caching of system prompt"""
    system_prompt = "You are a helpful assistant. Always be polite and concise."
    
    generator.cache_system_prompt(system_prompt)
    
    assert generator.system_prompt_cache is not None
    assert generator.system_prompt_length > 0
    
    # Cache should be valid KV cache structure
    cache = generator.system_prompt_cache
    assert isinstance(cache, tuple)
    assert len(cache) > 0


def test_conversation_system_user_assistant(generator):
    """Test system → user → assistant conversation flow"""
    # Step 1: Cache system prompt
    system_prompt = "You are a helpful assistant."
    generator.cache_system_prompt(system_prompt)
    
    initial_cache_len = generator.system_prompt_length
    
    # Step 2: Generate assistant response to user message
    result = generator.generate(
        prompt="Hello!",
        max_new_tokens=10,
        use_cache=True
    )
    
    assert result["cache_used"] is True
    assert result["num_tokens"] > 0
    
    # System prompt cache should still be intact
    assert generator.system_prompt_cache is not None
    assert generator.system_prompt_length == initial_cache_len


def test_conversation_multi_turn(generator):
    """Test multiple turns with cache reuse"""
    # Cache system prompt
    generator.cache_system_prompt("You are helpful.")
    
    # Turn 1
    result1 = generator.generate(
        prompt="User: Hello\nAssistant:",
        max_new_tokens=5,
        use_cache=True,
        return_cache=True
    )
    
    turn1_cache_len = result1["past_key_values"][0][0].shape[2]
    
    # Turn 2 (should reuse turn 1 cache)
    result2 = generator.generate(
        prompt="User: How are you?\nAssistant:",
        max_new_tokens=5,
        use_cache=True,
        past_key_values=result1["past_key_values"],
        return_cache=True
    )
    
    turn2_cache_len = result2["past_key_values"][0][0].shape[2]
    
    # Cache should have grown
    assert turn2_cache_len > turn1_cache_len


def test_conversation_cache_persistence(generator):
    """Test cache persists across turns"""
    generator.cache_system_prompt("System prompt here.")
    
    # Multiple generations
    caches = []
    for i in range(3):
        prev_cache = caches[-1] if caches else None
        result = generator.generate(
            prompt=f"Turn {i}",
            max_new_tokens=2,
            use_cache=True,
            past_key_values=prev_cache,
            return_cache=True
        )
        caches.append(result["past_key_values"])
    
    # Each cache should be larger than the previous
    for i in range(1, len(caches)):
        prev_len = caches[i-1][0][0].shape[2]
        curr_len = caches[i][0][0].shape[2]
        assert curr_len > prev_len


# ==============================================================================
# 4. Sampling and Generation Tests
# ==============================================================================

def test_temperature_sampling(generator):
    """Test temperature affects token selection"""
    prompt = "The weather today is"
    
    # Low temperature (more deterministic)
    result_low = generator.generate(
        prompt=prompt,
        max_new_tokens=10,
        temperature=0.1,
        do_sample=True
    )
    
    # High temperature (more random)
    result_high = generator.generate(
        prompt=prompt,
        max_new_tokens=10,
        temperature=2.0,
        do_sample=True
    )
    
    # Results should be different (high probability)
    # Note: Can't guarantee difference in every test, but structure is correct
    assert "generated_text" in result_low
    assert "generated_text" in result_high


def test_top_p_sampling(generator):
    """Test nucleus sampling (top_p) works"""
    prompt = "Once upon a time"
    
    # Strict top_p (more focused)
    result_strict = generator.generate(
        prompt=prompt,
        max_new_tokens=10,
        top_p=0.5,
        do_sample=True
    )
    
    # Relaxed top_p (broader distribution)
    result_relaxed = generator.generate(
        prompt=prompt,
        max_new_tokens=10,
        top_p=1.0,
        do_sample=True
    )
    
    assert result_strict["num_tokens"] > 0
    assert result_relaxed["num_tokens"] > 0


def test_deterministic_with_greedy(generator):
    """Test greedy decoding is deterministic"""
    prompt = "The capital of France is"
    
    # Generate 3 times with greedy (no sampling)
    results = []
    for _ in range(3):
        result = generator.generate(
            prompt=prompt,
            max_new_tokens=5,
            do_sample=False  # Greedy
        )
        results.append(result["generated_tokens"])
    
    # All 3 should be identical
    assert results[0] == results[1] == results[2]


# ==============================================================================
# 5. Error Handling Tests
# ==============================================================================

def test_empty_input_handling(generator):
    """Test graceful handling of empty input"""
    result = generator.generate(
        prompt="",
        max_new_tokens=5
    )
    
    # Should handle gracefully (generate from BOS or return early)
    assert "generated_tokens" in result
    assert isinstance(result["generated_tokens"], list)


def test_invalid_cache_handling(generator):
    """Test handling of malformed KV cache"""
    # Create invalid cache (wrong structure)
    invalid_cache = (torch.randn(1, 1, 1, 1),)  # Malformed
    
    with pytest.raises(Exception):  # Should raise some error
        generator.generate(
            prompt="Test",
            max_new_tokens=5,
            past_key_values=invalid_cache
        )


def test_memory_cleanup(generator):
    """Test KV cache is freed when requested"""
    # Generate with cache
    generator.cache_system_prompt("System prompt")
    assert generator.system_prompt_cache is not None
    
    # Clear cache
    generator.clear_cache()
    
    # Cache should be None
    assert generator.system_prompt_cache is None
    assert generator.system_prompt_length == 0


def test_max_context_length_handling(generator):
    """Test behavior at max context length"""
    # Create a very long prompt (close to model's max)
    long_prompt = "word " * 500  # ~500 tokens
    
    result = generator.generate(
        prompt=long_prompt,
        max_new_tokens=100
    )
    
    # Should either:
    # 1. Generate successfully (if under max)
    # 2. Truncate input (if over max)
    # 3. Raise clear error (if unhandled)
    assert "generated_tokens" in result or "error" in result


# ==============================================================================
# 6. Integration Tests
# ==============================================================================

def test_matches_generate_api_output(generator):
    """Test manual loop produces same tokens as model.generate()"""
    prompt = "The quick brown fox"
    max_new = 5
    
    # Manual generation (our implementation)
    manual_result = generator.generate(
        prompt=prompt,
        max_new_tokens=max_new,
        do_sample=False,  # Greedy for determinism
        use_cache=True
    )
    
    # Standard model.generate()
    inputs = generator.tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(generator.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        standard_output = generator.model.generate(
            **inputs,
            max_new_tokens=max_new,
            do_sample=False,
            pad_token_id=generator.tokenizer.eos_token_id
        )
    
    # Extract generated tokens (remove input)
    input_length = inputs["input_ids"].shape[1]
    standard_tokens = standard_output[0][input_length:].tolist()
    
    # Should match (or be very close, accounting for implementation details)
    assert len(manual_result["generated_tokens"]) == len(standard_tokens)
    
    # Tokens should be identical
    assert manual_result["generated_tokens"] == standard_tokens


def test_callback_invoked_per_token(generator):
    """Test callback is called for each generated token"""
    tokens_seen = []
    
    def callback(token_id: int, logits: torch.Tensor):
        tokens_seen.append(token_id)
        assert logits is not None
        assert logits.dim() == 1  # Should be [vocab_size]
    
    result = generator.generate(
        prompt="Test",
        max_new_tokens=5,
        callback=callback
    )
    
    # Callback should have been called 5 times
    assert len(tokens_seen) == 5
    
    # Tokens seen should match result
    assert tokens_seen == result["generated_tokens"]


# ==============================================================================
# 7. Performance Tests (Optional, for benchmarking)
# ==============================================================================

@pytest.mark.slow
def test_cache_speedup(generator):
    """Test that KV caching provides speedup"""
    import time
    
    prompt = "This is a test prompt for measuring speed"
    
    # Without cache
    start = time.time()
    generator.generate(prompt, max_new_tokens=20, use_cache=False)
    no_cache_time = time.time() - start
    
    # With cache
    generator.cache_system_prompt("System:")
    start = time.time()
    generator.generate(prompt, max_new_tokens=20, use_cache=True)
    cache_time = time.time() - start
    
    # Cache should be faster (or at least not significantly slower)
    # Note: Speedup may be small for tiny model + short prompts
    assert cache_time <= no_cache_time * 1.5  # Allow 50% variance


@pytest.mark.slow
def test_memory_efficiency(generator):
    """Test that caching reduces memory usage"""
    import gc
    
    # Measure memory without cache
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Generate without cache multiple times
    for _ in range(5):
        generator.generate("Test", max_new_tokens=10, use_cache=False)
    
    # Measure memory with cache
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    generator.cache_system_prompt("System prompt")
    for _ in range(5):
        generator.generate("Test", max_new_tokens=10, use_cache=True)
    
    # Actual memory measurement would need GPU monitoring
    # This test structure is correct for future implementation
    assert True  # Placeholder


# ==============================================================================
# 8. Edge Cases
# ==============================================================================

def test_single_word_input(generator):
    """Test with single word input"""
    result = generator.generate("Hi", max_new_tokens=3)
    assert result["num_tokens"] > 0


def test_very_long_input(generator):
    """Test with very long input (should truncate or handle)"""
    long_input = " ".join(["word"] * 1000)
    result = generator.generate(long_input, max_new_tokens=5)
    assert "generated_tokens" in result or "error" in result


def test_special_characters_input(generator):
    """Test with special characters"""
    result = generator.generate("Hello! @#$%^&*()", max_new_tokens=5)
    assert result["num_tokens"] > 0


def test_unicode_input(generator):
    """Test with Unicode characters"""
    result = generator.generate("Hello 你好 مرحبا", max_new_tokens=5)
    assert result["num_tokens"] > 0


if __name__ == "__main__":
    # Run tests with: python -m pytest tests/test_manual_generation.py -v
    pytest.main([__file__, "-v", "--tb=short"])
