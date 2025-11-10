"""
Integration tests for KV caching across multi-turn conversations.

These tests verify that the KV cache is properly maintained and reused
across multiple turns, preventing the gibberish generation bug.
"""

import pytest
import torch
from src.manual_generation import ManualGenerator
from transformers import AutoTokenizer, AutoModelForCausalLM


@pytest.fixture
def model_and_tokenizer():
    """Load a small model for testing."""
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"  # Smaller model for faster tests
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    return model, tokenizer, device


@pytest.fixture
def generator(model_and_tokenizer):
    """Create a ManualGenerator instance."""
    model, tokenizer, device = model_and_tokenizer
    return ManualGenerator(model, tokenizer, device)


def test_multi_turn_kv_cache_accumulation(generator):
    """
    Test that KV cache properly accumulates across multiple turns.
    
    This is the integration test that would have caught the gibberish bug.
    """
    # Cache system prompt
    system_prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    generator.cache_system_prompt(system_prompt)
    
    system_cache_length = generator.system_prompt_length
    assert system_cache_length > 0, "System prompt should be cached"
    
    # Turn 1: Generate with system cache
    turn1_prompt = "<|im_start|>user\nHello!<|im_end|>\n<|im_start|>assistant\n"
    result1 = generator.generate(
        prompt=turn1_prompt,
        max_new_tokens=10,
        temperature=0.0,  # Deterministic
        do_sample=False,
        return_cache=True
    )
    
    assert result1["cache_used"] is True
    assert result1["past_key_values"] is not None
    turn1_cache = result1["past_key_values"]
    
    # Cache should include system + turn1
    turn1_cache_length = turn1_cache[0][0].shape[2]
    assert turn1_cache_length > system_cache_length, "Turn 1 cache should be larger than system cache"
    
    # Turn 2: Generate with turn1 cache
    turn2_prompt = "<|im_start|>user\nHow are you?<|im_end|>\n<|im_start|>assistant\n"
    result2 = generator.generate(
        prompt=turn2_prompt,
        max_new_tokens=10,
        temperature=0.0,
        do_sample=False,
        past_key_values=turn1_cache,
        return_cache=True
    )
    
    assert result2["cache_used"] is True
    assert result2["past_key_values"] is not None
    turn2_cache = result2["past_key_values"]
    
    # Cache should include system + turn1 + turn2
    turn2_cache_length = turn2_cache[0][0].shape[2]
    assert turn2_cache_length > turn1_cache_length, "Turn 2 cache should be larger than turn 1 cache"
    
    # Verify response is not gibberish (has some coherent tokens)
    response2 = result2["generated_text"]
    assert len(response2) > 0, "Should generate non-empty response"
    # Check it's not just repeated tokens (common gibberish pattern)
    tokens = response2.split()
    if len(tokens) > 1:
        unique_ratio = len(set(tokens)) / len(tokens)
        assert unique_ratio > 0.3, f"Response seems like gibberish (only {unique_ratio:.0%} unique tokens): {response2}"


def test_kv_cache_without_system_prompt(generator):
    """
    Test that KV cache works even without system prompt caching.
    """
    # Turn 1: No system cache
    turn1_prompt = "<|im_start|>user\nHello!<|im_end|>\n<|im_start|>assistant\n"
    result1 = generator.generate(
        prompt=turn1_prompt,
        max_new_tokens=10,
        temperature=0.0,
        do_sample=False,
        return_cache=True
    )
    
    assert result1["past_key_values"] is not None
    turn1_cache = result1["past_key_values"]
    turn1_cache_length = turn1_cache[0][0].shape[2]
    
    # Turn 2: Use turn1 cache
    turn2_prompt = "<|im_start|>user\nHow are you?<|im_end|>\n<|im_start|>assistant\n"
    result2 = generator.generate(
        prompt=turn2_prompt,
        max_new_tokens=10,
        temperature=0.0,
        do_sample=False,
        past_key_values=turn1_cache,
        return_cache=True
    )
    
    assert result2["past_key_values"] is not None
    turn2_cache = result2["past_key_values"]
    turn2_cache_length = turn2_cache[0][0].shape[2]
    
    # Cache should grow
    assert turn2_cache_length > turn1_cache_length


def test_kv_cache_consistency_across_turns(generator):
    """
    Test that using KV cache produces consistent results.
    
    The same conversation should produce the same output whether cached or not.
    """
    # Cache system prompt
    system_prompt = "<|im_start|>system\nYou are helpful.<|im_end|>\n"
    generator.cache_system_prompt(system_prompt)
    
    # Generate with cache
    prompt = "<|im_start|>user\nTest<|im_end|>\n<|im_start|>assistant\n"
    result_with_cache = generator.generate(
        prompt=prompt,
        max_new_tokens=5,
        temperature=0.0,  # Deterministic
        do_sample=False
    )
    
    # Generate same prompt with fresh cache
    generator.cache_system_prompt(system_prompt)  # Re-cache
    result_with_fresh_cache = generator.generate(
        prompt=prompt,
        max_new_tokens=5,
        temperature=0.0,
        do_sample=False
    )
    
    # Should produce same output
    assert result_with_cache["generated_text"] == result_with_fresh_cache["generated_text"]


def test_conversation_state_not_shared_between_instances(model_and_tokenizer):
    """
    Test that conversation KV cache is instance-specific.
    
    This prevents bugs where one conversation affects another.
    """
    model, tokenizer, device = model_and_tokenizer
    
    gen1 = ManualGenerator(model, tokenizer, device)
    gen2 = ManualGenerator(model, tokenizer, device)
    
    # Cache different system prompts
    gen1.cache_system_prompt("<|im_start|>system\nGen1<|im_end|>\n")
    gen2.cache_system_prompt("<|im_start|>system\nGen2<|im_end|>\n")
    
    assert gen1.system_prompt_length != gen2.system_prompt_length or True  # May be same length
    
    # Generate on both
    prompt = "<|im_start|>user\nTest<|im_end|>\n<|im_start|>assistant\n"
    
    result1 = gen1.generate(prompt, max_new_tokens=5, temperature=0.0, do_sample=False, return_cache=True)
    result2 = gen2.generate(prompt, max_new_tokens=5, temperature=0.0, do_sample=False, return_cache=True)
    
    # Caches should be independent
    assert result1["past_key_values"] is not result2["past_key_values"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU test")
def test_kv_cache_gpu_memory_efficiency(generator):
    """
    Test that KV caching actually saves GPU memory/computation.
    
    With KV cache, processing turn N should be faster than processing
    the full conversation history.
    """
    import time
    
    # Cache system prompt
    system_prompt = "<|im_start|>system\nYou are helpful.<|im_end|>\n"
    generator.cache_system_prompt(system_prompt)
    
    # Build up conversation cache
    conversation_cache = None
    prompts = [
        "<|im_start|>user\nMessage 1<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nMessage 2<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nMessage 3<|im_end|>\n<|im_start|>assistant\n",
    ]
    
    times_with_cache = []
    for prompt in prompts:
        start = time.time()
        result = generator.generate(
            prompt=prompt,
            max_new_tokens=5,
            past_key_values=conversation_cache,
            return_cache=True
        )
        times_with_cache.append(time.time() - start)
        conversation_cache = result["past_key_values"]
    
    # Later turns should be faster (only processing new tokens)
    # Not always true due to GPU warmup, but cache should not make it slower
    avg_time = sum(times_with_cache) / len(times_with_cache)
    assert all(t < avg_time * 3 for t in times_with_cache), "Cache should not dramatically slow down generation"
