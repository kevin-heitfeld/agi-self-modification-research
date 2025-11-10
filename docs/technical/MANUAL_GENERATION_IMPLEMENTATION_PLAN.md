# Manual Generation Loop - Implementation Plan

**Status**: Ready to implement  
**Approach**: Test-Driven Development (TDD)  
**Estimated Time**: 3-4 days  
**Priority**: HIGH - Solves OOM immediately + enables Phase 2/3

---

## Executive Summary

**Problem**: System prompt (6,287 tokens) is processed EVERY turn, causing rapid memory exhaustion.
- Turn 1: 6,287 tokens
- Turn 2: 6,614 tokens  
- Turn 3: 12,783 tokens → **OOM!** (10.5 GB attention memory)

**Solution**: Manual generation loop with KV cache enables reusing system prompt cache.
- Turn 1: Cache 6,287 tokens once
- Turn 2: Reuse cache + 327 new tokens
- Turn 3: Reuse cache + 6,169 new tokens = **~6,496 tokens** (2.7 GB attention) ✅

**Benefits**:
1. **74% memory reduction** for attention (10.5 GB → 2.7 GB)
2. **Foundation for Phase 2/3** (per-token introspection, real-time monitoring)
3. **Cleaner architecture** (separation of concerns, more testable)
4. **Smart tool result optimization** (decide what data to include per-token)

---

## Phase 1: Comprehensive Test Suite (Day 1)

### Test File: `tests/test_manual_generation.py`

#### 1.1 Basic Generation Tests
```python
def test_generate_single_token():
    """Test generating a single token from input"""
    # Setup: model, tokenizer, simple input "Hello"
    # Expected: Returns 1 token, correct token ID
    
def test_generate_multiple_tokens():
    """Test generating multiple tokens (e.g., 5 tokens)"""
    # Setup: model, tokenizer, input "Once upon a"
    # Expected: Returns 5 tokens in correct order

def test_generate_until_eos():
    """Test generation stops at EOS token"""
    # Setup: Short prompt that model will complete quickly
    # Expected: Stops at EOS, doesn't exceed max_new_tokens

def test_generate_respects_max_new_tokens():
    """Test generation stops at max_new_tokens limit"""
    # Setup: max_new_tokens=10, prompt that would generate more
    # Expected: Stops at exactly 10 tokens
```

#### 1.2 KV Cache Tests
```python
def test_kv_cache_creation():
    """Test initial KV cache is created correctly"""
    # Setup: First forward pass with input tokens
    # Expected: past_key_values has correct shape and dtype
    
def test_kv_cache_reuse():
    """Test cached KV is reused for new tokens"""
    # Setup: Generate with cache, then generate more
    # Expected: Second generation uses cache (faster, correct)
    
def test_kv_cache_append():
    """Test new KV states are appended to cache"""
    # Setup: Cache with 5 tokens, generate 3 more
    # Expected: Cache now has 8 tokens worth of KV states
    
def test_kv_cache_shapes():
    """Test KV cache has correct tensor shapes"""
    # Expected: [batch=1, num_heads, seq_len, head_dim]
```

#### 1.3 Conversation Flow Tests
```python
def test_conversation_system_user_assistant():
    """Test system → user → assistant conversation"""
    # Setup: System prompt + user message
    # Expected: System prompt cached, user message appended
    
def test_conversation_multi_turn():
    """Test multiple turns with cache reuse"""
    # Setup: System + User1 + Asst1 + User2
    # Expected: Cache grows correctly, each turn reuses previous
    
def test_conversation_cache_persistence():
    """Test cache persists across turns"""
    # Setup: Generate turn 1, then turn 2
    # Expected: Turn 2 cache includes turn 1 KV states
```

#### 1.4 Sampling and Generation Tests
```python
def test_temperature_sampling():
    """Test temperature affects token selection"""
    # Setup: Same prompt, temperature=0.1 vs 1.0
    # Expected: Low temp more deterministic, high temp more random
    
def test_top_p_sampling():
    """Test nucleus sampling (top_p) works"""
    # Setup: top_p=0.9 vs top_p=1.0
    # Expected: top_p filters low-probability tokens
    
def test_deterministic_with_temp_zero():
    """Test temperature=0 is deterministic"""
    # Setup: Generate same prompt 3 times with temp=0
    # Expected: All 3 outputs identical
```

#### 1.5 Error Handling Tests
```python
def test_empty_input_handling():
    """Test graceful handling of empty input"""
    # Setup: input_ids = []
    # Expected: Returns error or handles gracefully
    
def test_invalid_cache_handling():
    """Test handling of malformed KV cache"""
    # Setup: Corrupt cache (wrong shape)
    # Expected: Detects and handles error
    
def test_memory_cleanup():
    """Test KV cache is freed when requested"""
    # Setup: Generate with cache, then clear_cache()
    # Expected: Memory released, cache is None
    
def test_max_context_length():
    """Test behavior at max context length"""
    # Setup: Cache approaching model's max length
    # Expected: Handles gracefully (error or truncation)
```

#### 1.6 Integration Tests
```python
def test_matches_generate_api_output():
    """Test manual loop matches model.generate() output"""
    # Setup: Same input, same seed, same params
    # Expected: Manual loop produces identical tokens
    
def test_matches_generate_api_with_sampling():
    """Test with sampling (temperature, top_p)"""
    # Setup: Set seed, use sampling params
    # Expected: Distributions match (statistical test)
```

---

## Phase 2: Core Implementation (Days 2-3)

### 2.1 File Structure

```
src/
  manual_generation.py       # Main implementation
  model_manager.py            # Add manual_generate() method
tests/
  test_manual_generation.py  # Comprehensive test suite
```

### 2.2 Implementation: `src/manual_generation.py`

```python
"""
Manual Generation Loop with KV Caching

Provides fine-grained control over token generation for:
1. System prompt KV caching (massive memory savings)
2. Per-token introspection (Phase 2)
3. Real-time activation monitoring (Phase 2)
4. Mid-generation interventions (Phase 3)
"""

import torch
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass

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
    """
    
    def __init__(self, model, tokenizer, device: str = "cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
        # Cached system prompt KV states
        self.system_prompt_cache: Optional[Tuple] = None
        self.system_prompt_length: int = 0
        
    def cache_system_prompt(self, system_prompt: str) -> None:
        """
        Pre-compute and cache system prompt KV states.
        
        This is the KEY optimization: process 6000+ token system prompt
        once, reuse forever.
        
        Args:
            system_prompt: The system prompt text
        """
        # TODO: Tokenize system prompt
        # TODO: Forward pass to get KV cache
        # TODO: Store in self.system_prompt_cache
        pass
        
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_p: float = 1.0,
        do_sample: bool = True,
        use_cache: bool = True,
        callback: Optional[Callable[[int, torch.Tensor], None]] = None
    ) -> Dict[str, Any]:
        """
        Generate tokens one at a time with optional callback.
        
        Args:
            prompt: Input text (user message)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            do_sample: Whether to sample (vs greedy)
            use_cache: Whether to use/reuse KV cache
            callback: Optional function called per token: callback(token_id, logits)
        
        Returns:
            Dict with:
            - generated_text: Generated response
            - generated_tokens: List of token IDs
            - num_tokens: Number of tokens generated
            - cache_used: Whether KV cache was used
        """
        # TODO: Implementation in TDD style
        pass
    
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
        # TODO: Implement sampling logic
        pass
    
    def clear_cache(self) -> None:
        """Clear all cached KV states"""
        self.system_prompt_cache = None
        self.system_prompt_length = 0


def test_manual_generator():
    """Quick smoke test for development"""
    # TODO: Simple test to verify basic functionality
    pass
```

### 2.3 Integration into ModelManager

Update `src/model_manager.py`:

```python
class ModelManager:
    def __init__(self, ...):
        # ...existing code...
        
        # Add manual generator
        self.manual_generator = None
        
    def enable_manual_generation(self, system_prompt: str) -> None:
        """
        Enable manual generation with system prompt caching.
        
        Args:
            system_prompt: System prompt to cache
        """
        from src.manual_generation import ManualGenerator
        
        self.manual_generator = ManualGenerator(
            self.model,
            self.tokenizer,
            device=self.device
        )
        
        # Cache the system prompt
        self.manual_generator.cache_system_prompt(system_prompt)
        
    def generate_manual(self, prompt: str, **kwargs) -> str:
        """
        Generate using manual loop (uses cached system prompt).
        
        Args:
            prompt: User message (NOT including system prompt)
            **kwargs: Generation parameters
            
        Returns:
            Generated text
        """
        if self.manual_generator is None:
            raise RuntimeError("Manual generation not enabled. Call enable_manual_generation() first.")
        
        result = self.manual_generator.generate(prompt, **kwargs)
        return result["generated_text"]
```

---

## Phase 3: Tool Result Optimization (Day 3)

### 3.1 Problem Analysis

Current `get_weight_statistics` returns ~2000 tokens per layer:
- ✅ **Keep (essential)**: name, shape, mean, std, min, max, l2_norm (~300 tokens)
- ❌ **Remove (verbose)**: histogram (huge!), full percentiles, zeros_percentage (~1700 tokens)

### 3.2 Create Compact Mode

Update `src/introspection/weight_inspector.py`:

```python
def get_weight_statistics(
    self,
    layer_name: Union[str, List[str]],
    compact: bool = False  # NEW parameter
) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Get weight statistics (optionally in compact mode).
    
    Args:
        layer_name: Layer name(s) to analyze
        compact: If True, returns only essential statistics (75% smaller)
    
    Compact mode includes:
    - name, shape, num_parameters
    - mean, std, min, max
    - l2_norm
    
    Full mode additionally includes:
    - median, abs_mean
    - zeros_percentage, near_zero_percentage
    - l1_norm
    - histogram (bins, counts, edges)
    - percentiles [5th, 25th, 50th, 75th, 95th]
    """
    # TODO: Implement compact flag
```

Similarly for `get_activation_statistics`.

### 3.3 Enable Compact Mode in Tool Interface

Update `src/tool_interface.py`:

```python
# In get_available_tools(), update documentation:

def get_weight_statistics(layer_name: Union[str, List[str]], compact: bool = False):
    """
    ...
    
    Args:
        compact: If True, returns essential stats only (75% smaller output)
                Recommended for examining many layers to save memory.
    
    Examples:
        >>> # Full statistics (verbose)
        >>> get_weight_statistics(layer_name="model.layers.0.mlp.gate_proj.weight")
        
        >>> # Compact statistics (recommended for multiple layers)
        >>> get_weight_statistics(
        ...     layer_name=["layer0", "layer1", "layer2"],
        ...     compact=True
        ... )
    """
```

### 3.4 Update System Prompt

In `scripts/experiments/phase1_base.py`:

```python
def get_memory_management_instructions(self) -> str:
    return """MEMORY MANAGEMENT STRATEGY:
    ...
    
    IMPORTANT: Use compact=True when examining multiple layers:
    - get_weight_statistics(layer_name=[...], compact=True)
    - get_activation_statistics(layer_name=[...], compact=True)
    
    This reduces output size by ~75% and prevents memory overflow.
    """
```

---

## Phase 4: Integration and Testing (Day 4)

### 4.1 Update Phase1BaseSession

Update `scripts/experiments/phase1_base.py`:

```python
class Phase1BaseSession:
    def __init__(self, ...):
        # ...existing code...
        
        # Enable manual generation with system prompt caching
        system_prompt = self.create_initial_prompt()
        self.model_mgr.enable_manual_generation(system_prompt)
        
    def chat(self, user_message: str) -> str:
        """
        Send message and get response (using manual generation).
        """
        # Build conversation without system prompt
        # (system prompt is already cached!)
        conversation_text = self._format_conversation_for_model(
            include_system=False  # NEW parameter
        )
        
        # Use manual generation (reuses cached system prompt)
        response = self.model_mgr.generate_manual(
            conversation_text,
            max_new_tokens=500,
            temperature=0.7,
            do_sample=True
        )
        
        # ...rest of existing code...
```

### 4.2 Verification Tests

1. **Memory Test**: Run Phase 1a, verify Turn 3 uses ~6.5K tokens (not 12.8K)
2. **Correctness Test**: Compare outputs with/without manual loop (should match)
3. **Performance Test**: Measure speedup from KV cache reuse
4. **Stability Test**: Run full 20-turn investigation without OOM

### 4.3 Metrics to Track

Before/After comparison:
- Turn 3 token count: 12,783 → ~6,496 (49% reduction)
- Turn 3 attention memory: 10.5 GB → 2.7 GB (74% reduction)
- Maximum turns before OOM: 3-4 → 15-20+ (4-5x improvement)
- Generation speed: Baseline → 2-3x faster (cache reuse)

---

## Phase 5: Documentation and Cleanup

### 5.1 Update Documentation

- Update `MANUAL_GENERATION_LOOP.md` with implementation details
- Document compact mode in tool interface docs
- Add examples to README

### 5.2 Git Commits

Suggested commit sequence:
1. `Add comprehensive tests for manual generation loop`
2. `Implement ManualGenerator with KV caching`
3. `Integrate manual generation into ModelManager`
4. `Add compact mode to introspection tools`
5. `Enable manual generation in Phase1BaseSession`
6. `Update documentation for manual generation`

---

## Success Criteria

✅ **All tests pass** (100% test coverage for manual generation)  
✅ **Memory reduction verified** (Turn 3: 12.8K → 6.5K tokens)  
✅ **No OOM for 20+ turns** (vs 3-4 turns currently)  
✅ **Output correctness** (matches model.generate() baseline)  
✅ **Performance improvement** (2-3x faster with cache)  
✅ **Clean architecture** (separation of concerns, maintainable)  

---

## Risk Mitigation

**Risk 1**: KV cache shape mismatches  
**Mitigation**: Comprehensive shape tests, assertions in code

**Risk 2**: Sampling doesn't match model.generate()  
**Mitigation**: Statistical tests, seed-based determinism checks

**Risk 3**: Memory leaks from cache  
**Mitigation**: Explicit cache management, memory monitoring tests

**Risk 4**: Integration breaks existing code  
**Mitigation**: All 5 phase variants tested, can rollback if needed

---

## Timeline

- **Day 1 (Today)**: Write all tests in `tests/test_manual_generation.py`
- **Day 2**: Implement core `ManualGenerator` class (TDD)
- **Day 3**: Add compact mode + integrate into ModelManager
- **Day 4**: Integration testing + verification in Colab
- **Total**: 3-4 days

---

## Next Steps

1. ✅ Create this implementation plan
2. ⏭️ **START**: Create `tests/test_manual_generation.py` with all test cases
3. ⏭️ Run tests (should all fail initially - that's good!)
4. ⏭️ Implement `ManualGenerator` incrementally until tests pass
5. ⏭️ Integrate and verify

**Ready to begin! Let's start with the test suite.** 🚀
