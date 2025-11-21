# H2O KV Cache Eviction - Implementation Plan

## Overview

Implement Heavy-Hitter Oracle (H2O) based KV cache eviction to enable unlimited conversation length while capping GPU memory usage.

**Key Benefits:**
- ✅ Keep full conversation text visible to model (no message deletion)
- ✅ Intelligent eviction based on actual attention patterns
- ✅ Model-driven decisions on what to keep
- ✅ We already track attention weights for introspection!

---

## Architecture

### Current System:
```
Conversation grows → Hit token limit → Prune old messages → Clear entire KV cache
```

### New System:
```
Conversation grows → Hit cache limit → Evict low-attention tokens → Keep conversation intact
```

---

## Implementation Phases

### Phase 1: Attention Tracking Infrastructure ✅ (Already Have!)

**What we already have:**
- `ActivationMonitor.capture_activations()` - Can capture attention weights
- Returns `attention_weights` dict when available
- Already exposed via introspection API

**What we need to add:**
- Track attention weights DURING generation (not just post-hoc analysis)
- Accumulate attention scores across turns
- Store cumulative attention per token

### Phase 2: H2O Cache Manager (New Component)

Create `src/memory/h2o_cache_manager.py`:

```python
class H2OCacheManager:
    """
    Heavy-Hitter Oracle KV Cache Eviction Manager
    
    Tracks cumulative attention scores and intelligently evicts
    low-attention tokens when cache reaches capacity.
    """
    
    def __init__(self, max_cache_tokens: int = 7000, system_prompt_tokens: int = 6000):
        """
        Args:
            max_cache_tokens: Maximum KV cache size (in tokens)
            system_prompt_tokens: Number of system prompt tokens (always keep)
        """
        self.max_cache_tokens = max_cache_tokens
        self.system_prompt_tokens = system_prompt_tokens
        
        # Track cumulative attention per token position
        self.attention_scores: Dict[int, float] = {}
        
        # Current cache state
        self.cached_tokens: Set[int] = set()
        self.total_tokens: int = 0
    
    def update_attention_scores(self, attention_weights: torch.Tensor):
        """
        Update cumulative attention scores from latest generation.
        
        Args:
            attention_weights: Attention tensor [layers, heads, query_len, key_len]
        """
        # Sum across layers and heads: [query_len, key_len]
        attention_sum = attention_weights.sum(dim=(0, 1))
        
        # Accumulate attention RECEIVED by each key position
        for key_pos in range(attention_sum.shape[1]):
            attention_received = attention_sum[:, key_pos].sum().item()
            self.attention_scores[key_pos] = self.attention_scores.get(key_pos, 0) + attention_received
    
    def select_tokens_to_keep(self) -> List[int]:
        """
        Select which tokens to keep in cache based on H2O policy.
        
        Returns:
            List of token positions to keep
        """
        if self.total_tokens <= self.max_cache_tokens:
            return list(range(self.total_tokens))  # Keep all
        
        # Always keep: System prompt + Recent window
        recent_window = 1000
        keep_tokens = set()
        
        # 1. System prompt (attention sink)
        keep_tokens.update(range(min(self.system_prompt_tokens, self.total_tokens)))
        
        # 2. Recent window
        recent_start = max(0, self.total_tokens - recent_window)
        keep_tokens.update(range(recent_start, self.total_tokens))
        
        # 3. High-attention tokens from conversation middle
        remaining_slots = self.max_cache_tokens - len(keep_tokens)
        conversation_start = self.system_prompt_tokens
        conversation_end = recent_start
        
        if conversation_start < conversation_end and remaining_slots > 0:
            # Get attention scores for conversation middle
            middle_tokens = range(conversation_start, conversation_end)
            token_scores = [(t, self.attention_scores.get(t, 0)) for t in middle_tokens]
            
            # Sort by attention (descending) and take top K
            token_scores.sort(key=lambda x: x[1], reverse=True)
            top_tokens = [t for t, _ in token_scores[:remaining_slots]]
            keep_tokens.update(top_tokens)
        
        return sorted(keep_tokens)
    
    def evict_cache(self, past_key_values: Tuple) -> Tuple:
        """
        Evict low-attention tokens from KV cache.
        
        Args:
            past_key_values: Current KV cache tuple
            
        Returns:
            Evicted KV cache (same structure, but with selective tokens)
        """
        keep_positions = self.select_tokens_to_keep()
        
        # TODO: Implement actual cache slicing
        # This requires indexing into the KV cache structure
        # and extracting only the specified token positions
        
        return past_key_values  # Placeholder
```

### Phase 3: Integration with ManualGenerator

Modify `src/manual_generation.py`:

```python
class ManualGenerator:
    def __init__(self, ...):
        # ... existing code ...
        
        # Add H2O cache manager
        self.h2o_cache = H2OCacheManager(
            max_cache_tokens=7000,
            system_prompt_tokens=len(self.system_prompt_tokens)
        )
        self.enable_h2o_eviction = False  # Feature flag
    
    def generate(self, prompt, max_new_tokens=100, **kwargs):
        # ... existing tokenization ...
        
        # Enable attention tracking
        if self.enable_h2o_eviction:
            kwargs['output_attentions'] = True
            kwargs['return_dict_in_generate'] = True
        
        # Generate
        result = self.model.generate(
            input_ids=input_ids,
            past_key_values=self.conversation_kv_cache,
            **kwargs
        )
        
        # Track attention if H2O is enabled
        if self.enable_h2o_eviction and hasattr(result, 'attentions'):
            self.h2o_cache.update_attention_scores(result.attentions)
            self.h2o_cache.total_tokens = len(input_ids[0]) + max_new_tokens
        
        # Update KV cache
        new_kv_cache = result.get('past_key_values')
        
        # Apply H2O eviction if needed
        if self.enable_h2o_eviction and new_kv_cache is not None:
            if self.h2o_cache.total_tokens > self.h2o_cache.max_cache_tokens:
                new_kv_cache = self.h2o_cache.evict_cache(new_kv_cache)
        
        self.conversation_kv_cache = new_kv_cache
        
        # ... rest of existing code ...
```

### Phase 4: Remove Conversation Pruning

Modify `scripts/experiments/phase1_base.py`:

```python
def manage_memory(self):
    """Memory management with H2O cache eviction (no message pruning)"""
    
    # OLD: Prune messages when limit reached
    # if total_tokens > limit:
    #     self.prune_conversation()
    
    # NEW: Let H2O handle cache eviction automatically
    # Messages stay, only KV cache is evicted
    
    if self.generator.enable_h2o_eviction:
        # H2O handles everything automatically
        # Just log current state
        cache_size = self.generator.h2o_cache.total_tokens
        logger.info(f"H2O Cache: {cache_size} tokens, "
                   f"keeping {len(self.generator.h2o_cache.cached_tokens)} in cache")
    else:
        # Fallback to old pruning logic
        self.prune_conversation()
```

---

## Technical Challenges

### 1. **KV Cache Structure Manipulation**

The KV cache is a nested tuple structure:
```python
past_key_values = (
    (key_layer0, value_layer0),  # Layer 0
    (key_layer1, value_layer1),  # Layer 1
    ...
)

# Where each key/value is: [batch_size, num_heads, seq_len, head_dim]
```

**Challenge:** Need to slice along `seq_len` dimension for selected tokens

**Solution:**
```python
def evict_cache(self, past_key_values: Tuple, keep_positions: List[int]) -> Tuple:
    evicted_cache = []
    for layer_kv in past_key_values:
        key, value = layer_kv
        # Index along seq_len dimension (dim=2)
        evicted_key = key[:, :, keep_positions, :]
        evicted_value = value[:, :, keep_positions, :]
        evicted_cache.append((evicted_key, evicted_value))
    return tuple(evicted_cache)
```

### 2. **Attention Mask Alignment**

When tokens are evicted, attention mask must match:
```python
# If we keep tokens [0-99, 500-599, 900-999]
# Attention mask must also have 300 positions (not 1000)
attention_mask = attention_mask[:, keep_positions]
```

### 3. **Flash Attention Compatibility**

- Flash Attention 2 doesn't support `output_attentions=True`
- Need to temporarily switch to eager attention for tracking

**Solution:**
```python
# During generation, use eager mode if H2O is enabled
if self.enable_h2o_eviction:
    original_attn = self.model.config._attn_implementation
    self.model.config._attn_implementation = 'eager'
    
    # ... generate with attention tracking ...
    
    self.model.config._attn_implementation = original_attn
```

**Performance impact:** ~20-30% slower with eager attention, but worth it for unlimited conversations.

---

## Configuration

Add to `configs/current_config.json`:

```json
{
  "kv_cache": {
    "eviction_strategy": "h2o",
    "max_cache_tokens": 7000,
    "enable_h2o": true,
    "track_attention_weights": true
  }
}
```

Add to Colab notebook:

```python
# Memory Management Strategy
USE_H2O_CACHE = True  # Enable H2O eviction (keeps full conversation)

if USE_H2O_CACHE:
    generator.enable_h2o_eviction = True
    generator.h2o_cache.max_cache_tokens = 7000
```

---

## Testing Plan

### Test 1: Basic Eviction
```python
# Generate 10,000 tokens of conversation
# Verify cache stays at 7000 tokens
# Verify all messages remain in conversation
```

### Test 2: Attention Tracking
```python
# Verify attention scores accumulate correctly
# Check heavy hitters are identified
# Validate eviction selects right tokens
```

### Test 3: Quality Comparison
```python
# Compare responses with H2O vs full cache
# Measure coherence and context retention
# Validate no degradation in quality
```

### Test 4: Long Conversations
```python
# Run 50+ turn conversations
# Verify no OOM errors
# Check conversation remains coherent
```

---

## Expected Outcomes

**Memory:**
- Cache capped at ~7 GB (7000 tokens)
- Never exceeds limit regardless of conversation length
- No more OOM errors

**Quality:**
- Full conversation visible to model
- Important context retained via attention-based selection
- Better coherence than message pruning

**Performance:**
- ~20-30% slower due to eager attention
- But enables unlimited conversation length
- Trade-off is worth it for long research sessions

---

## Migration Path

1. **Phase 1 (Now):** Keep existing pruning, add H2O in parallel (feature flag)
2. **Phase 2 (Testing):** Test H2O with small experiments
3. **Phase 3 (Gradual):** Default to H2O for new experiments
4. **Phase 4 (Complete):** Remove old pruning logic entirely

---

## References

- **H2O Paper:** Zhang et al., "H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models" (2023)
- **StreamingLLM:** Xiao et al., "Efficient Streaming Language Models with Attention Sinks" (2023)
- **Our activation tracking:** `src/introspection/activation_monitor.py`

---

*Document created: November 21, 2025*  
*Author: AGI Self-Modification Research Team*
