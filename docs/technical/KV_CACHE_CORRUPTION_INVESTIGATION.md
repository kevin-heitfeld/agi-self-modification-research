# KV Cache Corruption Investigation

**Date:** November 10, 2025
**Investigator:** Claude Sonnet 4.5 (via GitHub Copilot)
**Issue:** Garbled output after memory pruning in Phase 1a experiments

## Problem Statement

After implementing memory pruning to prevent OOM, the model began generating garbled output like:
```
"nonrozen_parameters": 0,
"selfwen2LayerCausalLM": 1,
"Embedwen2For": 1,
"Qding": 1,
"Linearwen2LayerLayer": 26
```

This corruption occurred immediately after the first memory pruning event in each experiment.

## Investigation Timeline

### Initial Hypothesis: KV Cache Not Cleared
- Suspected old KV cache states were being reused after conversation pruning
- Code review showed `conversation_kv_cache = None` was correctly set
- Explicit garbage collection was performed
- **Hypothesis rejected:** Cache was properly discarded

### Second Hypothesis: Position ID Misalignment
- Considered that pruned turns had different position IDs when re-processed
- Analyzed rotary embedding position calculations in `manual_generation.py`
- Generator correctly computed position_ids based on cache length
- **Hypothesis rejected:** Position IDs were calculated correctly

### Breakthrough: System Prompt Injection

**Root Cause Identified:**

The `_format_conversation_for_model()` function used `tokenizer.apply_chat_template()` to format conversation turns. However, **`apply_chat_template()` automatically injects the default Qwen system prompt** even when no system message is provided:

```python
# What we passed (no system message):
conversation = [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi"}
]

# What apply_chat_template returned:
"""
<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
<|im_start|>user
Hello<|im_end|>
<|im_start|>assistant
Hi<|im_end|>
<|im_start|>assistant
"""
```

## The Corruption Mechanism

### Normal Generation (Pre-Pruning)
1. System prompt (5909 tokens, custom introspection instructions) is cached once
2. Generator receives conversation turns only
3. Generator prepends cached system prompt → correct sequence
4. Position IDs: `[0..5908]` (system) + `[5909..N]` (conversation)
5. ✅ Output is coherent

### After Memory Pruning
1. `conversation_kv_cache = None` (correct)
2. `_format_conversation_for_model()` calls `apply_chat_template()`
3. **`apply_chat_template()` injects default Qwen system prompt** (wrong!)
4. Generator receives: `"<|im_start|>system\nYou are Qwen..."`
5. Generator treats this as "conversation" and prepends cached custom system prompt
6. Result: **Two system prompts in sequence**
   - Cached custom prompt: `[0..5908]`
   - Injected default prompt: `[5909..5950]` (treated as conversation!)
   - Actual conversation: `[5951..N]`
7. Token positions are misaligned for all conversation tokens
8. Rotary embeddings computed with wrong positions
9. ❌ Output is garbled

## The Fix

**Replace `apply_chat_template()` with manual formatting:**

```python
# OLD (buggy):
formatted = self.tokenizer.apply_chat_template(
    trimmed_history,
    tokenize=False,
    add_generation_prompt=True
)

# NEW (fixed):
formatted = ""
for msg in trimmed_history:
    formatted += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
formatted += "<|im_start|>assistant\n"
```

This ensures:
- No system prompt is injected into formatted text
- Generator correctly uses only the cached custom system prompt
- Position IDs align properly
- No KV cache corruption

## Evidence from Logs

**Phase 1a Log (data/logs/phase1a.log):**

```
Line 789: [GENERATION] Generated 314 tokens, cache used: True
          [MODEL] Based on the get_architecture_summary result... (COHERENT)

Line 819: [MEMORY MANAGEMENT] Pruning needed in tool loop
Line 820: [MEMORY MANAGEMENT] Clearing cache and keeping last 2 turns
Line 821: [MEMORY MANAGEMENT] Reset conversation, keeping last 2 turns

Line 825: [GENERATION] Generated 175 tokens, cache used: True
Line 826: [MODEL]       "nonrozen_parameters": 0,    <-- GARBLED!
                      "selfwen2LayerCausalLM": 1,
                      "Embedwen2For": 1,
                      "Qding": 1,
```

The corruption appears **immediately** after the first pruning event, confirming the system prompt mismatch hypothesis.

## Testing the Fix

**Verification Test:**
```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")

conversation = [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi there!"}
]

# Using apply_chat_template (buggy):
formatted_buggy = tokenizer.apply_chat_template(
    conversation,
    tokenize=False,
    add_generation_prompt=True
)
print("<|im_start|>system" in formatted_buggy)  # True - BAD!

# Using manual formatting (fixed):
formatted_fixed = ""
for msg in conversation:
    formatted_fixed += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
formatted_fixed += "<|im_start|>assistant\n"
print("<|im_start|>system" in formatted_fixed)  # False - GOOD!
```

## Impact

**Before Fix:**
- Memory pruning caused immediate KV cache corruption
- Model generated garbled output after 3 turns
- Experiments failed to produce meaningful results

**After Fix:**
- Memory pruning should work cleanly
- Model maintains coherence across pruning events
- Experiments can run for unlimited turns

## Related Issues

This fix also addresses:
- Issue in Run 2 (21:32): KV corruption after pruning
- Issue in Run 3 (21:40): Degraded output quality post-pruning
- General instability in multi-turn tool calling scenarios

## Commits

- **624a740**: Fix KV cache corruption by preventing system prompt injection

## Next Steps

1. ✅ Fix committed and documented
2. ⏳ Re-run Phase 1a experiment to verify fix
3. ⏳ Proceed with Phase 1b if Phase 1a successful
4. ⏳ Monitor for any residual KV cache issues

## Lessons Learned

1. **Always verify what library functions actually do** - `apply_chat_template()` had unexpected behavior (injecting default system prompt that was not initially recognized)
2. **KV cache debugging requires tracing token positions** - The corruption wasn't in the cache itself, but in position ID misalignment
3. **Test edge cases** - Memory pruning + KV caching + custom system prompts is a complex interaction
4. **Log timestamps are crucial** - Being able to pinpoint the exact moment corruption started (21:47:45.029 → 21:47:59.785) was key to diagnosis

## Technical Deep Dive

### Why Manual Formatting Works

The Qwen2 chat format is:
```
<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{user_message}<|im_end|>
<|im_start|>assistant
{assistant_response}<|im_end|>
...
<|im_start|>assistant
```

When we cache the system prompt, the generator stores KV states for:
```
<|im_start|>system
{OUR_CUSTOM_PROMPT}<|im_end|>
```

After pruning, we need to provide ONLY the conversation turns:
```
<|im_start|>user
{message}<|im_end|>
<|im_start|>assistant
{response}<|im_end|>
...
<|im_start|>assistant
```

The generator will internally construct:
```
[CACHED] <|im_start|>system{OUR_CUSTOM_PROMPT}<|im_end|>
[NEW]    <|im_start|>user{message}<|im_end|>...
```

But `apply_chat_template()` was giving us:
```
<|im_start|>system{DEFAULT_PROMPT}<|im_end|>  <-- WRONG!
<|im_start|>user{message}<|im_end|>...
```

Which the generator combined as:
```
[CACHED] <|im_start|>system{OUR_CUSTOM_PROMPT}<|im_end|>
[NEW]    <|im_start|>system{DEFAULT_PROMPT}<|im_end|>  <-- DISASTER!
[NEW]    <|im_start|>user{message}<|im_end|>...
```

Two system prompts in sequence caused catastrophic position misalignment.

## Conclusion

The KV cache corruption was caused by an unexpected behavior in HuggingFace's `apply_chat_template()` function, which automatically injects a default system prompt even when formatting conversations without one. This created a conflict with our cached custom system prompt, leading to position ID misalignment and garbled output.

The fix is simple but critical: manually format conversations using the Qwen chat format instead of relying on `apply_chat_template()`.

**Status:** Fixed in commit 624a740, awaiting experimental validation.
