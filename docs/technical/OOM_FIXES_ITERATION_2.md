# OOM Fixes - Iteration 2

**Date:** 2025-11-10
**Status:** Testing in progress
**Commit:** (pending)

## Problem Analysis from phase1a.log

### Issue 1: OOM After Only 5-6 Tool Calls ❌

**Evidence:**
```
2025-11-10 04:32:17,764 - [MEMORY OPTIMIZATION] Removed 4 old messages, pruned 1 old tool results
2025-11-10 04:32:20,458 - ERROR: CUDA out of memory. Tried to allocate 2.10 GiB.
GPU 0 has a total capacity of 14.74 GiB of which 2.10 GiB is free.
Process has 12.64 GiB memory in use.
```

**Analysis:**
- Model made only ~5-6 tool calls before OOM
- Memory optimization DID run (removed 4 messages, pruned 1 tool result)
- But pruning was **not aggressive enough**
- Still using 12.64 GiB (same as before fix!)

**Root Cause:**
Tool results are MASSIVE:
- Activation dumps: 3000-5000 tokens each
- Weight statistics: 2000-4000 tokens each
- Architecture info: 1000-2000 tokens each

Even with "last 3 exchanges" (6 messages), if each has 4000 token tool results:
- 3 tool results × 4000 tokens = 12,000 tokens just in recent results
- Plus system prompt (2000 tokens) = 14,000 tokens
- Plus conversation reasoning (5000+ tokens) = 19,000+ tokens total
- Plus max_new_tokens (700) = potential 19,700 tokens → ~40K attention matrix cells

**O(n²) attention complexity means:**
- 15,000 tokens → 225M attention cells
- 20,000 tokens → 400M attention cells (77% more!)
- Each cell = 2-4 bytes → 800MB-1.6GB just for attention in forward pass

### Issue 2: Model Not Using `record_observation()` ❌

**Evidence:**
```bash
$ grep -n "record_observation" phase1a.log
617: def record_observation(obs_type: str, category: str, description: str,
646:         >>> record_observation(
656:         >>> record_observation(
754: - `record_observation(obs_type="INTROSPECTION", ...)` - Save discoveries
```

All matches are from the tool documentation in system prompt. **Zero actual calls to record_observation!**

**Analysis:**
Model sees the tool in the documentation but never uses it. Why?
1. ✅ Tool is available in system prompt
2. ✅ Tool has clear documentation with examples
3. ❌ Model never receives explicit prompt/reminder to use it
4. ❌ Warning system doesn't actually warn the MODEL before pruning

**The Warning System Issue:**
Looking at code (lines 228-284), the warning IS sent to the model:
```python
warning_message = "[SYSTEM WARNING] You've made X turns. Old tool results are being removed..."
self.conversation_history.append({"role": "user", "content": warning_message})
```

BUT: The warning triggers at turn 5, then every 3 turns (5, 8, 11, 14...)
- Model crashed at turn ~6
- So warning triggered once at turn 5
- Model got ONE chance to respond
- Then crashed at turn 6

**The Pruning Reminder Issue:**
The replacement message `"[TOOL_RESULTS removed to save memory...]"` is inserted into PRUNED messages from OLD exchanges. The model already saw those exchanges when they were fresh. The reminder doesn't warn FUTURE generations - it just explains past pruning.

### Issue 3: No Input Token Logging in Main Loop ❌

**Evidence:**
```bash
$ grep "[GENERATION] Input tokens:" phase1a.log
# No results!
```

The logging was added to the pre-warning generation (line 248) but NOT to the main generation loop (line 315+).

**Impact:**
- Can't diagnose how many tokens were in context when OOM occurred
- Can't track if pruning is actually reducing token counts
- Can't validate our O(n²) theory

## Changes Made - Iteration 2

### 1. More Aggressive Pruning 🔧

**Before:**
```python
KEEP_RECENT_EXCHANGES = 3  # Last 3 exchanges = 6 messages
MAX_TOTAL_TURNS = 8
```

**After:**
```python
KEEP_RECENT_EXCHANGES = 2  # Last 2 exchanges = 4 messages (REDUCED)
MAX_TOTAL_TURNS = 6         # REDUCED from 8
```

**Expected Impact:**
- 33% fewer recent messages with full tool results
- 25% fewer total turns in context
- Memory savings: ~4,000-6,000 tokens → 16-36% less memory
- Attention savings: n² scaling → ~30-50% fewer attention cells

### 2. Reduced max_new_tokens 🔧

**Before:**
```python
max_new_tokens=700
```

**After:**
```python
max_new_tokens=500
```

**Expected Impact:**
- 28% shorter model responses
- Reduces total context length after generation
- Faster generation (fewer tokens to generate)

### 3. Added Input Token Logging 🔧

**Before:**
```python
# Generate response
conversation_text = self._format_conversation_for_model()
inputs = self.tokenizer(conversation_text, return_tensors="pt")
# ... no logging ...
with torch.no_grad():
    outputs = self.model.generate(**inputs, ...)
```

**After:**
```python
# Generate response
conversation_text = self._format_conversation_for_model()
inputs = self.tokenizer(conversation_text, return_tensors="pt")
input_length = inputs['input_ids'].shape[1]

# Log input size for monitoring
self.logger.info(f"[GENERATION] Input tokens: {input_length}, max_new_tokens: 500")

with torch.no_grad():
    outputs = self.model.generate(**inputs, ...)
```

**Expected Impact:**
- Can track token count evolution across turns
- Can verify pruning is working
- Can identify exact turn where OOM occurs
- Can validate if we're hitting token limits

## Still TODO - Issues Not Yet Fixed

### 1. Model Learning to Use `record_observation()` 🔄

**The Problem:**
Model has the tool but doesn't use it proactively. Current approach:
1. Warning triggers at turn 5+ (every 3 turns)
2. Model gets one turn to respond
3. But no explicit instruction to "save important findings NOW"

**Possible Solutions:**

**Option A: More Direct Warning (Recommended)**
```python
warning_message = """[SYSTEM WARNING] Memory limit approaching!

You've made {num_exchanges} investigation turns. To prevent data loss:
1. Use record_observation() NOW to save any important discoveries
2. Old tool results will be removed after this turn
3. You can query saved observations later with query_memory()

IMPORTANT: If you don't save findings now, they'll be lost!"""
```

**Option B: Automatic Summarization Prompt**
After every 2-3 tool calls, inject:
```python
"Before continuing, please record_observation() to save what you've learned so far."
```

**Option C: Teach in System Prompt (Long-term)**
Add to initial instructions:
```
MEMORY MANAGEMENT STRATEGY:
Every 2-3 tool calls, use record_observation() to save discoveries.
Example pattern:
1. Call get_architecture_summary()
2. Analyze results
3. Call record_observation(obs_type="INTROSPECTION", category="architecture", ...)
4. Continue with next investigation
```

### 2. Make Pruning Reminder Visible 🔄

**Current Issue:**
The message `"[TOOL_RESULTS removed to save memory...]"` replaces pruned tool results, but those are from OLD exchanges the model already processed.

**Possible Solutions:**

**Option A: Add Notice to User Message (Recommended)**
When pruning occurs, prepend to next user message:
```python
next_user_message = "[NOTE: Old tool results have been pruned from context] " + user_message
```

**Option B: Inject as Separate System Message**
After pruning, add:
```python
self.conversation_history.append({
    "role": "user",
    "content": "[SYSTEM] Previous tool results removed to save memory. Important: Use query_memory() to access saved observations."
})
```

### 3. Even More Aggressive Pruning? 🤔

If OOM still persists with current changes, consider:

**Option A: Keep Only 1 Recent Exchange**
```python
KEEP_RECENT_EXCHANGES = 1  # Only last exchange (2 messages)
MAX_TOTAL_TURNS = 4
```

**Option B: Summarize Tool Results**
Instead of removing old tool results entirely, replace with AI-generated summary:
```python
summary = summarize_tool_result(result)  # 100-200 tokens max
```

**Option C: Reduce max_new_tokens Further**
```python
max_new_tokens=300  # Force more concise responses
```

**Option D: Layer-by-Layer Examination**
Instead of returning full dumps, make tools return one layer at a time:
```python
# Instead of: get_activation_statistics(layer_name="model.layers.0.self_attn")
# Model must: get_layer_list() → iterate with get_single_layer_stats(layer_id=0)
```

## Memory Budget Analysis

### Current Constraints
- **GPU Total:** 14.74 GiB
- **Model Weights:** ~6 GB (Qwen 2.5 3B in FP16)
- **Available for Context:** ~8 GB
- **Context Needed:** Input tokens + KV cache + attention matrices + activations

### Token Budget with New Settings

**Aggressive Pruning (KEEP_RECENT_EXCHANGES=2, MAX_TOTAL_TURNS=6):**
- System prompt: 2,000 tokens
- Recent 2 tool results: 2 × 4,000 = 8,000 tokens
- Recent 2 assistant responses: 2 × 500 = 1,000 tokens
- Older assistant responses (pruned): 2 × 500 = 1,000 tokens
- **Total input: ~12,000 tokens**

**Attention Memory (Forward Pass):**
- Input tokens: 12,000
- max_new_tokens: 500
- Peak during generation: 12,500 tokens
- Attention matrix: 12,500² × num_heads (16) × bytes_per_element (2-4)
  - = 156M cells × 16 heads × 4 bytes
  - = ~10 GB **← STILL TOO HIGH!**

**The Problem:**
Even with aggressive pruning, attention memory alone approaches our limit!

### Why Attention is O(n²) Killer

```
Tokens | Attention Cells | Memory (16 heads, FP32)
-------|----------------|-------------------------
 5,000 | 25M            | 1.6 GB
10,000 | 100M           | 6.4 GB
12,000 | 144M           | 9.2 GB ← We're here
15,000 | 225M           | 14.4 GB ← Exceeds GPU!
20,000 | 400M           | 25.6 GB ← Way over
```

**The Math:**
- Each attention cell: 4 bytes (FP32 in attention computation)
- 16 attention heads
- Total: n² × 16 × 4 bytes

**At 12,000 tokens:**
- 12,000² = 144,000,000 cells per head
- × 16 heads = 2,304,000,000 cells
- × 4 bytes = 9.2 GB **just for attention!**

### Conclusion: Need EVEN More Aggressive Pruning

**Target: Stay under 10,000 tokens total input**

To achieve this with 4,000 token tool results:
- Can only keep 1-2 recent tool results
- Must prune more aggressively
- Or implement tool result summarization

**Recommended Next Step:**
Try `KEEP_RECENT_EXCHANGES = 1` if current fix doesn't work.

## Testing Plan

1. ✅ Commit changes (KEEP_RECENT_EXCHANGES=2, MAX_TOTAL_TURNS=6, max_new_tokens=500, logging added)
2. ⏳ Run Phase 1a in Colab with new settings
3. ⏳ Monitor logs for:
   - `[GENERATION] Input tokens:` values over time
   - Whether OOM still occurs
   - If so, at what token count
4. ⏳ Analyze results:
   - If OOM at >12K tokens → reduce to KEEP_RECENT_EXCHANGES=1
   - If OOM at <10K tokens → investigate other memory leak (activations? model state?)
   - If successful → test full investigation (20+ turns)
5. ⏳ Address `record_observation()` usage with improved warnings

## References

- Attention memory formula: `O(n² × h × b)` where n=tokens, h=heads, b=bytes
- Flash Attention paper: Shows attention is memory bottleneck for long context
- HuggingFace memory debugging: https://huggingface.co/docs/transformers/main/en/perf_train_gpu_one
- PyTorch CUDA management: https://pytorch.org/docs/stable/notes/cuda.html

---

**Next Update:** After testing in Colab with iteration 2 settings
