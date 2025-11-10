# OOM Fixes - Iteration 2

**Date:** 2025-11-10  
**Status:** Implementation complete - Ready for testing  
**Commits:** b03f743, 380d448, d244da2## Problem Analysis from phase1a.log

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

**The Warning System Issue (BUG FOUND AND FIXED):**
Looking at code (lines 228-284), the warning SHOULD have been sent, but had a critical bug:
```python
MAX_EXCHANGES_BEFORE_PRUNING = 5
if num_exchanges >= 5 and num_exchanges % 3 == 0:  # BUG HERE!
    warning_message = "[SYSTEM WARNING] You've made X turns..."
```

**THE BUG:** The condition `num_exchanges % 3 == 0` means:
- Turn 3: 3 % 3 = 0 ✓ (but >= 5 fails)
- Turn 5: 5 % 3 = 2 ✗ **SKIPPED!**
- Turn 6: 6 % 3 = 0 ✓
- Turn 9: 9 % 3 = 0 ✓

So warning would trigger at turns: 6, 9, 12, 15... (NOT at 5!)

**What Actually Happened:**
- Model made 4 exchanges (turns 1-4)
- OOM crashed at turn 5
- Warning was NEVER sent (would have triggered at turn 6)
- This explains why model never used `record_observation()`!

**FIXED in commit d244da2:**
```python
FIRST_WARNING_AT = 3
should_warn = num_exchanges >= 3 and (num_exchanges - 3) % 2 == 0
```
Now triggers at: 3, 5, 7, 9, 11... (every 2 turns starting at turn 3)

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

### 1. Fixed Critical Warning System Bug 🐛 **CRITICAL FIX**

**Commit:** d244da2

**Bug:**
```python
# OLD (BROKEN):
if num_exchanges >= 5 and num_exchanges % 3 == 0:
    # Triggers at: 6, 9, 12... (skips 5!)
```

**Fix:**
```python
# NEW (WORKING):
FIRST_WARNING_AT = 3
should_warn = num_exchanges >= 3 and (num_exchanges - 3) % 2 == 0
# Triggers at: 3, 5, 7, 9, 11...
```

**Impact:**
- Warning now fires at turn 3 (BEFORE OOM danger at turns 4-5)
- More frequent reminders (every 2 turns vs every 3)
- Model actually gets warned before data loss
- Explains why record_observation() was never used in phase1a.log

### 2. More Aggressive Pruning 🔧

**Commit:** b03f743

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

### 3. Reduced max_new_tokens 🔧

**Commit:** b03f743

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

### 4. Added Input Token Logging 🔧

**Commit:** b03f743

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

### 5. Teach Memory Management in System Prompt 📚

**Commits:** 380d448 (refactored to base class)

Added `get_memory_management_instructions()` method to `Phase1BaseSession`:
- Teaches model to use `record_observation()` every 2-3 tool calls
- Shows example workflow with concrete code
- Explains why (memory overflow, auto-removal after ~5 turns)
- Demonstrates `query_memory()` for retrieval
- Applied to all 5 phase variants (1a-1e)

**Benefits:**
- DRY principle: Single source of truth in base class
- Model learns pattern from the start
- Combined with warning system for reinforcement

### 6. Enhanced Warning Message 💬

**Commit:** 380d448

Made warning message more direct and urgent:
```python
"""[SYSTEM WARNING] Memory limit approaching!

You've made {num_exchanges} investigation turns. To prevent data loss:
1. Use record_observation() NOW to save any important discoveries
2. Old tool results will be removed after this turn
3. You can query saved observations later with query_memory()

IMPORTANT: If you don't save your findings now, they'll be lost forever!
Take this turn to record_observation() for any important discoveries."""
```

## Still TODO - Issues Not Yet Fixed

### 1. Test in Colab with All Fixes 🔄

**Now IMPLEMENTED (commits 380d448, d244da2):**
- ✅ Option A: More direct warning (implemented)
- ✅ Option C: Teach in system prompt (implemented)
- ✅ Fixed bug: Warning now actually fires (turns 3, 5, 7...)

**Remaining work:**
Monitor next test run to verify model actually uses `record_observation()` now that:
1. It's taught the pattern from the start
2. Warnings actually fire before OOM
3. Warnings are more direct and urgent

### 2. Even More Aggressive Pruning if Needed 🤔

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

## Summary of All Changes

### Commits
1. **b03f743**: More aggressive pruning + reduced tokens + logging
   - KEEP_RECENT_EXCHANGES: 3 → 2
   - MAX_TOTAL_TURNS: 8 → 6
   - max_new_tokens: 700 → 500
   - Added input token logging to main loop

2. **380d448**: Memory management teaching + enhanced warnings
   - Added `get_memory_management_instructions()` to base class
   - Applied to all 5 phase variants
   - Made warning message more direct and urgent

3. **d244da2**: Fixed critical warning system bug
   - Warning now triggers at turns: 3, 5, 7, 9, 11...
   - Previously would trigger at: 6, 9, 12... (too late!)
   - Explains why model never used `record_observation()` in phase1a

### Expected Impact
- **Memory**: 30-50% reduction from pruning + shorter responses
- **Warnings**: Actually fire before OOM (was broken)
- **Learning**: Model taught pattern from start + reminded every 2 turns
- **Monitoring**: Can track token counts to validate fixes

## Testing Plan

1. ✅ Commit all changes (b03f743, 380d448, d244da2)
2. ⏳ Run Phase 1a in Colab with new settings
3. ⏳ Monitor logs for:
   - `[SYSTEM WARNING TO MODEL] Turn 3` - verify warning fires
   - `[GENERATION] Input tokens:` values over time
   - Model calls to `record_observation()` - should see them now!
   - Whether OOM still occurs
   - If so, at what token count
4. ⏳ Analyze results:
   - If OOM at >12K tokens → reduce to KEEP_RECENT_EXCHANGES=1
   - If OOM at <10K tokens → investigate other memory leak (activations? model state?)
   - If model still doesn't use record_observation() → may need Option B (automatic prompts)
   - If successful → test full investigation (20+ turns)
5. ⏳ Verify model actually saves observations before pruning

## References

- Attention memory formula: `O(n² × h × b)` where n=tokens, h=heads, b=bytes
- Flash Attention paper: Shows attention is memory bottleneck for long context
- HuggingFace memory debugging: https://huggingface.co/docs/transformers/main/en/perf_train_gpu_one
- PyTorch CUDA management: https://pytorch.org/docs/stable/notes/cuda.html

## Why Option A (Prepend to User Message) Over Option B (Separate System Message)

When deciding how to notify model about pruning, Option A is better because:

1. **Token Efficiency** (critical in OOM context):
   - Option A: ~10 tokens added to existing message
   - Option B: ~25+ tokens + extra message overhead
   - We're fighting OOM - adding messages is counterproductive!

2. **Model Response Behavior**:
   - Option A: Model processes reminder + task together → one useful response
   - Option B: Model acknowledges system message → wasted generation, then actual work
   - Option B requires 2 generation cycles vs 1

3. **Conversation Flow**:
   - Option A: Seamless integration with natural flow
   - Option B: Awkward interruption with unnecessary acknowledgment

4. **Implementation**:
   - Option A: Simple - modify user message string
   - Option B: Complex - requires extra generation cycle and message management

**However, we didn't implement either** because:
- Warning system (now fixed) warns BEFORE pruning occurs
- Model taught memory management pattern from the start
- Replacement message in pruned exchanges is sufficient
- No need to add post-pruning notifications when we have pre-pruning warnings

---

**Next Update:** After testing in Colab with all iteration 2 fixes (b03f743, 380d448, d244da2)

**Key Things to Verify:**
1. ✅ Warning fires at turn 3 (check logs for "[SYSTEM WARNING TO MODEL] Turn 3")
2. ✅ Model uses record_observation() after warnings
3. ✅ Input token counts stay under 10K
4. ✅ No OOM for at least 10-15 turns
5. ✅ Model completes investigation successfully
