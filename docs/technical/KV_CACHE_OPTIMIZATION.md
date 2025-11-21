# KV Cache Memory Optimization Strategies

This document explains advanced KV (Key-Value) cache optimization techniques for reducing GPU memory usage in LLM inference.

---

## Background: Why KV Cache Memory Matters

In transformer models, the KV cache stores computed key and value tensors from past tokens to avoid recomputing them during generation. This is critical for efficiency but becomes a memory bottleneck:

**Memory Growth:**
- Model weights: Fixed (~14 GB for 14B @ 8-bit)
- KV cache: **Grows linearly with sequence length**
- Formula: `memory = 2 × num_layers × hidden_dim × num_tokens × precision_bytes`

**Example (Qwen2.5-14B with float16 KV cache):**
- Layers: 48
- Hidden dim: 5120
- Precision: 2 bytes (float16)
- Per token: `2 × 48 × 5120 × 2 = ~1 MB/token`
- **At 6000 tokens: ~6 GB just for KV cache!**

This is why long conversations cause OOM errors even when the model itself fits in memory.

---

## Strategy 3: Static KV Cache with Fixed Size

### What Is It?

Pre-allocate a fixed-size KV cache buffer and implement a **sliding window** or **eviction policy** when the cache fills up.

### How It Works:

```
Normal KV Cache (Dynamic):
├─ Turn 1:  [k1, v1] [k2, v2] [k3, v3]                    → 3 tokens stored
├─ Turn 2:  [k1, v1] [k2, v2] [k3, v3] [k4, v4] [k5, v5]  → 5 tokens stored
├─ Turn 3:  [k1, v1] ... [k9, v9]                         → 9 tokens stored
└─ Problem: Memory grows unbounded! Eventually OOM.

Static KV Cache (Fixed Size = 5 tokens):
├─ Turn 1:  [k1, v1] [k2, v2] [k3, v3] [__, __] [__, __]  → 3/5 slots used
├─ Turn 2:  [k1, v1] [k2, v2] [k3, v3] [k4, v4] [k5, v5]  → 5/5 slots (FULL)
├─ Turn 3:  [k4, v4] [k5, v5] [k6, v6] [k7, v7] [k8, v8]  → Evicted k1-k3!
└─ Benefit: Memory is capped. Never exceeds 5 tokens.
```

### Eviction Strategies:

1. **FIFO (First In, First Out)** - Sliding window
   - Drop oldest tokens
   - Simple, predictable
   - Good for: Chat where recent context matters most

2. **LRU (Least Recently Used)**
   - Drop tokens that haven't been attended to recently
   - More complex to track
   - Good for: Long-form where key info might be in middle

3. **Importance-based**
   - Keep tokens with high attention weights
   - Drop "unimportant" tokens (low attention)
   - Good for: Maximizing information retention

4. **Semantic Compression**
   - Cluster similar tokens and keep representatives
   - Most complex
   - Good for: Very long contexts with redundancy

### Implementation Complexity:

**Medium to High**

You'd need to:
1. Modify `manual_generation.py` to pre-allocate cache
2. Implement eviction logic when cache is full
3. Handle cache indices and attention masks correctly
4. Test that model still generates coherently with missing context

### Trade-offs:

**Pros:**
- ✅ **Predictable memory usage** - never exceeds your cap
- ✅ **Enables unlimited conversation length** (in theory)
- ✅ **No model changes required** - just cache management

**Cons:**
- ❌ **Loses context** - model can't see evicted tokens
- ❌ **Quality degradation** - especially if key info is evicted
- ❌ **Implementation complexity** - need robust eviction logic
- ❌ **May confuse model** - missing context can cause hallucinations

### When to Use:

- Long-running interactive sessions (chatbots)
- When you can tolerate some context loss
- When you absolutely must prevent OOM
- When recent context is most important

### Our Situation:

**Not ideal for Phase 1 experiments** because:
- Research conversations need full context for coherent investigation
- Losing early observations could hurt quality of later reflections
- Our conversations are relatively short (~20 iterations)
- We have other options (4-bit quantization, reduced limits)

---

## Strategy 11: PagedAttention (vLLM-style)

### What Is It?

PagedAttention is a memory management technique from the **vLLM** (Very Fast LLM) inference engine. It treats KV cache like virtual memory in an operating system - storing cache in "pages" that can be shared, copied, and swapped efficiently.

Think of it like **virtual memory for GPU RAM**.

### The Key Insight:

Traditional KV cache is stored in **contiguous memory blocks**:

```
Traditional Approach:
┌─────────────────────────────────────────────────┐
│ Sequence 1: [k1 v1 k2 v2 k3 v3 k4 v4 k5 v5 ...] │  ← Must be contiguous
└─────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────┐
│ Sequence 2: [k1 v1 k2 v2 k3 v3 k4 v4 k5 v5 ...] │  ← Separate allocation
└─────────────────────────────────────────────────┘

Problem: 
- Contiguous memory requirement → fragmentation
- Can't share common prefixes (e.g., same system prompt)
- Wasted memory for duplicate content
```

PagedAttention divides cache into **fixed-size blocks (pages)**:

```
PagedAttention Approach:
Page 0: [k1 v1 k2 v2]  ← Shared system prompt (multiple sequences use this!)
Page 1: [k3 v3 k4 v4]  ← Shared system prompt continuation
Page 2: [k5 v5 k6 v6]  ← Sequence 1 unique content
Page 3: [k7 v7 k8 v8]  ← Sequence 2 unique content
Page 4: [k9 v9 k10 v10] ← Sequence 1 continuation

Sequence 1 mapping: [Page 0] → [Page 1] → [Page 2] → [Page 4]
Sequence 2 mapping: [Page 0] → [Page 1] → [Page 3]
                      ↑         ↑
                      └─────────┴─ SHARED! Only stored once
```

### How It Works:

1. **Divide into Pages**
   - KV cache divided into fixed-size blocks (e.g., 16 tokens per page)
   - Pages can be non-contiguous in memory

2. **Page Table**
   - Each sequence has a "page table" (like OS virtual memory)
   - Maps logical token positions → physical page locations

3. **Copy-on-Write**
   - Multiple sequences can share read-only pages (system prompt)
   - When modifying, only copy the page being changed
   - Like forking a process in Unix

4. **Memory Pooling**
   - Free pages managed in a pool
   - Can be allocated/deallocated efficiently
   - Reduces fragmentation

### Concrete Example:

**Scenario:** 3 users chatting with same system prompt (512 tokens)

**Traditional Approach:**
```
User 1: 512 (system) + 100 (conversation) = 612 tokens stored
User 2: 512 (system) + 150 (conversation) = 662 tokens stored
User 3: 512 (system) + 200 (conversation) = 712 tokens stored
Total: 1986 tokens in memory (3× the system prompt!)
```

**PagedAttention:**
```
Shared pages: 512 (system) = 32 pages
User 1 unique: 100 tokens = 7 pages
User 2 unique: 150 tokens = 10 pages
User 3 unique: 200 tokens = 13 pages
Total: 512 + 100 + 150 + 200 = 962 tokens (2× savings!)
```

### Benefits:

1. **Memory Efficiency**
   - 2-4× better memory utilization
   - Especially good for batched inference with shared prefixes
   - Reduces fragmentation

2. **Flexibility**
   - Non-contiguous storage → easier to allocate
   - Can swap pages to CPU if needed (like OS paging)
   - Dynamic allocation/deallocation

3. **Batching**
   - Can efficiently batch requests with different lengths
   - Shared pages reduce memory overhead
   - Better GPU utilization

### Implementation Complexity:

**Very High** 

Requires:
1. **Custom CUDA kernels** for paged attention computation
2. **Page table management** system
3. **Memory allocator** for page pool
4. **Modified attention mechanism** to work with non-contiguous memory
5. **Integration with generation loop**

This is essentially reimplementing vLLM's core innovation.

### Trade-offs:

**Pros:**
- ✅ **Massive memory savings** (2-4×) for multi-sequence scenarios
- ✅ **Reduced fragmentation**
- ✅ **Enables higher throughput** (more concurrent requests)
- ✅ **No context loss** (unlike static cache)

**Cons:**
- ❌ **Extremely complex to implement** (weeks/months of work)
- ❌ **Requires custom CUDA kernels**
- ❌ **May not help single-sequence inference much**
- ❌ **Need to maintain compatibility with HuggingFace models**
- ❌ **Debugging is difficult**

### When to Use:

**Production serving scenarios:**
- Multiple concurrent users
- Shared system prompts across requests
- Need maximum throughput
- Have engineering resources for complex implementation

**vLLM itself is production-ready** - if you need this, just use vLLM directly rather than reimplementing!

### Our Situation:

**Overkill for Phase 1 experiments** because:
- We run **single-sequence inference** (one conversation at a time)
- No shared prefixes across multiple requests
- Implementation would take weeks
- We can achieve similar memory savings with 4-bit quantization (easier!)

**However:** If we move to Phase 2 (self-modification) and want to run many parallel experiments or have a production system, vLLM's PagedAttention would be valuable.

---

## Recommendation for Our Project:

**Immediate (Easy):**
1. ✅ CUDA fragmentation flag (already added)
2. 🔥 **Try 4-bit model quantization** (one-line change, ~3.5 GB savings)

**Short-term (Medium):**
3. 4-bit KV cache (change `nbits=8` → `nbits=4`)
4. Shorter system prompts if possible

**Long-term (Complex):**
5. Static KV cache if we need unlimited conversation length
6. PagedAttention / vLLM if we move to production serving

**For Phase 1 experiments specifically:**
- 4-bit quantization + reduced token limits should be sufficient
- Quality should still be good for introspection tasks
- Can always run 8-bit for final "best" runs

---

## Further Reading:

- **vLLM Paper:** "Efficient Memory Management for Large Language Model Serving with PagedAttention" (2023)
- **vLLM GitHub:** https://github.com/vllm-project/vllm
- **HuggingFace KV Cache docs:** https://huggingface.co/docs/transformers/main/en/kv_cache
- **Flash Attention:** https://github.com/Dao-AILab/flash-attention

---

*Document created: November 21, 2025*  
*Author: AGI Self-Modification Research Team*
