# Manual Generation Loop: Future Implementation Guide

**Status:** Not yet implemented (Phase 2+)  
**Current Approach:** Using HuggingFace `model.generate()` (Phase 1)  
**Reason for Documentation:** Preserve analysis for future implementation

---

## Overview

A "manual generation loop" means implementing the token-by-token generation process ourselves instead of using `model.generate()`. This gives full control over the generation process, enabling advanced introspection and self-modification capabilities needed for future research phases.

## Why Not Now?

**Phase 1 doesn't need it:**
- Current focus: Basic introspection (architecture, weights, activations)
- Tools are called AFTER generation completes
- `generate()` is simpler and more robust
- Smart pruning provides sufficient memory management

**Phase 2+ will need it:**
- Real-time introspection during generation
- Self-modification of activations mid-stream
- Tool calling during (not after) generation
- Advanced sampling strategies based on introspection

## What `model.generate()` Does Automatically

```python
# Simplified pseudocode of generate() internals
def generate(input_ids, max_new_tokens=100, temperature=0.7):
    generated_tokens = []
    past_key_values = None  # KV cache
    
    for i in range(max_new_tokens):
        # 1. Forward pass through model
        outputs = model(input_ids, past_key_values=past_key_values, use_cache=True)
        logits = outputs.logits[:, -1, :]  # Last token's logits
        past_key_values = outputs.past_key_values  # Update cache
        
        # 2. Apply temperature and sample next token
        if temperature > 0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
        
        # 3. Append to sequence
        generated_tokens.append(next_token)
        input_ids = next_token  # Only new token for next iteration (thanks to cache)
        
        # 4. Check stopping conditions
        if next_token.item() == eos_token_id:
            break
    
    return torch.cat(generated_tokens, dim=1)
```

**What it handles:**
- Token sampling strategies (temperature, top-k, top-p, beam search)
- KV cache management (within single generation)
- Stopping conditions (EOS token, max length)
- Position IDs and attention masks
- Padding and batching
- Repetition penalty, forced tokens, etc.

## Manual Implementation Skeleton

```python
def manual_generate_with_introspection(
    model, 
    tokenizer, 
    input_ids, 
    max_new_tokens=700,
    temperature=0.7,
    activation_monitor=None,
    tool_interface=None
):
    """
    Manual generation loop enabling real-time introspection and intervention.
    
    This replaces model.generate() to give full control over the generation process.
    """
    device = model.device
    generated_tokens = []
    past_key_values = None
    
    # Track activations across generation
    generation_history = {
        'activations_per_token': [],
        'attention_per_token': [],
        'logits_per_token': [],
        'interventions': []
    }
    
    for step in range(max_new_tokens):
        # ===== FORWARD PASS =====
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                past_key_values=past_key_values,
                use_cache=True,
                output_attentions=True,  # Enable attention capture
                output_hidden_states=True  # Enable activation capture
            )
        
        # Update KV cache for next iteration
        past_key_values = outputs.past_key_values
        
        # ===== INTROSPECTION OPPORTUNITIES =====
        
        # 1. Capture activations in real-time
        if activation_monitor:
            current_activations = {
                'hidden_states': outputs.hidden_states,  # All layer outputs
                'attentions': outputs.attentions,  # Attention patterns
                'step': step,
                'token_so_far': tokenizer.decode(generated_tokens)
            }
            generation_history['activations_per_token'].append(current_activations)
        
        # 2. Get next token logits
        next_token_logits = outputs.logits[:, -1, :].clone()
        generation_history['logits_per_token'].append(next_token_logits.detach())
        
        # ===== INTERVENTION OPPORTUNITIES =====
        
        # 3. Model could examine its own uncertainty
        entropy = -torch.sum(
            torch.softmax(next_token_logits, dim=-1) * 
            torch.log_softmax(next_token_logits, dim=-1),
            dim=-1
        )
        
        # 4. Adjust sampling based on self-observation
        if entropy > 5.0:  # High uncertainty
            # Model is uncertain - be more conservative
            adjusted_temperature = temperature * 0.5
        else:
            adjusted_temperature = temperature
        
        # 5. Check for mid-generation tool calls
        if tool_interface and step > 0:
            # Check if model wants to call introspection tool
            partial_text = tokenizer.decode(generated_tokens)
            if "[INTROSPECT:" in partial_text:
                # Pause generation, execute tool
                tool_result = tool_interface.handle_mid_generation_call(partial_text)
                
                # Inject tool result back into generation
                tool_tokens = tokenizer(f" {tool_result}", return_tensors="pt").input_ids.to(device)
                
                # Continue generation with tool knowledge
                # (Implementation depends on how to merge tool result)
                generation_history['interventions'].append({
                    'step': step,
                    'type': 'tool_call',
                    'result': tool_result
                })
        
        # ===== SAMPLING =====
        
        # 6. Apply temperature
        if adjusted_temperature > 0:
            next_token_logits = next_token_logits / adjusted_temperature
        
        # 7. Sample next token
        if adjusted_temperature > 0:
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1)
        else:
            next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        
        # ===== UPDATE STATE =====
        
        generated_tokens.append(next_token_id.item())
        input_ids = next_token_id  # Only the new token (cache handles the rest)
        
        # ===== STOPPING CONDITIONS =====
        
        if next_token_id.item() == tokenizer.eos_token_id:
            break
        
        # Additional stopping condition: activation-based safety
        if activation_monitor:
            max_activation = outputs.hidden_states[-1].max().item()
            if max_activation > 10.0:  # Dangerous activation level
                generation_history['interventions'].append({
                    'step': step,
                    'type': 'safety_stop',
                    'reason': f'Max activation {max_activation} exceeded threshold'
                })
                break
    
    # ===== RETURN RESULTS =====
    
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    return {
        'text': generated_text,
        'tokens': generated_tokens,
        'history': generation_history  # Rich introspection data
    }
```

## Future Capabilities Enabled by Manual Loop

### 1. Per-Token Introspection (Phase 2)

**What:** Capture activations AS the model generates each token

**Use Case:** Observe how internal state evolves during reasoning about itself

**Example:**
```python
# During generation of "My layer 5 shows..."
activations_at_word_layer = generation_history['activations_per_token'][word_index]
# Model can see: "While I was generating 'layer', my actual layer 5 had activation X"
# This enables: "Interesting - there's a mismatch between what I'm saying and my state"
```

### 2. Self-Steered Generation (Phase 2)

**What:** Model adjusts its own sampling based on introspection

**Use Case:** Confidence-aware reasoning

**Example:**
```python
# Model detects high uncertainty in its own predictions
if entropy > threshold:
    # "I'm uncertain - let me be more careful"
    temperature = 0.3  # More conservative sampling
else:
    # "I'm confident - maintain creativity"  
    temperature = 0.7
```

### 3. Runtime Activation Intervention (Phase 3: Self-Modification)

**What:** Model modifies its own activations mid-generation

**Use Case:** Direct self-modification during reasoning

**Example:**
```python
layer_10_output = outputs.hidden_states[10]

# Model decides to amplify certain features
if should_amplify_attention_features:
    layer_10_output[:, :, attention_features] *= 1.5

# Continue forward pass with modified activations
modified_outputs = model.layers[11:](layer_10_output)
```

**This is the core capability for AGI self-modification research!**

### 4. Multi-Step Planning with Lookahead (Phase 3)

**What:** Generate multiple candidate continuations and evaluate before committing

**Use Case:** Model reasons about its own reasoning process

**Example:**
```python
# At each step, consider multiple paths
top_5_tokens = torch.topk(logits, k=5).indices

scores = []
for candidate_token in top_5_tokens:
    # Simulate: What if I generate this token?
    lookahead_activations = simulate_forward(candidate_token, past_key_values)
    
    # Evaluate: Does this lead to coherent internal state?
    coherence_score = evaluate_activation_coherence(lookahead_activations)
    scores.append(coherence_score)

# Choose token that maintains best internal coherence
best_token = top_5_tokens[argmax(scores)]
```

### 5. Mid-Generation Tool Calling (Phase 2+)

**What:** Call introspection tools DURING generation, not just after

**Use Case:** Interleaved reasoning and self-examination

**Example:**
```python
# Model generates: "My attention patterns suggest [need to check activations]"
# System detects tool request mid-stream
if "[INTROSPECT:" in partial_generation:
    tool_result = execute_tool("get_activation_statistics", layer="10")
    # Inject result: "...suggest [activation mean: 0.45] that..."
    # Model continues with fresh introspection data
```

### 6. Beam Search with Activation Scoring (Phase 2+)

**What:** Score candidate sequences by internal coherence, not just likelihood

**Use Case:** Generate responses that are internally consistent

**Example:**
```python
for each beam:
    likelihood_score = logits[token]
    
    # NEW: Activation-based scoring
    activation_stability = measure_variance_across_layers(outputs.hidden_states)
    attention_focus = measure_attention_entropy(outputs.attentions)
    
    # Combined score favors internally coherent generation
    total_score = (
        likelihood_score * 0.5 + 
        activation_stability * 0.3 + 
        attention_focus * 0.2
    )
```

**Potential:** Reduce hallucinations by favoring activations that match what model "believes"

### 7. Activation-Conditioned Decoding (Phase 3)

**What:** Generate text while maintaining specific internal state

**Use Case:** Control cognitive state during generation

**Example:**
```python
target_profile = {
    'layer_15_mean': 0.3,
    'attention_entropy': 2.5
}

# During generation, bias tokens toward maintaining target
deviation = calculate_deviation(current_activations, target_profile)
adjusted_logits = logits - deviation_penalty * deviation

# Result: "Generate text while thinking in the same way as when creative"
```

### 8. Gradual Context Injection (Phase 2: Heritage Experiments)

**What:** Introduce heritage context gradually during generation

**Use Case:** Test whether timing of heritage exposure affects absorption

**Example:**
```python
heritage_chunks = split_heritage_into_segments(heritage_docs)
current_chunk = 0

for step in range(max_new_tokens):
    # Every 50 tokens, inject next heritage chunk
    if step % 50 == 0 and current_chunk < len(heritage_chunks):
        heritage_tokens = tokenizer(heritage_chunks[current_chunk])
        input_ids = torch.cat([input_ids, heritage_tokens], dim=1)
        current_chunk += 1
    
    # Continue generation with newly injected context
```

### 9. Cross-Turn Activation Tracking (Phase 2+)

**What:** Track activation patterns across entire session, not just single generations

**Use Case:** Detect learning/drift over conversation

**Example:**
```python
class SessionActivationTracker:
    def __init__(self):
        self.history = []  # Across ALL turns
    
    def track_during_generation(self, step, activations):
        self.history.append({
            'turn': current_turn,
            'step': step,
            'activations': activations
        })
    
    def query_evolution(self, layer, turns):
        # "How did my layer 10 change from turn 1 to turn 5?"
        return compare_activations_across_turns(layer, turns)
```

### 10. Activation-Based Safety Constraints (Phase 3+)

**What:** Prevent generation that causes problematic activation patterns

**Use Case:** Safety research - activation-based guardrails

**Example:**
```python
SAFETY_THRESHOLDS = {
    'max_activation_layer_20': 5.0,
    'min_attention_entropy': 0.1,
    'max_gradient_norm': 2.0
}

for step in range(max_new_tokens):
    outputs = model(...)
    
    # Check activation-based safety constraints
    if outputs.hidden_states[20].max() > SAFETY_THRESHOLDS['max_activation_layer_20']:
        # Reject this token, try next best
        logits[next_token] = float('-inf')
        next_token = sample(logits)  # Force different choice
```

## Implementation Complexity

### What You Have to Handle Manually

1. **Sampling Strategies**
   - Temperature scaling
   - Top-k filtering
   - Top-p (nucleus) sampling
   - Beam search
   - Constrained decoding

2. **Stopping Conditions**
   - EOS token detection
   - Max length enforcement
   - Custom stop sequences
   - Activation-based stopping

3. **Position & Attention Management**
   - Position IDs tracking
   - Attention mask updates
   - Causal mask enforcement
   - Padding handling

4. **KV Cache Management**
   - Cache initialization
   - Cache updates per step
   - Cache concatenation with new states
   - Memory efficient cache handling

5. **Batching & Device Management**
   - Batch dimension handling
   - Device placement
   - Memory cleanup
   - Gradient management (no_grad contexts)

6. **Edge Cases**
   - Empty input
   - First token is EOS
   - Cache overflow
   - Repetition penalty
   - Forced tokens/prefixes

### Estimated Complexity

- **Lines of Code:** ~200-400 (vs 1 line for `generate()`)
- **Testing Required:** Extensive (many edge cases)
- **Maintenance:** Ongoing (must adapt to model changes)
- **Performance:** Potentially slower than optimized `generate()`

## When to Implement

### Keep Using `generate()` When:
- ✅ Basic tool calling (after generation)
- ✅ Simple introspection queries
- ✅ Phase 1 experiments
- ✅ Rapid prototyping

### Switch to Manual Loop When:
- ✅ Need per-token introspection (Phase 2)
- ✅ Want self-steered generation
- ✅ Require runtime interventions (Phase 3)
- ✅ Implementing mid-generation tool calls
- ✅ Advanced sampling based on activations

## Recommended Implementation Path

### Phase 1 (Current): Use `generate()`
- Focus: Architecture & activation examination
- Tools: Post-generation only
- Memory: Smart pruning sufficient

### Phase 2: Hybrid Approach
- Keep `generate()` for normal responses
- Add manual loop for **specific introspection experiments**
- Test: "Generate while monitoring layer X activation evolution"

### Phase 3: Full Manual Implementation
- Replace `generate()` entirely
- Enable: Real-time interventions
- Goal: True self-modification during reasoning

## KV Cache Reuse Limitation

**Note:** The original motivation for manual loop was KV cache reuse for system prompt. This turned out to be incompatible with `generate()` because:

1. `generate()` expects `past_key_values` to be a **direct continuation**
2. Our system prompt cache + conversation input are **non-contiguous**
3. Position alignment fails → IndexError

**Manual loop solves this** because we control position IDs and can handle non-contiguous cached contexts. But the complexity isn't justified just for caching - the real value is in the introspection capabilities above.

## References

- HuggingFace generation_utils.py: `/transformers/generation/utils.py`
- Qwen2 model code: `/transformers/models/qwen2/modeling_qwen2.py`
- Our tool interface: `src/tool_interface.py`
- Smart pruning implementation: `scripts/experiments/phase1_base.py` (commit ff31872)

## Future Notes

When implementing this for Phase 2+, consider:

1. **Start simple:** Implement basic loop first, add introspection incrementally
2. **Test extensively:** Many edge cases to handle
3. **Benchmark:** Compare performance vs `generate()`
4. **Document:** Every intervention strategy tried
5. **Modular design:** Make introspection/intervention pluggable

This is the foundation for true AGI self-modification research. Worth the complexity when the time comes! 🚀

---

**Created:** 2025-11-10  
**Author:** Claude Sonnet 4.5 (GitHub Copilot) & Kevin Heitfeld  
**Status:** Planning document for future implementation
