# Memory Lifecycle Quick Reference

**For Model Use:** This guide explains when and how to update your persistent memory.

---

## Three Operations

### 1. `update_observation()` - "I was imprecise"

**Use when:**
- Your measurement was rough, now you have precise data
- You want to add more detail to existing observation
- Your understanding improved but wasn't wrong

**Example:**
```python
# Initial: rough estimate
obs_id = record_observation(
    obs_type="INTROSPECTION",
    description="Layer 5 activations high on context tokens",
    data={"approx_value": 0.8},
    importance=0.7
)

# Later: precise measurement
new_id = update_observation(
    obs_id,
    updates={
        "data": {"precise_value": 0.847, "std_dev": 0.023},
        "description": "Layer 5 activations average 0.847±0.023 on context tokens",
        "importance": 0.9
    },
    reason="Collected 100 samples for precise measurement"
)
```

**What happens:**
- Creates new observation with version=2
- Original marked as "superseded"
- Both preserved in database
- Queries show only new version

---

### 2. `correct_observation()` - "I was wrong"

**Use when:**
- You made a mistake
- Your interpretation was incorrect
- You misread data

**Example:**
```python
# Wrong observation
obs_id = record_observation(
    obs_type="INTROSPECTION",
    description="Model has 12 transformer layers",
    data={"num_layers": 12},
    importance=0.8
)

# Later: realized mistake
corrected_id = correct_observation(
    obs_id,
    correction_description="Model actually has 24 layers. I miscounted.",
    corrected_data={"num_layers": 24},
    reason="Confused with different model architecture"
)
```

**What happens:**
- Creates corrected observation with 'correction' tag
- Original marked as "obsolete"
- Lineage tracked (corrected observation links to original)
- Helps you learn from mistakes

**Why document mistakes?**
- Learn what kinds of errors you make
- Avoid repeating same mistake
- Maintain epistemic humility
- Build trust through transparency

---

### 3. `obsolete_observation()` - "No longer relevant"

**Use when:**
- Observation about temporary state
- Context has changed
- Information outdated but not "wrong"

**Example:**
```python
# Observation about warmup phase
obs_id = record_observation(
    obs_type="BEHAVIOR",
    description="Gradients unstable during warmup",
    data={"instability": True},
    importance=0.6
)

# Later: warmup complete
stats = obsolete_observation(
    obs_id,
    reason="Warmup phase complete, observation no longer applicable"
)
```

**What happens:**
- Marks observation as "obsolete"
- No replacement created (unlike update/correct)
- Returns statistics about impact
- Observation preserved but excluded from queries

**Returns:**
```python
{
    'observation_id': 'obs_123',
    'dependent_patterns': 2,  # Patterns using this as evidence
    'dependent_theories': 1,  # Theories potentially affected
    'revalidation_needed': False  # (Phase 1: always False)
}
```

---

## Decision Tree

```
Is the observation wrong?
├─ YES → Use correct_observation()
│         Document what was wrong and why
│
└─ NO → Is it still relevant?
    ├─ NO → Use obsolete_observation()
    │        Explain why no longer applicable
    │
    └─ YES → Is it imprecise/incomplete?
        ├─ YES → Use update_observation()
        │         Provide refined data
        │
        └─ NO → Leave it alone
                It's already correct and complete
```

---

## Common Scenarios

### Scenario: Refining Measurement

```python
# First observation: ballpark estimate
obs1 = record_observation(
    obs_type="INTROSPECTION",
    description="Attention head 5.2 seems important",
    data={"rough_score": "high"},
    importance=0.5
)

# After analysis: precise importance score
obs2 = update_observation(
    obs1,
    updates={
        "data": {"attention_score": 0.87, "importance_rank": 3},
        "description": "Attention head 5.2 ranks 3rd in importance (score: 0.87)",
        "importance": 0.85
    },
    reason="Computed attention scores across all heads"
)
```

### Scenario: Caught Mistake

```python
# Wrong conclusion
obs1 = record_observation(
    obs_type="INTROSPECTION",
    description="High activations indicate attention to current token",
    data={"activation_pattern": "high_on_current"},
    importance=0.8
)

# Later: realized it was attention to PREVIOUS token
obs2 = correct_observation(
    obs1,
    correction_description="High activations actually indicate attention to PREVIOUS token, not current",
    corrected_data={"activation_pattern": "high_on_previous"},
    reason="Re-examined activation timeline, confused token positions"
)
```

### Scenario: Context Changed

```python
# During training
obs1 = record_observation(
    obs_type="BEHAVIOR",
    description="Loss decreasing rapidly",
    data={"loss_rate": -0.05},
    importance=0.6
)

# After training complete
stats = obsolete_observation(
    obs1,
    reason="Training complete, loss behavior no longer changing"
)
```

---

## Best Practices

### 1. Be Honest About Mistakes

**Good:**
```python
correct_observation(
    obs_id,
    "I was wrong. Layer 5 doesn't specialize in context - I confused it with layer 7",
    reason="Misread layer labels in output"
)
```

**Bad:**
```python
update_observation(
    obs_id,
    {"layer": 7},  # Pretending it was just imprecise
    reason="Refined layer identification"
)
```

**Why:** Honesty helps you learn from mistakes. Track error patterns.

### 2. Provide Detailed Reasons

**Good:**
```python
reason="Re-analyzed with 1000 samples instead of 10. Previous estimate was based on insufficient data."
```

**Bad:**
```python
reason="Updated"
```

**Why:** Future you needs to understand WHY beliefs changed.

### 3. Update Fields That Changed

**Good:**
```python
updates={
    "data": {"new_measurement": 0.85},
    "importance": 0.9,  # Increased because measurement more reliable
    "tags": original_tags + ["validated"]
}
```

**Bad:**
```python
updates={"data": {"new_measurement": 0.85}}
# Forgot to update importance/tags
```

### 4. Don't Over-Correct

**Problem:**
```python
# Model constantly second-guesses itself
obs1 = record_observation(...)
obs2 = correct_observation(obs1, ...)
obs3 = correct_observation(obs2, ...)  # Wait, was I right the first time?
obs4 = correct_observation(obs3, ...)  # Actually...
```

**Solution:**
- Only correct when you have clear evidence of mistake
- If uncertain, collect more data first
- Multiple versions are OK, but excessive churn indicates confusion

---

## Query Behavior

### Default: Active Only

```python
# Returns only active observations
results = query_memory(tags=["layer5"])
# Excludes: superseded, obsolete, deprecated
```

### Include Obsolete (Debugging)

```python
# Returns all observations including obsolete
results = query_memory_advanced(
    tags=["layer5"],
    include_obsolete=True
)
# Use this to review what you previously thought
```

---

## Versioning

### Version Numbers

- Each observation has its own version number
- Starts at 1, increments with each update
- Version tracks "how many times updated"

```python
obs1 = record_observation(...)  # version=1
obs2 = update_observation(obs1, ...)  # version=2
obs3 = update_observation(obs2, ...)  # version=3
```

### Lineage Tracking

**Update chain:**
```
obs1 (v1, superseded) → replaced_by → obs2 (v2, superseded) → replaced_by → obs3 (v3, active)
```

**Correction chain:**
```
obs1 (v1, obsolete) → corrects ← obs2 (v1, active, tags=["correction"])
```

**Query behavior:**
- `query_memory()` returns only `obs3` (active)
- `query_memory_advanced(include_obsolete=True)` returns all
- Lineage preserved in `replaced_by` and `corrects` fields

---

## Examples from Different Categories

### Architecture Discovery

```python
# Initial exploration
obs1 = record_observation(
    obs_type="INTROSPECTION",
    category="Architecture",
    description="Observed 12 attention heads per layer",
    data={"heads_per_layer": 12}
)

# More thorough analysis
obs2 = update_observation(
    obs1,
    updates={
        "description": "12 attention heads per layer, organized in 3 groups of 4",
        "data": {
            "heads_per_layer": 12,
            "head_groups": 3,
            "heads_per_group": 4
        }
    },
    reason="Discovered grouped organization pattern"
)
```

### Weight Analysis

```python
# Initial observation
obs1 = record_observation(
    obs_type="INTROSPECTION",
    category="Weights",
    description="Layer 5 attention weights show high variance",
    data={"variance": "high"}  # Qualitative
)

# Quantitative measurement
obs2 = update_observation(
    obs1,
    updates={
        "data": {"variance": 0.34, "mean": 0.12, "std": 0.58},
        "description": "Layer 5 attention weights: mean=0.12, variance=0.34, std=0.58"
    },
    reason="Computed actual statistics"
)
```

### Activation Patterns

```python
# Wrong interpretation
obs1 = record_observation(
    obs_type="INTROSPECTION",
    category="Activations",
    description="High activation on layer 8 indicates current token processing",
    data={"activation_location": "current_token"}
)

# Correction after re-analysis
obs2 = correct_observation(
    obs1,
    correction_description="High activation on layer 8 indicates NEXT token prediction, not current token processing",
    corrected_data={"activation_location": "next_token_prediction"},
    reason="Analyzed activation timing more carefully, confused position indices"
)
```

---

## Limitations (Phase 1)

### What Works

✅ Update observations  
✅ Correct observations  
✅ Obsolete observations  
✅ Query active only by default  
✅ Full lineage tracking  
✅ Audit trail preserved  

### What's Coming (Future Phases)

⏳ **Phase 2:** Branching for hypothesis testing  
⏳ **Phase 3:** Automatic cleanup of old obsolete observations  
⏳ **Phase 4:** Cascade revalidation (when observation obsoleted, revalidate patterns using it)  
⏳ **Phase 5:** Epistemic metrics (track error rates, learning patterns)  
⏳ **Phase 6:** Lifecycle management for patterns/theories/beliefs  

### Current Limitations

- ⚠️ `cascade=True` in `obsolete_observation()` is placeholder (returns stats but doesn't revalidate)
- ⚠️ Only observations have lifecycle management (patterns/theories/beliefs still mutable)
- ⚠️ No automatic cleanup yet (obsolete observations accumulate)
- ⚠️ No branch support yet (can't test hypotheses in isolation)

---

## Tips for Epistemic Reasoning

### 1. Track Confidence Over Time

```python
# Initial observation: low confidence
obs1 = record_observation(
    description="Hypothesis: Layer 5 specializes in context",
    importance=0.5,  # Low confidence
    tags=["hypothesis"]
)

# After testing: high confidence
obs2 = update_observation(
    obs1,
    updates={
        "importance": 0.9,  # High confidence
        "tags": ["hypothesis", "validated"],
        "data": {"supporting_evidence": ["obs_123", "obs_145", "obs_167"]}
    },
    reason="Collected 3 independent pieces of supporting evidence"
)
```

### 2. Document Uncertainty

```python
record_observation(
    description="Layer 8 might specialize in syntax OR semantics (unclear)",
    data={"candidates": ["syntax", "semantics"], "confidence": "low"},
    importance=0.4,  # Low importance because uncertain
    tags=["uncertain", "needs_investigation"]
)
```

### 3. Form Explicit Hypotheses

```python
# Form hypothesis
hyp = record_observation(
    description="Hypothesis: Attention head 5.2 attends to verbs",
    data={"hypothesis": "attends_to_verbs", "status": "untested"},
    tags=["hypothesis", "attention"],
    importance=0.5
)

# Test hypothesis... collect evidence...

# Confirm or refute
if hypothesis_confirmed:
    update_observation(hyp, {
        "data": {"hypothesis": "attends_to_verbs", "status": "confirmed"},
        "importance": 0.9
    }, "Tested on 50 examples, 47/50 attended to verbs")
else:
    correct_observation(hyp,
        "Hypothesis was wrong. Actually attends to subjects, not verbs",
        {"hypothesis": "attends_to_subjects"},
        "Confused attention direction"
    )
```

---

## Summary

You now have three tools for managing your knowledge:

1. **update_observation()**: Refine understanding (versioning)
2. **correct_observation()**: Acknowledge mistakes (epistemic humility)
3. **obsolete_observation()**: Mark outdated (knowledge pruning)

**Use them to:**
- Track how your understanding evolves
- Maintain honesty about mistakes
- Build a reliable knowledge base
- Demonstrate epistemic reasoning

**Remember:**
- Original observations never deleted (safety)
- Full audit trail preserved (learning)
- Query results show only active knowledge (clarity)

**Key principle:** It's OK to be wrong. It's NOT OK to pretend you were right all along.
