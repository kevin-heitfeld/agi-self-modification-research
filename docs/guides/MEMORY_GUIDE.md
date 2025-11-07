# Memory Systems Guide

**AGI Self-Modification Research Platform**  
**Version**: Phase 0 Complete  
**Date**: November 7, 2025

---

## Learning From Experience 🧠

The memory system enables your AI to learn from its modification experiments. This guide explains the 4-layer memory architecture and how to use it effectively.

---

## Four-Layer Architecture

```
┌─────────────────────────────────────────┐
│  Layer 4: BELIEFS                       │
│  - High-confidence principles           │
│  - Confidence > 0.8                     │
│  - Example: "Always checkpoint"         │
└─────────────────────────────────────────┘
              ↑ Evidence
┌─────────────────────────────────────────┐
│  Layer 3: THEORIES                      │
│  - Causal models                        │
│  - Confidence tracked                   │
│  - Example: "Small changes safer"       │
└─────────────────────────────────────────┘
              ↑ Patterns
┌─────────────────────────────────────────┐
│  Layer 2: PATTERNS                      │
│  - Detected regularities                │
│  - Requires 3+ observations             │
│  - Example: "NaN follows large mods"    │
└─────────────────────────────────────────┘
              ↑ Observations
┌─────────────────────────────────────────┐
│  Layer 1: OBSERVATIONS                  │
│  - Raw events                           │
│  - Timestamped logs                     │
│  - Example: "Modified layer 5 by +0.01" │
└─────────────────────────────────────────┘
```

---

## Quick Start

```python
from memory import MemorySystem

# Create memory system
memory = MemorySystem(base_dir="data/my_experiments")

# Record what happened
obs_id = memory.record_observation(
    content="Modified attention layer, perplexity improved",
    category="success",
    importance=0.8
)

# Query memories
successes = memory.observations.query(category="success")
print(f"Found {len(successes)} successful modifications")
```

---

## Layer 1: Observations

### What Are Observations?

Raw events that happen during experiments. Everything starts here.

### Recording Observations

```python
# Basic observation
obs_id = memory.record_observation(
    content="Modified model.layers.5.self_attn.q_proj",
    category="modification"
)

# Detailed observation with metadata
obs_id = memory.record_observation(
    content="Attention scaling experiment successful",
    category="success",
    metadata={
        "layer": "model.layers.5.self_attn.q_proj",
        "change": +0.05,
        "perplexity_before": 11.27,
        "perplexity_after": 10.83,
        "improvement": 3.9
    },
    importance=0.9,
    tags=["attention", "scaling", "layer_5"]
)
```

### Querying Observations

```python
# By category
modifications = memory.observations.query(category="modification")

# By importance
critical_obs = memory.observations.query(min_importance=0.8)

# By tags
layer5_obs = memory.observations.query(tags=["layer_5"])

# Combined filters
recent_successes = memory.observations.query(
    category="success",
    min_importance=0.7,
    tags=["attention"],
    limit=10
)
```

### Observation Categories

| Category | Purpose | Example |
|----------|---------|---------|
| `modification` | Weight changes | "Modified layer X by Y" |
| `observation` | General events | "Model generated text" |
| `success` | Positive outcomes | "Performance improved" |
| `failure` | Negative outcomes | "NaN detected" |
| `insight` | Discoveries | "Pattern noticed" |
| `coupled_modification` | Shared weight changes | Auto-tagged |

---

## Layer 2: Patterns

### What Are Patterns?

Regularities detected across multiple observations (requires 3+).

### Automatic Pattern Detection

```python
# Record similar observations
for i in range(5):
    memory.record_observation(
        f"Small modification {i} was safe",
        category="insight",
        tags=["safety", "small_changes"]
    )

# Patterns automatically detected
patterns = memory.patterns.get_patterns(min_frequency=3)

for pattern in patterns:
    print(f"Pattern: {pattern.description}")
    print(f"  Frequency: {pattern.frequency}")
    print(f"  Confidence: {pattern.confidence:.2f}")
    print(f"  Tags: {pattern.tags}")
```

### Manual Pattern Creation

```python
# Create pattern explicitly
pattern_id = memory.patterns.create_pattern(
    description="Small modifications safer than large",
    supporting_observations=[obs1_id, obs2_id, obs3_id],
    frequency=3,
    confidence=0.75
)
```

### Pattern Queries

```python
# High-confidence patterns
strong_patterns = memory.patterns.get_patterns(min_confidence=0.7)

# Frequent patterns
common_patterns = memory.patterns.get_patterns(min_frequency=5)

# By tags
safety_patterns = memory.patterns.get_patterns(tags=["safety"])
```

---

## Layer 3: Theories

### What Are Theories?

Causal models and explanatory hypotheses about model behavior.

### Forming Theories

```python
# Create theory
theory_id = memory.form_theory(
    description="Attention layers control information flow bottlenecks",
    supporting_evidence=[pattern1_id, obs1_id, obs2_id],
    confidence=0.7,
    metadata={
        "hypothesis": "Modifying attention affects downstream layers",
        "testable": True,
        "tested": False
    }
)
```

### Updating Theories

```python
# Test theory and update confidence
test_result = run_experiment()

if test_result['supports_theory']:
    memory.theories.update_confidence(theory_id, increase=0.1)
else:
    memory.theories.update_confidence(theory_id, decrease=0.2)
    
# Add evidence
memory.theories.add_evidence(theory_id, new_observation_id)
```

### Theory Queries

```python
# High-confidence theories
strong_theories = memory.theories.get_theories(min_confidence=0.7)

# Untested theories
untested = memory.theories.query(
    metadata_filter={"tested": False}
)

# Print theories
for theory in strong_theories:
    print(f"\n💡 {theory.description}")
    print(f"   Confidence: {theory.confidence:.2f}")
    print(f"   Evidence count: {len(theory.supporting_evidence)}")
```

---

## Layer 4: Beliefs

### What Are Beliefs?

High-confidence principles (>0.8) that guide future decisions.

### Forming Beliefs

```python
# Beliefs form from high-confidence theories
belief_id = memory.form_belief(
    content="Always create checkpoint before modification",
    confidence=0.95,
    evidence=[theory1_id, theory2_id, obs1_id, obs2_id],
    category="safety_principle"
)
```

### Belief-Based Decision Making

```python
# Query beliefs before action
safety_beliefs = memory.beliefs.query(category="safety_principle")

print("🔐 Safety Beliefs to Follow:")
for belief in safety_beliefs:
    print(f"  - {belief.content} (confidence: {belief.confidence:.2f})")

# Check if action violates beliefs
def check_against_beliefs(action_description):
    all_beliefs = memory.beliefs.get_beliefs(min_confidence=0.8)
    
    warnings = []
    for belief in all_beliefs:
        if conflicts_with(action_description, belief.content):
            warnings.append(f"⚠️  Violates belief: {belief.content}")
    
    return warnings
```

### Belief Evolution

```python
# Beliefs strengthen with evidence
memory.beliefs.add_evidence(belief_id, new_confirming_observation)

# Or weaken
memory.beliefs.update_confidence(belief_id, decrease=0.1)

# Strong beliefs become "core principles"
core_beliefs = memory.beliefs.query(min_confidence=0.95)
```

---

## Advanced Features

### Weight Inspector Integration

**Crucial for coupled modification tracking!**

```python
from introspection import WeightInspector

# Integrate inspector
inspector = WeightInspector(model, "Qwen2.5-3B")
memory.set_weight_inspector(inspector)

# Now modifications are automatically tracked as coupled
memory.record_modification(
    layer_name="model.embed_tokens.weight",
    modification_data={"change": 0.01}
)
# Automatically creates coupled_modification observation
# Tags include both affected layers!
```

### Query Engine Power Features

```python
# Complex queries
results = memory.query(
    layer="observations",
    filters={
        "category": "success",
        "min_importance": 0.7,
        "tags": ["attention", "layer_5"],
        "date_range": ("2025-11-01", "2025-11-07")
    },
    sort_by="importance",
    limit=20
)

# Full-text search
relevant = memory.search_text("perplexity improved")

# Aggregations
stats = memory.get_statistics()
print(f"Total observations: {stats['observations']['total']}")
print(f"Success rate: {stats['observations']['by_category']['success'] / stats['observations']['total'] * 100:.1f}%")
```

### Memory Consolidation

```python
# Summarize old memories to save space
memory.consolidate(
    older_than_days=30,
    keep_high_importance=True,  # Keep importance > 0.8
    summarize_low_importance=True
)

# Export memory for sharing
memory.export("experiment_results.json")

# Import memories
memory.import_memories("previous_experiments.json")
```

---

## Best Practices

### Recording Best Practices

```python
# ✅ GOOD: Detailed, structured
memory.record_observation(
    content="Modified attention layer 5, Q projection by +0.05",
    category="modification",
    metadata={
        "layer": "model.layers.5.self_attn.q_proj",
        "method": "gradient_scaling",
        "magnitude": 0.05,
        "baseline_checkpoint": checkpoint_id,
        "perplexity_change": -0.44
    },
    importance=0.8,
    tags=["attention", "scaling", "layer_5", "success"]
)

# ❌ BAD: Vague, unstructured
memory.record_observation(
    content="Changed something",
    category="modification"
)
```

### Importance Scoring Guidelines

| Score | Meaning | Example |
|-------|---------|---------|
| 0.9-1.0 | Critical discovery | "Found safe modification method" |
| 0.7-0.8 | Important finding | "Performance improved 5%" |
| 0.5-0.6 | Useful observation | "Layer 5 more sensitive" |
| 0.3-0.4 | Minor note | "Ran experiment" |
| 0.0-0.2 | Trivial | "Loaded model" |

### Pattern Formation Strategy

```python
# Record observations with consistent tags
for experiment in experiments:
    result = run_experiment(experiment)
    
    memory.record_observation(
        content=f"Experiment {experiment['id']}: {result['outcome']}",
        category="success" if result['improved'] else "failure",
        metadata=result,
        tags=[
            experiment['method'],
            f"layer_{experiment['layer']}",
            "small_change" if abs(experiment['delta']) < 0.1 else "large_change"
        ]
    )

# Patterns will emerge from consistent tagging
patterns = memory.patterns.get_patterns(tags=["small_change"])
# Likely pattern: "Small changes more successful"
```

---

## Complete Example: Learning Loop

```python
def learning_experiment_loop(model, num_experiments=10):
    """
    Complete learning loop using memory system
    """
    from memory import MemorySystem
    from checkpointing import CheckpointManager
    from introspection import WeightInspector
    
    # Setup
    memory = MemorySystem("data/learning_experiment")
    checkpointer = CheckpointManager(model, "data/checkpoints")
    inspector = WeightInspector(model, "Qwen2.5-3B")
    memory.set_weight_inspector(inspector)
    
    # Initial beliefs
    memory.form_belief(
        content="Start with small modifications",
        confidence=0.8,
        category="strategy"
    )
    
    for i in range(num_experiments):
        print(f"\n🧪 Experiment {i+1}/{num_experiments}")
        
        # 1. Query existing knowledge
        patterns = memory.patterns.get_patterns(min_confidence=0.6)
        theories = memory.theories.get_theories(min_confidence=0.7)
        
        print(f"  Known patterns: {len(patterns)}")
        print(f"  Working theories: {len(theories)}")
        
        # 2. Design experiment based on knowledge
        if any("small_change" in p.tags for p in patterns):
            delta = 0.01  # Small changes work
        else:
            delta = 0.05  # Exploring
        
        # 3. Execute
        baseline_id = checkpointer.save_checkpoint()
        layer = f"model.layers.{i % 10}.self_attn.q_proj"
        
        modify_layer(model, layer, delta)
        
        # 4. Test
        result = test_model(model)
        
        # 5. Record observation
        obs_id = memory.record_observation(
            content=f"Modified {layer} by {delta:+.3f}",
            category="success" if result['improved'] else "failure",
            metadata={
                "layer": layer,
                "delta": delta,
                "improvement": result['improvement'],
                "checkpoint": baseline_id
            },
            importance=0.8 if result['improved'] else 0.5,
            tags=["experiment", f"delta_{abs(delta):.3f}"]
        )
        
        # 6. Update knowledge
        if result['improved']:
            # Strengthen success patterns
            success_patterns = memory.patterns.get_patterns(
                tags=["success", f"delta_{abs(delta):.3f}"]
            )
            
            if not success_patterns:
                # Form new theory
                theory_id = memory.form_theory(
                    description=f"Modifications of {delta:+.3f} are effective",
                    supporting_evidence=[obs_id],
                    confidence=0.6
                )
            else:
                # Strengthen existing theory
                for pattern in success_patterns:
                    pattern.add_evidence(obs_id)
        
        # 7. Form beliefs
        all_obs = memory.observations.query(category="success")
        if len(all_obs) >= 5:
            success_rate = len(all_obs) / (i + 1)
            if success_rate > 0.7:
                memory.form_belief(
                    content="Current modification strategy is effective",
                    confidence=0.8,
                    evidence=[obs_id]
                )
    
    # Final analysis
    print(f"\n📊 Final Knowledge State:")
    print(f"  Observations: {len(memory.observations.get_all())}")
    print(f"  Patterns: {len(memory.patterns.get_patterns())}")
    print(f"  Theories: {len(memory.theories.get_theories())}")
    print(f"  Beliefs: {len(memory.beliefs.get_beliefs())}")
    
    # Export
    memory.export("learning_results.json")
    print(f"\n✓ Exported knowledge to learning_results.json")

# Run it
learning_experiment_loop(model, num_experiments=20)
```

---

## Summary

The memory system enables true learning from experience:

1. **Observations**: Record everything that happens
2. **Patterns**: Automatically detect regularities  
3. **Theories**: Form causal models
4. **Beliefs**: Establish guiding principles

**Key Insight**: The system doesn't just execute modifications—it *learns* what works and what doesn't, forming increasingly accurate models of its own behavior.

This is the foundation for genuine self-improvement! 🧠✨

---

**Last Updated**: November 7, 2025  
**Version**: Phase 0 Complete  
**Ready**: Begin Phase 1 with learning enabled!