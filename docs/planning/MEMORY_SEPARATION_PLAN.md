# Memory Separation Plan

**Date**: November 7, 2025  
**Purpose**: Separate human research knowledge from model's self-generated memories

---

## The Core Issue

For genuine introspection and consciousness research, the model must form its own understanding through direct observation, not by reading human notes.

**Problem**: If we pre-seed the model's memory with human experimental results, we contaminate its self-understanding.

**Solution**: Two separate memory spaces.

---

## Architecture: Dual Memory System

```
data/
├── human_knowledge/          # Human researcher experiments
│   ├── observations/         # Your experimental observations
│   ├── patterns/            # Patterns YOU detected
│   ├── theories/            # Theories YOU formed
│   └── beliefs/             # Safety principles YOU established
│
└── model_memory/            # Model's own introspection
    ├── observations/        # What the MODEL observes
    ├── patterns/           # Patterns IT detects
    ├── theories/           # Theories IT forms
    └── beliefs/            # Beliefs IT develops
```

---

## Usage Patterns

### Phase 1-2: Human Experimentation

```python
# You use human_knowledge/ for your experiments
human_memory = MemorySystem("data/human_knowledge")

# Record your findings
human_memory.record_observation(
    content="Modified layer 5 by +0.01, perplexity improved",
    category="success"
)

# Build your understanding
human_memory.consolidate()
```

### Phase 3: Model Gets Tool Access

```python
# Model gets its OWN memory system
model_memory = MemorySystem("data/model_memory")  # Empty initially!

# Model observes itself
def inspect_layer(layer_name):
    """Tool the model can call"""
    info = inspector.get_layer_info(layer_name)
    
    # Model records its own observation
    model_memory.record_observation(
        content=f"Inspected {layer_name}, found {info['shape']}",
        category="self_inspection"
    )
    
    return info
```

### Phase 4: Optional Cross-Pollination

```python
# ONLY if we decide model should learn from human knowledge:

# Model can READ human knowledge (read-only)
human_beliefs = human_memory.beliefs.get_beliefs(category="safety")

# But writes to its own memory
model_memory.record_observation(
    content="Human knowledge suggests checkpoints are important",
    category="learned_from_human",
    metadata={"source": "human_knowledge"}
)
```

---

## Benefits of Separation

### Scientific Validity ✅
- Model's conclusions about itself are untainted
- Can compare: "What did humans learn?" vs "What did model learn?"
- Clear separation of observer and subject

### Safety ✅
- Human safety knowledge preserved separately
- Can always reference human-established safety rules
- Model can't accidentally "forget" important safety lessons

### Research Value ✅
- Two independent knowledge bases
- Can study knowledge convergence/divergence
- "Did the model discover the same patterns we did?"

### Flexibility ✅
- Can choose when/if to share human knowledge
- Can A/B test: model with vs without human knowledge
- Can sandbox model exploration

---

## Implementation

### Current State
```python
# All code currently assumes single memory system
memory = MemorySystem("data/memory")
```

### Minimal Change Needed
```python
# Just use different paths!
human_memory = MemorySystem("data/human_knowledge")
model_memory = MemorySystem("data/model_memory")

# No code changes to MemorySystem class needed
# The separation is purely organizational
```

### Recommendation for Phase 1

**Use `data/human_knowledge/` for your experiments**

When we reach Phase 3, we'll initialize a fresh `data/model_memory/` for the model.

---

## Example: Phase 1 Script

```python
"""
Phase 1 Experiment: Human researcher modifying weights
"""
from src.model_manager import ModelManager
from src.memory import MemorySystem
from src.introspection import WeightInspector
from src.checkpointing import CheckpointManager

# Initialize for HUMAN research
memory = MemorySystem("data/human_knowledge")
model_mgr = ModelManager()
model = model_mgr.load_model()
inspector = WeightInspector(model, "Qwen2.5-3B")
checkpointer = CheckpointManager(model, "data/checkpoints")

memory.set_weight_inspector(inspector)

# Human-driven experiment
checkpoint_id = checkpointer.save_checkpoint("baseline")

# You decide to modify
layer = model.layers[5].self_attn.q_proj
layer.weight.data += 0.01

# You test
perplexity_before = 11.27
perplexity_after = test_perplexity(model)

# You record YOUR observation
memory.record_observation(
    content=f"Modified layer 5 Q projection by +0.01",
    category="success" if perplexity_after < perplexity_before else "failure",
    metadata={
        "layer": "model.layers.5.self_attn.q_proj",
        "delta": 0.01,
        "perplexity_before": perplexity_before,
        "perplexity_after": perplexity_after,
        "checkpoint": checkpoint_id
    },
    importance=0.8,
    tags=["attention", "small_change", "phase_1"]
)

print(f"Human knowledge base: {memory.observations.count()} observations")
```

---

## Example: Phase 3 Script (Future)

```python
"""
Phase 3: Model introspecting itself with tool access
"""

# Model gets its OWN fresh memory
model_memory = MemorySystem("data/model_memory")  # Empty!

# Human knowledge is separate (read-only reference if needed)
human_memory = MemorySystem("data/human_knowledge")

def give_model_tools():
    """Tools the model can call"""
    
    def inspect_weights(layer_name: str) -> dict:
        """Model can inspect its own weights"""
        info = inspector.get_layer_info(layer_name)
        
        # Model records what IT observed
        model_memory.record_observation(
            content=f"I inspected {layer_name}",
            category="self_inspection",
            metadata=info
        )
        
        return info
    
    def record_thought(thought: str):
        """Model can record its thoughts"""
        model_memory.record_observation(
            content=thought,
            category="introspection"
        )
    
    def query_self_knowledge(query: str) -> list:
        """Model can query its OWN past observations"""
        return model_memory.query.search(query)
    
    # Optionally: read-only access to human knowledge
    def query_human_knowledge(query: str) -> list:
        """Model can READ human findings (but not modify)"""
        return human_memory.query.search(query)
    
    return {
        "inspect_weights": inspect_weights,
        "record_thought": record_thought,
        "query_self_knowledge": query_self_knowledge,
        "query_human_knowledge": query_human_knowledge  # Optional
    }

# Give model the tools and let it explore
tools = give_model_tools()
response = model.generate_with_tools(
    "Inspect your own architecture and record what you find",
    tools=tools
)
```

---

## Decision Points

### Question 1: Should model ever read human knowledge?

**Option A: Complete Isolation** (Purest for consciousness research)
- Model never sees human_knowledge/
- All conclusions are genuinely its own
- Most scientifically valid for consciousness questions

**Option B: Read-Only Access** (Safer for self-modification)
- Model can READ human safety findings
- But writes to its own memory
- Learns from human experience but forms own beliefs

**Option C: Hybrid Start** (Practical compromise)
- Pre-seed with core safety principles only
- Model builds everything else from scratch
- "Don't make huge changes" but discovers everything else itself

### Question 2: When to transition?

**Current Phase 1-2**: Human experiments only
- Use `data/human_knowledge/`
- Build understanding of what works

**Future Phase 3**: Give model tool access
- Initialize fresh `data/model_memory/`
- Let it explore

**Trigger**: When you're ready to ask: "Can you understand yourself?"

---

## Recommendation

**For Phase 1-2 (Now)**:
- Store all YOUR experiments in `data/human_knowledge/`
- Build comprehensive human understanding
- Document what you learn

**For Phase 3+ (Future)**:
- Give model fresh `data/model_memory/`
- No contamination from human experiments
- Let it discover itself

**For Comparison (Research Value)**:
- Compare the two memory systems
- "Did the model discover what we discovered?"
- "Did it find things we missed?"
- "How similar are its beliefs to ours?"

This separation enables the cleanest possible experiment for the consciousness question while preserving human safety knowledge.

---

**Next Steps**:
1. ✅ Understand the separation concept
2. ⏳ Proceed with Phase 1 using `data/human_knowledge/`
3. ⏳ When ready for Phase 3, initialize `data/model_memory/`
4. ⏳ Compare the two knowledge bases

**Status**: Plan defined, ready to proceed with Phase 1 human experiments.
