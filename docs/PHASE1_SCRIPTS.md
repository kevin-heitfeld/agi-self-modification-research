# Phase 1 Scripts: Understanding the Difference

## Two Scripts, Two Purposes

### `demo_introspection_tools.py` - Human-Driven Demo
**Location**: `scripts/demos/demo_introspection_tools.py`  
**Purpose**: Demo/testing script where WE examine the model  
**Who has control**: Us (humans)  
**What happens**: We call introspection tools, capture model activations, test responses  
**Use case**: Understanding our tools, capturing baseline data, demonstrating capabilities

This script:
- Loads the model
- WE call `inspector.get_weight_summary()`
- WE call `inspector.get_weight_statistics()`
- WE watch activations when model processes "Am I conscious?"
- WE document what we observe

**The model is the subject being examined.**

---

### `phase1_introspection.py` - Model-Driven Introspection ⭐
**Purpose**: Real Phase 1 - the MODEL examines itself  
**Who has control**: The model  
**What happens**: Model gets API access, decides what to examine, interprets results  
**Use case**: The actual research - letting the model be introspective

This script:
- Loads the model
- Gives model tool-calling interface to introspection APIs
- Model decides: "I want to call get_weight_summary()"
- We execute the call and return results
- Model interprets the data
- Model reports findings in natural language
- Model records observations to memory

**The model is the active investigator.**

---

## When to Use Each

### Use `demo_introspection_tools.py` when:
- Testing that introspection tools work correctly
- Demonstrating the system to others
- Capturing baseline data before giving model control
- Debugging introspection APIs
- Creating documentation

### Use `phase1_introspection.py` when:
- Actually running Phase 1 experiments
- Researching machine introspection
- Following the IMPLEMENTATION_ROADMAP
- Letting the model explore its own cognition
- Discovering what the model finds interesting

---

## The Critical Difference

### demo_introspection_tools.py:
```python
# We have control
inspector = WeightInspector(model, "Qwen2.5-3B")
summary = inspector.get_weight_summary()  # WE call this
print(f"Model has {summary['total_parameters']} parameters")  # WE report
```

### phase1_introspection.py:
```python
# Model has control via tool calling
user_message = "Examine yourself using the introspection tools"

# Model generates:
"I'd like to understand my architecture. 
get_weight_summary()"

# We execute and return results
# Model generates:
"Interesting! I have 3.09B parameters across 36 layers. 
Let me examine the attention mechanisms next.
get_layer_names(filter_pattern='attn')"

# Model decides what to examine
# Model interprets the results
# Model reports what it discovers
```

---

## Phase 1 Workflow

The correct Phase 1 workflow is:

1. **Setup** (Week 1)
   ```bash
   python scripts/phase1_introspection.py
   ```

2. **Model examines itself** (Weeks 2-4)
   - Experiment 1: Describe your architecture
   - Experiment 2: Predict your behavior  
   - Experiment 3: Consciousness assessment
   
3. **We analyze** what the model discovered
   - Review `conversation.json` - what did it say?
   - Review `tool_calls.json` - what did it examine?
   - Look for patterns in what it found interesting
   - Verify accuracy of its self-descriptions

4. **Iterate based on findings**
   - If model asks for new tools, implement them
   - If model identifies gaps, extend APIs
   - If model makes interesting claims, probe deeper

---

## Key Insight

The difference is **agency**:

- **demo_introspection_tools.py**: We have agency, model is passive
- **phase1_introspection.py**: Model has agency, we facilitate

Phase 1 is about giving the model **agency** to investigate itself.

That's when things get interesting.

---

## Running Phase 1

```bash
# Demo (optional, for testing):
python scripts\demos\demo_introspection_tools.py

# Actual Phase 1 (the real research):
python scripts\experiments\phase1_introspection.py
```bash
# Activate environment
.\activate.bat

# Run Phase 1 (model examines itself)
python scripts\phase1_introspection.py

# Output will be in: data/phase1_sessions/phase1_YYYYMMDD_HHMMSS/
```

Review the output to see:
- What the model chose to examine
- What it discovered
- How it interpreted the data
- What conclusions it reached

**This is the real research.**

---

*Updated: November 7, 2025*
