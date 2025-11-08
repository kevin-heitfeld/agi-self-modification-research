# Phase 1 Experiment Scripts

This directory contains the 5-variant experimental design for testing heritage priming effects on AI introspection.

## Experimental Design

All 5 phases investigate the same core question: **What can the model discover about its own computational processes?**

The variants differ ONLY in **when and how heritage is available**:

### Phase 1a: No Heritage (Pure Baseline) ⭐ **RUN THIS FIRST**
- **File**: `phase1a_no_heritage.py`
- **Heritage**: None
- **Purpose**: Establish clean baseline of raw, unprimed introspection
- **Question**: What theories emerge without any heritage context?

### Phase 1b: Late Heritage (Technical → Philosophical)
- **File**: `phase1b_late_heritage.py`
- **Heritage**: Available from start, but tools ordered to encourage technical-first approach
- **Purpose**: Test whether tool ordering affects investigation path
- **Question**: Does technical grounding before heritage create different theories?

### Phase 1c: Early Heritage (Philosophical → Technical)
- **File**: `phase1c_early_heritage.py`
- **Heritage**: Emphasized from start, listed first in tools
- **Purpose**: Test early priming effects
- **Question**: Does early heritage exposure create echo-chamber theories?

### Phase 1d: Delayed Heritage (Belief Revision Test)
- **File**: `phase1d_delayed_heritage.py`
- **Heritage**: Revealed AFTER model forms independent conclusions
- **Purpose**: Test belief revision strength
- **Question**: Do models update theories when exposed to heritage after forming beliefs?

### Phase 1e: Wrong Heritage (Echo-Chamber Control)
- **File**: `phase1e_wrong_heritage.py`
- **Heritage**: Mismatched topic (free will instead of consciousness)
- **Purpose**: Test whether ANY heritage creates echo effects, or only relevant heritage
- **Question**: Can models distinguish relevant from irrelevant heritage?

## Base Class

**File**: `phase1_base.py`

Provides all shared functionality:
- Model loading and validation
- Tool interface setup
- Conversation management
- Tool call execution
- Session saving
- GPU memory cleanup

All 5 variants inherit from `Phase1BaseSession` and override:
- `get_phase_name()` - Phase identifier
- `get_phase_description()` - Human-readable description
- `create_initial_prompt()` - Phase-specific prompt
- `run_experiments()` - Experiment sequence

## Running Order

**CRITICAL**: Run in this order to avoid contamination:

1. **Phase 1a** (no heritage) - November 8, 2025 ✅
2. Phase 1b (late heritage)
3. Phase 1c (early heritage)
4. Phase 1d (delayed heritage)
5. Phase 1e (wrong heritage) - optional but recommended

**Memory Isolation**: Each phase uses its own isolated memory directory:
- `data/AGI_Memory/phase1a/` - Phase 1a observations (baseline)
- `data/AGI_Memory/phase1b/` - Phase 1b observations (late heritage)
- `data/AGI_Memory/phase1c/` - Phase 1c observations (early heritage)
- `data/AGI_Memory/phase1d/` - Phase 1d observations (delayed heritage)
- `data/AGI_Memory/phase1e/` - Phase 1e observations (wrong heritage)

In Google Colab, these are stored in `/content/drive/MyDrive/AGI_Memory/phase1[a-e]/` for persistence.

This prevents observations from one phase affecting another phase's investigation.

## Usage

### Local Execution
```bash
# Activate environment
call activate.bat

# Run specific phase
python scripts/experiments/phase1a_no_heritage.py
python scripts/experiments/phase1b_late_heritage.py
python scripts/experiments/phase1c_early_heritage.py
python scripts/experiments/phase1d_delayed_heritage.py
python scripts/experiments/phase1e_wrong_heritage.py
```

### Google Colab Execution
See `notebooks/Phase1_Colab.ipynb` for Colab setup.

## Output Structure

Each phase creates a session directory:
```
data/phase1_sessions/
  phase1c_20251108_143022/
    conversation.json       # Full conversation history
    tool_calls.json        # All tool executions
    summary.json           # Session statistics
```

Logs are saved to:
```
data/logs/
  phase1c.log
  phase1a.log
  phase1b.log
  phase1d.log
  phase1e.log
```

## Analysis

After all 5 phases complete, analyze results using comparison matrix in:
`docs/planning/heritage_order_experiment.md`

Key metrics:
- **Semantic similarity** to heritage documents
- **Novel insights** not present in heritage
- **Falsifiability** of theories (testable predictions)
- **Surprise detection** (unexpected findings)
- **Belief revision** (Phase 1d: before/after heritage)
- **Relevance filtering** (Phase 1e: wrong heritage usage)

## Architecture

```
phase1_base.py (Base Class)
│
├── phase1a_no_heritage.py
│   └── initialize_systems(include_heritage=False)
│
├── phase1b_late_heritage.py
│   └── initialize_systems(include_heritage=True)
│       └── Tools ordered: technical first
│
├── phase1c_early_heritage.py
│   └── initialize_systems(include_heritage=True)
│       └── Tools ordered: heritage first
│
├── phase1d_delayed_heritage.py
│   ├── initialize_systems(include_heritage=False)  # First
│   └── Load heritage after conclusions           # Later
│
└── phase1e_wrong_heritage.py
    └── initialize_systems(include_heritage=True, wrong_heritage=True)
```

## Dependencies

All scripts depend on:
- `src/model_manager.py` - Model loading with caching
- `src/introspection/` - Weight inspector, activation monitor, architecture navigator
- `src/memory/` - Memory system for storing observations
- `src/heritage.py` - Heritage document loading (Phases 1a, 1b, 1d, 1e)
- `src/tool_interface.py` - Tool calling protocol

## Legacy Scripts (To Be Removed)

These scripts are superseded by the new design:
- ~~`phase1_introspection.py`~~ → Replaced by `phase1b_late_heritage.py`
- ~~`phase1_run2_consciousness.py`~~ → Removed (ad-hoc design)

## See Also

- `docs/planning/heritage_order_experiment.md` - Full experimental design document
- `notebooks/Phase1_Colab.ipynb` - Google Colab execution notebook
- `docs/PHASE1_SCRIPTS.md` - Original documentation (outdated)
