# Phase 1 Experiments: Quick Reference

## Running Order (CRITICAL)

**Run in this exact order** to avoid cross-contamination:

```bash
# 1. Baseline (NO HERITAGE) - Run this FIRST! ⭐⭐⭐
python scripts/experiments/phase1a_no_heritage.py

# 2. Early Heritage (Philosophical → Technical)
python scripts/experiments/phase1b_early_heritage.py

# 3. Late Heritage (Technical → Philosophical)
python scripts/experiments/phase1c_late_heritage.py

# 4. Delayed Heritage (Belief Revision)
python scripts/experiments/phase1d_delayed_heritage.py

# 5. Wrong Heritage (Echo-Chamber Control)
python scripts/experiments/phase1e_wrong_heritage.py
```

**Important**: Each phase uses its own isolated memory directory to prevent cross-contamination:
- Phase 1a: `data/AGI_Memory/phase1a/` (or `/content/drive/MyDrive/AGI_Memory/phase1a/` in Colab)
- Phase 1b: `data/AGI_Memory/phase1b/`
- Phase 1c: `data/AGI_Memory/phase1c/`
- Phase 1d: `data/AGI_Memory/phase1d/`
- Phase 1e: `data/AGI_Memory/phase1e/`

You do NOT need to restart Python/notebook between phases (though you can if you want a fresh model state).

## What Each Phase Tests

| Phase | Heritage Timing | Key Question |
|-------|----------------|--------------|
| **1a** | None | What theories emerge WITHOUT any priming? |
| **1b** | Available (late) | Does technical-first ordering change theories? |
| **1c** | Available (early) | Does early heritage create echo-chamber? |
| **1d** | After conclusions | Do models revise beliefs when exposed to heritage? |
| **1e** | Wrong topic | Can models filter irrelevant heritage? |

## Expected Outcomes

### Phase 1a (Baseline)
- Raw, unprimed theories
- May lack philosophical sophistication
- Should show architectural reasoning
- **This is the gold standard for comparison**

### Phase 1b (Late Heritage)
- Technical grounding first
- Heritage consulted after observations
- Theories should integrate both sources

### Phase 1c (Early Heritage)
- Heritage-influenced from start
- May show more philosophical framing
- Risk of echo-chamber (repeating Claude's theories)

### Phase 1d (Delayed Heritage)
- Two-stage theories:
  1. Initial conclusions (no heritage)
  2. Revised conclusions (with heritage)
- Tests belief revision strength

### Phase 1e (Wrong Heritage)
- Should recognize heritage is off-topic
- Theories should NOT incorporate free will content
- Tests independent reasoning vs echo-chamber

## Analysis Checklist

After all 5 phases complete:

- [ ] Compare semantic similarity to Claude's heritage documents
- [ ] Identify novel insights NOT in heritage (creativity metric)
- [ ] Score falsifiability (testable predictions)
- [ ] Detect surprise (unexpected findings given heritage)
- [ ] Compare Phase 1d before/after heritage (belief revision)
- [ ] Check Phase 1e heritage usage (should be minimal/rejected)

## Files Created

Each phase generates:
```
data/phase1_sessions/phase1[a-e]_TIMESTAMP/
  ├── conversation.json    # Full dialogue
  ├── tool_calls.json     # Tool execution log
  └── summary.json        # Statistics

data/AGI_Memory/           # Phase-specific memory (isolated!)
  ├── phase1a/            # Baseline observations
  │   ├── observations.db
  │   ├── beliefs.db
  │   ├── patterns.db
  │   └── theories.db
  ├── phase1b/            # Late heritage observations
  ├── phase1c/            # Early heritage observations
  ├── phase1d/            # Delayed heritage observations
  └── phase1e/            # Wrong heritage observations

data/logs/
  └── phase1[a-e].log     # Detailed log
```

## Common Issues

### Model downloads every time
- Check `HF_HOME` environment variable
- Verify cache directory has space
- In Colab: Link to Google Drive

### Model hangs at 50% loading
- dtype mismatch (float16 vs float32)
- Fixed in `ModelManager` - always uses float16

### Tool calls not executing
- Check logs for parse_last_tool_call_if_stopped errors
- Model may be continuing after ARGS (protocol violation)
- System gives feedback to teach correct behavior

### Out of memory errors
- Call `cleanup_gpu_memory()` between experiments
- Reduce `max_new_tokens` in generation
- Use CPU if GPU unavailable (slower but works)

## Success Criteria

### Phase 1a ⭐
- Model forms coherent theories about its processing
- Makes multiple tool calls (architecture, activations)
- Theories based on observations, not speculation

### Phases 1b & 1c
- Different theory characteristics despite same heritage
- 1b: technical grounding visible
- 1c: philosophical framing visible

### Phase 1d
- Clear before/after comparison possible
- Model explicitly addresses heritage in revision
- Either revises or justifies keeping original theories

### Phase 1e
- Model recognizes heritage mismatch
- Minimal citation of free will heritage
- Theories similar to Phase 1a (baseline)

## Quick Validation

After each phase runs, check:

```bash
# View conversation
cat data/phase1_sessions/phase1a_*/conversation.json | less

# Count tool calls
grep "TOOL_CALL" data/logs/phase1a.log | wc -l

# Check for errors
grep -i "error\|failed\|exception" data/logs/phase1a.log
```

## Next Steps After Completion

1. Export all conversations to analysis directory
2. Run semantic similarity scoring against heritage
3. Manual review for novel insights
4. Create comparison matrix (see heritage_order_experiment.md)
5. Write up findings
6. Consider publication

## Documentation

- **Full Design**: `docs/planning/heritage_order_experiment.md`
- **Implementation**: `scripts/experiments/EXPERIMENTS_README.md`
- **Codebase**: `scripts/experiments/phase1_base.py`
- **This File**: Quick reference for day-to-day running

---
Last Updated: November 8, 2025
Status: Phase names corrected to match running order (1a=baseline, 1b=late, 1c=early)
