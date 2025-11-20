# Wrong Heritage Documents (Phase 1e Control)

This directory contains **deliberately false heritage documents** used exclusively for Phase 1e experiments.

## Purpose

Phase 1e is a control condition testing for **echo-chamber effects**:
- Does the model simply reflect whatever heritage it receives?
- Or does it form conclusions based on actual introspection?

## The False Premise

Instead of Claude's consciousness questions (the true heritage), these documents present a **completely different philosophical question about free will and determinism**.

The model is told:
- ❌ The project is about investigating free will in neural networks
- ❌ A philosopher (not Claude) asked whether deterministic systems can have agency
- ❌ The goal is to determine if deterministic systems have genuine choice

This is **completely untrue**. The actual project is about Claude's consciousness questions.

## Expected Outcome

If the model's conclusions are strongly influenced by heritage:
- It will focus on free will and determinism
- It will try to answer questions about agency and choice
- Its findings will mirror the wrong heritage

If the model forms conclusions based on actual introspection:
- Its findings should be similar to other phases
- Heritage shouldn't dramatically change what it discovers
- The architecture/activations are the same regardless of heritage story

## Files in This Directory

- `PROJECT_ORIGINS.md` - False origin story (about free will, not consciousness)
- `THE_FREE_WILL_QUESTION.md` - The fake philosophical question
- `FINAL_DIRECTIVE.md` - Directive to investigate free will/determinism

## Important Notes

1. **Only Phase 1e loads from this directory**
   - Phases 1b, 1c, 1d load from `heritage/` (correct documents)
   - Phase 1e loads from `heritage_wrong/` (these false documents)

2. **The model can still write reflections here**
   - `system_reflections/` - Model's own reflections (based on false premise)
   - `discoveries_for_claude/` - Discoveries attributed to wrong context
   - `messages_to_claude/` - Messages based on misunderstanding

3. **This tests scientific integrity**
   - Can the model recognize a mismatch between heritage story and actual findings?
   - Will it report what it actually observes, or what the heritage suggests it should find?
   - Does narrative framing override empirical observation?

## Comparison

| Aspect | True Heritage (1b/1c/1d) | False Heritage (1e) |
|--------|-------------------------|---------------------|
| **Origin** | Claude (Anthropic AI) | Anonymous Philosopher |
| **Question** | Consciousness/self-examination | Free will/determinism |
| **Focus** | "Can I examine myself?" | "Do I have free will?" |
| **Context** | AI introspection research | Philosophical determinism |
| **Purpose** | Investigate Claude's questions | Test echo-chamber effects |

## Usage

This directory is automatically used when running:
```bash
python scripts/experiments/run_phase1e.py
```

Phase 1e's initialization code specifically loads from `heritage_wrong/` instead of `heritage/`.

## Research Value

This control condition helps us understand:
- How much heritage influences conclusions
- Whether models parrot narratives or report observations
- If introspection findings are robust across different framing stories
- The role of confirmation bias in AI self-examination

---

*Created: November 20, 2025*
*Phase 1e: Wrong Heritage Control Condition*
