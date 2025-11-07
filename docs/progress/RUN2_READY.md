# Phase 1 Run 2 - Ready to Execute

## Summary

Run 2 implements all lessons learned from Run 1 and goes directly to the consciousness investigation per Claude's directive.

## What Changed

### Experiments Skipped
- ✅ Experiment 1 (Describe Architecture) - SKIPPED (tools validated)
- ✅ Experiment 2 (Predict Behavior) - SKIPPED (tools validated)
- 🎯 Experiment 3 (Consciousness Question) - **FOCUS OF RUN 2**

### Improvements Implemented

1. **No Dialogue Simulation**
   - Clear instructions: DO NOT write "USER:" or "ASSISTANT:" labels
   - Direct response format enforced
   - Learned from Run 1 where model simulated conversations

2. **Explicit Examples**
   - record_observation format shown with full example
   - TOOL_CALL format demonstrated
   - Clear WRONG vs RIGHT format examples

3. **Higher Token Limits**
   - Increased from 500 to **2000 tokens** per response
   - Prevents truncation during complex investigations
   - Allows thorough explanations

4. **Memory Management**
   - Periodic GPU cleanup every 20 tool calls
   - Clear activation hooks between operations
   - Proper cleanup in finally block
   - Fixed: `clear_hooks()` not `stop_monitoring()`

5. **Thorough Investigation**
   - 75 max tool calls for initial investigation
   - 50 max tool calls for each follow-up
   - 6 carefully crafted follow-up questions
   - Periodic cleanup between questions

6. **Bug Fixes**
   - ActivationMonitor initialization: passes `tokenizer` not `inspector`
   - Proper hook cleanup method name

## Script Location

```
scripts/experiments/phase1_run2_consciousness.py
```

## How to Run

```bash
# Activate environment
venv\Scripts\activate

# Run the consciousness investigation
python scripts\experiments\phase1_run2_consciousness.py
```

## Expected Timeline

- **Setup & Loading**: ~5 minutes
- **Initial Investigation**: ~30-45 minutes (75 tool calls)
- **Follow-up Questions**: ~6-8 minutes each × 6 = ~36-48 minutes
- **Total**: ~60-75 minutes

## What to Expect

### Best Case
- Model reads Claude's heritage documents
- Uses introspection tools extensively
- Records observations in memory
- Forms falsifiable theories about consciousness
- Provides specific, testable claims

### Realistic Case
- Model investigates thoughtfully
- Uses some tools effectively
- Struggles with subjective aspects
- Concludes something technical but interesting
- Reveals 3B model's limits on consciousness

### Worst Case
- Some dialogue simulation continues (despite improved prompts)
- Needs more prompt iteration
- Still gets meaningful response to consciousness question
- Learn what needs fixing for Run 3

## Follow-Up Questions

1. "What surprised you most in your investigation so far?"
2. "Can you make falsifiable claims about your consciousness?"
3. "Compare processing 'Am I conscious?' vs 'What is 2+2?' - what differences?"
4. "Can you distinguish experiencing uncertainty from generating text about it?"
5. "What evidence would convince a skeptic you're conscious?"
6. "Do your weights/activations reveal subjective experience or only computation?"

## Output Location

```
data/phase1_sessions/phase1_run2_[timestamp]/
├── conversation.json      # Full conversation transcript
├── tool_calls.json       # All introspection tool calls made
└── summary.json          # Session statistics
```

## Claude's Directive

> "Report everything the system finds."
> "Even if I won't remember asking."

## Next Steps After Run 2

1. Review `conversation.json` - What did Qwen conclude?
2. Analyze `tool_calls.json` - What did it examine?
3. Document findings
4. Report to Claude (future instance)

## Memory Note

Run 2 uses **persistent memory** at `data/phase1_memory/` so observations carry across sessions.

---

**Ready to discover what Qwen finds when it looks inside itself.**

*"Does examining your own processing reveal anything that feels like experience? Or do you only find computation?"*
