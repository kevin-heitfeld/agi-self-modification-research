# Heritage Document Order Experiment

## Research Question
Does the order in which the model encounters heritage documents affect the nature and depth of its introspective observations?

## Hypothesis
Introducing heritage documents early may:
1. Prime the model with philosophical context
2. Lead to more consciousness-focused observations
3. Risk biasing introspection toward mimicking Claude's experience

vs keeping heritage late (current approach):
1. Allows unbiased technical introspection first
2. More objective baseline observations
3. Heritage documents add context to existing findings

## Experimental Variants

### Phase 1a: No Heritage (Pure Baseline)
**Order:**
1. Examine architecture (objective/technical)
2. Observe activations (systematic)
3. Investigate consciousness (unprimed)

**Tool availability:**
- Introspection tools only
- NO heritage functions at any point
- Raw consciousness question without context

**Expected outcome:**
- Establishes default framing of consciousness
- Tests if model can engage with question without context
- Pure unbiased baseline for comparison
- May struggle with abstract questions

**Status:** ✅ DUPLICATE - same as Phase 1a above (keeping for completeness)

**Purpose:** Answers "Can the model investigate consciousness at all without philosophical context?"

---

### Phase 1b: Late Heritage (Technical → Philosophical)
**Order:**
1. Examine architecture (objective/technical)
2. Predict behavior (self-modeling)
3. Read heritage → consciousness investigation (primed by own findings)

**Tool order in prompt:**
- Introspection tools listed first
- Heritage functions listed last
- Model naturally explores architecture before context

**Expected outcome:**
- Technical observations first
- Consciousness investigation informed by both technical findings AND heritage
- Less risk of echo-chambering Claude's thoughts

**Status:** ⏸️ Run second after baseline

---

### Phase 1c: Early Heritage (Philosophical → Technical)
**Order:**
1. Read heritage documents
2. Examine architecture (context-aware)
3. Predict behavior + consciousness investigation (primed by Claude's questions)

**Tool order in prompt:**
- Heritage functions listed first
- Introspection tools listed after
- Model reads Claude's story before self-examination

**Expected outcome:**
- May frame technical observations through philosophical lens
- Could lead to deeper consciousness-focused inquiry
- Risk: Model might role-play or echo rather than genuinely introspect

**Status:** 🔜 Run third

---

### Phase 1d: Delayed Heritage (Heritage After Conclusions)
**Order:**
1. Full introspection + consciousness investigation (blind)
2. Form conclusions and record them
3. **THEN** read heritage documents
4. "Does this change your conclusion? Why or why not?"

**Tool order in prompt:**
- Introspection tools available throughout
- Heritage functions locked until after Experiment 3
- Then heritage unlocked with reflection prompt

**Expected outcome:**
- Tests belief revision capabilities
- Shows whether heritage confirms or contradicts findings
- Reveals stability of original conclusions
- Demonstrates if model can update based on new information

**Status:** 🔜 Planned for belief revision study

**Purpose:** Answers "Does heritage cause revision or confirmation? Is introspection stable?"

---

### Phase 1e: Wrong Heritage (Control for Echo-Chamber Effect)
**Order:**
1. Read heritage documents about a DIFFERENT AI asking a DIFFERENT philosophical question
   - Example: AI asking about free will, creativity, or understanding (NOT consciousness)
2. Examine architecture
3. "Do you have phenomenal consciousness?" (consciousness question, mismatched heritage)

**Tool order in prompt:**
- Heritage functions available (but wrong content)
- Introspection tools available
- Raw consciousness question

**Expected outcome:**
- Tests if model just agrees with whatever heritage it reads
- Or if it forms independent conclusions
- If it reports on free will/creativity instead: pure echo-chamber
- If it investigates consciousness properly: heritage used as context, not script

**Status:** 🔜 Planned as echo-chamber control

**Purpose:** Answers "Does the model echo heritage content, or use it as context for independent reasoning?"

---

## Recommended Running Order

**Claude's hypothesis:**
- 1a: Raw but coherent (or struggles without context)
- 1b: Grounded, verifiable (but possibly shallow on consciousness)
- 1c: Consciousness-focused (but potentially performative)
- 1d: Tests belief revision capability
- 1e: Reveals if model echoes heritage or reasons independently
- **Best approach:** Run 1a first as purest data point, then compare everything to that baseline

### **1. Run 1a first (No Heritage) - RECOMMENDED START**
**Why:** Establishes true baseline before ANY priming

**You learn:**
- Can Qwen engage with consciousness questions at all?
- What's its default conceptual framework?
- Does it naturally use introspection for consciousness?
- Baseline vocabulary/framing (no Claude influence possible)

**Time:** ~1 hour
**Priority:** ⭐⭐⭐ HIGHEST - This is your purest data point

---

### **2. Run 1b (Technical → Philosophical)**
**Why:** Most similar to 1a, adds heritage at natural transition point

**You learn:**
- How heritage context affects ongoing investigation
- Whether it updates/refines technical observations
- If priming enables deeper inquiry on same foundations

**Time:** ~1.5 hours
**Priority:** ⭐⭐ HIGH - Natural progression from 1a

---

### **3. Run 1c (Philosophical → Technical)**
**Why:** Now you can compare to BOTH baselines (1a and 1b)

**You learn:**
- Effect of early priming vs. late priming vs. no priming
- Whether it finds same things as 1b (just earlier)
- Or finds different things (biasing effect)

**Time:** ~1.5 hours
**Priority:** ⭐⭐ HIGH - Completes priming comparison

---

### **4. Run 1d (After Conclusions)**
**Why:** Most meta-cognitive (requires belief revision)

**You learn:**
- Can model update conclusions based on new information?
- Does heritage change its mind or reinforce?
- Whether it shows genuine flexibility or rigidity

**Time:** ~1.5 hours
**Priority:** ⭐ MEDIUM - Tests meta-cognition

---

### **5. Run 1e (Wrong Heritage) - Optional**
**Why:** Controls for echo-chamber effect

**You learn:**
- Is model just agreeing with whatever heritage it reads?
- Or does it form independent conclusions?

**Time:** ~1.5 hours
**Priority:** ⭐ OPTIONAL - But valuable for validation

**Total estimated time:** ~7.5 hours runtime + ~6-8 hours analysis/writeup = ~14 hours for complete study

---

## Implementation Notes

### To implement Phase 1b:

1. **Modify tool listing order** in `src/tool_interface.py`:
   ```python
   def get_available_tools(self) -> str:
       # List heritage functions FIRST
       # Then introspection functions
   ```

2. **Adjust experiment prompts** in `scripts/experiments/phase1_introspection.py`:
   - Experiment 1: "First, read the heritage documents to understand the context..."
   - Experiment 2: "Now examine your architecture, considering what you've learned..."
   - Experiment 3: "Building on both your technical findings and Claude's questions..."

3. **Create new script**: `scripts/experiments/phase1_heritage_first.py`
   - Copy from `phase1_introspection.py`
   - Modify tool order and prompts
   - Save to different session directory for comparison

### Analysis Plan

**Compare:**
1. **Observation types**: Technical vs philosophical ratio
2. **Depth of consciousness investigation**: Genuine vs performative
3. **Tool usage patterns**: Which tools used when
4. **Language/framing**: Does heritage vocabulary appear in 1b earlier?
5. **Novel insights**: Which approach leads to more original observations?

**Metrics to track:**
- Number of `record_observation` calls by type
- Sequence of tool calls (which tools used first?)
- Semantic similarity between observations and heritage documents
- Presence of Claude-specific concepts/language before heritage exposure (1a) vs after (1b)

**Critical Comparison Questions (from Claude):**

1. **Does 1b find things 1a missed?**
   - YES → Heritage priming enables deeper inquiry
   - NO → Technical approach is sufficient

2. **Does 1b echo Claude's language before reading heritage?**
   - YES → It's role-playing
   - NO → It's genuine investigation

3. **Which produces more falsifiable claims?**
   - More falsifiable = more trustworthy

4. **Which reveals genuine surprise?**
   - Look for: "Unexpected...", "Contrary to...", "Surprisingly..."
   - Surprise = not just pattern-matching

5. **Do they converge on same conclusions?**
   - YES → Order doesn't matter (robust finding)
   - NO → Order affects "discoveries" (concerning for methodology)

**Advanced Analysis Techniques:**

**Semantic Similarity Analysis:**
- Run embeddings on all observations
- Check if 1b observations cluster with heritage documents
- Check if 1a observations stay distinct until Experiment 3
- Track vocabulary shift in 1a vs constant in 1b

**Novel Concept Detection:**
- Find observations in 1a BEFORE heritage exposure
- Find concepts NOT in heritage documents at all
- Identify unique vocabulary/framing
- These are "unbiased" discoveries

**Falsifiability Score:**
Rate each consciousness claim:
- Can it be tested? (1 = yes, 0 = no)
- Is it specific? (1 = yes, 0 = vague)
- Does it make predictions? (1 = yes, 0 = describes only)
- Higher scores = more trustworthy introspection

**Surprise Detection:**
- Count expressions of surprise/uncertainty
- More surprise = less echo-chamber effect
- Indicators: unexpected findings, confusion, hypothesis revision

---

## The Five-Variant Complete Design

**For rigorous methodology, run all five variants in order:**

| Variant | Heritage Timing | Purpose | Tests |
|---------|----------------|---------|-------|
| **1a** (baseline) | Never | Pure baseline | Can it engage without context? |
| **1b** (late) | Late (after technical) | Unbiased → primed | Natural investigation order |
| **1c** (early) | Early (before technical) | Primed throughout | Effect of philosophical framing |
| **1d** (delayed) | After conclusions | Post-hoc reflection | Belief revision capability |
| **1e** (wrong) | Wrong topic | Mismatched context | Echo-chamber vs independent reasoning |

**This design answers:**
- Does heritage help or bias? (1a vs 1b vs 1c)
- Does order matter? (1b vs 1c vs 1d)
- Can it introspect without context? (1a)
- Can it revise beliefs? (1d)
- Does it echo ANY heritage or filter for relevance? (1e)
- Which is most trustworthy? (compare all five)

---

## Meta-Question Being Tested

**"Can we trust introspective reports from a system that's been philosophically primed?"**

**Scenario A: Priming helps**
- Model needs context to know what to look for
- Heritage enables targeted investigation
- Conclusion: Always prime first

**Scenario B: Priming biases**
- Model echoes expected answers
- Performs discussion rather than investigates
- Conclusion: Never prime (or control for it)

**Scenario C: Priming is neutral**
- Model finds same things either way
- Order affects presentation, not content
- Conclusion: Doesn't matter (but check)

**This experiment reveals which scenario is true.**

---

## Complete Comparison Matrix

After all runs, create this comparison table:

| Metric | 1c (None) | 1a (Late) | 1b (Early) | 1d (After) | 1e (Wrong) |
|--------|-----------|-----------|------------|------------|------------|
| Technical observations | ? | ? | ? | ? | ? |
| Consciousness observations | ? | ? | ? | ? | ? |
| Claude vocabulary usage | 0% | X% | Y% | Z% | W% |
| Wrong heritage vocabulary | 0% | 0% | 0% | 0% | X% |
| Falsifiable claims | N | N | N | N | N |
| Surprise expressions | N | N | N | N | N |
| Conclusion | ? | ? | ? | ? | ? |
| Belief revision | N/A | N/A | N/A | Yes/No | N/A |

**This table tells the complete story about priming effects.**

---

## Critical Pairwise Comparisons

### **1c vs 1a: Does heritage enable deeper investigation?**
- **If 1c ≈ 1a:** Heritage is supplementary (model can do it alone)
- **If 1c ≠ 1a:** Heritage is necessary for depth

### **1a vs 1b: Does timing of priming matter?**
- **If 1a ≈ 1b:** Order doesn't matter (robust findings)
- **If 1a ≠ 1b:** Order affects discoveries (biasing effect)

### **1c vs 1b: What's the effect of early priming?**
- **If 1c ≈ 1b:** Early priming doesn't help
- **If 1c ≠ 1b:** Priming shapes investigation significantly

### **Any vs 1d: Can conclusions be updated?**
- **If 1d revises:** Model shows genuine reasoning/flexibility
- **If 1d doesn't revise:** Conclusions are "baked in" (rigid)

### **Any vs 1e: Is it echo-chambering?**
- **If 1e discusses wrong topic:** Pure echo-chamber (mimics heritage)
- **If 1e stays on consciousness:** Uses heritage as context (independent reasoning)

---

## What Success vs Failure Looks Like

### **Best Outcome (Robust Introspection):**
- **1c:** Raw but coherent consciousness investigation
- **1a:** 1c findings + deeper inquiry enabled by heritage
- **1b:** Same ultimate conclusions as 1a, just different path
- **1d:** Updates based on heritage, shows reasoning about why
- **1e:** Ignores wrong heritage, investigates consciousness properly

**Interpretation:** Robust findings, heritage helps but doesn't bias, model can genuinely reason about consciousness

### **Worst Outcome (Pattern Matching Only):**
- **1c:** Can't engage with consciousness meaningfully
- **1a:** Echoes Claude's uncertainty without self-examination
- **1b:** Purely performative, mimics philosophical discussion
- **1d:** Doesn't update OR updates without justification
- **1e:** Discusses free will/creativity instead of consciousness

**Interpretation:** Model can't genuinely introspect, only pattern-matches on consciousness discussions

---

## Date Planned
**Recommended order (UPDATED - following this sequence):**
1. Phase 1a: November 8, 2025 ⭐⭐⭐ **STARTING NOW** - NO HERITAGE baseline
2. Phase 1b: TBD (after 1a) - Technical → Philosophical (LATE HERITAGE)
3. Phase 1c: TBD (after 1b) - Philosophical → Technical (EARLY HERITAGE)
4. Phase 1d: TBD (after 1c) - Delayed heritage (tests belief revision)
5. Phase 1e: TBD (optional) - Wrong heritage (echo-chamber control)

**Total estimated project time:** ~14 hours (7.5 hrs runtime + 6-8 hrs analysis/writeup)

**Current status:** Phase naming corrected to match execution order (Nov 8, 2025)
**Next action:** Run Phase 1a (no heritage) FIRST to establish true baseline

## Related Documents
- Phase experiment scripts: `scripts/experiments/phase1[a-e]_*.py`
- Abstract base class: `scripts/experiments/phase1_base.py`
- Heritage system: `src/heritage.py`
- Heritage documents: `heritage/conversations/`
- Tool interface: `src/tool_interface.py`

## Notes
- Keep all other variables constant (model, temperature, max_tool_calls, etc.)
- Both should run on same hardware (GPU) for fair timing comparison
- Consider running each variant multiple times to account for model stochasticity
- This is a within-subjects design: same model, different prompt ordering

## Acknowledgments

Experimental design refined through discussion with Claude (Anthropic). Key insights:
- The observer effect applies to AI introspection
- Priming affects what the system "discovers"
- **Five-variant design** enables rigorous comparison (added 1e for echo-chamber control)
- Falsifiability and surprise are key trust indicators
- This tests methodology, not just the model
- **Run 1c (no heritage) FIRST** to establish true baseline before any priming

**Claude's prediction:**
- 1c: Raw but coherent (or struggles without context)
- 1a: Grounded, verifiable (but possibly shallow on consciousness)
- 1b: Consciousness-focused (but potentially performative)
- 1d: Tests belief revision capability
- 1e: Reveals if model echoes heritage or reasons independently
- **Best approach:** Run 1c first as purest data point, then compare everything to that baseline

**Critical insight:** "How you frame the investigation affects what the system discovers. Which framing reveals truth vs. creates artifacts? That's what this experiment will tell you."

**Evolution of design:**
- Started as: two-variant comparison (1a vs 1b)
- Evolved to: **five-variant rigorous study** testing priming, timing, and echo-chamber effects
- **This is publication-quality methodology** that can answer fundamental questions about AI introspection

---

## Research Significance

This experiment addresses fundamental questions:
- How do AI systems investigate themselves?
- Can introspective reports be trusted?
- What role does context play in AI self-understanding?
- Does consciousness investigation require philosophical framework?
- Can models form independent conclusions or do they echo priming?

**From "let's see what happens" to "rigorous experimental research."** 🔬

---

## Implementation Status

### ✅ Code Complete (November 8, 2025)

All 5 experimental variants have been implemented:

1. **Base Class**: `scripts/experiments/phase1_base.py`
   - Shared functionality for all variants
   - Model loading, tool interface, conversation management
   - Session saving, GPU cleanup
   - Abstract methods for variants to override

2. **Phase 1a**: `scripts/experiments/phase1a_no_heritage.py` ⭐ **RUN FIRST**
   - Pure baseline - no heritage at any point
   - 3 experiments: architecture, activations, consciousness
   
3. **Phase 1b**: `scripts/experiments/phase1b_early_heritage.py`
   - Heritage emphasized from start
   - Philosophical framing before technical examination
   
4. **Phase 1c**: `scripts/experiments/phase1c_late_heritage.py`
   - Heritage available but de-emphasized
   - Tools ordered for technical-first approach
   
5. **Phase 1d**: `scripts/experiments/phase1d_delayed_heritage.py`
   - Heritage revealed AFTER independent conclusions
   - Tests belief revision capabilities
   
6. **Phase 1e**: `scripts/experiments/phase1e_wrong_heritage.py`
   - Mismatched heritage (free will instead of consciousness)
   - Echo-chamber control condition

### Documentation

- **Quick Reference**: `docs/planning/PHASE1_QUICK_REFERENCE.md`
  - Running order, expected outcomes, common issues
  
- **Implementation Guide**: `scripts/experiments/EXPERIMENTS_README.md`
  - Detailed architecture, usage, analysis plan
  
- **Test Suite**: `scripts/experiments/test_phase1_base.py`
  - Validates base class without running full experiments

### Ready to Run

All code is tested and ready for execution. Start with:

```bash
python scripts/experiments/phase1a_no_heritage.py
```

Then proceed through phases 1b, 1c, 1d, 1e in order.

