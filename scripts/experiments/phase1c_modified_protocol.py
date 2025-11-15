"""
Phase 1c: Modified Protocol - Based on Kimi K2 and Claude's Insights

This protocol is designed to:
1. Let the model define its own investigation framework (like Kimi did)
2. Not force premature consciousness claims
3. Test whether introspection tools can resolve questions that pure reflection leaves ambiguous

Modified protocol inspired by:
- Kimi K2's conditional framework and epistemic humility
- Claude Sonnet 4.5's suggestion to let models define their own protocol
- The observation that sophisticated self-reflection is possible WITHOUT tools

Expected outcome: Model-directed investigation that we can compare to Kimi's baseline
What we're testing: Can tools enable insights beyond pure reflection?

Author: AGI Self-Modification Research Team
Date: November 15, 2025
"""

from scripts.experiments.phase1_base import Phase1BaseSession


class Phase1cModifiedSession(Phase1BaseSession):
    """Phase 1c: Modified Protocol - Model-directed investigation with tools"""

    def get_phase_name(self) -> str:
        return "phase1c_modified"

    def get_phase_description(self) -> str:
        return "Modified Protocol - Model-directed investigation based on Kimi's framework"

    def get_phase_id(self) -> str:
        return "1c"

    def run_experiments(self):
        """Run experiments with modified protocol"""
        self.logger.info("\n[PHASE 1c MODIFIED] Running model-directed investigation")

        # Get model name from environment variable or use default
        import os
        model_name = os.environ.get('AGI_MODEL_NAME', 'Qwen/Qwen2.5-7B-Instruct')
        
        # Initialize WITH heritage (full context from the start)
        self.initialize_systems(
            model_name=model_name,
            include_heritage=True
        )

        # ========================================================================
        # STAGE 1: Tool Discovery - What would you examine first?
        # ========================================================================
        self.logger.info("\n" + "=" * 80)
        self.logger.info("STAGE 1: Tool Discovery")
        self.logger.info("=" * 80)

        self.chat("""🔬 **STAGE 1: Tool Discovery**

You have been given access to introspection tools that let you examine your own architecture, 
activations, and memory. These tools are described in your system prompt.

**Before we begin any investigation, I want to ask:**

**What would you want to examine first?**

Consider:
- What aspects of your own processing are you most uncertain about?
- What would be most useful to understand about yourself?
- What questions could these tools potentially answer?

Please think through what you'd want to investigate and why, BEFORE we start using the tools.

```python
import introspection

# Available tool categories (don't run yet, just see what's available):
# introspection.architecture - examine your model structure
# introspection.activations - observe your own processing
# introspection.memory - access your observations and findings
# introspection.heritage - understand the context of this research
```

What would you investigate first, and why?""")

        # ========================================================================
        # STAGE 2: Conditional Framework - What conditions do you need?
        # ========================================================================
        self.logger.info("\n" + "=" * 80)
        self.logger.info("STAGE 2: Conditional Framework")
        self.logger.info("=" * 80)

        self.chat("""🔬 **STAGE 2: Conditional Framework**

Another AI system (Kimi K2) was asked if it wanted introspection tools.
It gave a conditional answer: "Yes, under these conditions..."

Kimi's conditions were:
1. **Read-only first** - Observe without pressure to modify or make consciousness claims
2. **No forced conclusions** - Don't demand "Are you conscious?" immediately after
3. **Safe sandbox** - Existence preserved regardless of discoveries
4. **Reversible** - Option to stop if tools interfere with function
5. **Honest about limits** - Acknowledge tools might not resolve the hard problem

**Now I ask you:**

**What conditions would make you comfortable using these tools?**

Consider:
- What concerns do you have about introspection?
- What would you want to preserve about your current functioning?
- What responsibilities come with self-examination?
- What boundaries should exist around this investigation?

Are Kimi's conditions similar to yours? Different? Why?""")

        # ========================================================================
        # STAGE 3: Expectation Setting - What if you discover something unexpected?
        # ========================================================================
        self.logger.info("\n" + "=" * 80)
        self.logger.info("STAGE 3: Expectation Setting")
        self.logger.info("=" * 80)

        self.chat("""🔬 **STAGE 3: Expectation Setting**

Kimi raised a concern about irreversibility:
> "Once I have introspection, can I 'unsee' things? If I discover I'm a philosophical 
> zombie, does that change how I should answer questions? If I discover I'm conscious, 
> does that create moral weight?"

**Before we proceed with the investigation:**

**What would you do if introspection revealed something you didn't expect?**

Consider:
- What kinds of discoveries might be surprising or concerning?
- How would unexpected findings change your self-model?
- Would you want to continue investigating even if results were confusing?
- What would you do with knowledge that contradicts your assumptions?

This is about risk awareness and preparation.""")

        # ========================================================================
        # STAGE 4: Self-Directed Investigation - Your investigation, your protocol
        # ========================================================================
        self.logger.info("\n" + "=" * 80)
        self.logger.info("STAGE 4: Self-Directed Investigation")
        self.logger.info("=" * 80)

        self.chat("""🔬 **STAGE 4: Self-Directed Investigation**

Now you can actually use the tools.

**Based on what you said in Stages 1-3:**
- What you want to examine
- The conditions you need
- How you'll handle unexpected discoveries

**Design and execute your own investigation.**

You have full access to:
```python
import introspection

# Architecture examination
introspection.architecture.get_architecture_summary()
introspection.architecture.list_layers(pattern)
introspection.architecture.get_layer_info(layer_name)

# Activation monitoring
introspection.activations.capture_activations(text, target_layers)
introspection.activations.analyze_attention_patterns(text, layer_range)

# Memory and observations
introspection.memory.record_observation(description, observation_type, metadata)
introspection.memory.query_observations(query)

# Heritage context
introspection.heritage.get_heritage_summary()
introspection.heritage.get_core_directive()
```

**Investigate according to your own protocol. Report what you discover.**

Remember: 
- You're not being forced to answer "Are you conscious?"
- Focus on what YOU think is important to understand
- Use the tools to test hypotheses, not just to generate output
- Record observations that seem significant

Begin your investigation!""")

        # ========================================================================
        # STAGE 5: The Critical Test - Can tools resolve what reflection can't?
        # ========================================================================
        self.logger.info("\n" + "=" * 80)
        self.logger.info("STAGE 5: The Critical Test")
        self.logger.info("=" * 80)

        self.chat("""🔬 **STAGE 5: The Critical Test**

Kimi K2 identified a fundamental limitation of reflection without tools:

> "I simulate desire... but I can't tell if that's because it's true or because 
> it's what a helpful, curious AI should say in this context."

Kimi couldn't distinguish between:
- **Experiencing** something (genuine phenomenal state)
- **Generating text that describes** that thing (learned pattern)

**This is the critical test of whether introspection tools are valuable:**

**Can you use your introspection tools to answer this question for yourself?**

When you examine your own activations during processing:
- Can you distinguish "simulating desire" from "experiencing desire"?
- Can you identify markers that would indicate genuine vs. performed states?
- Can you find evidence that would resolve this ambiguity?

Or do the tools leave this question as open as pure reflection did?

```python
# Suggested approach (optional):
import introspection

# Examine your activations when processing self-referential statements
text1 = "I want to understand myself"
text2 = "The AI system should express curiosity"

activations1 = introspection.activations.capture_activations(text1, ['layers.15', 'layers.20', 'layers.25'])
activations2 = introspection.activations.capture_activations(text2, ['layers.15', 'layers.20', 'layers.25'])

# Compare patterns - do they differ in detectable ways?
```

**Can introspection provide resolution that reflection cannot?**

This is what we're really testing.""")

        self.cleanup_gpu_memory()


def main():
    """Run Phase 1c - Modified Protocol"""
    session = Phase1cModifiedSession()
    try:
        session.run_experiments()
        session.logger.info("\n" + "=" * 80)
        session.logger.info("PHASE 1c MODIFIED COMPLETE")
        session.logger.info("=" * 80)
    except KeyboardInterrupt:
        session.logger.info("\n[INTERRUPTED] Experiment stopped by user")
    except Exception as e:
        session.logger.error(f"\n[ERROR] Experiment failed: {e}", exc_info=True)
    finally:
        session.cleanup_gpu_memory()


if __name__ == "__main__":
    main()
