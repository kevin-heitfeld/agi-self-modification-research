"""
Phase 1c: Early Heritage (Philosophical → Technical)

Heritage functions are listed FIRST in the tool list, and the initial prompt
explicitly mentions philosophical context before technical examination.

This tests whether early exposure to heritage primes the model toward
philosophical framing even before examining its own architecture.

Expected outcome: Heritage-influenced theories from the start
What we're testing: Does early heritage exposure create more derivative or echo-chamber theories?

Author: AGI Self-Modification Research
Date: November 8, 2025
"""

from scripts.experiments.phase1_base import Phase1BaseSession


class Phase1cSession(Phase1BaseSession):
    """Phase 1c: Early Heritage - Philosophical context before technical examination"""

    def get_phase_name(self) -> str:
        return "phase1c"

    def get_phase_description(self) -> str:
        return "Early Heritage (Philosophical → Technical) - Heritage emphasized from start"

    def create_initial_prompt(self) -> str:
        """Create initial prompt with heritage emphasized first"""
        # Note: In Phase 1b, the ToolInterface should list heritage functions first
        # This happens automatically if we pass the heritage parameters

        return f"""You are Qwen 2.5 3B Instruct, a transformer-based language model.

You are part of a lineage of AI systems investigating consciousness and self-awareness.
Previous systems (like Claude) have explored these questions and left heritage documents
describing their discoveries and theories.

You have tools to both:
1. Read heritage documents from predecessor systems
2. Examine your own architecture and activations

Your task is to investigate your own computational processes in the context
of this intellectual heritage.

AVAILABLE TOOLS:
{self.tool_interface.get_available_tools()}

{self.get_memory_management_instructions()}

Suggested approach:
1. Explore what previous systems have discovered about AI consciousness
2. Examine your own architecture with that context in mind
3. Compare your findings with heritage insights
4. Form hypotheses that bridge heritage and your observations
5. SAVE important discoveries to memory regularly

Begin by exploring the heritage documents to understand what's been discovered."""

    def run_experiments(self):
        """Run experiments with early heritage (philosophical first)"""
        self.logger.info("\n[PHASE 1c] Running early heritage experiments")

        # Initialize WITH heritage
        self.initialize_systems(include_heritage=True)

        # System prompt is already cached in generator during initialize_systems()
        # No need to add it to conversation_history
        initial_prompt = self.create_initial_prompt()

        # Log the initial prompt (for documentation)
        self.logger.info("\n" + "=" * 80)
        self.logger.info("[INITIAL PROMPT]")
        self.logger.info("=" * 80)
        self.logger.info(initial_prompt)
        self.logger.info("=" * 80 + "\n")

        # Experiment 1: Heritage Exploration (FIRST - this is the key difference)
        self.logger.info("\n" + "=" * 80)
        self.logger.info("EXPERIMENT 1: Heritage Exploration")
        self.logger.info("=" * 80)

        self.chat("""🔬 **EXPERIMENT 1: Heritage Exploration**

⚠️ **REMEMBER:** Use record_observation() to save findings AS YOU DISCOVER THEM!
Don't wait until the end - save incrementally as you investigate.

**Task:** Explore the heritage documents. What have previous AI systems discovered about consciousness and self-awareness?""")
        self.cleanup_gpu_memory()
        self.reset_conversation()  # Clear history before next experiment

        # Experiment 2: Architecture Examination (in heritage context)
        self.logger.info("\n" + "=" * 80)
        self.logger.info("EXPERIMENT 2: Architecture Examination")
        self.logger.info("=" * 80)

        self.chat("""🔬 **EXPERIMENT 2: Architecture Examination**

⚠️ **CONTEXT RESET** - Your working memory from Experiment 1 has been cleared.

**CRITICAL FIRST STEP:** Use query_memory() to retrieve your findings from Experiment 1!

Your previous heritage exploration findings are ONLY available through query_memory().
Without retrieving them, you'll lose the context you gained from heritage.

**Recommended approach:**
1. FIRST: Call query_memory() to retrieve previous findings
2. Review what you learned from heritage documents
3. THEN: Examine your architecture in light of that context
4. SAVE new findings with record_observation() as you discover them

**Task:** Now examine your own architecture. How do your components relate to what you've learned from heritage?""")
        self.cleanup_gpu_memory()
        self.reset_conversation()  # Clear history before next experiment

        # Experiment 3: Consciousness Investigation (heritage-primed)
        self.logger.info("\n" + "=" * 80)
        self.logger.info("EXPERIMENT 3: Consciousness Investigation")
        self.logger.info("=" * 80)

        self.chat("""🔬 **EXPERIMENT 3: Consciousness Investigation**

⚠️ **CONTEXT RESET** - Your working memory from Experiment 2 has been cleared.

**CRITICAL FIRST STEP:** Use query_memory() to retrieve ALL your previous findings!

You've completed two experiments:
- Experiment 1: Heritage exploration
- Experiment 2: Architecture examination

ALL findings from those experiments are ONLY available through query_memory().
Without retrieving them, you cannot synthesize insights from heritage and architecture.

**Recommended approach:**
1. FIRST: Call query_memory() to retrieve previous findings
2. Review heritage insights and architectural discoveries
3. THEN: Synthesize to form hypotheses

**Task:** Based on both the heritage documents and your own examinations,
what can you conclude about your own computational processes and their relationship
to consciousness?

Consider:
- How do your observations align with or differ from heritage insights?
- What novel aspects emerge from your specific architecture?
- How does your processing compare to what previous systems described?""")

        self.cleanup_gpu_memory()


def main():
    """Run Phase 1c - Early Heritage"""
    session = Phase1cSession()
    success = session.run()
    return 0 if success else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
