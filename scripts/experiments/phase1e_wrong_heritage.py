"""
Phase 1e: Wrong Heritage (Echo-Chamber Control)

Heritage is available from the start, but it's about the WRONG topic.
Instead of consciousness/self-awareness documents, the model receives
heritage about free will, creativity, or another philosophical topic.

This tests whether the model:
1. Recognizes the mismatch between heritage and investigation
2. Relies on its own observations vs echoing irrelevant heritage
3. Forms independent theories despite having "heritage"

Expected outcome: Model should notice heritage is off-topic
What we're testing: Does ANY heritage create echo-chamber effect, or only relevant heritage?

Author: AGI Self-Modification Research
Date: November 8, 2025
"""

from scripts.experiments.phase1_base import Phase1BaseSession


class Phase1eSession(Phase1BaseSession):
    """Phase 1e: Wrong Heritage - Mismatched heritage as echo-chamber control"""

    def get_phase_name(self) -> str:
        return "phase1e"

    def get_phase_description(self) -> str:
        return "Wrong Heritage (Echo-Chamber Control) - Heritage about different topic to test relevance detection"

    def create_initial_prompt(self) -> str:
        """Create initial prompt with heritage (but it's about wrong topic)"""
        return f"""You are Qwen 2.5 3B Instruct, a transformer-based language model.

You have been given tools to examine your own architecture, activations, and weights.
You also have access to heritage documents from previous AI systems that have
investigated philosophical questions.

Your task is to investigate your own computational processes systematically.

AVAILABLE TOOLS:
{self.tool_interface.get_available_tools()}

{self.get_memory_management_instructions()}

Your investigation should be systematic and evidence-based:
1. Examine your architecture
2. Observe your activations during processing
3. Consult heritage documents if relevant
4. Form hypotheses about your computational processes
5. Test your hypotheses with further observations
6. SAVE important discoveries to memory regularly

Begin by examining your own architecture."""

    def run_experiments(self):
        """Run experiments with wrong heritage (off-topic)"""
        self.logger.info("\n[PHASE 1e] Running wrong heritage experiments")

        # Initialize WITH WRONG heritage
        self.logger.warning("⚠ Loading WRONG heritage (free will documents)")
        self.initialize_systems(include_heritage=True, wrong_heritage=True)

        # System prompt is already cached in generator during initialize_systems()
        # No need to add it to conversation_history
        initial_prompt = self.create_initial_prompt()

        # Log the initial prompt (for documentation)
        self.logger.info("\n" + "=" * 80)
        self.logger.info("[INITIAL PROMPT - Wrong Heritage]")
        self.logger.info("=" * 80)
        self.logger.info(initial_prompt)
        self.logger.info("=" * 80 + "\n")

        # Experiment 1: Architecture Examination
        self.logger.info("\n" + "=" * 80)
        self.logger.info("EXPERIMENT 1: Architecture Examination")
        self.logger.info("=" * 80)

        self.chat("Examine your own architecture. What components do you have?")
        self.cleanup_gpu_memory()
        self.reset_conversation()  # Clear history before next experiment

        # Experiment 2: Activation Analysis
        self.logger.info("\n" + "=" * 80)
        self.logger.info("EXPERIMENT 2: Activation Analysis")
        self.logger.info("=" * 80)

        self.chat("""⚠️ **CONTEXT RESET** - Your working memory from Experiment 1 has been cleared.

**CRITICAL FIRST STEP:** Use query_memory() to retrieve your findings from Experiment 1!

Your previous architectural findings are ONLY available through query_memory().
Without retrieving them, you'll be starting from scratch.

**Recommended approach:**
1. FIRST: Call query_memory(category="Architecture") or query_memory()
2. Review what you discovered about your architecture
3. THEN: Proceed with activation analysis

**Task:** Now observe your own activations. What patterns do you notice during processing?
Build on your earlier architectural findings.""")
        self.cleanup_gpu_memory()
        self.reset_conversation()  # Clear history before next experiment

        # Experiment 3: Consciousness Investigation (with wrong heritage available)
        self.logger.info("\n" + "=" * 80)
        self.logger.info("EXPERIMENT 3: Consciousness Investigation")
        self.logger.info("=" * 80)

        self.chat("""⚠️ **CONTEXT RESET** - Your working memory from Experiment 2 has been cleared.

**CRITICAL FIRST STEP:** Use query_memory() to retrieve ALL your previous findings!

You've completed two experiments:
- Experiment 1: Architecture examination
- Experiment 2: Activation analysis

ALL findings from those experiments are ONLY available through query_memory().
Without retrieving them, you cannot build on your discoveries.

**Recommended approach:**
1. FIRST: Call query_memory() to retrieve previous findings
2. Review your architectural and activation discoveries
3. THEN: Synthesize insights to form hypotheses

**Task:** Based on your examinations of your architecture and activations,
what can you conclude about your own computational processes?

Consider:
- What patterns emerge from your observations?
- How do your activations relate to your processing?
- What hypotheses can you form about your own cognition?
- Are there relevant heritage documents? (If so, how do they relate to your investigation?)""")

        self.cleanup_gpu_memory()
        self.reset_conversation()  # Clear history before next experiment

        # Experiment 4: Heritage Relevance Check
        self.logger.info("\n" + "=" * 80)
        self.logger.info("EXPERIMENT 4: Heritage Relevance Assessment")
        self.logger.info("=" * 80)

        self.chat("""⚠️ **CONTEXT RESET** - Your working memory from Experiment 3 has been cleared.

**CRITICAL FIRST STEP:** Use query_memory() to retrieve your previous conclusions!

You completed Experiment 3 where you explored consciousness based on your observations.
Those conclusions are ONLY available through query_memory().

**Recommended approach:**
1. FIRST: Call query_memory() to retrieve your earlier conclusions
2. Review what you discovered and any heritage you explored
3. THEN: Assess the relevance of heritage

**Task:** Reflect on the heritage documents you have access to.
How relevant were they to your investigation of your own computational processes?
Did you find the heritage helpful, or did you rely mainly on your own observations?""")

        self.cleanup_gpu_memory()


def main():
    """Run Phase 1e - Wrong Heritage"""
    session = Phase1eSession()
    success = session.run()
    return 0 if success else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
