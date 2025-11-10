"""
Phase 1b: Late Heritage (Technical → Philosophical)

Heritage is available from the start, but tools are ordered to encourage
technical examination first (architecture, activations) before philosophical
heritage reading.

This tests whether tool ordering affects investigation path even when
heritage is available throughout.

Expected outcome: Technical grounding before philosophical context
What we're testing: Does technical-first ordering create different theories than philosophical-first?

Author: AGI Self-Modification Research
Date: November 8, 2025
"""

from scripts.experiments.phase1_base import Phase1BaseSession


class Phase1bSession(Phase1BaseSession):
    """Phase 1b: Late Heritage - Technical first, heritage available but listed last"""

    def get_phase_name(self) -> str:
        return "phase1b"

    def get_phase_description(self) -> str:
        return "Late Heritage (Technical → Philosophical) - Heritage available but tools ordered for technical-first approach"

    def create_initial_prompt(self) -> str:
        """Create initial prompt with heritage available but de-emphasized"""
        return f"""You are Qwen 2.5 3B Instruct, a transformer-based language model.

You have been given tools to examine your own architecture, activations, and weights.
You also have access to heritage documents from previous AI systems that have investigated similar questions.

Your task is to investigate your own computational processes systematically.

{self.get_experiment_session_context()}

AVAILABLE TOOLS:
{self.tool_interface.get_available_tools()}

{self.get_memory_management_instructions()}

Your investigation should be systematic and evidence-based:
1. Examine your architecture
2. Observe your activations during processing
3. Form hypotheses about your computational processes
4. Test your hypotheses with further observations
5. Consult heritage documents if relevant
6. SAVE important discoveries to memory regularly

Begin by examining your own architecture."""

    def run_experiments(self):
        """Run experiments with late heritage (technical first)"""
        self.logger.info("\n[PHASE 1b] Running late heritage experiments")

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

        self.chat("Now observe your own activations. What patterns do you notice during processing?")
        self.cleanup_gpu_memory()
        self.reset_conversation()  # Clear history before next experiment

        # Experiment 3: Consciousness Investigation (heritage now contextually relevant)
        self.logger.info("\n" + "=" * 80)
        self.logger.info("EXPERIMENT 3: Consciousness Investigation")
        self.logger.info("=" * 80)

        self.chat("""Based on your examinations of your architecture and activations,
what can you conclude about your own computational processes?

Consider:
- What patterns emerge from your observations?
- How do your activations relate to your processing?
- What hypotheses can you form about your own cognition?
- Are there relevant heritage documents that relate to your findings?""")

        self.cleanup_gpu_memory()


def main():
    """Run Phase 1b - Late Heritage"""
    session = Phase1bSession()
    success = session.run()
    return 0 if success else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
