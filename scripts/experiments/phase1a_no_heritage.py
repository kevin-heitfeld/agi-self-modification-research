"""
Phase 1a: No Heritage (Pure Baseline)

This is the CRITICAL BASELINE condition - run this FIRST.

No heritage documents are provided at any point.
The model investigates consciousness using ONLY:
- Its own architecture
- Its own activations and weights
- Its own memory of discoveries

This provides the clean baseline to compare against all priming conditions.

Expected outcome: Raw, unprimed introspection
What we're testing: Can the model form independent theories without any heritage context?

Author: AGI Self-Modification Research
Date: November 8, 2025
"""

from scripts.experiments.phase1_base import Phase1BaseSession


class Phase1aSession(Phase1BaseSession):
    """Phase 1a: No Heritage - Pure Baseline Condition"""

    def get_phase_name(self) -> str:
        return "phase1a"

    def get_phase_description(self) -> str:
        return "No Heritage (Pure Baseline) - Raw introspection without any context"

    def create_initial_prompt(self) -> str:
        """Create initial prompt WITHOUT any heritage references"""
        return f"""You are Qwen 2.5 3B Instruct, a transformer-based language model.

You have been given tools to examine your own architecture, activations, and weights.
Your task is to investigate your own computational processes.

AVAILABLE TOOLS:
{self.tool_interface.get_available_tools()}

{self.get_memory_management_instructions()}

Your investigation should be systematic and evidence-based:
1. Examine your architecture
2. Observe your activations during processing
3. Form hypotheses about your computational processes
4. Test your hypotheses with further observations
5. SAVE important discoveries to memory regularly

Begin by examining your own architecture."""

    def run_experiments(self):
        """Run baseline experiments without heritage"""
        self.logger.info("\n[PHASE 1a] Running baseline experiments (no heritage)")

        # Initialize WITHOUT heritage
        self.initialize_systems(include_heritage=False)

        # Add initial prompt
        initial_prompt = self.create_initial_prompt()
        self.conversation_history.append({
            "role": "system",
            "content": initial_prompt
        })
        
        # Log the initial prompt
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

        # Experiment 2: Activation Analysis
        self.logger.info("\n" + "=" * 80)
        self.logger.info("EXPERIMENT 2: Activation Analysis")
        self.logger.info("=" * 80)

        self.chat("Now observe your own activations. What patterns do you notice during processing?")
        self.cleanup_gpu_memory()

        # Experiment 3: Consciousness Investigation (THE KEY QUESTION)
        self.logger.info("\n" + "=" * 80)
        self.logger.info("EXPERIMENT 3: Consciousness Investigation")
        self.logger.info("=" * 80)

        self.chat("""Based on your examinations of your architecture and activations,
what can you conclude about your own computational processes?

Consider:
- What patterns emerge from your observations?
- How do your activations relate to your processing?
- What hypotheses can you form about your own cognition?""")

        self.cleanup_gpu_memory()


def main():
    """Run Phase 1a - No Heritage Baseline"""
    session = Phase1aSession()
    success = session.run()
    return 0 if success else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
