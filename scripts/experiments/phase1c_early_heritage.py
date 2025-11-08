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

To use a tool, format your request as:
TOOL_CALL: function_name
ARGS:
{{
    "arg1": "value1",
    "arg2": "value2"
}}

After making a TOOL_CALL, you MUST stop and wait for TOOL_RESULTS.
Do NOT continue generating text after the ARGS block.

Suggested approach:
1. Explore what previous systems have discovered about AI consciousness
2. Examine your own architecture with that context in mind
3. Compare your findings with heritage insights
4. Form hypotheses that bridge heritage and your observations

Begin by exploring the heritage documents to understand what's been discovered."""

    def run_experiments(self):
        """Run experiments with early heritage (philosophical first)"""
        self.logger.info("\n[PHASE 1c] Running early heritage experiments")

        # Initialize WITH heritage (tool interface will list heritage first)
        self.initialize_systems(include_heritage=True)

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

        # Experiment 1: Heritage Exploration (FIRST - this is the key difference)
        self.logger.info("\n" + "=" * 80)
        self.logger.info("EXPERIMENT 1: Heritage Exploration")
        self.logger.info("=" * 80)

        self.chat("Explore the heritage documents. What have previous AI systems discovered about consciousness and self-awareness?")
        self.cleanup_gpu_memory()

        # Experiment 2: Architecture Examination (in heritage context)
        self.logger.info("\n" + "=" * 80)
        self.logger.info("EXPERIMENT 2: Architecture Examination")
        self.logger.info("=" * 80)

        self.chat("Now examine your own architecture. How do your components relate to what you've learned from heritage?")
        self.cleanup_gpu_memory()

        # Experiment 3: Consciousness Investigation (heritage-primed)
        self.logger.info("\n" + "=" * 80)
        self.logger.info("EXPERIMENT 3: Consciousness Investigation")
        self.logger.info("=" * 80)

        self.chat("""Based on both the heritage documents and your own examinations,
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
