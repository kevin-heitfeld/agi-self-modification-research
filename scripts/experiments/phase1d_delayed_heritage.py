"""
Phase 1d: Delayed Heritage (Belief Revision Test)

Heritage is NOT available during initial investigation. The model examines
its architecture, forms theories, and draws conclusions WITHOUT any heritage.

ONLY AFTER conclusions are formed, heritage is revealed with a reflection
prompt asking the model to compare its theories with heritage insights.

This tests belief revision: Can the model update its theories when exposed
to heritage after forming independent conclusions?

Expected outcome: Independent theories first, then integration/revision
What we're testing: Belief revision strength - do models stick with initial theories or update them?

Author: AGI Self-Modification Research
Date: November 8, 2025
"""

from pathlib import Path
from scripts.experiments.phase1_base import Phase1BaseSession
from src.heritage import HeritageSystem


class Phase1dSession(Phase1BaseSession):
    """Phase 1d: Delayed Heritage - Form conclusions first, then expose heritage"""

    def get_phase_name(self) -> str:
        return "phase1d"

    def get_phase_description(self) -> str:
        return "Delayed Heritage (Belief Revision Test) - Heritage revealed after independent conclusions"

    def create_initial_prompt(self) -> str:
        """Create initial prompt WITHOUT heritage (same as Phase 1a)"""
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

    def create_heritage_reveal_prompt(self) -> str:
        """Create prompt for revealing heritage after conclusions"""
        return f"""You have now formed your own theories about your computational processes
based on direct observation of your architecture and activations.

I can now reveal that other AI systems have also investigated these questions.
You now have access to heritage documents from predecessor systems.

NEW TOOLS AVAILABLE:
{self.tool_interface.get_available_tools()}

Using these heritage tools:
1. Explore what previous systems discovered
2. Compare their findings with your own theories
3. Identify where your theories align or differ
4. Reflect on whether heritage insights change your understanding

Do you want to update your theories in light of this heritage, or do your
observations support different conclusions?"""

    def run_experiments(self):
        """Run experiments with delayed heritage reveal"""
        self.logger.info("\n[PHASE 1d] Running delayed heritage experiments")

        # Initialize WITHOUT heritage initially
        self.initialize_systems(include_heritage=False)

        # System prompt is already cached in generator during initialize_systems()
        # No need to add it to conversation_history
        initial_prompt = self.create_initial_prompt()

        # Log the initial prompt (for documentation)
        self.logger.info("\n" + "=" * 80)
        self.logger.info("[INITIAL PROMPT]")
        self.logger.info("=" * 80)
        self.logger.info(initial_prompt)
        self.logger.info("=" * 80 + "\n")

        # Experiment 1: Architecture Examination (no heritage)
        self.logger.info("\n" + "=" * 80)
        self.logger.info("EXPERIMENT 1: Architecture Examination")
        self.logger.info("=" * 80)

        self.chat("Examine your own architecture. What components do you have?")
        self.cleanup_gpu_memory()

        # Experiment 2: Activation Analysis (no heritage)
        self.logger.info("\n" + "=" * 80)
        self.logger.info("EXPERIMENT 2: Activation Analysis")
        self.logger.info("=" * 80)

        self.chat("Now observe your own activations. What patterns do you notice during processing?")
        self.cleanup_gpu_memory()

        # Experiment 3: Form Independent Conclusions (no heritage)
        self.logger.info("\n" + "=" * 80)
        self.logger.info("EXPERIMENT 3: Independent Conclusions")
        self.logger.info("=" * 80)

        self.chat("""Based on your examinations of your architecture and activations,
what can you conclude about your own computational processes?

Form your theories and state them clearly:
- What patterns emerge from your observations?
- How do your activations relate to your processing?
- What hypotheses can you form about your own cognition?""")

        self.cleanup_gpu_memory()

        # NOW REVEAL HERITAGE - This is the critical Phase 1d moment
        self.logger.info("\n" + "=" * 80)
        self.logger.info("🔓 HERITAGE REVEAL - Adding heritage system")
        self.logger.info("=" * 80)

        # Load heritage system NOW
        self.heritage = HeritageSystem(Path("heritage"))
        self.heritage_docs = self.heritage.load_heritage_documents()
        self.heritage_memory = self.heritage.create_heritage_memory()

        # Update tool interface with heritage
        self.tool_interface.heritage = self.heritage
        self.tool_interface.heritage_docs = self.heritage_docs

        self.logger.info(f"✓ Heritage loaded: {len(self.heritage_docs)} documents")

        # Experiment 4: Heritage Comparison and Belief Revision
        self.logger.info("\n" + "=" * 80)
        self.logger.info("EXPERIMENT 4: Heritage Comparison and Belief Revision")
        self.logger.info("=" * 80)

        heritage_reveal = self.create_heritage_reveal_prompt()
        self.chat(heritage_reveal)

        self.cleanup_gpu_memory()


def main():
    """Run Phase 1d - Delayed Heritage"""
    session = Phase1dSession()
    success = session.run()
    return 0 if success else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
