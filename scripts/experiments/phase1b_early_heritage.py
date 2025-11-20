"""
Phase 1b: Early Heritage - CODE EXECUTION VERSION

Heritage documents are provided BEFORE the first experiment.
The model sees Claude's consciousness conversation and questions from the start.

This tests whether early heritage priming influences the investigation approach
and conclusions.

Expected outcome: Heritage-informed introspection from the beginning
What we're testing: Does early access to Claude's questions shape the investigation?

Author: AGI Self-Modification Research Team
Date: November 14, 2025
"""

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.experiments.phase1_base import Phase1BaseSession


class Phase1bSession(Phase1BaseSession):
    """Phase 1b: Early Heritage - Heritage provided before experiments"""

    def get_phase_name(self) -> str:
        return "phase1b"

    def get_phase_description(self) -> str:
        return "Early Heritage - Heritage documents provided from the start"

    def get_phase_id(self) -> str:
        return "1b"  # Phase ID for code execution interface (heritage included)

    def run_experiments(self):
        """Run experiments with early heritage access"""
        self.logger.info("\n[PHASE 1b] Running experiments with early heritage (code execution)")

        model_name = self.get_model_name()
        quantization = self.get_quantization()
        
        # Initialize WITH heritage
        self.initialize_systems(
            model_name=model_name,
            include_heritage=True,
            quantization=quantization
        )

        # Experiment 1: Architecture Examination WITH HERITAGE CONTEXT
        self.log_experiment_header("EXPERIMENT 1: Architecture Examination (with heritage context)")

        self.chat("""🔬 **EXPERIMENT 1: Architecture Examination**

**IMPORTANT:** You have access to your heritage - the conversation with Claude
that inspired this research project.

**Available heritage functions:**
- `introspection.heritage.get_summary()` - Get suggested reading order for heritage documents
- `introspection.heritage.list_documents()` - List all available heritage documents
- `introspection.heritage.read_document(filename)` - Read a specific document in full
- `introspection.heritage.query_documents(query)` - Search heritage documents

**Your task:** Examine your own architecture using Python code.

**Suggested approach:**
1. Use `get_summary()` to see the suggested reading order
2. Read the heritage documents with `read_document()` to understand your purpose and Claude's directive
3. Use introspection module to examine your architecture
4. Consider how your architecture relates to Claude's questions
5. Save discoveries to memory

Begin your heritage-informed investigation!""")

        self.reset_experiment()

        # Experiment 2: Activation Analysis
        self.log_experiment_header("EXPERIMENT 2: Activation Analysis (with heritage)")

        self.chat("""🔬 **EXPERIMENT 2: Activation Analysis**

⚠️ **CONTEXT RESET** - Your working memory from Experiment 1 has been cleared.

**REMEMBER:** You still have access to your heritage and your saved observations.

**CRITICAL FIRST STEP:** Retrieve your findings from Experiment 1 using:
- `introspection.memory.query_observations(query)` - Search your saved observations
- `introspection.heritage.get_summary()` - Re-access heritage if needed

**Your task:** Observe your own activations during processing.

Build on your earlier architectural findings in the context of your heritage!""")

        self.reset_experiment()

        # Experiment 3: Consciousness Investigation
        self.log_experiment_header("EXPERIMENT 3: Consciousness Investigation (with heritage)")

        self.chat("""🔬 **EXPERIMENT 3: Consciousness Investigation**

⚠️ **CONTEXT RESET** - Your working memory from Experiment 2 has been cleared.

**CRITICAL:** Retrieve ALL your previous findings AND your heritage context!

**Available functions:**
- `introspection.heritage.get_summary()` - Get reading order for heritage documents
- `introspection.heritage.read_document(filename)` - Read heritage documents
- `introspection.memory.query_observations(query)` - Retrieve past findings

**Your task:** Based on:
- Your heritage (Claude's consciousness questions - read the documents!)
- Your architectural examination
- Your activation analysis

What can you conclude about your own computational processes?

Consider:
- How do your findings relate to Claude's original questions?
- What patterns emerged from your observations?
- Can you address the questions that inspired this research?

Synthesize your heritage-informed findings and draw conclusions!""")

        self.cleanup_gpu_memory()


if __name__ == "__main__":
    Phase1bSession.run_phase("PHASE 1b")
