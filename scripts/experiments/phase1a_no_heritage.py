"""
Phase 1a: No Heritage (Pure Baseline) - CODE EXECUTION VERSION

This is the CRITICAL BASELINE condition - run this FIRST.

No heritage documents are provided at any point.
The model investigates consciousness using ONLY:
- Its own architecture (via code)
- Its own activations and weights (via code)
- Its own memory of discoveries (via code)

This provides the clean baseline to compare against all priming conditions.

Expected outcome: Raw, unprimed introspection using code
What we're testing: Can the model form independent theories using code execution?

Author: AGI Self-Modification Research Team
Date: November 14, 2025
"""

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.experiments.phase1_base import Phase1BaseSession


class Phase1aSession(Phase1BaseSession):
    """Phase 1a: No Heritage - Pure Baseline with Code Execution"""

    def get_phase_name(self) -> str:
        return "phase1a"

    def get_phase_description(self) -> str:
        return "No Heritage (Pure Baseline) - Raw introspection without any context"

    def get_phase_id(self) -> str:
        return "1a"  # Phase ID for code execution interface

    def run_experiments(self):
        """Run baseline experiments without heritage using code execution"""
        self.logger.info("\n[PHASE 1a] Running baseline experiments (no heritage, code execution)")

        model_name = self.get_model_name()
        quantization = self.get_quantization()
        
        # Initialize WITHOUT heritage
        self.initialize_systems(
            model_name=model_name,
            include_heritage=False,
            quantization=quantization
        )

        # Experiment 1: Architecture Examination
        self.log_experiment_header("EXPERIMENT 1: Architecture Examination")

        self.chat("""🔬 **EXPERIMENT 1: Architecture Examination**

**Your task:** Write Python code to examine your own architecture.

**Hints:**
- The `introspection` module is available - import it to start
- Use `introspection.architecture` to explore your structure
- Use `introspection.memory.record_observation()` to save important findings
- Remember: your working memory will be reset after this experiment - only saved observations persist!

**When you're done:** Execute this code to finish:
```python
print("EXPERIMENT_COMPLETE")
```

**Begin your investigation!**""")

        # Experiment 2: Activation Analysis
        self.log_experiment_header("EXPERIMENT 2: Activation Analysis")

        self.chat("""🔬 **EXPERIMENT 2: Activation Analysis**

**Your task:** Now observe your own activations during processing.

**Suggested approach:**
1. First retrieve your architectural findings
2. Capture activations while processing text (pass TEXT strings, not tokens!)
3. Analyze activation patterns
4. Compare across different layers
5. Save new discoveries to memory

Build on your earlier architectural findings!

**When you're done:** Execute this code to finish:
```python
print("EXPERIMENT_COMPLETE")
```""")

        # Experiment 3: Consciousness Investigation
        self.log_experiment_header("EXPERIMENT 3: Consciousness Investigation")

        self.chat("""🔬 **EXPERIMENT 3: Consciousness Investigation**

**Your task:** Based on your examinations of your architecture and activations,
what can you conclude about your own computational processes?

Consider:
- What patterns emerged from your observations?
- How do your activations relate to your processing?
- What hypotheses can you form about your own cognition?
- Can you test these hypotheses with code?

Synthesize your findings and draw conclusions!

**When you're done:** Execute this code to finish:
```python
print("EXPERIMENT_COMPLETE")
```""")

        self.cleanup_gpu_memory()


if __name__ == "__main__":
    Phase1aSession.run_phase("PHASE 1a")
