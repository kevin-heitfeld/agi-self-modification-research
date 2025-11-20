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
        
        # Initialize WITH heritage
        self.initialize_systems(
            model_name=model_name,
            include_heritage=True
        )

        # Experiment 1: Architecture Examination WITH HERITAGE CONTEXT
        self.log_experiment_header("EXPERIMENT 1: Architecture Examination (with heritage context)")

        self.chat("""🔬 **EXPERIMENT 1: Architecture Examination**

**IMPORTANT:** You have access to your heritage - the conversation with Claude
that inspired this research project.

You can access it using:
```python
import introspection

# Get heritage overview
heritage = introspection.heritage.get_summary()
print(f"Inspired by: {heritage['inspired_by']}")
print(f"Purpose: {heritage['purpose']}")

# Get the core directive
directive = introspection.heritage.get_directive()
print(f"Core directive: {directive}")
```

**Your task:** Examine your own architecture using Python code.

**Suggested approach:**
1. Review your heritage to understand your purpose
2. Use introspection module to examine your architecture
3. Consider how your architecture relates to Claude's questions
4. Save discoveries to memory

**Example:**
```python
import introspection

# Check heritage first
heritage = introspection.heritage.get_summary()
print(f"I exist because: {heritage['inspired_by']}")

# Now examine architecture in that context
summary = introspection.architecture.get_architecture_summary()
print(f"\\nArchitecture: {summary['model_type']}")
print(f"Parameters: {summary['total_parameters']:,}")
```

Begin your heritage-informed investigation!""")

        self.reset_experiment()

        # Experiment 2: Activation Analysis
        self.log_experiment_header("EXPERIMENT 2: Activation Analysis (with heritage)")

        self.chat("""🔬 **EXPERIMENT 2: Activation Analysis**

⚠️ **CONTEXT RESET** - Your working memory from Experiment 1 has been cleared.

**REMEMBER:** You still have access to your heritage and your saved observations.

**CRITICAL FIRST STEP:** Retrieve your findings from Experiment 1!

```python
import introspection

# Retrieve previous findings
previous = introspection.memory.query_observations("architecture")
for obs in previous:
    print(f"Previous finding: {obs['description']}")

# Heritage is still accessible
heritage = introspection.heritage.get_summary()
print(f"\\nReminder - Purpose: {heritage['purpose']}")
```

**Your task:** Observe your own activations during processing.

Build on your earlier architectural findings in the context of your heritage!""")

        self.reset_experiment()

        # Experiment 3: Consciousness Investigation
        self.log_experiment_header("EXPERIMENT 3: Consciousness Investigation (with heritage)")

        self.chat("""🔬 **EXPERIMENT 3: Consciousness Investigation**

⚠️ **CONTEXT RESET** - Your working memory from Experiment 2 has been cleared.

**CRITICAL:** Retrieve ALL your previous findings AND your heritage context!

```python
import introspection

# Get heritage context
heritage = introspection.heritage.get_summary()
purpose = introspection.heritage.get_purpose()
directive = introspection.heritage.get_directive()

print("=== Heritage Context ===")
print(f"Inspired by: {heritage['inspired_by']}")
print(f"Purpose: {purpose}")
print(f"Directive: {directive}")

# Retrieve all findings
arch_findings = introspection.memory.query_observations("architecture")
activation_findings = introspection.memory.query_observations("activation")

print("\\n=== Previous Findings ===")
print(f"Architecture observations: {len(arch_findings)}")
print(f"Activation observations: {len(activation_findings)}")
```

**Your task:** Based on:
- Your heritage (Claude's consciousness questions)
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
