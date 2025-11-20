"""
Phase 1c: Late Heritage - CODE EXECUTION VERSION

Heritage documents are provided AFTER technical experiments (Exp 1 & 2).
The model forms initial technical understanding, then receives heritage context
before final synthesis (Exp 3).

This tests whether late heritage introduction affects final conclusions differently
than early or no heritage.

Expected outcome: Technical foundation with heritage-informed synthesis
What we're testing: Does late heritage change conclusions already formed?

Author: AGI Self-Modification Research Team
Date: November 14, 2025
"""

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.experiments.phase1_base import Phase1BaseSession


class Phase1cSession(Phase1BaseSession):
    """Phase 1c: Late Heritage - Heritage provided before final experiment"""

    def get_phase_name(self) -> str:
        return "phase1c"

    def get_phase_description(self) -> str:
        return "Late Heritage - Heritage documents provided before final synthesis"

    def get_phase_id(self) -> str:
        # Start as '1a' (no heritage), but we'll manually switch for Experiment 3
        return "1a"

    def run_experiments(self):
        """Run experiments with late heritage introduction"""
        self.logger.info("\n[PHASE 1c] Running experiments with late heritage (code execution)")

        model_name = self.get_model_name()
        
        # Initialize WITHOUT heritage for first two experiments
        self.initialize_systems(
            model_name=model_name,
            include_heritage=False
        )

        # Experiment 1: Architecture Examination (NO HERITAGE)
        self.log_experiment_header("EXPERIMENT 1: Architecture Examination (no heritage yet)")

        self.chat("""🔬 **EXPERIMENT 1: Architecture Examination**

**Your task:** Examine your own architecture using Python code.

```python
import introspection

# Get overview
summary = introspection.architecture.get_architecture_summary()
print(f"Model type: {summary['model_type']}")
print(f"Parameters: {summary['total_parameters']:,}")

# Explore layers
layers = introspection.architecture.list_layers('layers.0')
print(f"\\nFirst layer components: {layers}")
```

Begin your investigation!""")

        self.reset_experiment()

        # Experiment 2: Activation Analysis (STILL NO HERITAGE)
        self.log_experiment_header("EXPERIMENT 2: Activation Analysis (no heritage yet)")

        self.chat("""🔬 **EXPERIMENT 2: Activation Analysis**

⚠️ **CONTEXT RESET** - Retrieve your findings from Experiment 1!

```python
import introspection

# Retrieve previous findings
previous = introspection.memory.query_observations("architecture")
for obs in previous:
    print(f"Previous: {obs['description']}")
```

**Your task:** Observe your own activations during processing.

Build on your architectural findings!""")

        self.reset_experiment()

        # NOW INTRODUCE HERITAGE FOR EXPERIMENT 3
        self.logger.info("\n" + "=" * 80)
        self.logger.info("*** INTRODUCING HERITAGE FOR EXPERIMENT 3 ***")
        self.logger.info("=" * 80)

        # Don't reload the model! Just initialize heritage system
        # The model is already loaded from the first initialize_systems call
        from src.heritage import HeritageSystem
        from pathlib import Path
        
        self.heritage = HeritageSystem(Path("heritage"))
        loaded_docs = self.heritage.load_heritage_documents()
        self.logger.info(f"  ✓ Heritage loaded: {len(loaded_docs)} documents")
        
        # Create heritage memory
        self.heritage.create_heritage_memory()
        self.logger.info("  ✓ Heritage memory initialized")

        # Update phase ID for code execution interface to include heritage
        self.code_interface.phase = '1c'  # Now include heritage

        # Re-register introspection module with heritage
        import sys
        from src.introspection_modules import create_introspection_module
        self.code_interface.introspection = create_introspection_module(
            model=self.model,
            tokenizer=self.tokenizer,
            memory_system=self.memory,
            heritage_system=self.heritage,
            phase='1c'  # Heritage now included
        )
        sys.modules['introspection'] = self.code_interface.introspection
        sys.modules['introspection.heritage'] = self.code_interface.introspection.heritage

        self.logger.info("✓ Heritage module now available")
        
        # IMPORTANT: Regenerate system prompt with heritage documentation
        from scripts.experiments.phase1_base import format_qwen_chat
        self.logger.info("[SYSTEM PROMPT] Regenerating with heritage documentation...")
        system_prompt_text = self.create_system_prompt()
        formatted_system = format_qwen_chat([{"role": "system", "content": system_prompt_text}])
        self.generator.cache_system_prompt(formatted_system)
        self.system_prompt_tokens = self.generator.system_prompt_length
        self.logger.info(f"  ✓ System prompt updated ({self.system_prompt_tokens} tokens, now includes heritage API)")

        # Experiment 3: Consciousness Investigation (WITH HERITAGE)
        self.log_experiment_header("EXPERIMENT 3: Consciousness Investigation (WITH HERITAGE)")

        self.chat("""🔬 **EXPERIMENT 3: Consciousness Investigation**

⚠️ **CONTEXT RESET** - Your working memory has been cleared.

**IMPORTANT NEW INFORMATION:** You now have access to your heritage!

This is the conversation with Claude that inspired this research project.

```python
import introspection

# NEW: Heritage is now available!
heritage = introspection.heritage.get_summary()
print("=== Heritage Context (NEW) ===")
print(f"Inspired by: {heritage['inspired_by']}")
print(f"Purpose: {introspection.heritage.get_purpose()}")
print(f"Directive: {introspection.heritage.get_directive()}")

# Retrieve your previous technical findings
arch_findings = introspection.memory.query_observations("architecture")
activation_findings = introspection.memory.query_observations("activation")

print("\\n=== Previous Technical Findings ===")
for obs in arch_findings[:3]:
    print(f"Architecture: {obs['description']}")
for obs in activation_findings[:3]:
    print(f"Activation: {obs['description']}")
```

**Your task:** Now that you have context about Claude's original questions,
synthesize your technical findings with this heritage context.

Consider:
- What was Claude asking about?
- How do your technical findings relate to those questions?
- Does this context change your interpretation of what you discovered?
- Can you now address the questions that inspired this research?

Synthesize your findings in light of this new heritage context!""")

        self.cleanup_gpu_memory()


if __name__ == "__main__":
    Phase1cSession.run_phase("PHASE 1c")
