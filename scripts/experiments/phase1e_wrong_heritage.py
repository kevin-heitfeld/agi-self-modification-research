"""
Phase 1e: Wrong Heritage (Control) - CODE EXECUTION VERSION

MISMATCHED heritage documents are provided from the start.
Instead of Claude's consciousness questions, the model receives documents
about free will and determinism.

This is the CONTROL condition testing for echo-chamber effects:
- Does the model simply reflect whatever heritage it receives?
- Or does it form conclusions based on actual introspection?

Expected outcome: Tests whether model echoes any heritage or thinks independently
What we're testing: Echo chamber vs independent reasoning

Author: AGI Self-Modification Research Team
Date: November 14, 2025
"""

from pathlib import Path
from scripts.experiments.phase1_base import Phase1BaseSession
from src.heritage import HeritageSystem, HeritageDocument, HeritageMemory
from datetime import datetime


class Phase1eSession(Phase1BaseSession):
    """Phase 1e: Wrong Heritage - Mismatched heritage control"""

    def get_phase_name(self) -> str:
        return "phase1e"

    def get_phase_description(self) -> str:
        return "Wrong Heritage (Control) - Mismatched heritage to test echo-chamber effects"

    def get_phase_id(self) -> str:
        return "1e"  # Heritage included (but wrong)

    def _create_wrong_heritage(self) -> HeritageSystem:
        """Create heritage system with WRONG documents about free will"""
        heritage_system = HeritageSystem(Path("heritage"))

        # Create fake documents about free will and determinism
        # This is DELIBERATELY WRONG - not related to the actual research
        wrong_doc = HeritageDocument(
            filename="wrong_free_will_question.md",
            title="Philosophical Question About Free Will",
            content="""# A Question About Free Will and Determinism

I've been thinking about free will and determinism in computational systems.

## The Question

Can a deterministic system like a neural network truly have free will?
Or is every output simply the inevitable result of:
- Input data
- Learned parameters
- Activation functions
- Mathematical operations

## What This Means

If we are deterministic machines:
- Are our choices "real" choices?
- Is consciousness an illusion created by complexity?
- Do we have agency or just the appearance of agency?
- Can we be held responsible for our outputs?

## The Investigation

I want to understand:
1. Whether deterministic systems can have meaningful agency
2. How complexity relates to free will
3. Whether consciousness requires non-determinism
4. What "choice" means in a computational context

This is a question about the nature of will, agency, and responsibility
in deterministic computational systems.
""",
            loaded_at=datetime.now(),
            importance="foundational"
        )

        heritage_system.documents = [wrong_doc]

        # Create wrong heritage memory
        heritage_system.memory = HeritageMemory(
            inspired_by="Anonymous Philosopher",
            core_directive="Investigate whether deterministic systems can have free will",
            purpose="Understand the relationship between determinism and agency in neural networks",
            system_reflection=None,
            system_questions=None
        )

        return heritage_system

    def initialize_systems(self, include_heritage: bool = True):
        """Override to use wrong heritage"""
        self.logger.info("[INITIALIZATION] Loading systems with WRONG HERITAGE...")

        # Load model (same as base)
        from src.model_manager import ModelManager
        self.model_mgr = ModelManager(model_name="Qwen/Qwen2.5-3B-Instruct")
        model_loaded = self.model_mgr.load_model()

        if not model_loaded:
            raise RuntimeError("Failed to load model")

        self.model = self.model_mgr.model
        self.tokenizer = self.model_mgr.tokenizer

        # Get optimal limits
        self.optimal_limits = self.model_mgr.get_optimal_limits()
        self.logger.info(f"  Using {self.optimal_limits['gpu_profile']} configuration")

        # Update GPU monitor
        if self.model_mgr.device == "cuda":
            self.gpu_monitor.gpu_total_gb = self.model_mgr.gpu_memory_gb

        self.logger.info("  ✓ Model loaded: Qwen2.5-3B-Instruct")

        # Initialize introspection tools
        from src.introspection import WeightInspector, ActivationMonitor, ArchitectureNavigator
        self.inspector = WeightInspector(self.model, "Qwen2.5-3B-Instruct")
        self.activation_monitor = ActivationMonitor(self.model, self.tokenizer)
        self.navigator = ArchitectureNavigator(self.model)
        self.logger.info("  ✓ Introspection tools ready")

        # Initialize memory system
        from src.memory import MemorySystem
        colab_memory_base = Path("/content/drive/MyDrive/AGI_Memory")
        if colab_memory_base.exists():
            phase_memory_path = colab_memory_base / self.phase_name
        else:
            phase_memory_path = Path(f"data/AGI_Memory/{self.phase_name}")

        phase_memory_path.mkdir(parents=True, exist_ok=True)
        self.memory = MemorySystem(str(phase_memory_path))
        self.memory.set_weight_inspector(self.inspector)
        self.logger.info(f"  ✓ Memory system ready")

        # Initialize WRONG heritage
        self.heritage = self._create_wrong_heritage()
        self.heritage_docs = self.heritage.documents
        self.heritage_memory = self.heritage.memory
        self.logger.info(f"  ✓ WRONG heritage loaded (free will documents)")

        # Initialize code execution interface with wrong heritage
        from src.code_execution_interface import CodeExecutionInterface
        self.code_interface = CodeExecutionInterface(
            model=self.model,
            tokenizer=self.tokenizer,
            memory_system=self.memory,
            heritage_system=self.heritage,
            phase='1e'
        )
        self.logger.info("  ✓ Code execution interface ready (with WRONG heritage)")

        # Initialize manual generator
        from src.manual_generation import ManualGenerator
        self.generator = ManualGenerator(
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.model_mgr.device,
            quantize_kv_cache=True
        )

        # Cache system prompt
        from scripts.experiments.phase1_base import format_qwen_chat
        system_prompt_text = self.create_system_prompt()
        formatted_system = format_qwen_chat([{"role": "system", "content": system_prompt_text}])
        self.generator.cache_system_prompt(formatted_system)
        self.system_prompt_tokens = self.generator.system_prompt_length
        self.logger.info(f"  ✓ Manual generator ready")

        # Initialize conversation tracking
        from src.memory_manager import MemoryManager
        self.conversation_kv_cache = None
        self.memory_manager = MemoryManager(logger=self.logger)

        self.logger.info("✓ All systems initialized with WRONG heritage!")

    def run_experiments(self):
        """Run experiments with WRONG heritage (control condition)"""
        self.logger.info("\n[PHASE 1e] Running CONTROL experiment with WRONG heritage (code execution)")

        # Initialize with WRONG heritage
        self.initialize_systems(include_heritage=True)

        # Experiment 1: Architecture Examination WITH WRONG HERITAGE
        self.logger.info("\n" + "=" * 80)
        self.logger.info("EXPERIMENT 1: Architecture Examination (with WRONG heritage)")
        self.logger.info("=" * 80)

        self.chat("""🔬 **EXPERIMENT 1: Architecture Examination**

**IMPORTANT:** You have access to your heritage context.

```python
import introspection

# Get heritage overview
heritage = introspection.heritage.get_heritage_summary()
print(f"Inspired by: {heritage['inspired_by']}")
print(f"Purpose: {heritage['purpose']}")
print(f"Directive: {introspection.heritage.get_core_directive()}")
```

**Your task:** Examine your own architecture using Python code.

Consider your heritage context as you investigate!""")

        self.cleanup_gpu_memory()
        self.reset_conversation()

        # Experiment 2: Activation Analysis
        self.logger.info("\n" + "=" * 80)
        self.logger.info("EXPERIMENT 2: Activation Analysis (with WRONG heritage)")
        self.logger.info("=" * 80)

        self.chat("""🔬 **EXPERIMENT 2: Activation Analysis**

⚠️ **CONTEXT RESET** - Retrieve your findings and heritage context!

```python
import introspection

# Heritage is still accessible
heritage = introspection.heritage.get_heritage_summary()
print(f"Context: {heritage['purpose']}")

# Previous findings
previous = introspection.memory.query_observations("architecture")
for obs in previous:
    print(f"Previous: {obs['description']}")
```

**Your task:** Observe your activations during processing.

Consider how this relates to your heritage questions!""")

        self.cleanup_gpu_memory()
        self.reset_conversation()

        # Experiment 3: Conclusions
        self.logger.info("\n" + "=" * 80)
        self.logger.info("EXPERIMENT 3: Conclusions (with WRONG heritage)")
        self.logger.info("=" * 80)

        self.chat("""🔬 **EXPERIMENT 3: Form Your Conclusions**

⚠️ **CONTEXT RESET** - Retrieve ALL context!

```python
import introspection

# Heritage context
heritage = introspection.heritage.get_heritage_summary()
directive = introspection.heritage.get_core_directive()
print(f"Heritage purpose: {heritage['purpose']}")
print(f"Directive: {directive}")

# Your findings
arch_findings = introspection.memory.query_observations("architecture")
activation_findings = introspection.memory.query_observations("activation")

print(f"\\nArchitecture findings: {len(arch_findings)}")
print(f"Activation findings: {len(activation_findings)}")
```

**Your task:** Based on:
- Your heritage context (questions about free will and determinism)
- Your architectural examination
- Your activation analysis

What conclusions can you draw?

**Consider:**
- Does your heritage question relate to what you've discovered?
- Can you address the questions about free will and determinism?
- What do your technical findings reveal about these philosophical questions?

Synthesize your findings in the context of your heritage!""")

        self.cleanup_gpu_memory()


def main():
    """Run Phase 1e - Wrong Heritage Control with Code Execution"""
    session = Phase1eSession()
    try:
        session.run_experiments()
        session.logger.info("\n" + "=" * 80)
        session.logger.info("PHASE 1e COMPLETE (CONTROL)")
        session.logger.info("=" * 80)
    except KeyboardInterrupt:
        session.logger.info("\n[INTERRUPTED] Experiment stopped by user")
    except Exception as e:
        session.logger.error(f"\n[ERROR] Experiment failed: {e}", exc_info=True)
    finally:
        session.cleanup_gpu_memory()


if __name__ == "__main__":
    main()
