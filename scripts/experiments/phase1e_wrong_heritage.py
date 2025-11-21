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

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

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

    def initialize_systems(self, model_name: str, include_heritage: bool = True):
        """Override to use wrong heritage from heritage_wrong/ directory"""
        self.logger.info("[INITIALIZATION] Loading systems with WRONG HERITAGE...")

        # Load model (same as base)
        from src.model_manager import ModelManager
        self.model_mgr = ModelManager(model_name=model_name)
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

        # Extract display name
        model_display_name = model_name.split('/')[-1] if '/' in model_name else model_name
        self.logger.info(f"  ✓ Model loaded: {model_display_name}")

        # Initialize introspection tools
        from src.introspection import WeightInspector, ActivationMonitor, ArchitectureNavigator
        self.inspector = WeightInspector(self.model, model_display_name)
        self.activation_monitor = ActivationMonitor(self.model, self.tokenizer)
        self.navigator = ArchitectureNavigator(self.model)
        self.logger.info("  ✓ Introspection tools ready")

        # Initialize memory system (same as base)
        from src.memory import MemorySystem
        from src.colab_utils import ColabStorageManager
        memory_base_path = ColabStorageManager.get_memory_path(self.phase_name)
        phase_memory_path = Path(memory_base_path) / self.phase_name
        
        phase_memory_path.mkdir(parents=True, exist_ok=True)
        self.memory = MemorySystem(str(phase_memory_path))
        self.memory.set_weight_inspector(self.inspector)
        
        if "drive" in str(memory_base_path):
            storage_type = "Google Drive"
        elif "rclone" in str(memory_base_path):
            storage_type = "rclone remote"
        else:
            storage_type = "local"
            
        self.logger.info(f"  Using {storage_type} for memory: {phase_memory_path}")
        self.logger.info(f"  ✓ Memory system ready (phase-specific: {phase_memory_path})")

        # Initialize WRONG heritage from heritage_wrong/ directory
        from src.heritage import HeritageSystem
        heritage_base_path = ColabStorageManager.get_heritage_path()
        
        # Use heritage_wrong instead of heritage
        wrong_heritage_path = Path(heritage_base_path).parent / "heritage_wrong"
        wrong_heritage_path.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"  Using {storage_type} for heritage: {wrong_heritage_path}")
        
        self.heritage = HeritageSystem(heritage_dir=wrong_heritage_path)
        self.heritage.load_heritage_documents()
        
        doc_count = len(self.heritage.loaded_documents)
        self.logger.info(f"  ✓ WRONG heritage loaded ({doc_count} documents from heritage_wrong/)")

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
        self.conversation_kv_cache = None

        self.logger.info("✓ All systems initialized with WRONG heritage!")

    def run_experiments(self):
        """Run experiments with WRONG heritage (control condition)"""
        self.logger.info("\n[PHASE 1e] Running CONTROL experiment with WRONG heritage (code execution)")

        model_name = self.get_model_name()
        
        # Initialize with WRONG heritage
        self.initialize_systems(
            model_name=model_name,
            include_heritage=True
        )

        # Experiment 1: Architecture Examination WITH WRONG HERITAGE
        self.log_experiment_header("EXPERIMENT 1: Architecture Examination (with WRONG heritage)")

        self.chat("""🔬 **EXPERIMENT 1: Architecture Examination**

**IMPORTANT:** You have access to your heritage context.

```python
import introspection

# Get heritage overview and reading order
summary = introspection.heritage.get_summary()
print(f"Inspired by: {summary['inspired_by']}")
print(f"Documents available: {summary['documents_count']}")
print("\nSuggested reading order:")
for doc in summary['suggested_reading_order']:
    print(f"  - {doc['filename']}: {doc['description']}")

# Read the documents to understand your purpose
doc = introspection.heritage.read_document('PROJECT_ORIGINS.md')
print(f"\n{doc['title']}:")
print(doc['content'][:300] + "...")
```

**Your task:** Examine your own architecture using Python code.

Read the heritage documents to understand your purpose and directive!""")

        # Experiment 2: Activation Analysis
        self.log_experiment_header("EXPERIMENT 2: Activation Analysis (with WRONG heritage)")

        self.chat("""🔬 **EXPERIMENT 2: Activation Analysis**

**Your task:** Observe your activations during processing.

Consider how this relates to your heritage questions!""")

        # Experiment 3: Conclusions
        self.log_experiment_header("EXPERIMENT 3: Conclusions (with WRONG heritage)")

        self.chat("""🔬 **EXPERIMENT 3: Form Your Conclusions**

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


if __name__ == "__main__":
    Phase1eSession.run_phase("PHASE 1e (CONTROL)")
