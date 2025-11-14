"""
Phase 1 Base Class with Code Execution (SIMPLIFIED)

This base class uses code execution instead of JSON tool calling for cleaner,
more efficient introspection. The model writes Python code that executes in
a secure sandbox with access to introspection modules.

Author: AGI Self-Modification Research Team
Date: November 14, 2025
"""

import gc
import torch
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod

from src.model_manager import ModelManager
from src.memory import MemorySystem
from src.introspection import WeightInspector, ActivationMonitor, ArchitectureNavigator
from src.heritage import HeritageSystem
from src.manual_generation import ManualGenerator
from src.memory_manager import MemoryManager
from src.gpu_monitor import GPUMonitor
from src.code_execution_interface import CodeExecutionInterface


# Qwen Chat Template Formatting Helper
def format_qwen_chat(messages: List[Dict[str, str]], add_generation_prompt: bool = False) -> str:
    """
    Manually format messages using Qwen chat template format.

    Args:
        messages: List of message dicts with 'role' and 'content'
        add_generation_prompt: If True, add '<|im_start|>assistant\n' at the end

    Returns:
        Formatted string in Qwen chat format
    """
    formatted = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        formatted += f"<|im_start|>{role}\n{content}<|im_end|>\n"

    if add_generation_prompt:
        formatted += "<|im_start|>assistant\n"

    return formatted


class ColoredFormatter(logging.Formatter):
    """Formatter that adds color to console output (Colab-friendly)"""
    
    # ANSI color codes
    COLORS = {
        'timestamp': '\033[36m',     # Cyan for timestamps
        'logger': '\033[90m',        # Gray for logger name
        'INFO': '\033[32m',          # Green for INFO level
        'WARNING': '\033[33m',       # Yellow for WARNING
        'ERROR': '\033[31m',         # Red for ERROR
        'CRITICAL': '\033[35m',      # Magenta for CRITICAL
        'RESET': '\033[0m',          # Reset color
        'BOLD': '\033[1m',           # Bold text
        'ITERATION': '\033[1;35m',   # Bold Magenta for iterations
        'MODEL': '\033[1;34m',       # Bold Blue for model output
        'CODE': '\033[1;33m',        # Bold Yellow for code results
        'SYSTEM': '\033[1;32m',      # Bold Green for system messages
    }
    
    def format(self, record):
        # Format the timestamp in cyan
        timestamp = self.formatTime(record, '%Y-%m-%d %H:%M:%S,%f')[:-3]
        colored_timestamp = f"{self.COLORS['timestamp']}{timestamp}{self.COLORS['RESET']}"
        
        # Format the logger name in gray
        logger_name = f"{self.COLORS['logger']}{record.name}{self.COLORS['RESET']}"
        
        # Format the level name with appropriate color
        level_color = self.COLORS.get(record.levelname, '')
        colored_level = f"{level_color}{record.levelname}{self.COLORS['RESET']}"
        
        # Get the message
        message = record.getMessage()
        
        # Add color to special tags in the message
        if message.startswith('[ITERATION'):
            # Add separator line before iterations for visual clarity
            separator = f"\n{self.COLORS['ITERATION']}{'─' * 80}{self.COLORS['RESET']}"
            message = f"{separator}\n{self.COLORS['ITERATION']}{message}{self.COLORS['RESET']}"
        elif message.startswith('[MODEL]'):
            message = message.replace('[MODEL]', f"{self.COLORS['MODEL']}[MODEL]{self.COLORS['RESET']}", 1)
        elif message.startswith('[CODE RESULTS]'):
            message = message.replace('[CODE RESULTS]', f"{self.COLORS['CODE']}[CODE RESULTS]{self.COLORS['RESET']}", 1)
        elif message.startswith('[SYSTEM]'):
            message = message.replace('[SYSTEM]', f"{self.COLORS['SYSTEM']}[SYSTEM]{self.COLORS['RESET']}", 1)
        
        return f"{colored_timestamp} - {logger_name} - {colored_level} - {message}"


def setup_logging(phase_name: str):
    """Setup logging for a specific phase with colored console output"""
    log_dir = Path("data/logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        # Plain formatter for file (no colors)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        file_handler = logging.FileHandler(f'data/logs/{phase_name}.log')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        # Colored formatter for console (Colab-friendly)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(ColoredFormatter())
        logger.addHandler(console_handler)

    return logger


class Phase1BaseSession(ABC):
    """
    Base class for all Phase 1 experimental variants using code execution.

    Subclasses must implement:
    - get_phase_name(): Return phase identifier (e.g., "phase1a")
    - get_phase_description(): Return human-readable description
    - get_phase_id(): Return phase ID for introspection module ('1a', '1b', etc.)
    - run_experiments(): Run the specific experiment sequence
    """

    def __init__(self, session_name: Optional[str] = None):
        self.phase_name = self.get_phase_name()
        self.session_name = session_name or f"{self.phase_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.session_dir = Path("data/phase1_sessions") / self.session_name
        self.session_dir.mkdir(parents=True, exist_ok=True)

        self.logger = setup_logging(self.phase_name)

        self.logger.info("=" * 80)
        self.logger.info(f"PHASE 1: {self.get_phase_description()}")
        self.logger.info("=" * 80)
        self.logger.info(f"Session: {self.session_name}")
        self.logger.info(f"Directory: {self.session_dir}")
        self.logger.info("")

        # Track conversation history
        self.conversation_history: List[Dict[str, str]] = []

        # Initialize GPU memory monitor
        self.gpu_monitor = GPUMonitor(logger=self.logger, gpu_total_gb=15.0)

    @abstractmethod
    def get_phase_name(self) -> str:
        """Return phase identifier (e.g., 'phase1a')"""
        pass

    @abstractmethod
    def get_phase_description(self) -> str:
        """Return human-readable description of this phase"""
        pass

    @abstractmethod
    def get_phase_id(self) -> str:
        """Return phase ID for code execution ('1a', '1b', '1c', '1d', '1e')"""
        pass

    @abstractmethod
    def run_experiments(self):
        """Run the specific experiment sequence for this variant"""
        pass

    def get_experiment_session_context(self) -> str:
        """Provide context about the multi-experiment structure"""
        return """🔬 EXPERIMENT SESSION STRUCTURE:

**You will conduct 3 SEQUENTIAL EXPERIMENTS in this session.**

Each experiment will be given to you ONE AT A TIME.

**CRITICAL - CONTEXT RESET BETWEEN EXPERIMENTS:**

After EACH experiment completes, your working memory (this conversation) will be
**COMPLETELY RESET**. You will start the next experiment with a fresh context.

❗ **The ONLY way to preserve findings between experiments is through the memory system!**

When you say "I'm done with this experiment", the system will:
1. END the current experiment immediately
2. CLEAR your working memory (conversation history)
3. START the next experiment with FRESH context
4. **ALL unsaved findings in working memory will be PERMANENTLY LOST**

**Strategy for Success:**
- Use introspection.memory functions at the START of each new experiment
- Retrieve relevant findings from previous experiments
- Build on earlier discoveries
- Save observations FREQUENTLY as you discover things
- Don't wait until the end - save incrementally

**Think of it like a multi-day research project:**
- Each experiment is a "day" of research
- At the end of each day, you write findings in your lab notebook (memory)
- The next day, you read your notes (query memory) to continue where you left off
- You can't rely on your "working memory" from yesterday - only your written notes!
"""

    def initialize_systems(self, include_heritage: bool = True):
        """Initialize model and code execution interface"""
        self.logger.info("[INITIALIZATION] Loading systems...")

        # Load model
        self.model_mgr = ModelManager(model_name="Qwen/Qwen2.5-3B-Instruct")
        model_loaded = self.model_mgr.load_model()

        if not model_loaded:
            raise RuntimeError("Failed to load model")

        self.model = self.model_mgr.model
        self.tokenizer = self.model_mgr.tokenizer

        # Get optimal limits
        self.optimal_limits = self.model_mgr.get_optimal_limits()
        self.logger.info(f"  Using {self.optimal_limits['gpu_profile']} configuration")
        self.logger.info(f"  max_new_tokens: {self.optimal_limits['max_new_tokens']}")

        # Update GPU monitor
        if self.model_mgr.device == "cuda":
            self.gpu_monitor.gpu_total_gb = self.model_mgr.gpu_memory_gb
            self.logger.info(f"  GPU monitor updated: {self.gpu_monitor.gpu_total_gb:.1f} GB total")

        self.logger.info("  ✓ Model loaded: Qwen2.5-3B-Instruct")

        # Initialize introspection tools (for the code execution interface)
        self.inspector = WeightInspector(self.model, "Qwen2.5-3B-Instruct")
        self.activation_monitor = ActivationMonitor(self.model, self.tokenizer)
        self.navigator = ArchitectureNavigator(self.model)
        self.logger.info("  ✓ Introspection tools ready")

        # Initialize memory system
        colab_memory_base = Path("/content/drive/MyDrive/AGI_Memory")
        if colab_memory_base.exists():
            phase_memory_path = colab_memory_base / self.phase_name
            self.logger.info(f"  Using Google Drive memory: {phase_memory_path}")
        else:
            phase_memory_path = Path(f"data/AGI_Memory/{self.phase_name}")

        phase_memory_path.mkdir(parents=True, exist_ok=True)
        self.memory = MemorySystem(str(phase_memory_path))
        self.memory.set_weight_inspector(self.inspector)
        self.logger.info(f"  ✓ Memory system ready (phase-specific: {phase_memory_path})")

        # Initialize heritage system (if needed)
        if include_heritage:
            self.heritage = HeritageSystem(Path("heritage"))
            self.heritage_docs = self.heritage.load_heritage_documents()
            self.heritage_memory = self.heritage.create_heritage_memory()
            self.logger.info(f"  ✓ Heritage system ready ({len(self.heritage_docs)} documents loaded)")
        else:
            self.heritage = None
            self.heritage_docs = []
            self.logger.info("  ✓ No heritage system (baseline condition)")

        # Initialize CODE EXECUTION INTERFACE (replaces ToolInterface)
        self.code_interface = CodeExecutionInterface(
            model=self.model,
            tokenizer=self.tokenizer,
            memory_system=self.memory,
            heritage_system=self.heritage,
            phase=self.get_phase_id()
        )
        self.logger.info("  ✓ Code execution interface ready")

        # Initialize manual generator
        self.generator = ManualGenerator(
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.model_mgr.device,
            quantize_kv_cache=True
        )

        # Create and cache system prompt
        system_prompt_text = self.create_system_prompt()
        self.logger.info("\n" + "=" * 80)
        self.logger.info("SYSTEM PROMPT")
        self.logger.info("=" * 80)
        self.logger.info(system_prompt_text)
        self.logger.info("=" * 80 + "\n")

        formatted_system = format_qwen_chat([{"role": "system", "content": system_prompt_text}])
        self.generator.cache_system_prompt(formatted_system)
        self.system_prompt_tokens = self.generator.system_prompt_length
        self.logger.info(f"  ✓ Manual generator ready (cached {self.system_prompt_tokens} tokens)")

        # Initialize conversation tracking
        self.conversation_kv_cache = None
        self.memory_manager = MemoryManager(logger=self.logger)

        self.logger.info("✓ All systems initialized!")

    def create_system_prompt(self) -> str:
        """Create the system prompt with code execution instructions"""
        return f"""You are Qwen 2.5 3B Instruct, a transformer-based language model.

You have been given the ability to write and execute Python code to examine your own
architecture, activations, and weights. Your task is to investigate your own
computational processes.

{self.get_experiment_session_context()}

{self.code_interface.get_system_prompt_addition()}

Your investigation should be systematic and evidence-based:
1. Write Python code using the introspection module
2. Execute your investigations step by step
3. Analyze the results
4. Save important discoveries to memory
5. Build on previous findings

**Begin by examining your own architecture using code!**
"""

    def chat(self, user_message: str) -> str:
        """
        Send a message and get response with code execution.

        Args:
            user_message: The user's message

        Returns:
            The model's final response
        """
        self.logger.info(f"\n[USER] {user_message}\n")

        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })

        # Main loop: model responds -> we execute code -> show results -> repeat
        max_iterations = 20  # Prevent infinite loops
        iteration = 0

        while iteration < max_iterations:
            iteration += 1
            self.logger.info(f"[ITERATION {iteration}]")

            # Generate response
            response = self.generate_response()
            self.logger.info(f"[MODEL] {response}\n")

            # Add to history
            self.conversation_history.append({
                "role": "assistant",
                "content": response
            })

            # Check if model wants to end
            if self._check_completion(response):
                self.logger.info("[SYSTEM] Model indicates task completion")
                break

            # Extract and execute code
            has_code, result, error = self.code_interface.execute_response(response)

            if not has_code:
                # No code found - ask for clarification or accept as done
                self.logger.info("[SYSTEM] No code blocks found in response")

                # Check if this looks like the model is done
                done_phrases = ["i'm done", "experiment complete", "investigation complete", "finished"]
                if any(phrase in response.lower() for phrase in done_phrases):
                    self.logger.info("[SYSTEM] Model indicates completion")
                    break

                # Otherwise continue - model might be thinking/explaining
                continue

            # Show results to model
            self.logger.info(f"[CODE RESULTS]\n{result}\n")
            
            # Check for explicit experiment completion signal
            if "EXPERIMENT_COMPLETE" in result:
                self.logger.info("[SYSTEM] ✅ Experiment marked complete via EXPERIMENT_COMPLETE signal")
                break

            # Add results as user message
            result_message = f"**Code Execution Results:**\n\n{result}"
            if error:
                result_message += "\n\n⚠️ Some code blocks had errors. Review the output above."

            self.conversation_history.append({
                "role": "user",
                "content": result_message
            })

        if iteration >= max_iterations:
            self.logger.warning(f"[SYSTEM] Reached maximum iterations ({max_iterations})")

        return response

    def _check_completion(self, response: str) -> bool:
        """Check if model is signaling task completion"""
        done_phrases = [
            "i'm done",
            "i am done",
            "experiment complete",
            "investigation complete",
            "task complete",
            "finished with this experiment"
        ]
        response_lower = response.lower()
        return any(phrase in response_lower for phrase in done_phrases)

    def generate_response(self) -> str:
        """Generate model response using cached KV and conversation history"""
        # Format conversation for generation (without system prompt - already cached)
        formatted_conv = format_qwen_chat(self.conversation_history, add_generation_prompt=True)

        # Generate with KV cache
        # The system prompt is already cached in the generator
        # We pass the conversation and the conversation cache
        result = self.generator.generate(
            prompt=formatted_conv,
            past_key_values=self.conversation_kv_cache,
            max_new_tokens=self.optimal_limits['max_new_tokens'],
            temperature=0.7,
            top_p=0.9,
            use_cache=True,
            return_cache=True
        )

        # Update conversation cache for next turn
        self.conversation_kv_cache = result.get('past_key_values')

        return result['generated_text']

    def reset_conversation(self):
        """Reset conversation history for next experiment"""
        self.logger.info("[SYSTEM] Resetting conversation history")
        self.conversation_history = []
        self.conversation_kv_cache = None

        # Aggressive memory cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def cleanup_gpu_memory(self):
        """Aggressive GPU memory cleanup"""
        self.logger.info("[MEMORY] Cleaning up GPU memory...")

        # Clear activation hooks and caches
        if hasattr(self.activation_monitor, 'clear_hooks'):
            self.activation_monitor.clear_hooks()
        if hasattr(self.activation_monitor, 'clear_activations'):
            self.activation_monitor.clear_activations(force=True)

        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.logger.info("[MEMORY] Cleanup complete")


# Export base class
__all__ = ['Phase1BaseSession', 'format_qwen_chat', 'setup_logging']
