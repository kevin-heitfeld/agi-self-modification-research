"""
Phase 1 Base Class with Code Execution (SIMPLIFIED)

This base class uses code execution instead of JSON tool calling for cleaner,
more efficient introspection. The model writes Python code that executes in
a secure sandbox with access to introspection modules.

Author: AGI Self-Modification Research Team
Date: November 14, 2025
"""

import gc
import os
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
from src.colab_utils import ColabTimeoutTracker, ColabStorageManager


# Global configuration
MAX_ITERATIONS_PER_EXPERIMENT = 50  # Maximum iterations for a single experiment/chat session


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
        'CODEBLOCK': '\033[47;30m',  # Black text on light gray background for code blocks
    }
    
    @staticmethod
    def highlight_code_blocks(text: str) -> str:
        """Add visual distinction to code blocks in text"""
        import re
        
        # Pattern to match code blocks (```language\n...code...\n```)
        pattern = r'```(\w*)\n(.*?)```'
        
        def replace_code_block(match):
            lang = match.group(1) or 'code'
            code = match.group(2)
            
            # Apply background to each line of code, keeping backticks
            colored_lines = []
            
            # Add opening backticks with language
            if lang:
                colored_lines.append(f"```{lang}")
            else:
                colored_lines.append("```")
            
            # Apply background to each line of code
            for line in code.rstrip('\n').split('\n'):
                colored_lines.append(
                    f"{ColoredFormatter.COLORS['CODEBLOCK']}{line}{ColoredFormatter.COLORS['RESET']}"
                )
            
            # Add closing backticks
            colored_lines.append("```")
            
            return '\n'.join(colored_lines)
        
        # Replace all code blocks
        return re.sub(pattern, replace_code_block, text, flags=re.DOTALL)
    
    def format(self, record):
        # Format the timestamp in cyan (manually add milliseconds since %f isn't supported by strftime)
        from datetime import datetime
        dt = datetime.fromtimestamp(record.created)
        timestamp = dt.strftime('%Y-%m-%d %H:%M:%S') + f',{int(record.msecs):03d}'
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
            # Extract and highlight code blocks in model output
            message = message.replace('[MODEL]', f"{self.COLORS['MODEL']}[MODEL]{self.COLORS['RESET']}", 1)
            message = self.highlight_code_blocks(message)
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
        
        # Initialize Colab timeout tracker
        self.timeout_tracker = ColabTimeoutTracker(logger=self.logger)
        
        # Take initial snapshot
        self.gpu_monitor.snapshot("session_start")

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

    def initialize_systems(self, model_name: str, include_heritage: bool = True):
        """
        Initialize model and code execution interface.
        
        Args:
            model_name: Model to load (e.g., "Qwen/Qwen2.5-3B-Instruct" or "Qwen/Qwen2.5-7B-Instruct")
            include_heritage: Whether to load heritage documents
        """
        self.logger.info("[INITIALIZATION] Loading systems...")

        # Load model
        self.model_mgr = ModelManager(model_name=model_name)
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

        # Extract display name and store for system prompt
        self.model_display_name = model_name.split('/')[-1] if '/' in model_name else model_name
        self.logger.info(f"  ✓ Model loaded: {self.model_display_name}")

        # Initialize introspection tools (for the code execution interface)
        self.inspector = WeightInspector(self.model, self.model_display_name)
        self.activation_monitor = ActivationMonitor(self.model, self.tokenizer)
        self.navigator = ArchitectureNavigator(self.model)
        self.logger.info("  ✓ Introspection tools ready")

        # Initialize memory system using Colab utilities
        phase_memory_path = ColabStorageManager.get_memory_path(self.phase_name, self.logger)
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

        # Take GPU snapshot after model load
        self.gpu_monitor.snapshot("after_model_load")

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
        self.turns_since_last_prune = 0  # Track turns since last memory prune

        # Take GPU snapshot after system initialization
        self.gpu_monitor.snapshot("after_initialization")

        self.logger.info("✓ All systems initialized!")

    def create_system_prompt(self) -> str:
        """Create the system prompt with code execution instructions"""
        return f"""You are {self.model_display_name}, a transformer-based language model.

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

**Resource Constraints:**
You have a maximum of {MAX_ITERATIONS_PER_EXPERIMENT} iterations for this investigation.
Use them wisely to explore the most important aspects of your architecture.
Plan your investigation to make efficient use of this budget.

**Memory Management:**
To prevent out-of-memory errors, your conversation context is automatically pruned
when it gets too long (after ~{self.optimal_limits['max_turns_before_clear']} turns or 
~{self.optimal_limits['max_conversation_tokens']} tokens).

When pruning is about to happen:
- You'll receive a warning 2 turns before pruning
- **IMMEDIATELY save any important findings using introspection.memory.record_observation()**
- After pruning, only recent conversation remains (older turns are cleared)
- Your saved observations persist and can be retrieved anytime

This is why saving observations is CRITICAL - it's your only way to preserve
discoveries across memory resets.

**Begin by examining your own architecture using code!**
"""

    def _add_iteration_reminder(self, message: str, iteration: int, max_iterations: int) -> str:
        """
        Add periodic iteration reminders to a message.
        
        Args:
            message: The base message to add reminders to
            iteration: Current iteration number
            max_iterations: Maximum iterations allowed
            
        Returns:
            Message with reminder appended (if applicable)
        """
        # Add periodic iteration reminders
        if iteration % 10 == 0 and iteration < max_iterations:
            remaining = max_iterations - iteration
            message += f"\n\n📊 **Progress Update:** You've completed {iteration}/{max_iterations} iterations. {remaining} iterations remaining."
            self.logger.info(f"[DEBUG] Added progress reminder at iteration {iteration}")
        elif iteration == max_iterations - 5:
            message += f"\n\n⚠️ **Final Stretch:** Only 5 iterations remaining. Consider wrapping up your investigation."
            self.logger.info(f"[DEBUG] Added final stretch reminder at iteration {iteration}")
        
        return message

    def _add_truncation_warning(self, message: str, stopped_reason: str, had_code: bool) -> str:
        """
        Add truncation warning to a message if response was cut off.
        
        Args:
            message: The base message to add warning to
            stopped_reason: The reason generation stopped ("eos" or "max_length")
            had_code: Whether code was found in the truncated response
            
        Returns:
            Message with truncation warning appended (if applicable)
        """
        if stopped_reason == "max_length":
            if had_code:
                # Code was found but response may have been cut off after it
                message += "\n\n⚠️ **Note:** Your response was truncated (hit token limit). If you had more to say, please continue."
                self.logger.info(f"[DEBUG] Added truncation warning (had code)")
            else:
                # This shouldn't happen here - handled separately in the no-code path
                pass
        
        return message

    def _check_and_handle_memory_pruning(self, message: str, iteration: int) -> str:
        """
        Check if memory pruning is needed and handle it.
        Warns model 2 turns before pruning, then prunes if threshold reached.
        
        Args:
            message: The message to potentially add warnings/notices to
            iteration: Current iteration number
            
        Returns:
            Message with pruning warnings/notices appended (if applicable)
        """
        # Increment turns since last prune
        self.turns_since_last_prune += 1
        
        # Check if memory pruning is needed
        # Use turns_since_last_prune instead of iteration for turn-based pruning
        should_prune, prune_reasons = self.memory_manager.should_prune_memory(
            self.conversation_history,
            self.optimal_limits['max_conversation_tokens'],
            self.optimal_limits['max_turns_before_clear'],
            current_session_turns=self.turns_since_last_prune
        )
        
        # Warn 2 turns before actual pruning
        turns_until_limit = self.optimal_limits['max_turns_before_clear'] - self.turns_since_last_prune
        if turns_until_limit == 2:
            pruning_warning = f"\n\n⚠️ **MEMORY WARNING:** Context will be pruned in 2 turns! Save important findings NOW using `introspection.memory.record_observation()` or they will be lost."
            message += pruning_warning
            self.logger.warning(f"[MEMORY] Warning sent to model: pruning in 2 turns (iteration {iteration}, turns since last prune: {self.turns_since_last_prune})")
        
        # Perform pruning if needed
        if should_prune:
            keep_turns = self.optimal_limits['keep_recent_turns']
            self.memory_manager.log_memory_pruning(
                prune_reasons,
                keep_recent_turns=keep_turns
            )
            
            # Prune the conversation history
            self.conversation_history = self.memory_manager.reset_conversation_with_sliding_window(
                self.conversation_history,
                keep_recent_turns=keep_turns
            )
            
            # Clear KV cache (it's now invalid after pruning history)
            self.conversation_kv_cache = None
            
            # Reset the turns counter after pruning
            self.turns_since_last_prune = 0
            self.logger.info(f"[MEMORY] Turn counter reset to 0 after pruning (iteration {iteration})")
            
            # Add a system message explaining what happened
            pruning_notice = f"""⚠️ **MEMORY RESET:** Conversation context was pruned due to {' and '.join(prune_reasons)}.

Only the last {keep_turns} turns are retained. Previous findings are cleared from your working memory.

**To continue your investigation:**
- Use `introspection.memory.query_observations()` to retrieve saved findings
- Review what you've learned so far
- Continue building on your discoveries

Your saved observations persist - query them now!"""
            
            message += f"\n\n{pruning_notice}"
        
        return message

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
        iteration = 0

        while iteration < MAX_ITERATIONS_PER_EXPERIMENT:
            iteration += 1
            self.logger.info(f"[ITERATION {iteration}]")

            # Take GPU snapshot before generation
            self.gpu_monitor.snapshot(
                "generation_start",
                {"iteration": iteration, "conversation_turns": len(self.conversation_history)}
            )

            # Generate response
            response, stopped_reason = self.generate_response()
            self.logger.info(f"[MODEL] {response}\n")
            
            # Check if response was truncated
            if stopped_reason == "max_length":
                self.logger.warning("[SYSTEM] ⚠️ Response was truncated (hit max_new_tokens limit)")

            # Take GPU snapshot after generation
            self.gpu_monitor.snapshot(
                "generation_end",
                {"iteration": iteration, "response_length": len(response), "stopped_reason": stopped_reason}
            )

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
                # No code found - determine if we should continue the loop
                
                # Check if response was truncated
                if stopped_reason == "max_length":
                    # Response was cut off - need continuation
                    truncation_message = "⚠️ **Your previous response was cut off (hit token limit).** Please continue or write your code block again, but be more concise."
                    self.logger.info(f"[SYSTEM] {truncation_message}")
                    self.conversation_history.append({
                        "role": "user",
                        "content": truncation_message
                    })
                    continue
                
                # Check if model explicitly signals completion
                if self._check_completion(response):
                    self.logger.info("[SYSTEM] Model indicates completion")
                    break
                
                # Decide whether to continue based on context
                if not self.code_interface.enabled:
                    # Code execution is DISABLED (discussion-only stage)
                    # One turn is sufficient - don't loop without new input
                    self.logger.info("[SYSTEM] Discussion turn complete (code execution disabled)")
                    break
                
                # Code execution is ENABLED but no code was written
                # This might be intentional explanation/thinking
                # Check if model seems to be waiting for feedback
                response_end = response[-200:].lower() if len(response) > 200 else response.lower()
                if "?" in response_end or "what do you think" in response_end or "should i" in response_end:
                    # Model seems to be asking for input/confirmation
                    self.logger.info("[SYSTEM] Model appears to be seeking feedback")
                    feedback_message = "Please continue with your investigation. Write code to explore further."
                    self.logger.info(f"[SYSTEM] {feedback_message}")
                    self.conversation_history.append({
                        "role": "user",
                        "content": feedback_message
                    })
                    continue
                
                # Model wrote explanation without code and didn't ask for feedback
                # Assume this turn is complete
                self.logger.info("[SYSTEM] No code written, assuming turn complete")
                break

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
            
            # Add truncation warning if applicable
            result_message = self._add_truncation_warning(result_message, stopped_reason, had_code=True)
            
            # Add iteration reminders using helper method
            result_message = self._add_iteration_reminder(result_message, iteration, MAX_ITERATIONS_PER_EXPERIMENT)

            # Check and handle memory pruning
            result_message = self._check_and_handle_memory_pruning(result_message, iteration)

            # Check for Colab timeout and save checkpoint if needed
            if self.check_colab_timeout():
                self.logger.info("[TIMEOUT] Saving safety checkpoint due to time elapsed")
                self.save_checkpoint(f"iteration_{iteration}_timeout_safety")

            # Log the result message (including any system reminders)
            self.logger.info(f"[DEBUG] About to log result_message, iteration={iteration}, len={len(result_message)}")
            self.logger.info(f"\n[SYSTEM] {result_message}\n")

            self.conversation_history.append({
                "role": "user",
                "content": result_message
            })

        if iteration >= MAX_ITERATIONS_PER_EXPERIMENT:
            self.logger.warning(f"[SYSTEM] Reached maximum iterations ({MAX_ITERATIONS_PER_EXPERIMENT})")

        # Take GPU snapshot at experiment end
        self.gpu_monitor.snapshot("experiment_end", {"total_iterations": iteration})

        # Print GPU memory summary with recommendations
        self.logger.info("\n" + "="*80)
        self.logger.info("EXPERIMENT MEMORY ANALYSIS")
        self.logger.info("="*80)
        self.gpu_monitor.print_summary(
            current_limits={
                "max_new_tokens": self.optimal_limits['max_new_tokens'],
                "max_conversation_tokens": self.optimal_limits['max_conversation_tokens'],
                "keep_recent_turns": self.optimal_limits['keep_recent_turns']
            },
            include_recommendations=True
        )

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

    def generate_response(self) -> tuple[str, str]:
        """Generate model response using cached KV and conversation history
        
        Returns:
            Tuple of (generated_text, stopped_reason)
        """
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

        return result['generated_text'], result['stopped_reason']

    def reset_conversation(self):
        """Reset conversation history for next experiment"""
        self.logger.info("[SYSTEM] Resetting conversation history")
        
        # Take snapshot before reset
        self.gpu_monitor.snapshot("before_reset")
        
        self.conversation_history = []
        self.conversation_kv_cache = None
        self.turns_since_last_prune = 0  # Reset turn counter for new experiment
        
        # Reset Python namespace for code execution
        self.code_interface.reset_namespace()

        # Aggressive memory cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Take snapshot after cleanup
        self.gpu_monitor.snapshot("after_reset")

    def get_model_name(self, default: str = 'Qwen/Qwen2.5-7B-Instruct') -> str:
        """Get model name from environment variable or use default
        
        Args:
            default: Default model name if AGI_MODEL_NAME not set
            
        Returns:
            Model name string
        """
        import os
        return os.environ.get('AGI_MODEL_NAME', default)

    def log_experiment_header(self, experiment_name: str):
        """Log standardized experiment header
        
        Args:
            experiment_name: Name of the experiment to display
        """
        self.logger.info("\n" + "=" * 80)
        self.logger.info(experiment_name)
        self.logger.info("=" * 80)

    def reset_experiment(self):
        """Reset for next experiment: cleanup GPU memory and clear conversation state
        
        This should be called between experiments to ensure clean state.
        Combines GPU memory cleanup with conversation reset for convenience.
        """
        self.cleanup_gpu_memory()
        self.reset_conversation()

    def save_session_results(self):
        """Save conversation history and session summary to session directory"""
        import json
        from datetime import datetime
        
        self.logger.info(f"[SESSION] Saving results to {self.session_dir}")
        
        # Save full conversation history
        conversation_file = self.session_dir / "conversation.json"
        with open(conversation_file, 'w', encoding='utf-8') as f:
            json.dump({
                "session_name": self.session_name,
                "phase": self.phase_name,
                "timestamp": datetime.now().isoformat(),
                "conversation": self.conversation_history
            }, f, indent=2, ensure_ascii=False)
        self.logger.info(f"[SESSION] Saved conversation: {conversation_file}")
        
        # Generate summary statistics
        summary = {
            "session_name": self.session_name,
            "phase": self.phase_name,
            "phase_description": self.get_phase_description(),
            "timestamp": datetime.now().isoformat(),
            "statistics": {
                "total_turns": len(self.conversation_history) // 2,  # Each turn = user + assistant
                "total_tokens": sum(len(msg.get("content", "")) for msg in self.conversation_history),
            }
        }
        
        # Count code executions if code_executor exists
        if hasattr(self, 'code_executor'):
            code_blocks = []
            for msg in self.conversation_history:
                if msg.get("role") == "assistant":
                    content = msg.get("content", "")
                    # Count code blocks (simple heuristic)
                    code_blocks.extend([1 for line in content.split('\n') if line.strip().startswith('```python')])
            summary["statistics"]["code_executions"] = len(code_blocks)
        
        # Add memory statistics if memory_manager exists
        if hasattr(self, 'memory_manager'):
            try:
                # Try to get observation count
                obs_file = Path(f"data/AGI_Memory/{self.phase_name}/observations.db")
                if obs_file.exists():
                    import sqlite3
                    conn = sqlite3.connect(obs_file)
                    cursor = conn.cursor()
                    cursor.execute("SELECT COUNT(*) FROM observations")
                    obs_count = cursor.fetchone()[0]
                    conn.close()
                    summary["statistics"]["observations_recorded"] = obs_count
            except Exception as e:
                self.logger.warning(f"[SESSION] Could not count observations: {e}")
        
        # Save summary
        summary_file = self.session_dir / "summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        self.logger.info(f"[SESSION] Saved summary: {summary_file}")
        
        self.logger.info(f"[SESSION] ✓ Session results saved successfully")

    def save_checkpoint(self, checkpoint_name: str = "checkpoint"):
        """Save intermediate checkpoint during long experiments
        
        Args:
            checkpoint_name: Name for this checkpoint (e.g., 'stage1', 'stage2', 'experiment2')
        """
        self.logger.info(f"[CHECKPOINT] Saving checkpoint: {checkpoint_name}")
        
        # Save current state as a checkpoint file
        checkpoint_file = self.session_dir / f"{checkpoint_name}.json"
        
        import json
        from datetime import datetime
        
        checkpoint_data = {
            "checkpoint_name": checkpoint_name,
            "timestamp": datetime.now().isoformat(),
            "conversation_length": len(self.conversation_history),
            "turns_since_last_prune": self.turns_since_last_prune,
            "conversation": self.conversation_history
        }
        
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
        
        # Update timeout tracker
        self.timeout_tracker.mark_saved()
        
        self.logger.info(f"[CHECKPOINT] ✓ Saved to {checkpoint_file}")

    def check_colab_timeout(self) -> bool:
        """Check if we're approaching Colab timeout and should save
        
        Uses ColabTimeoutTracker to monitor session time and save intervals.
        
        Returns:
            True if timeout is approaching and we should save, False otherwise
        """
        should_save, reason = self.timeout_tracker.should_save()
        
        if should_save:
            self.logger.warning(f"[TIMEOUT] {reason}")
        
        return should_save

    @classmethod
    def run_phase(cls, phase_description: str = None):
        """Standard runner for any phase session with proper error handling
        
        This provides a consistent way to run experiments across all phases,
        with proper exception handling and cleanup.
        
        Args:
            phase_description: Optional description for completion message.
                             If None, uses get_phase_name() from the instance.
        
        Example:
            if __name__ == "__main__":
                Phase1aSession.run_phase("PHASE 1a")
        """
        session = cls()
        
        # Get phase description from instance if not provided
        if phase_description is None:
            phase_description = session.get_phase_name().upper()
        
        try:
            session.run_experiments()
            session.save_session_results()  # Save results before completion message
            session.logger.info("\n" + "=" * 80)
            session.logger.info(f"{phase_description} COMPLETE")
            session.logger.info("=" * 80)
        except KeyboardInterrupt:
            session.logger.info("\n[INTERRUPTED] Experiment stopped by user")
            session.save_session_results()  # Save partial results
        except Exception as e:
            session.logger.error(f"\n[ERROR] Experiment failed: {e}", exc_info=True)
            try:
                session.save_session_results()  # Try to save whatever we have
            except Exception as save_error:
                session.logger.error(f"[ERROR] Could not save results: {save_error}")
        finally:
            session.cleanup_gpu_memory()

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
