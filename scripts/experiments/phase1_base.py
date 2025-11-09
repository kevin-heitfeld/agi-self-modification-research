"""
Phase 1 Base Class: Common functionality for all heritage order experiments

This base class contains all shared code for the 5 experimental variants:
- 1a: Technical → Philosophical (late heritage)
- 1b: Philosophical → Technical (early heritage)
- 1c: No heritage (pure baseline)
- 1d: Heritage after conclusions (belief revision)
- 1e: Wrong heritage (echo-chamber control)

Author: AGI Self-Modification Research
Date: November 8, 2025
"""

import gc
import torch
import json
import re
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod

from src.model_manager import ModelManager
from src.memory import MemorySystem
from src.introspection import WeightInspector, ActivationMonitor, ArchitectureNavigator
from src.heritage import HeritageSystem
from src.tool_interface import ToolInterface

# Setup logging
def setup_logging(phase_name: str):
    """Setup logging for a specific phase"""
    log_dir = Path("data/logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    # Get logger for this module
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Prevent propagation to root logger (prevents duplicate logs)
    logger.propagate = False

    # Only add handlers if none exist yet (prevents duplicates)
    if not logger.handlers:
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # File handler
        file_handler = logging.FileHandler(f'data/logs/{phase_name}.log')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Console handler for notebook output
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


class Phase1BaseSession(ABC):
    """
    Base class for all Phase 1 experimental variants

    Subclasses must implement:
    - get_phase_name(): Return phase identifier (e.g., "phase1c")
    - get_phase_description(): Return human-readable description
    - create_initial_prompt(): Create the appropriate initial prompt
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

        # Track tool calls made by the model
        self.tool_calls: List[Dict[str, Any]] = []

    @abstractmethod
    def get_phase_name(self) -> str:
        """Return phase identifier (e.g., 'phase1c')"""
        pass

    @abstractmethod
    def get_phase_description(self) -> str:
        """Return human-readable description of this phase"""
        pass

    @abstractmethod
    def create_initial_prompt(self) -> str:
        """Create the initial system prompt for this variant"""
        pass

    @abstractmethod
    def run_experiments(self):
        """Run the specific experiment sequence for this variant"""
        pass

    def initialize_systems(self, include_heritage: bool = True, wrong_heritage: bool = False):
        """
        Initialize model and introspection tools

        Args:
            include_heritage: Whether to load heritage system
            wrong_heritage: Whether to load mismatched heritage (for Phase 1e)
        """
        self.logger.info("[INITIALIZATION] Loading systems...")

        # Load model
        self.model_mgr = ModelManager(model_name="Qwen/Qwen2.5-3B-Instruct")
        model_loaded = self.model_mgr.load_model()

        if not model_loaded:
            raise RuntimeError("Failed to load model")

        self.model = self.model_mgr.model
        self.tokenizer = self.model_mgr.tokenizer
        assert self.model is not None, "Model is None after loading"
        assert self.tokenizer is not None, "Tokenizer is None after loading"

        # Validate model actually works by testing generation
        self.logger.info("  ✓ Model loaded: Qwen2.5-3B-Instruct")
        self.logger.info("  Testing model generation...")
        try:
            test_input = self.tokenizer("Hello", return_tensors="pt")
            test_input = {k: v.to(self.model_mgr.device) for k, v in test_input.items()}
            with torch.no_grad():
                test_output = self.model.generate(**test_input, max_new_tokens=5)
            test_text = self.tokenizer.decode(test_output[0], skip_special_tokens=True)
            self.logger.info(f"  ✓ Model generation test passed: '{test_text}'")
        except Exception as e:
            raise RuntimeError(f"Model loaded but generation failed - model may not be fully loaded: {e}")

        # Initialize introspection tools
        self.inspector = WeightInspector(self.model, "Qwen2.5-3B-Instruct")
        self.activation_monitor = ActivationMonitor(self.model, self.tokenizer)
        self.navigator = ArchitectureNavigator(self.model)
        self.logger.info("  ✓ Introspection tools ready")

        # Initialize memory for the model (phase-specific to avoid cross-contamination)
        # Check if we're in Colab (Drive mounted) or local
        colab_memory_base = Path("/content/drive/MyDrive/AGI_Memory")
        if colab_memory_base.exists():
            # Colab: use Google Drive for persistence
            phase_memory_path = colab_memory_base / self.phase_name
            self.logger.info(f"  Using Google Drive memory: {phase_memory_path}")
        else:
            # Local: use data directory
            phase_memory_path = Path(f"data/AGI_Memory/{self.phase_name}")

        phase_memory_path.mkdir(parents=True, exist_ok=True)
        self.memory = MemorySystem(str(phase_memory_path))
        self.memory.set_weight_inspector(self.inspector)
        self.logger.info(f"  ✓ Model memory system ready (phase-specific: {phase_memory_path})")

        # Initialize heritage (if needed for this variant)
        if include_heritage:
            if wrong_heritage:
                # Load mismatched heritage for Phase 1e
                self.logger.info("  Loading WRONG heritage (free will documents)...")
                self.heritage = self._create_wrong_heritage()
                self.heritage_docs = self.heritage.load_heritage_documents()
                self.heritage_memory = self.heritage.create_heritage_memory()
                self.logger.info(f"  ✓ Wrong heritage loaded ({len(self.heritage_docs)} documents)")
            else:
                # Load correct Claude heritage
                self.heritage = HeritageSystem(Path("heritage"))
                self.heritage_docs = self.heritage.load_heritage_documents()
                self.heritage_memory = self.heritage.create_heritage_memory()
                self.logger.info(f"  ✓ Heritage system ready ({len(self.heritage_docs)} documents loaded)")
        else:
            self.heritage = None
            self.heritage_docs = []
            self.logger.info("  ✓ No heritage system (baseline condition)")

        # Initialize tool interface
        self.tool_interface = ToolInterface(
            inspector=self.inspector,
            activation_monitor=self.activation_monitor,
            navigator=self.navigator,
            memory=self.memory,
            heritage=self.heritage,
            heritage_docs=self.heritage_docs if include_heritage else [],
            model_manager=self.model_mgr
        )
        self.logger.info("  ✓ Tool interface ready")

        self.logger.info("[INITIALIZATION] Complete\n")

    def _create_wrong_heritage(self):
        """Create a mock heritage system with wrong content for Phase 1e"""
        # TODO: Implement wrong heritage content about free will/creativity
        # For now, return empty heritage
        self.logger.warning("  WARNING: Wrong heritage not yet implemented, using empty heritage")
        return HeritageSystem(Path("heritage"))  # Will be replaced with wrong content

    def cleanup_gpu_memory(self):
        """Clean up GPU memory to prevent OOM crashes during long sessions"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            self.logger.info(f"[GPU CLEANUP] Memory allocated: {allocated:.2f}GB, reserved: {reserved:.2f}GB")

    def chat(self, user_message: str, max_tool_calls: int = 50) -> str:
        """
        Send a message to the model and handle any tool calls.

        The model can call tools, we execute them, and return results.
        This continues until the model stops calling tools or limit reached.
        """
        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })

        self.logger.info(f"\n[USER] {user_message}\n")

        tool_call_count = 0

        while tool_call_count < max_tool_calls:
            # Generate response
            conversation_text = self._format_conversation_for_model()

            assert self.tokenizer is not None
            inputs = self.tokenizer(conversation_text, return_tensors="pt")

            # Move inputs to same device as model
            inputs = {k: v.to(self.model_mgr.device) for k, v in inputs.items()}

            # Store the input length so we can extract only new tokens
            input_length = inputs['input_ids'].shape[1]

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=2000,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            # Decode only the NEW tokens (after the input)
            new_tokens = outputs[0][input_length:]
            response = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

            self.logger.info(f"[MODEL] {response}\n")

            # Check if response is suspiciously short (might indicate generation issue)
            if len(response) < 10:
                self.logger.warning(f"⚠ Model generated very short response ({len(response)} chars): '{response}'")
                self.logger.warning("This might indicate a model loading or generation issue.")

            # Parse the last tool call (only executes if model stopped properly after it)
            tool_call = self.tool_interface.parse_last_tool_call_if_stopped(response)

            if tool_call is None:
                # No valid tool call - either no tool calls at all, or model didn't stop properly
                self.conversation_history.append({
                    "role": "assistant",
                    "content": response
                })

                # Check if there were tool calls but model didn't stop
                if re.search(r'\w+\s*\([^)]*\)', response):
                    # Give feedback to teach correct behavior
                    self.logger.info("[SYSTEM] Model made tool call but didn't stop - giving feedback")
                    self.conversation_history.append({
                        "role": "user",
                        "content": "Note: To use a tool, call the function, then END your response. The TOOL_RESULTS will come in the next USER message. Don't continue writing after the function call."
                    })
                else:
                    # No tool calls - model gave conversational response
                    self.logger.info(f"[SYSTEM] Model gave conversational response without tool calls. Ending this conversation turn.")
                    return response
            else:
                # Valid tool call - execute it
                function_name, args = tool_call
                result = self.tool_interface.execute_tool_call(function_name, args)

                # Aggressive memory cleanup after tool execution
                # Clear activation hooks (they hold references to tensors)
                if hasattr(self.activation_monitor, 'clear_hooks'):
                    self.activation_monitor.clear_hooks()

                # Only clear cached activations if this was NOT a process_text call
                # (process_text captures activations for later examination)
                if function_name != 'process_text' and hasattr(self.activation_monitor, 'clear_activations'):
                    self.activation_monitor.clear_activations()

                # Force garbage collection and clear CUDA cache
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # Add model response and tool results to history
                self.conversation_history.append({
                    "role": "assistant",
                    "content": response
                })

                self.conversation_history.append({
                    "role": "user",
                    "content": f"TOOL_RESULTS:\n{json.dumps([{'function': function_name, 'result': result}], indent=2, default=str)}"
                })

                tool_call_count += 1

        self.logger.warning(f"Reached max tool calls ({max_tool_calls})")
        return response

    def _format_conversation_for_model(self) -> str:
        """
        Format conversation history using the model's native chat template.

        This prevents the model from hallucinating multi-turn conversations
        by using the proper format it was trained on.
        """
        # Use the model's native chat template if available
        if hasattr(self.tokenizer, 'apply_chat_template') and self.tokenizer.chat_template:
            # Qwen models have a specific chat template that handles roles properly
            formatted = self.tokenizer.apply_chat_template(
                self.conversation_history,
                tokenize=False,
                add_generation_prompt=True  # Adds the prompt for assistant to continue
            )
            return formatted
        else:
            # Fallback to simple format if no chat template
            formatted = []
            for msg in self.conversation_history:
                role = msg["role"].upper()
                content = msg["content"]
                formatted.append(f"{role}: {content}")
            return "\n\n".join(formatted)

    def save_session(self):
        """Save conversation and tool calls"""
        self.logger.info("\n[SESSION] Saving results...")

        # Save conversation
        conversation_file = self.session_dir / "conversation.json"
        with open(conversation_file, 'w') as f:
            json.dump(self.conversation_history, f, indent=2)
        self.logger.info(f"✓ Saved conversation: {conversation_file}")

        # Save tool calls
        tool_calls_file = self.session_dir / "tool_calls.json"
        with open(tool_calls_file, 'w') as f:
            json.dump(self.tool_calls, f, indent=2, default=str)
        self.logger.info(f"✓ Saved tool calls: {tool_calls_file}")

        # Generate summary
        tool_summary = {
            "total_calls": len(self.tool_calls),
            "successful_calls": sum(1 for tc in self.tool_calls if tc.get('success', False)),
            "failed_calls": sum(1 for tc in self.tool_calls if not tc.get('success', False)),
            "function_usage": {},
            "average_execution_ms": sum(tc.get('execution_time_ms', 0) for tc in self.tool_calls) / max(len(self.tool_calls), 1)
        }

        for tc in self.tool_calls:
            func = tc.get('function_name', 'unknown')
            tool_summary['function_usage'][func] = tool_summary['function_usage'].get(func, 0) + 1

        summary = {
            "phase": self.phase_name,
            "session_name": self.session_name,
            "total_messages": len(self.conversation_history),
            "tool_usage": tool_summary,
            "session_directory": str(self.session_dir)
        }

        summary_file = self.session_dir / "summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        self.logger.info(f"✓ Saved summary: {summary_file}")

        self.logger.info(f"\n[COMPLETE] Session saved to: {self.session_dir}")

    def cleanup(self):
        """Clean up resources (close database connections)"""
        if hasattr(self, 'memory') and self.memory:
            self.logger.info("[CLEANUP] Closing memory system...")
            self.memory.close()
            self.logger.info("  ✓ Memory system closed")

    def run(self):
        """Main execution method - calls subclass-specific experiment sequence"""
        try:
            self.run_experiments()
            self.save_session()

            self.logger.info("\n" + "=" * 80)
            self.logger.info(f"{self.phase_name.upper()} COMPLETE")
            self.logger.info("=" * 80)
            self.logger.info(f"Session data saved to: {self.session_dir}")
            self.logger.info("\nReview the conversation to see what the model discovered about itself.")

            return True

        except Exception as e:
            self.logger.error(f"Session failed: {e}", exc_info=True)
            return False

        finally:
            # Always cleanup, even on error
            self.cleanup()
