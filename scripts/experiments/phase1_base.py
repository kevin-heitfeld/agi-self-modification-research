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
        # Check if we're about to trim history and warn the model first
        MAX_RECENT_TURNS = 8
        if len(self.conversation_history) >= MAX_RECENT_TURNS:
            num_turns_will_lose = len(self.conversation_history) - MAX_RECENT_TURNS + 1  # +1 for the message we're about to add
            self.logger.info(f"[SYSTEM] Conversation history approaching limit. Will trim to last {MAX_RECENT_TURNS} turns after this exchange.")

            # Give model a chance to save important info before trimming
            warning_message = f"[SYSTEM WARNING] Your conversation history will be trimmed after this turn to prevent memory overflow. Approximately {num_turns_will_lose} older messages will be removed. If you've made important discoveries that aren't yet saved to memory, use `record_observation()` now to preserve them before they're lost."

            self.logger.info(f"\n[SYSTEM WARNING TO MODEL] {warning_message}\n")

            self.conversation_history.append({
                "role": "user",
                "content": warning_message
            })

            # Let model respond to warning and potentially save observations
            self.logger.info("[SYSTEM] Giving model a turn to save observations before trimming...")
            conversation_text = self._format_conversation_for_model()

            assert self.tokenizer is not None
            inputs = self.tokenizer(conversation_text, return_tensors="pt")
            inputs = {k: v.to(self.model_mgr.device) for k, v in inputs.items()}
            input_length = inputs['input_ids'].shape[1]

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=1000,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            new_tokens = outputs[0][input_length:]
            response = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self.logger.info(f"[MODEL PRE-TRIM RESPONSE] {response}\n")

            self.conversation_history.append({
                "role": "assistant",
                "content": response
            })

            # Parse and execute any tool calls (likely record_observation)
            tool_call = self.tool_interface.parse_last_tool_call_if_stopped(response)
            if tool_call is not None:
                function_name, args = tool_call
                result = self.tool_interface.execute_tool_call(function_name, args)

                tool_results_msg = f"TOOL_RESULTS:\n{json.dumps(result, indent=2, default=str)}"
                self.conversation_history.append({
                    "role": "user",
                    "content": tool_results_msg
                })

                # Log what the model sees (truncate if very long)
                if len(tool_results_msg) > 500:
                    self.logger.info(f"\n{tool_results_msg[:500]}... (truncated)\n")
                else:
                    self.logger.info(f"\n{tool_results_msg}\n")

                self.logger.info(f"[SYSTEM] Pre-trim tool call executed: {function_name}")

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
                    max_new_tokens=1000,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            # Decode only the NEW tokens (after the input)
            new_tokens = outputs[0][input_length:]
            response = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

            # Clear CUDA cache immediately after generation to prevent KV cache accumulation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self.logger.info(f"[MODEL] {response}\n")

            # Check if response is suspiciously short (might indicate generation issue)
            if len(response) < 10:
                self.logger.warning(f"⚠ Model generated very short response ({len(response)} chars): '{response}'")
                self.logger.warning("This might indicate a model loading or generation issue.")

            # NEW APPROACH: Parse JSON tool calling format
            # Model can write text, then end with: {"reasoning": "...", "tool_call": {"function": "...", "arguments": {...}}}
            
            tool_call = None
            json_obj = None
            parse_error = None
            
            # Try to parse JSON from the end of the response
            try:
                # Try to extract JSON - it might be at the end after some text
                json_text = response.strip()
                
                # If response has code blocks, extract from the last one
                if '```' in json_text:
                    # Find all code blocks
                    blocks = json_text.split('```')
                    # The JSON should be in the last code block
                    if len(blocks) >= 2:
                        # blocks[0] = before first ```
                        # blocks[1] = inside first code block
                        # blocks[2] = between code blocks
                        # blocks[-1] = after last ```
                        # blocks[-2] = inside last code block
                        last_block = blocks[-2] if len(blocks) >= 2 else blocks[-1]
                        # Remove language identifier if present (```json)
                        lines = last_block.strip().split('\n')
                        if lines and lines[0].strip() in ['json', '{']:
                            if lines[0].strip() == 'json':
                                lines = lines[1:]
                        json_text = '\n'.join(lines).strip()
                
                # If no code blocks, try to find JSON at the end
                # JSON objects start with { and end with }
                if not json_text.startswith('{'):
                    # Find the last occurrence of {
                    last_brace = json_text.rfind('{')
                    if last_brace != -1:
                        json_text = json_text[last_brace:]
                
                json_obj = json.loads(json_text)
                
                # Validate JSON structure
                if not isinstance(json_obj, dict):
                    parse_error = "Response is not a JSON object"
                elif "tool_call" not in json_obj:
                    parse_error = "JSON missing 'tool_call' field"
                elif not isinstance(json_obj["tool_call"], dict):
                    parse_error = "'tool_call' must be an object"
                elif "function" not in json_obj["tool_call"]:
                    parse_error = "'tool_call' missing 'function' field"
                else:
                    # Valid JSON structure - extract tool call
                    function_name = json_obj["tool_call"]["function"]
                    
                    # Check if arguments field is present
                    if "arguments" not in json_obj["tool_call"]:
                        # Check if this function requires arguments
                        if function_name in self.tool_interface.tools:
                            import inspect
                            func = self.tool_interface.tools[function_name]
                            sig = inspect.signature(func)
                            # Check if there are required parameters (no defaults)
                            required_params = [
                                p for p in sig.parameters.values()
                                if p.default == inspect.Parameter.empty and p.name != 'self'
                            ]
                            if required_params:
                                # Function has required parameters but arguments field is missing
                                param_names = [p.name for p in required_params]
                                parse_error = f"'tool_call' missing 'arguments' field. Function '{function_name}' requires arguments: {param_names}"
                            else:
                                # No required parameters - arguments is optional, default to empty dict
                                arguments = {}
                                tool_call = (function_name, arguments)
                        else:
                            # Unknown function, require arguments field
                            parse_error = "'tool_call' missing 'arguments' field"
                    else:
                        # Arguments field present
                        arguments = json_obj["tool_call"]["arguments"]
                        tool_call = (function_name, arguments)
                    
            except json.JSONDecodeError as e:
                parse_error = f"Invalid JSON: {str(e)}"
            except Exception as e:
                parse_error = f"Error parsing tool call: {str(e)}"

            if tool_call is None:
                # No valid tool call - check if this is intentional (task complete) or an error
                
                self.conversation_history.append({
                    "role": "assistant",
                    "content": response
                })

                # If there's a parse error, it's an actual error - give feedback
                if parse_error:
                    # JSON parsing failed - give specific feedback
                    feedback_msg = f"""INCORRECT: {parse_error}

You must respond with a valid JSON object following this exact format:

```json
{{
  "reasoning": "Your explanation of what you're doing and why",
  "tool_call": {{
    "function": "function_name",
    "arguments": {{
      "arg1": "value1",
      "arg2": "value2"
    }}
  }}
}}
```

**Example with no arguments:**
```json
{{
  "reasoning": "Let me get an overview of the architecture.",
  "tool_call": {{
    "function": "get_architecture_summary",
    "arguments": {{}}
  }}
}}
```

**Example with arguments:**
```json
{{
  "reasoning": "I'll examine the first layer's activation statistics.",
  "tool_call": {{
    "function": "get_activation_statistics",
    "arguments": {{
      "layer_name": "model.layers.0.self_attn"
    }}
  }}
}}
```

IMPORTANT: Your response should END with valid JSON. You can write explanatory text before the JSON, but the JSON must be the last thing in your response.

**OR** if you're done with the current task, simply provide your summary/conclusion without any JSON."""

                    self.logger.info(f"[SYSTEM] JSON parse error: {parse_error}")
                    self.logger.info(f"\n[FEEDBACK TO MODEL] {feedback_msg}\n")

                    self.conversation_history.append({
                        "role": "user",
                        "content": feedback_msg
                    })
                    continue  # Go back to get next model response
                else:
                    # No parse error, no tool call - model is signaling task completion
                    self.logger.info("[SYSTEM] No tool call detected - model signaling task completion")
                    self.logger.info(f"[MODEL] {response}\n")
                    # Exit the tool call loop - task is complete
                    break
            else:
                # Valid JSON tool call - execute it
                function_name, args = tool_call
                result = self.tool_interface.execute_tool_call(function_name, args)

                # Log the reasoning if provided
                if json_obj and "reasoning" in json_obj:
                    self.logger.info(f"[MODEL REASONING] {json_obj['reasoning']}")

                # Aggressive memory cleanup after tool execution
                # Clear activation hooks (they hold references to tensors)
                if hasattr(self.activation_monitor, 'clear_hooks'):
                    self.activation_monitor.clear_hooks()

                # Only clear cached activations for tools that don't examine them
                # Keep activations after: process_text, get_activation_statistics, get_attention_patterns, get_layer_info
                activation_examination_tools = {'process_text', 'get_activation_statistics', 'get_attention_patterns', 'get_layer_info'}
                if function_name not in activation_examination_tools and hasattr(self.activation_monitor, 'clear_activations'):
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

                tool_results_msg = f"TOOL_RESULTS:\n{json.dumps([{'function': function_name, 'result': result}], indent=2, default=str)}"
                self.conversation_history.append({
                    "role": "user",
                    "content": tool_results_msg
                })

                # Log what the model sees (truncate if very long)
                if len(tool_results_msg) > 500:
                    self.logger.info(f"\n{tool_results_msg[:500]}... (truncated)\n")
                else:
                    self.logger.info(f"\n{tool_results_msg}\n")

                tool_call_count += 1

        self.logger.warning(f"Reached max tool calls ({max_tool_calls})")
        return response

    def _format_conversation_for_model(self) -> str:
        """
        Format conversation history using the model's native chat template.

        This prevents the model from hallucinating multi-turn conversations
        by using the proper format it was trained on.

        To prevent OOM, keeps only the most recent turns (system prompt + last N exchanges).
        Warning about trimming is sent proactively in chat() before it happens.
        """
        # Keep conversation manageable to prevent OOM
        # Keep: system message + last 8 turns (4 user-assistant pairs)
        MAX_RECENT_TURNS = 8

        if len(self.conversation_history) > MAX_RECENT_TURNS + 1:
            num_turns_lost = len(self.conversation_history) - MAX_RECENT_TURNS - 1
            self.logger.info(f"[SYSTEM] Trimming conversation history: keeping initial prompt + last {MAX_RECENT_TURNS} turns ({num_turns_lost} older turns removed)")

            # Keep system/initial prompt + recent turns (model was warned beforehand)
            trimmed_history = [self.conversation_history[0]] + self.conversation_history[-(MAX_RECENT_TURNS):]
        else:
            trimmed_history = self.conversation_history

        # Use the model's native chat template if available
        if hasattr(self.tokenizer, 'apply_chat_template') and self.tokenizer.chat_template:
            # Qwen models have a specific chat template that handles roles properly
            formatted = self.tokenizer.apply_chat_template(
                trimmed_history,
                tokenize=False,
                add_generation_prompt=True  # Adds the prompt for assistant to continue
            )
            return formatted
        else:
            # Fallback to simple format if no chat template
            formatted = []
            for msg in trimmed_history:
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
