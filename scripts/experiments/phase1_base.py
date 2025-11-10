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
from src.manual_generation import ManualGenerator

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

    def get_memory_management_instructions(self) -> str:
        """
        Return memory management instructions for the model.

        This teaches the model to use record_observation() proactively
        to prevent data loss when old tool results are pruned.
        """
        return """MEMORY MANAGEMENT STRATEGY:
This investigation will span many tool calls. To prevent memory overflow:
- After every 2-3 tool calls, use record_observation() to save important discoveries
- Old tool results will be automatically removed from context after ~5 turns
- You can retrieve saved observations later using query_memory()

Example workflow:
1. Call get_architecture_summary() → examine results
2. Call record_observation(obs_type="INTROSPECTION", category="architecture", description="Key finding...", ...)
3. Call get_activation_statistics(...) → examine results
4. Call record_observation(...) to save findings
5. Continue investigation using saved observations as needed"""

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

        # Initialize manual generator with KV caching
        self.generator = ManualGenerator(
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.model_mgr.device
        )

        # CRITICAL FIX: Modify chat template to NOT inject default system message
        # Problem: Qwen's chat template adds "You are Qwen, created by Alibaba Cloud..."
        # if no system message is present. This interferes with our system prompt caching.
        #
        # Solution: Modify tokenizer.chat_template to remove default injection
        original_template = self.tokenizer.chat_template
        modified_template = original_template.replace(
            "        {{- '<|im_start|>system\\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\\n' }}",
            "        {# No default system message - using cached custom system prompt #}"
        )
        # Also modify the tools section
        modified_template = modified_template.replace(
            "        {{- 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.' }}",
            "        {{- '' }}"
        )
        self.tokenizer.chat_template = modified_template

        # Now cache the system prompt ONCE for all future generations
        # This saves ~6000 tokens being repeated on every turn
        system_prompt_text = self.create_initial_prompt()

        # Format system prompt with chat template
        system_message = [{"role": "system", "content": system_prompt_text}]
        formatted_system = self.tokenizer.apply_chat_template(
            system_message,
            tokenize=False,
            add_generation_prompt=False
        )

        self.generator.cache_system_prompt(formatted_system)
        self.logger.info(f"  ✓ Manual generator ready (cached {self.generator.system_prompt_length} tokens, modified template to prevent default system injection)")

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
        # Check if we're approaching the limit where tool results will be pruned
        # Warn EARLY and OFTEN - OOM can happen quickly with large tool results
        FIRST_WARNING_AT = 3  # Warn after just 3 exchanges (was 5, but OOM happens at ~4-5)
        num_exchanges = len([m for m in self.conversation_history if m["role"] == "assistant"])

        # Warn at turn 3, then every 2 turns (3, 5, 7, 9, 11...)
        # This ensures model gets warning BEFORE OOM, with frequent reminders
        should_warn = num_exchanges >= FIRST_WARNING_AT and (num_exchanges - FIRST_WARNING_AT) % 2 == 0

        if should_warn:
            # Warn model periodically that old tool results are being pruned
            warning_message = f"""[SYSTEM WARNING] Memory limit approaching!

You've made {num_exchanges} investigation turns. To prevent data loss:
1. Use record_observation() NOW to save any important discoveries from recent tool results
2. Old tool results will be removed from context after this turn
3. You can query saved observations later with query_memory()

IMPORTANT: If you don't save your findings now, they'll be lost forever!
Take this turn to record_observation() for any important discoveries you haven't yet saved."""

            self.logger.info(f"\n[SYSTEM WARNING TO MODEL] Turn {num_exchanges} - {warning_message}\n")

            self.conversation_history.append({
                "role": "user",
                "content": warning_message
            })

            # Let model respond to warning and potentially save observations
            self.logger.info("[SYSTEM] Giving model a turn to save observations...")
            conversation_text = self._format_conversation_for_model()

            # Use manual generator (system prompt already cached!)
            result = self.generator.generate(
                prompt=conversation_text,
                max_new_tokens=500,
                temperature=0.7,
                do_sample=True
            )

            response = result["generated_text"]
            num_tokens = result["num_tokens"]
            cache_used = result["cache_used"]

            self.logger.info(f"[GENERATION] Generated {num_tokens} tokens, cache used: {cache_used}")

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

            # DEBUG: Log what we're sending to the generator
            self.logger.info(f"[DEBUG] Conversation text length: {len(conversation_text)} chars")
            self.logger.info(f"[DEBUG] Conversation text:\n{conversation_text}")

            # Use manual generator (system prompt cached, template modified to prevent default injection)
            result = self.generator.generate(
                prompt=conversation_text,
                max_new_tokens=500,
                temperature=0.7,
                do_sample=True
            )

            response = result["generated_text"]
            num_tokens = result["num_tokens"]
            cache_used = result["cache_used"]

            self.logger.info(f"[GENERATION] Generated {num_tokens} tokens, cache used: {cache_used}")

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

                # If no code blocks, JSON must be at the very end
                # Find the start of the JSON object by looking for the outermost {
                if not json_text.startswith('{'):
                    # We need to find where the JSON object starts
                    # Strategy: try each '{' from the end and see if we can parse valid JSON from there
                    brace_positions = [i for i, char in enumerate(json_text) if char == '{']

                    if not brace_positions:
                        raise ValueError("No JSON object found in response")

                    # Try from the last { backwards to find the start of a valid JSON object
                    json_found = False
                    for pos in reversed(brace_positions):
                        candidate = json_text[pos:]
                        # Try to find the matching closing brace
                        brace_count = 0
                        json_end = -1
                        for i, char in enumerate(candidate):
                            if char == '{':
                                brace_count += 1
                            elif char == '}':
                                brace_count -= 1
                                if brace_count == 0:
                                    json_end = i + 1
                                    break

                        if json_end > 0:
                            # Found a complete {...} structure, use this
                            json_text = candidate[:json_end]
                            json_found = True
                            break

                    if not json_found:
                        raise ValueError("Could not find complete JSON object in response")
                else:
                    # json_text already starts with {, just need to find the matching }
                    brace_count = 0
                    json_end = -1
                    for i, char in enumerate(json_text):
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                json_end = i + 1
                                break

                    if json_end > 0:
                        json_text = json_text[:json_end]

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
                # Log the extracted JSON text for debugging
                self.logger.debug(f"[DEBUG] Extracted JSON text that failed to parse:\n{json_text}")
                self.logger.debug(f"[DEBUG] JSON text length: {len(json_text)}, first 200 chars: {json_text[:200]}")
            except Exception as e:
                parse_error = f"Error parsing tool call: {str(e)}"

            if tool_call is None:
                # No valid tool call - check if this is intentional (task complete) or an error

                self.conversation_history.append({
                    "role": "assistant",
                    "content": response
                })

                # If there's a parse error, ask model to clarify intent
                if parse_error:
                    # Check if we already asked for confirmation (to avoid infinite loop)
                    last_user_message = self.conversation_history[-2]["content"] if len(self.conversation_history) >= 2 else ""
                    already_asked_confirmation = "Are you done with this investigation" in last_user_message

                    if already_asked_confirmation:
                        # Model didn't clarify even after being asked - assume done
                        self.logger.info("[SYSTEM] No tool call after confirmation request - assuming task complete")
                        self.logger.info(f"[MODEL] {response}\n")
                        break
                    else:
                        # First time - ask for clarification
                        feedback_msg = f"""No tool call detected in your response.

**Are you:**
A) ✅ Done with this investigation (ready to move to next experiment)
B) ❌ Forgot to include a tool call (want to continue investigating)

**Please respond:**
- If **DONE (A)**: Just confirm "I'm done" or provide your summary
- If **FORGOT (B)**: Include your tool call in JSON format:

```json
{{
  "reasoning": "What I want to do next",
  "tool_call": {{
    "function": "function_name",
    "arguments": {{...}}
  }}
}}
```

Your previous response had: "{parse_error}"
"""

                        self.logger.info(f"[SYSTEM] No tool call detected, asking for clarification: {parse_error}")
                        self.logger.info(f"\n[FEEDBACK TO MODEL] Asking for clarification\n")

                        self.conversation_history.append({
                            "role": "user",
                            "content": feedback_msg
                        })
                        continue  # Go back to get clarification
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

                # Add full tool results (we'll prune old ones later to prevent OOM)
                tool_results_json = json.dumps([{'function': function_name, 'result': result}], indent=2, default=str)
                tool_results_msg = f"TOOL_RESULTS:\n{tool_results_json}"

                self.conversation_history.append({
                    "role": "user",
                    "content": tool_results_msg
                })

                # Log what the model sees (truncate if very long)
                if len(tool_results_msg) > 500:
                    self.logger.info(f"\n{tool_results_msg[:500]}... (truncated in log only)\n")
                else:
                    self.logger.info(f"\n{tool_results_msg}\n")

                tool_call_count += 1

        self.logger.warning(f"Reached max tool calls ({max_tool_calls})")
        return response

    def _format_conversation_for_model(self) -> str:
        """
        Format conversation history for manual generation.

        NOTE: System prompt is NOW CACHED in ManualGenerator and NOT included here!
        This saves massive amounts of memory (~6000 tokens per turn).

        We only format the recent conversation (user/assistant exchanges).
        The model's chat template will be applied by tokenizer.apply_chat_template.

        IMPORTANT: We modified the chat template to NOT inject a default system message,
        so the conversation can start with a user message without any issues.

        To prevent OOM, implements smart pruning:
        - System prompt: CACHED ONCE (not included in conversation)
        - Always keep: Last 2 full exchanges (4 messages - full tool results preserved)
        - Older messages: Keep model responses, REMOVE tool results (replace with notice)

        Rationale: Model should have saved important discoveries to memory.
        Recent tool results are needed for active analysis.
        Old tool results just bloat context without adding value.
        """
        # Keep recent exchanges fully intact
        KEEP_RECENT_EXCHANGES = 2  # Last 2 exchanges = 4 messages (2 assistant + 2 tool results)
        KEEP_RECENT_MESSAGES = KEEP_RECENT_EXCHANGES * 2

        # Maximum total conversation length (WITHOUT system prompt - it's cached!)
        MAX_TOTAL_TURNS = 6

        # IMPORTANT: Skip system prompt (index 0) - it's cached in ManualGenerator
        conversation_without_system = self.conversation_history[1:] if len(self.conversation_history) > 1 else []

        # If conversation is empty (only system prompt exists), return empty string
        if len(conversation_without_system) == 0:
            return ""

        if len(conversation_without_system) <= MAX_TOTAL_TURNS:
            # Short conversation - no pruning needed
            trimmed_history = conversation_without_system
        else:
            # Long conversation - smart pruning
            recent_messages = conversation_without_system[-KEEP_RECENT_MESSAGES:]

            # Calculate how many older messages we can keep (with tool results pruned)
            available_slots = MAX_TOTAL_TURNS - KEEP_RECENT_MESSAGES

            if available_slots > 0:
                # We have room for some pruned older messages
                older_messages = conversation_without_system[:-KEEP_RECENT_MESSAGES]

                # Take the most recent older messages and prune their tool results
                keep_older = older_messages[-available_slots:] if len(older_messages) > available_slots else older_messages

                pruned_older = []
                for msg in keep_older:
                    if msg["role"] == "user" and msg["content"].startswith("TOOL_RESULTS:"):
                        # Replace tool result with notice
                        pruned_older.append({
                            "role": "user",
                            "content": "[TOOL_RESULTS removed to save memory - you should have saved important findings to memory]"
                        })
                    else:
                        # Keep assistant messages (reasoning) and other user messages
                        pruned_older.append(msg)

                trimmed_history = pruned_older + recent_messages

                # Log the pruning
                num_removed = len(conversation_without_system) - len(trimmed_history)
                num_tool_results_pruned = sum(1 for msg in keep_older if msg["role"] == "user" and msg["content"].startswith("TOOL_RESULTS:"))
                self.logger.info(f"[MEMORY OPTIMIZATION] Removed {num_removed} old messages, pruned {num_tool_results_pruned} old tool results (kept last {KEEP_RECENT_EXCHANGES} exchanges fully intact, system prompt cached separately)")
            else:
                # No room for older messages - just keep recent
                trimmed_history = recent_messages
                num_removed = len(conversation_without_system) - len(trimmed_history)
                self.logger.info(f"[MEMORY OPTIMIZATION] Removed {num_removed} old messages (kept last {KEEP_RECENT_EXCHANGES} exchanges, system prompt cached separately)")

        # CRITICAL: When system prompt is cached, we must NOT use apply_chat_template
        # because it will inject an empty <|im_start|>system<|im_end|> block,
        # which confuses the model and causes gibberish generation!
        #
        # Instead, manually format the conversation turns using the chat format
        # WITHOUT any system message block.

        # Qwen chat format: <|im_start|>role\ncontent<|im_end|>\n
        formatted_parts = []
        for msg in trimmed_history:
            role = msg["role"]  # "user" or "assistant"
            content = msg["content"]
            # Each message includes its trailing newline
            formatted_parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")

        # Add generation prompt for assistant to continue
        # Format: <|im_start|>assistant\n (with single newline, no trailing newline)
        formatted_parts.append("<|im_start|>assistant\n")

        # Join without separator since each part already has proper formatting
        return "".join(formatted_parts)

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
