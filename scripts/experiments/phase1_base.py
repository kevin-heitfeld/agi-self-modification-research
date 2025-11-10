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
        return """🧠 MEMORY MANAGEMENT - HOW YOUR MEMORY WORKS:

**Your memory has two systems (like human memory):**

1. **Working Memory (this conversation):**
   - Holds recent context (last ~3 conversation turns)
   - **VERY limited capacity** - GPU memory constraint
   - Old turns are automatically pruned when conversation gets long
   - Think of this as your "active thoughts" or "scratch pad"
   - **IMPORTANT:** Keep responses concise! Memory fills quickly.

2. **Long-Term Memory (observations database):**
   - Unlimited capacity
   - Stores only what you explicitly save with record_observation()
   - Retrievable anytime with query_memory()
   - Think of this as your "research notes" or "lab notebook"

**CRITICAL: You must actively manage your memory!**

When you make important discoveries:
1. Use record_observation() to save findings to long-term memory
2. Include detailed descriptions and relevant data
3. Categorize properly (e.g., category="architecture", "activations", "weights")
4. **Be concise in your reasoning** - you have limited working memory

When conversation gets long (you'll receive warnings):
1. Save any unsaved important findings immediately
2. After pruning, old tool results disappear from working memory
3. Use query_memory() to retrieve previously saved observations

**Example workflow:**
```
Turn 1: Call get_architecture_summary()
Turn 2: Call record_observation(obs_type="INTROSPECTION", category="architecture", 
        description="Found 28 transformer layers with...", data={...})
Turn 3: Call get_activation_statistics(...)
Turn 4: Call record_observation() to save activation findings
...
Turn 8: [SYSTEM WARNING: Memory limit approaching]
Turn 9: Call record_observation() to save recent unsaved findings
Turn 10: [SYSTEM: Old turns pruned, working memory reset]
Turn 11: Call query_memory(category="architecture") to retrieve earlier findings
```

This is exactly how humans do research - we don't keep everything in our heads,
we write down important discoveries and look them up later!"""

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

        # Cache the system prompt ONCE for all future generations
        # This saves ~6000 tokens being repeated on every turn
        system_prompt_text = self.create_initial_prompt()

        # Format system prompt with chat template
        system_message = [{"role": "system", "content": system_prompt_text}]
        formatted_system = self.tokenizer.apply_chat_template(
            system_message,
            tokenize=False,
            add_generation_prompt=False
        )

        # Cache it
        self.generator.cache_system_prompt(formatted_system)
        self.logger.info(f"  ✓ Manual generator ready (cached {self.generator.system_prompt_length} tokens)")

        # Track the growing KV cache for multi-turn conversation
        # This starts as None, gets populated with system+turn1, then system+turn1+turn2, etc.
        self.conversation_kv_cache = None
        self.last_tool_called = None  # Track last tool to prevent immediate repetition

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

    def _prune_tool_result(self, result: Any, function_name: str) -> Any:
        """
        Intelligently prune verbose fields from tool results to save memory.
        
        Keeps: metadata, key findings, essential information
        Removes: verbose generated text, redundant data
        
        This is applied to OLD tool results when conversation gets long.
        Recent tool results are kept in full.
        """
        if not isinstance(result, dict):
            return result
        
        # Create a copy to avoid mutating the original
        pruned = result.copy()
        
        if function_name == "process_text":
            # The 'response' field contains the full generated text (can be 100+ tokens)
            # Model doesn't need this for analysis - it has the activations
            if "response" in pruned:
                response_text = pruned["response"]
                if len(response_text) > 200:
                    pruned["response"] = f"[Generated response of {len(response_text)} chars - removed to save memory]"
                    
        elif function_name in ["get_activation_statistics", "get_weight_statistics"]:
            # Statistics can be very detailed - keep summary, remove verbose details
            # This is acceptable for old results since model should have saved important findings
            pass  # Keep for now, could trim detailed stats if needed
            
        elif function_name == "get_layer_names":
            # Full list of 434+ layer names is huge
            if "layers" in pruned and isinstance(pruned["layers"], list) and len(pruned["layers"]) > 20:
                total = len(pruned["layers"])
                pruned["layers"] = f"[List of {total} layer names - removed to save memory. Use get_layer_names() again if needed]"
        
        return pruned

    def _estimate_conversation_tokens(self) -> int:
        """
        Estimate the number of tokens in the conversation history.
        
        This is a rough estimate: chars / 4 (approximation for tokenization)
        Used to decide when to reset KV cache.
        """
        total_chars = sum(len(msg.get("content", "")) for msg in self.conversation_history)
        return total_chars // 4  # Rough approximation: 1 token ≈ 4 chars

    def _reset_kv_cache_with_sliding_window(self, keep_recent_turns: int = 3):
        """
        Reset KV cache and trim conversation to recent turns only.
        
        This prevents OOM by maintaining a sliding window of context.
        The system prompt remains cached, so we only lose older conversation turns.
        
        Model should use record_observation() to save important findings before they
        slide out of the window.
        
        Args:
            keep_recent_turns: Number of recent conversation turns to keep (default: 3)
        """
        # Count conversation exchanges (user-assistant pairs)
        num_exchanges = len([m for m in self.conversation_history if m["role"] == "assistant"])
        
        if num_exchanges <= keep_recent_turns:
            # Not enough history to trim yet
            return
        
        # Calculate how many messages to keep (2 per exchange: user + assistant)
        keep_messages = keep_recent_turns * 2
        
        # Prune old tool results BEFORE trimming (so we keep metadata even from old turns)
        pruned_history = []
        for msg in self.conversation_history:
            if msg["role"] == "user" and msg["content"].startswith("TOOL_RESULTS:"):
                # This is a tool result - check if it's old enough to prune
                # Keep recent turns in full, prune older ones
                msg_index = self.conversation_history.index(msg)
                is_recent = msg_index >= len(self.conversation_history) - keep_messages
                
                if not is_recent:
                    # Old tool result - prune it
                    try:
                        tool_results = json.loads(msg["content"].replace("TOOL_RESULTS:\n", ""))
                        if isinstance(tool_results, list) and len(tool_results) > 0:
                            pruned_results = []
                            for tr in tool_results:
                                function_name = tr.get("function", "unknown")
                                result = tr.get("result", {})
                                pruned_result = self._prune_tool_result(result, function_name)
                                pruned_results.append({
                                    "function": function_name,
                                    "result": pruned_result
                                })
                            msg["content"] = f"TOOL_RESULTS:\n{json.dumps(pruned_results, indent=2, default=str)}"
                    except:
                        # If parsing fails, keep as is
                        pass
            pruned_history.append(msg)
        
        self.conversation_history = pruned_history
        
        # Trim to recent turns only
        self.conversation_history = self.conversation_history[-keep_messages:]
        
        # Discard the KV cache (system prompt cache remains)
        self.conversation_kv_cache = None
        
        self.logger.info(f"[MEMORY MANAGEMENT] Reset KV cache, keeping last {keep_recent_turns} turns")
        self.logger.info(f"[MEMORY MANAGEMENT] Pruned {num_exchanges - keep_recent_turns} old exchanges")
        self.logger.info(f"[MEMORY MANAGEMENT] Model should retrieve old findings with query_memory()")


    def chat(self, user_message: str, max_tool_calls: int = 50) -> str:
        """
        Send a message to the model and handle any tool calls.

        The model can call tools, we execute them, and return results.
        This continues until the model stops calling tools or limit reached.
        
        Implements sliding window context management to prevent OOM.
        """
        # Check conversation length and manage KV cache
        estimated_tokens = self._estimate_conversation_tokens()
        MAX_CONTEXT_TOKENS = 4000  # Reduced from 8000 - be more aggressive to prevent OOM
        
        if estimated_tokens > MAX_CONTEXT_TOKENS:
            # Conversation is getting too long - time to prune and reset cache
            num_exchanges = len([m for m in self.conversation_history if m["role"] == "assistant"])
            
            self.logger.warning(f"\n[MEMORY MANAGEMENT] Conversation has ~{estimated_tokens} tokens")
            self.logger.warning(f"[MEMORY MANAGEMENT] This exceeds safe limit of {MAX_CONTEXT_TOKENS} tokens")
            
            # Warn model to save important findings before we prune
            warning_message = f"""⚠️ MEMORY LIMIT REACHED

Your conversation history has grown to approximately {estimated_tokens} tokens.
To prevent out-of-memory errors, I need to prune old conversation turns.

**ACTION REQUIRED:**
1. Use record_observation() NOW to save any important discoveries you haven't yet saved
2. After this turn, old tool results will be removed from your working memory
3. You can retrieve saved observations later with query_memory()

**Think of this like human memory:**
- Working memory (this conversation): Limited, recent context only
- Long-term memory (observations): Unlimited, stores what you explicitly save

Please save your important findings now, then confirm "Ready for pruning" or continue your investigation."""

            self.logger.info(f"\n[SYSTEM WARNING] Sent memory warning to model\n")
            
            self.conversation_history.append({
                "role": "user",
                "content": warning_message
            })

            # Let model respond and potentially save observations
            conversation_text = self._format_conversation_for_model()
            result = self.generator.generate(
                prompt=conversation_text,
                max_new_tokens=300,  # Balance: enough for complete responses, not too large for OOM
                temperature=0.7,
                do_sample=True,
                past_key_values=self.conversation_kv_cache,
                return_cache=True
            )

            response = result["generated_text"]
            self.conversation_kv_cache = result.get("past_key_values")
            
            self.logger.info(f"[MODEL PRE-PRUNING RESPONSE] {response}\n")

            self.conversation_history.append({
                "role": "assistant",
                "content": response
            })

            # Execute any tool calls (likely record_observation)
            tool_call = self.tool_interface.parse_last_tool_call_if_stopped(response)
            if tool_call is not None:
                function_name, args = tool_call
                result = self.tool_interface.execute_tool_call(function_name, args)
                
                tool_results_msg = f"TOOL_RESULTS:\n{json.dumps(result, indent=2, default=str)}"
                self.conversation_history.append({
                    "role": "user",
                    "content": tool_results_msg
                })
                
                self.logger.info(f"[PRE-PRUNING TOOL CALL] {function_name} executed\n")

            # NOW prune the conversation and reset KV cache
            self._reset_kv_cache_with_sliding_window(keep_recent_turns=3)  # Reduced from 5 to 3
            
            # Clear CUDA cache after pruning
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()

        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })

        self.logger.info(f"\n[USER] {user_message}\n")

        tool_call_count = 0
        confirmation_attempts = 0  # Track how many times we've asked for clarification

        while tool_call_count < max_tool_calls:
            # Generate response
            conversation_text = self._format_conversation_for_model()

            # Use manual generator with KV caching
            # If we have a conversation cache, use it (includes system + all previous turns)
            # Otherwise, it will use just the system prompt cache
            result = self.generator.generate(
                prompt=conversation_text,
                max_new_tokens=300,  # Balance: enough for analysis + tool call, not too large for OOM
                temperature=0.7,
                do_sample=True,
                past_key_values=self.conversation_kv_cache,
                return_cache=True  # Get the updated cache back
            )

            response = result["generated_text"]
            num_tokens = result["num_tokens"]
            cache_used = result["cache_used"]

            # Update the conversation KV cache for next turn
            self.conversation_kv_cache = result.get("past_key_values")

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
                    # Check if we've already asked for confirmation once
                    if confirmation_attempts >= 1:
                        # Already asked once and still no valid tool call - assume done
                        self.logger.info("[SYSTEM] No tool call after confirmation request - assuming task complete")
                        self.logger.info(f"[MODEL] {response}\n")
                        break
                    else:
                        # First time - ask for clarification
                        confirmation_attempts += 1
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

                # Update last tool called and reset confirmation counter
                self.last_tool_called = function_name
                confirmation_attempts = 0  # Reset counter on successful tool call

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

                # Truncate long responses from process_text to prevent OOM
                # The 'response' field can be hundreds of tokens and isn't needed for introspection
                if function_name == "process_text" and isinstance(result, dict) and "response" in result:
                    response_text = result["response"]
                    if len(response_text) > 200:  # Truncate if longer than ~50 tokens
                        result["response"] = response_text[:200] + "... (truncated - full response not needed for activation analysis)"
                        self.logger.info(f"[MEMORY OPTIMIZATION] Truncated process_text response from {len(response_text)} to 200 chars")

                # Add tool results (with truncation applied above if needed)
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
        Format conversation history for model generation.

        System prompt is cached, so we only format the conversation turns
        (user and assistant messages).

        Returns:
            str: Formatted conversation text (WITHOUT system prompt)
        """
        # Get conversation without system (system is cached)
        conversation_without_system = [
            msg for msg in self.conversation_history
            if msg["role"] != "system"
        ]

        # Token budget management
        MAX_CONTEXT_TOKENS = 8000
        KEEP_RECENT_EXCHANGES = 2

        trimmed_history = conversation_without_system

        # Trim if needed
        if len(conversation_without_system) > KEEP_RECENT_EXCHANGES * 2:
            approx_tokens = sum(len(msg["content"]) for msg in conversation_without_system) // 4

            if approx_tokens > MAX_CONTEXT_TOKENS:
                recent_messages = conversation_without_system[-(KEEP_RECENT_EXCHANGES * 2):]
                keep_older = conversation_without_system[:-(KEEP_RECENT_EXCHANGES * 2)]

                pruned_older = []
                for msg in keep_older:
                    if msg["role"] == "user" and msg["content"].startswith("TOOL_RESULTS:"):
                        pruned_older.append({
                            "role": "user",
                            "content": "[TOOL_RESULTS removed to save memory - you should have saved important findings to memory]"
                        })
                    else:
                        pruned_older.append(msg)

                trimmed_history = pruned_older + recent_messages
                num_removed = len(conversation_without_system) - len(trimmed_history)
                self.logger.info(f"[MEMORY OPTIMIZATION] Removed {num_removed} old messages")

        # Format with chat template (no system message, just conversation)
        formatted = self.tokenizer.apply_chat_template(
            trimmed_history,
            tokenize=False,
            add_generation_prompt=True
        )

        return formatted

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
