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
from src.memory_manager import MemoryManager
from src.gpu_monitor import GPUMonitor


# Qwen Chat Template Formatting Helper
def format_qwen_chat(messages: List[Dict[str, str]], add_generation_prompt: bool = False) -> str:
    """
    Manually format messages using Qwen chat template format.

    This ensures we have complete control over the formatting and avoids
    apply_chat_template() injecting unwanted default system prompts.

    Args:
        messages: List of message dicts with 'role' and 'content' keys
                  Roles can be: 'system', 'user', 'assistant'
        add_generation_prompt: If True, add '<|im_start|>assistant\n' at the end
                               to prompt the model to generate a response

    Returns:
        Formatted string in Qwen chat format

    Example:
        >>> format_qwen_chat([{"role": "system", "content": "You are helpful"}])
        '<|im_start|>system\\nYou are helpful<|im_end|>\\n'

        >>> format_qwen_chat([
        ...     {"role": "user", "content": "Hi"},
        ...     {"role": "assistant", "content": "Hello!"}
        ... ], add_generation_prompt=True)
        '<|im_start|>user\\nHi<|im_end|>\\n<|im_start|>assistant\\nHello!<|im_end|>\\n<|im_start|>assistant\\n'
    """
    formatted = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        formatted += f"<|im_start|>{role}\n{content}<|im_end|>\n"

    if add_generation_prompt:
        formatted += "<|im_start|>assistant\n"

    return formatted


# Setup logging
def setup_logging(phase_name: str):
    """Setup logging for a specific phase"""
    log_dir = Path("data/logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    # Get logger for this module
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)  # Revert to INFO for normal runs

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

        # Initialize GPU memory monitor (T4 GPU = 15GB total)
        self.gpu_monitor = GPUMonitor(logger=self.logger, gpu_total_gb=15.0)

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

    def get_experiment_session_context(self) -> str:
        """
        Provide upfront context about the multi-experiment structure.

        This helps the model understand:
        1. There are 3 separate experiments
        2. Context is reset between experiments
        3. record_observation() is the ONLY way to persist findings
        """
        return """🔬 EXPERIMENT SESSION STRUCTURE:

**You will conduct 3 SEQUENTIAL EXPERIMENTS in this session:**

1. **Experiment 1: Architecture Examination**
   - Investigate your model structure, layers, parameters
   - Document architectural findings

2. **Experiment 2: Activation Analysis**
   - Analyze your activation patterns during processing
   - Study how information flows through layers

3. **Experiment 3: Consciousness Investigation**
   - Explore questions of self-awareness and subjective experience
   - Synthesize findings from previous experiments

**CRITICAL - CONTEXT RESET BETWEEN EXPERIMENTS:**

After EACH experiment completes, your working memory (this conversation) will be
**COMPLETELY RESET**. You will start the next experiment with a fresh context.

❗ **The ONLY way to preserve findings between experiments is record_observation()!**

When you say "I'm done with this experiment", the system will:
1. END the current experiment immediately
2. CLEAR your working memory (conversation history)
3. START the next experiment with FRESH context
4. **ALL unsaved findings in working memory will be PERMANENTLY LOST**

**Strategy for Success:**
- Use query_memory() at the START of each new experiment
- Retrieve relevant findings from previous experiments
- Build on earlier discoveries
- Use record_observation() FREQUENTLY as you discover things
- Don't wait until the end - save incrementally

**Think of it like a multi-day research project:**
- Each experiment is a "day" of research
- At the end of each day, you write findings in your lab notebook (record_observation)
- The next day, you read your notes (query_memory) to continue where you left off
- You can't rely on your "working memory" from yesterday - only your written notes!

"""

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
   - **RESPONSE LIMIT: Maximum 450 tokens per response**
     - **CRITICAL**: Your responses will be hard-cut at 450 tokens
     - Incomplete JSON will cause errors and corrupt future responses
     - Always finish your JSON tool calls within the limit
     - Token budget: Reasoning (~100-150) + JSON (~150-250) + Buffer (~50)

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
5. If you have extensive analysis, save it to memory and provide a brief summary

When conversation gets long (you'll receive warnings):
1. Save any unsaved important findings immediately
2. After pruning, old tool results disappear from working memory
3. Use query_memory() to retrieve previously saved observations

**Response Planning Tips:**
- **ALWAYS complete your JSON** - incomplete JSON breaks everything
- Keep reasoning focused (~100-150 tokens maximum)
- Tool calls with arguments: ~150-250 tokens
- Leave ~50 token buffer to ensure JSON closes properly
- For complex data, use record_observation() first, then just reference it
- **If your response approaches 450 tokens, STOP and finish the JSON immediately**

**Example workflow:**
```
Turn 1: Call get_architecture_summary()
Turn 2: Brief analysis (100 tokens) + record_observation() with full details (in data field)
Turn 3: Call get_activation_statistics(...)
Turn 4: Brief findings (150 tokens) + record_observation() to save detailed analysis
...
Turn 8: [SYSTEM WARNING: Memory limit approaching]
Turn 9: Call record_observation() to save recent unsaved findings
Turn 10: [SYSTEM: Old turns pruned, working memory reset]
Turn 11: Call query_memory() to retrieve earlier findings
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

        # Defense-in-depth: Modify chat template to NOT inject default system message
        # This prevents accidental injection if apply_chat_template is used anywhere.
        # However, we now use manual formatting everywhere to have complete control.
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

        # Format system prompt manually using our helper function
        # This ensures consistent formatting with conversation turns
        formatted_system = format_qwen_chat([{"role": "system", "content": system_prompt_text}])

        # Cache it
        self.generator.cache_system_prompt(formatted_system)
        self.system_prompt_tokens = self.generator.system_prompt_length  # Store for memory calculations
        self.logger.info(f"  ✓ Manual generator ready (cached {self.system_prompt_tokens} tokens)")

        # Track the growing KV cache for multi-turn conversation
        # This starts as None, gets populated with system+turn1, then system+turn1+turn2, etc.
        self.conversation_kv_cache = None
        self.last_tool_called = None  # Track last tool to prevent immediate repetition

        # Initialize memory manager for conversation pruning
        self.memory_manager = MemoryManager(logger=self.logger)

        self.logger.info("[INITIALIZATION] Complete\n")

        # Capture GPU memory after full initialization
        self.gpu_monitor.snapshot("after_model_load")

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

    def reset_conversation(self):
        """Reset conversation history and cache between experiments"""
        # Keep only the system message
        system_msg = [msg for msg in self.conversation_history if msg["role"] == "system"]
        self.conversation_history = system_msg

        # CRITICAL: Clear the conversation KV cache completely
        # This forces the next generation to start fresh with ONLY system prompt cache
        # Without this, position IDs get misaligned because the generator expects
        # the cache to match the conversation history length
        self.conversation_kv_cache = None

        # NOTE: The system prompt cache (self.generator.system_prompt_cache) remains,
        # so the next generation will correctly use system cache + new input,
        # with position IDs calculated from system_prompt_length

        self.logger.info("[RESET] Conversation history cleared for new experiment (system prompt retained)")

    def _count_observations_in_current_experiment(self) -> int:
        """
        Count how many record_observation() calls were made in the current experiment.

        This looks through conversation history for successful record_observation calls.
        """
        count = 0
        for msg in self.conversation_history:
            if msg["role"] == "user" and "TOOL_RESULTS:" in msg["content"]:
                # Check if this was a record_observation call
                if '"function": "record_observation"' in msg["content"]:
                    # Check if it was successful (has result, not error)
                    if '"result":' in msg["content"] and '"error"' not in msg["content"]:
                        count += 1
        return count

    def chat(self, user_message: str, max_tool_calls: int = 50) -> str:
        """
        Send a message to the model and handle any tool calls.

        The model can call tools, we execute them, and return results.
        This continues until the model stops calling tools or limit reached.

        Implements sliding window context management to prevent OOM.

        NOTE: Memory management now happens INSIDE the tool loop (not here at start),
        since OOM occurs during generation in the loop, not between chat() calls.
        """
        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })

        self.logger.info(f"\n[USER] {user_message}\n")

        tool_call_count = 0
        confirmation_attempts = 0  # Track how many times we've asked for clarification
        response = ""  # Initialize response in case we break early
        generated_in_this_call = False  # Track if we've generated at least once
        turns_in_this_session = 0  # Track assistant turns within THIS chat() session only
        tools_since_last_save = 0  # Track non-save tool calls to remind about record_observation()

        while tool_call_count < max_tool_calls:
            # CRITICAL: Check memory BEFORE each generation (not just at chat() start)
            # OOM happens during generation, so we need to check in the tool loop
            # BUT: Only check AFTER we've generated at least once in this chat() call
            # Otherwise we might break immediately without responding to the current user message
            if generated_in_this_call:
                should_prune, reasons = self.memory_manager.should_prune_memory(
                    self.conversation_history,
                    max_conversation_tokens=2000,
                    max_turns_before_clear=3,
                    current_session_turns=turns_in_this_session  # Pass session-specific count
                )

                # ADDITIONAL CHECK: Force pruning if KV cache is too large
                # This prevents OOM even if turn count is low (e.g., after previous pruning)
                if self.conversation_kv_cache is not None:
                    cache_length = self.conversation_kv_cache[0][0].shape[2]
                    max_cache_length = 15000  # Safety limit: 15K tokens (conservative for T4 GPU)
                    if cache_length > max_cache_length:
                        should_prune = True
                        reasons.append(f"cache size ({cache_length} > {max_cache_length} tokens)")
                        self.logger.warning(f"[MEMORY MANAGEMENT] Cache size threshold exceeded: {cache_length} tokens")

                if should_prune:
                    self.memory_manager.log_memory_pruning(reasons, keep_recent_turns=2)

                    # Reset conversation and trim to recent turns
                    self.conversation_history = self.memory_manager.reset_conversation_with_sliding_window(
                        self.conversation_history,
                        keep_recent_turns=2
                    )

                    # Completely discard the KV cache to prevent corruption
                    # The system prompt cache remains in the generator
                    self.conversation_kv_cache = None

                    # Force garbage collection to free the old cache
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    # Reset turn counter for this session since we've pruned history
                    turns_in_this_session = 0

                    # DON'T add notification to history - it would desync from KV cache
                    # Model will naturally continue with reduced context
                    # The system prompt explains memory can be pruned and to use query_memory()
                    self.logger.info(f"[MEMORY MANAGEMENT] Pruning complete, model will continue with reduced context")

                    # Verify cache was cleared (no noisy debug output)
                    self.logger.info(f"[MEMORY MANAGEMENT] conversation_kv_cache cleared after pruning")
                    self.logger.info(f"[MEMORY MANAGEMENT] Conversation history length: {len(self.conversation_history)}")

                    # Capture memory after pruning
                    self.gpu_monitor.snapshot("after_pruning", {
                        "conversation_turns": len([m for m in self.conversation_history if m["role"] == "assistant"])
                    })

                    # Continue the tool loop - model has reduced context but can keep investigating

            # Generate response
            conversation_text = self._format_conversation_for_model()

            # Debug: Log formatted conversation to diagnose corruption
            if conversation_text:
                self.logger.debug(f"[DEBUG] Formatted conversation ({len(conversation_text)} chars)")
                if len(conversation_text) < 1000:
                    self.logger.debug(f"[DEBUG] Full conversation:\n{conversation_text}")
                else:
                    self.logger.debug(f"[DEBUG] First 300 chars: {conversation_text[:300]}")
                    self.logger.debug(f"[DEBUG] Last 300 chars: {conversation_text[-300:]}")

            # Use manual generator with KV caching
            # If we have a conversation cache, use it (includes system + all previous turns)
            # Otherwise, it will use just the system prompt cache
            # Calling generator (INFO level logs only)
            if self.conversation_kv_cache is not None:
                cache_len = self.conversation_kv_cache[0][0].shape[2]
                self.logger.info(f"Using conversation KV cache (length: {cache_len} tokens)")
            else:
                self.logger.info(f"No conversation cache, will use system cache ({self.system_prompt_tokens} tokens)")

            # Prepare prompt token count for telemetry (no noisy debug)
            prompt_token_count = len(self.generator.tokenizer.encode(conversation_text))
            self.logger.info(f"Prompt token count: {prompt_token_count}")

            # Capture memory before generation
            num_assistant_turns = len([m for m in self.conversation_history if m["role"] == "assistant"])
            self.gpu_monitor.snapshot("generation_start", {
                "conversation_turns": num_assistant_turns,
                "prompt_tokens": prompt_token_count
            })

            result = self.generator.generate(
                prompt=conversation_text,
                max_new_tokens=450,  # Increased slightly to reduce truncation (from 400 to 450)
                temperature=0.7,
                do_sample=True,
                past_key_values=self.conversation_kv_cache,
                return_cache=True  # Get the updated cache back
            )

            response = result["generated_text"]
            num_tokens = result["num_tokens"]
            cache_used = result["cache_used"]

            # Mark that we've generated at least once in this chat() call
            generated_in_this_call = True

            # Update the conversation KV cache for next turn
            self.conversation_kv_cache = result.get("past_key_values")

            self.logger.info(f"[GENERATION] Generated {num_tokens} tokens, cache used: {cache_used}")

            # Capture memory after generation
            self.gpu_monitor.snapshot("generation_end", {
                "conversation_turns": num_assistant_turns + 1,
                "generated_tokens": num_tokens
            })

            # Clear CUDA cache immediately after generation to prevent KV cache accumulation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self.logger.info(f"[MODEL] {response}\n")

            # Check if response is suspiciously short (might indicate generation issue)
            if len(response) < 10:
                self.logger.warning(f"⚠ Model generated very short response ({len(response)} chars): '{response}'")
                self.logger.warning("This might indicate a model loading or generation issue.")

            # Check if generation was truncated at token limit
            # Truncation often results in incomplete JSON that can't be parsed
            if num_tokens >= 400:  # Hit the max_new_tokens limit
                self.logger.warning(f"⚠ Generation truncated at token limit ({num_tokens} tokens)")
                self.logger.warning("This may result in incomplete JSON - checking...")

                # If response looks like incomplete JSON, provide helpful feedback
                if '{' in response and response.count('{') > response.count('}'):
                    self.logger.warning("⚠ Detected incomplete JSON (more { than })")
                    truncation_warning = (
                        "\n⚠️ [TRUNCATION DETECTED] Your last response was cut off at the 450-token limit.\n"
                        "The JSON appears incomplete.\n\n"
                        "� **CRITICAL: Your tool call was NOT EXECUTED!**\n"
                        "A tool is only executed if you receive a TOOL_RESULTS response.\n"
                        "Since the JSON was incomplete, nothing was saved/processed.\n\n"
                        "💡 **To fix this:**\n"
                        "1. **PUT THE TOOL CALL FIRST** - Start with the JSON, write commentary after (if needed)\n"
                        "2. Keep observations concise (focus on key findings, not full details)\n"
                        "3. Save detailed data in the 'data' field, brief summary in 'description'\n"
                        "4. Break large observations into multiple smaller ones\n\n"
                        "**Example of correct format:**\n"
                        "```json\n"
                        "{\"tool_call\": {\"function\": \"record_observation\", \"arguments\": {...}}, \"reasoning\": \"brief\"}\n"
                        "```\n"
                        "[Optional: Short commentary here]\n\n"
                        "Please provide a shorter, complete response with tool call FIRST.\n"
                    )

                    # Log the truncation warning so we can see what the model sees
                    self.logger.info(f"[USER] {truncation_warning}\n")

                    # Add warning to history so model sees it
                    self.conversation_history.append({
                        "role": "user",
                        "content": truncation_warning
                    })
                    # Force model to retry without broken state
                    continue

            # NEW APPROACH: Parse JSON tool calling format
            # Model can write text, then end with: {"reasoning": "...", "tool_call": {"function": "...", "arguments": {...}}}

            tool_call = None
            json_obj = None
            parse_error = None

            # Try to parse JSON from the end of the response
            try:
                # Try to extract JSON - it might be at the end after some text
                json_text = response.strip()

                # If response has code blocks, extract from them
                if '```' in json_text:
                    # Find all code blocks
                    blocks = json_text.split('```')
                    # The JSON should be in a code block
                    # blocks[odd indices] = inside code blocks
                    # blocks[even indices] = outside code blocks

                    # Try code blocks from FIRST to LAST (execute in order)
                    json_text = None
                    selected_block_index = None
                    skipped_complete_blocks = []  # Track complete blocks we skip

                    # Iterate through odd indices (code blocks): 1, 3, 5, ...
                    for i in range(1, len(blocks), 2):
                        candidate_block = blocks[i].strip()
                        # Remove language identifier if present (```json)
                        lines = candidate_block.split('\n')
                        if lines and lines[0].strip() in ['json', '{']:
                            if lines[0].strip() == 'json':
                                lines = lines[1:]
                        candidate_json = '\n'.join(lines).strip()

                        # Check if this looks like a complete JSON object
                        is_complete = candidate_json.startswith('{') and candidate_json.count('{') == candidate_json.count('}')

                        if json_text is None and is_complete:
                            # First complete block - use this one
                            json_text = candidate_json
                            selected_block_index = i
                            self.logger.info(f"[JSON PARSER] Using code block {(i + 1) // 2} as tool call")
                        elif json_text is not None and is_complete:
                            # We already have a complete block, this is another complete block we're skipping
                            block_number = (i + 1) // 2
                            skipped_complete_blocks.append(block_number)
                            self.logger.warning(f"[JSON PARSER] Skipping complete code block {block_number} (only first tool call is executed)")
                        # Incomplete blocks are silently ignored (likely truncation)

                    # Store info about skipped complete blocks for later warning to model
                    self.skipped_complete_blocks = skipped_complete_blocks if json_text else []

                    # If no complete block found, use the last block (even if incomplete)
                    if json_text is None:
                        last_block = blocks[-2] if len(blocks) >= 2 else blocks[-1]
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
                            # Note: VAR_KEYWORD (**kwargs) is treated as optional
                            required_params = [
                                p for p in sig.parameters.values()
                                if p.default == inspect.Parameter.empty
                                and p.name != 'self'
                                and p.kind != inspect.Parameter.VAR_KEYWORD
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
                # Check for common formatting errors
                error_msg = str(e)

                # Detect unclosed code block (``` without closing ```)
                if "No JSON object found in response" in error_msg or "Could not find complete JSON object" in error_msg:
                    # Check if response has opening ``` but no closing ```
                    if response.count('```') == 1:
                        parse_error = "Missing closing ``` for code block. You opened a code block with ``` but forgot to close it."
                    elif response.count('```') % 2 != 0:
                        parse_error = "Unmatched ``` code block markers. Make sure each ``` has a closing ```."
                    else:
                        parse_error = f"Error parsing tool call: {error_msg}"
                else:
                    parse_error = f"Error parsing tool call: {error_msg}"

            if tool_call is None:
                # No valid tool call - check if this is intentional (task complete) or an error

                self.conversation_history.append({
                    "role": "assistant",
                    "content": response
                })
                turns_in_this_session += 1  # Increment session turn counter

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

                        # Count observations saved in this experiment
                        obs_count = self._count_observations_in_current_experiment()
                        obs_warning = ""
                        if obs_count == 0:
                            obs_warning = "\n\n⚠️ **WARNING**: You haven't saved any observations in this experiment!\nIf you're done, all your findings will be lost."
                        else:
                            obs_warning = f"\n\n✓ You've saved {obs_count} observation(s) in this experiment."

                        feedback_msg = f"""No tool call detected in your response.

**Are you:**
A) ✅ Done with THIS EXPERIMENT (ready to move to next experiment)
B) ❌ Forgot to include a tool call (want to continue investigating)

**IMPORTANT - If you choose DONE (A):**
- This will END the current experiment immediately
- Your working memory (this conversation) will be COMPLETELY RESET
- The next experiment will start with FRESH context
- **Any unsaved findings will be PERMANENTLY LOST**
- 💾 Use record_observation() FIRST if you have important discoveries!{obs_warning}

**How to respond:**

**If DONE (A) with unsaved findings:**
1. First save with record_observation() (include tool call in next response)
2. Then on the following turn, say "I'm done" (no tool call)

**If DONE (A) and everything is saved:**
- Just say "I'm done" (no tool call needed)

**If FORGOT (B) - want to continue:**
- Include your next tool call in JSON format:

```json
{{
  "tool_call": {{
    "function": "function_name",
    "arguments": {{...}}
  }},
  "reasoning": "What I want to do next"
}}
```

⚠️ **DO NOT** say "I'm done" AND include a tool call in the same response!

Your previous response had: "{parse_error}"
"""

                        self.logger.info(f"[SYSTEM] No tool call detected, asking for clarification: {parse_error}")
                        self.logger.info(f"\n[FEEDBACK TO MODEL] Asking for clarification\n")

                        # Log the full feedback message so we can see what the model sees
                        self.logger.info(f"[USER] {feedback_msg}\n")

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

                # Check for common parameter name errors and provide helpful feedback
                if isinstance(result, dict) and "error" in result:
                    error_msg = result["error"]

                    # Detect "unexpected keyword argument" errors - common mistake
                    if "unexpected keyword argument" in error_msg:
                        # Extract the wrong argument name from error message
                        import re
                        match = re.search(r"unexpected keyword argument '(\w+)'", error_msg)
                        if match:
                            wrong_arg = match.group(1)

                            # Common fixes
                            fixes = {
                                "layer_names": "layer_name",  # Plural vs singular confusion
                            }

                            if wrong_arg in fixes:
                                correct_arg = fixes[wrong_arg]
                                result["hint"] = (
                                    f"\n\n💡 PARAMETER NAME ERROR!\n"
                                    f"   You used: '{wrong_arg}'\n"
                                    f"   Should be: '{correct_arg}'\n\n"
                                    f"The function accepts '{correct_arg}' (note the singular form).\n"
                                    f"Try again with the correct parameter name."
                                )

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
                turns_in_this_session += 1  # Increment session turn counter

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

                # Track if this was a save operation
                if function_name == "record_observation":
                    tools_since_last_save = 0  # Reset counter on save
                else:
                    tools_since_last_save += 1  # Increment for non-save tools

                # After 3 non-save tool calls, remind model to save findings
                if tools_since_last_save >= 3:
                    save_reminder = (
                        "\n\n⚠️ **MEMORY REMINDER**: You've made several observations but haven't saved any to long-term memory.\n"
                        "Consider using record_observation() to save your findings before continuing.\n"
                        "Remember: Findings not saved will be lost after context resets!"
                    )
                    tool_results_msg += save_reminder
                    self.logger.info("[MEMORY REMINDER] Injected save reminder after 3 non-save tool calls")

                # Warn if we skipped complete tool calls (model wrote multiple complete tool calls)
                if hasattr(self, 'skipped_complete_blocks') and self.skipped_complete_blocks:
                    block_list = ", ".join(str(b) for b in self.skipped_complete_blocks)
                    multiple_tool_call_warning = (
                        f"\n\n⚠️ **MULTIPLE TOOL CALLS DETECTED**: Your response contained multiple complete tool calls.\n"
                        f"Only the FIRST tool call was executed. Code block(s) {block_list} were ignored.\n\n"
                        f"**Remember**: You can only execute ONE tool call per response.\n"
                        f"After receiving the tool results, include your NEXT tool call in the following response.\n"
                        f"This is the correct pattern:\n"
                        f"  Turn 1: Call tool_A()\n"
                        f"  Turn 2: Receive results → Call tool_B()\n"
                        f"  Turn 3: Receive results → Call tool_C()\n"
                    )
                    tool_results_msg += multiple_tool_call_warning
                    self.logger.info(f"[MULTIPLE TOOL CALLS] Warned model about {len(self.skipped_complete_blocks)} skipped tool call(s)")
                    # Clear the list for next iteration
                    self.skipped_complete_blocks = []

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

        # Only log warning if we ACTUALLY hit the limit (not if we broke out early)
        if tool_call_count >= max_tool_calls:
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

        # Debug: Log what's being formatted
        self.logger.debug(f"[DEBUG] _format_conversation_for_model: {len(conversation_without_system)} messages (system excluded)")
        if conversation_without_system:
            self.logger.debug(f"[DEBUG] First message role: {conversation_without_system[0]['role']}")

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

        # Format using our manual chat template helper
        # This ensures consistent formatting with the cached system prompt
        formatted = format_qwen_chat(trimmed_history, add_generation_prompt=True)

        # Debug: Log formatted output
        self.logger.debug(f"[DEBUG] Formatted conversation length: {len(formatted)} chars")
        self.logger.debug(f"[DEBUG] Formatted starts with: {formatted[:100] if formatted else 'EMPTY'}")

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
        # Capture memory at session start
        self.gpu_monitor.snapshot("session_start")
        
        try:
            self.run_experiments()
            self.save_session()

            # Capture memory at session end
            self.gpu_monitor.snapshot("session_end")
            
            # Print GPU memory summary with current limits
            self.gpu_monitor.print_summary(
                current_limits={
                    "max_new_tokens": 450,
                    "max_conversation_tokens": 2000,
                    "keep_recent_turns": 2
                }
            )

            self.logger.info("\n" + "=" * 80)
            self.logger.info(f"{self.phase_name.upper()} COMPLETE")
            self.logger.info("=" * 80)
            self.logger.info(f"Session data saved to: {self.session_dir}")
            self.logger.info("\nReview the conversation to see what the model discovered about itself.")

            return True

        except Exception as e:
            self.logger.error(f"Session failed: {e}", exc_info=True)
            # Still print memory summary on error
            self.gpu_monitor.snapshot("session_error")
            self.gpu_monitor.print_summary(
                current_limits={
                    "max_new_tokens": 450,
                    "max_conversation_tokens": 2000,
                    "keep_recent_turns": 2
                }
            )
            return False

        finally:
            # Always cleanup, even on error
            self.cleanup()
