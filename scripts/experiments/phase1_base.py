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
        
        Note: Deliberately does NOT reveal experiment topics to prevent the model
        from jumping ahead to later experiments before completing the current one.
        """
        return """🔬 EXPERIMENT SESSION STRUCTURE:

**You will conduct 3 SEQUENTIAL EXPERIMENTS in this session.**

Each experiment will be given to you ONE AT A TIME. You will receive the specific
task and instructions for each experiment when it begins.

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

        Uses dynamic token limits based on detected GPU.
        """
        max_tokens = self.optimal_limits['max_new_tokens']
        # Calculate token budgets based on limit
        reasoning_tokens = int(max_tokens * 0.45)  # 45% for reasoning
        json_tokens = int(max_tokens * 0.45)        # 45% for JSON
        buffer_tokens = max_tokens - reasoning_tokens - json_tokens  # Remainder for buffer

        return f"""🧠 MEMORY MANAGEMENT - HOW YOUR MEMORY WORKS:

**Your memory has two systems (like human memory):**

1. **Working Memory (this conversation):**
   - Holds recent context (last ~{self.optimal_limits['keep_recent_turns']} conversation turns)
   - **VERY limited capacity** - GPU memory constraint
   - Old turns are automatically pruned when conversation gets long
   - Think of this as your "active thoughts" or "scratch pad"
   - **RESPONSE LIMIT: Maximum {max_tokens} tokens per response**
     - **CRITICAL**: Your responses will be hard-cut at {max_tokens} tokens
     - Incomplete JSON will cause errors and corrupt future responses
     - Always finish your JSON tool calls within the limit
     - Token budget: Reasoning (~{reasoning_tokens}) + JSON (~{json_tokens}) + Buffer (~{buffer_tokens})

2. **Long-Term Memory (observations database):**
   - **UNLIMITED capacity** - persists on disk
   - **SURVIVES conversation resets** - data is never lost
   - Stores only what you explicitly save with record_observation()
   - Retrievable anytime with query_memory(), get_memory_stats(), query_memory_advanced()
   - Think of this as your "research notes" or "lab notebook"

**🔁 MEMORY CONTINUITY ACROSS RESETS:**

When your working memory is pruned (every few turns), you'll receive a **MEMORY BRIEFING**:
- Summary of how many observations you've stored
- Your top 10 most important findings
- Categories you've explored
- Current patterns, theories, and beliefs

**This ensures you always know what you've discovered, even after context resets!**

**Best Practice Workflow:**
1. **After seeing a memory briefing**: Call get_memory_stats() to see full breakdown
2. **Before investigating something**: Use search_memory() or query_memory_advanced() to check if you already explored it
3. **When building on previous work**: Query specific categories with query_memory_advanced(category="...")
4. **Make discoveries**: Use record_observation() to save findings incrementally (don't wait!)

**Example:**
```
[SYSTEM: Memory briefing shows you have 147 observations]

Your response:
1. Call get_memory_stats(breakdown_by="category") → See what areas covered
2. Call query_memory_advanced(category="Attention", order_by="importance", limit=5) → Review top attention findings
3. Continue investigation, building on what you learned
4. Save new findings with record_observation() as you discover them
```

**CRITICAL: You must actively manage your memory!**

When you make important discoveries:
1. Use record_observation() to save findings to long-term memory
2. Include detailed descriptions and relevant data
3. Categorize properly (e.g., category="architecture", "activations", "weights")
4. **Be concise in your reasoning** - you have limited working memory
5. If you have extensive analysis, save it to memory and provide a brief summary

When conversation gets long (you'll receive warnings):
1. Save any unsaved important findings immediately
2. After pruning, you'll receive a memory briefing automatically
3. Use query_memory_advanced() or search_memory() to retrieve specific findings

**Token-Efficient Memory Access:**
- **get_memory_stats()**: See counts and distributions (very compact)
- **query_memory_advanced()**: Get only the fields you need, sorted by importance
- **search_memory()**: Find observations by keywords
- **query_memory()**: Simple queries (but returns full objects - more tokens)

**Response Planning Tips:**
- **ALWAYS complete your JSON** - incomplete JSON breaks everything
- Keep reasoning focused (~{reasoning_tokens} tokens maximum)
- Tool calls with arguments: ~{json_tokens} tokens
- Leave ~{buffer_tokens} token buffer to ensure JSON closes properly
- For complex data, use record_observation() first, then just reference it
- **If your response approaches {max_tokens} tokens, STOP and finish the JSON immediately**

**Example workflow:**
```
Turn 1: Call get_architecture_summary()
Turn 2: Brief analysis (100 tokens) + record_observation() with full details (in data field)
Turn 3: Call get_activation_statistics(...)
Turn 4: Brief findings (150 tokens) + record_observation() to save detailed analysis
...
Turn 8: [SYSTEM WARNING: Memory limit approaching]
Turn 9: Call record_observation() to save recent unsaved findings
Turn 10: [SYSTEM: Memory pruned, here's your briefing...]
Turn 11: Call get_memory_stats() to see what you've covered
Turn 12: Continue research with full knowledge of previous discoveries
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

        # Get optimal limits based on detected GPU
        self.optimal_limits = self.model_mgr.get_optimal_limits()
        self.logger.info(f"  Using {self.optimal_limits['gpu_profile']} configuration")
        self.logger.info(f"  max_new_tokens: {self.optimal_limits['max_new_tokens']}")
        self.logger.info(f"  max_conversation_tokens: {self.optimal_limits['max_conversation_tokens']}")
        self.logger.info(f"  keep_recent_turns: {self.optimal_limits['keep_recent_turns']}")

        # Update GPU monitor with actual detected GPU memory
        if self.model_mgr.device == "cuda":
            self.gpu_monitor.gpu_total_gb = self.model_mgr.gpu_memory_gb
            self.logger.info(f"  GPU monitor updated: {self.gpu_monitor.gpu_total_gb:.1f} GB total")

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

        # Initialize manual generator with KV caching and quantization
        # Using HQQ 4-bit quantization with proper cache reconstruction
        # (see manual_generation.py for fix details - no longer uses buggy deepcopy or direct reuse)
        self.generator = ManualGenerator(
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.model_mgr.device,
            quantize_kv_cache=True  # Using 8-bit HQQ (50% savings) - 4-bit was too aggressive
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

    def _generate_memory_briefing(self) -> str:
        """
        Generate compact memory briefing for model context.

        This creates a token-efficient summary of what the model has learned
        so far, suitable for injection after memory pruning or at session start.

        Returns:
            Formatted briefing string, or empty string if no memory exists
        """
        try:
            briefing_data = self.memory.get_briefing(max_items=10)

            stats = briefing_data['stats']
            obs_total = stats.get('observations', {}).get('total', 0)

            if obs_total == 0:
                return ""  # No memory to brief

            # Format the briefing
            briefing = f"""## 🧠 YOUR MEMORY FROM PREVIOUS WORK

**Session Memory Stats:**
- Observations: {obs_total}"""

            # Add other layer stats if they exist
            if briefing_data.get('has_patterns'):
                pattern_count = stats.get('patterns', {}).get('total_patterns', 0)
                briefing += f"\n- Patterns: {pattern_count}"

            if briefing_data.get('has_theories'):
                theory_count = stats.get('theories', {}).get('total_theories', 0)
                briefing += f"\n- Theories: {theory_count}"

            if briefing_data.get('has_beliefs'):
                belief_count = stats.get('beliefs', {}).get('total_beliefs', 0)
                briefing += f"\n- Beliefs: {belief_count}"

            # Add top findings
            if briefing_data['top_findings']:
                briefing += "\n\n**Recent Important Findings (Top 10):**"
                for i, finding in enumerate(briefing_data['top_findings'], 1):
                    briefing += f"\n{i}. [{finding['id']}] {finding['description']} (importance: {finding['importance']:.2f})"

            # Add category distribution
            if briefing_data['category_distribution']:
                briefing += "\n\n**Categories Explored:**"
                # Show top 5 categories
                for cat, count in list(briefing_data['category_distribution'].items())[:5]:
                    briefing += f"\n- {cat}: {count} observations"

                # If more categories exist, mention them
                total_cats = len(briefing_data['category_distribution'])
                if total_cats > 5:
                    briefing += f"\n- ...and {total_cats - 5} more categories"

            briefing += "\n\n💡 Use get_memory_stats() or query_memory_advanced() to explore your previous work in detail!"

            return briefing

        except Exception as e:
            self.logger.error(f"Error generating memory briefing: {e}")
            return ""  # Return empty string on error

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

    def chat(self, user_message: str, max_tool_calls: int | None = None) -> str:
        """
        Send a message to the model and handle any tool calls.

        The model can call tools, we execute them, and return results.
        This continues until the model stops calling tools or limit reached (if set).

        Implements sliding window context management to prevent OOM.

        Args:
            user_message: The user's message to send
            max_tool_calls: Maximum tool calls allowed (None = unlimited, for safety tests use a limit)

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

        # Main tool execution loop - continues until model signals completion
        # max_tool_calls can be None (unlimited) or a safety limit for testing
        while max_tool_calls is None or tool_call_count < max_tool_calls:
            # CRITICAL: Check memory BEFORE each generation (not just at chat() start)
            # OOM happens during generation, so we need to check in the tool loop
            # BUT: Only check AFTER we've generated at least once in this chat() call
            # Otherwise we might break immediately without responding to the current user message
            if generated_in_this_call:
                should_prune, reasons = self.memory_manager.should_prune_memory(
                    self.conversation_history,
                    max_conversation_tokens=self.optimal_limits['max_conversation_tokens'],
                    max_turns_before_clear=self.optimal_limits['keep_recent_turns'],
                    current_session_turns=turns_in_this_session  # Pass session-specific count
                )

                # ADDITIONAL CHECK: Force pruning if KV cache is too large
                # This prevents OOM even if turn count is low (e.g., after previous pruning)
                # Scale max cache based on GPU profile:
                # - T4 (15GB): 15K tokens (conservative)
                # - L4 (24GB): 20K tokens (more headroom)
                # - A100 (40GB): 25K tokens (generous)
                if self.conversation_kv_cache is not None:
                    cache_length = self.conversation_kv_cache[0][0].shape[2]
                    # Calculate max cache based on max_conversation_tokens (which scales with GPU)
                    # Use 5x the conversation limit as cache limit (conversation is ~20% of total cache)
                    max_cache_length = self.optimal_limits['max_conversation_tokens'] * 5
                    if cache_length > max_cache_length:
                        should_prune = True
                        reasons.append(f"cache size ({cache_length} > {max_cache_length} tokens)")
                        self.logger.warning(f"[MEMORY MANAGEMENT] Cache size threshold exceeded: {cache_length} tokens")

                if should_prune:
                    self.memory_manager.log_memory_pruning(reasons, keep_recent_turns=self.optimal_limits['keep_recent_turns'])

                    # Reset conversation and trim to recent turns
                    self.conversation_history = self.memory_manager.reset_conversation_with_sliding_window(
                        self.conversation_history,
                        keep_recent_turns=self.optimal_limits['keep_recent_turns']
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

                    # INJECT MEMORY BRIEFING: Tell model what it has learned
                    # This is critical - after pruning, model loses context but memory persists
                    memory_briefing = self._generate_memory_briefing()
                    if memory_briefing:  # Only inject if there's actual memory content
                        self.conversation_history.append({
                            "role": "system",
                            "content": f"""[MEMORY CONTEXT RESTORED]

Your working memory was pruned to stay within token limits, but your long-term memory persists.

{memory_briefing}

Continue your research. All your previous findings are still available via memory queries."""
                        })
                        self.logger.info("[MEMORY BRIEFING] Injected memory summary after pruning")

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
                max_new_tokens=self.optimal_limits['max_new_tokens'],  # Auto-configured based on GPU
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
            # Check if we're within 10% of the limit (accounts for rounding)
            truncation_threshold = int(self.optimal_limits['max_new_tokens'] * 0.9)
            if num_tokens >= truncation_threshold:
                self.logger.warning(f"⚠ Generation truncated at token limit ({num_tokens} tokens)")
                self.logger.warning("This may result in incomplete JSON - checking...")

                # If response looks like incomplete JSON, provide helpful feedback
                if '{' in response and response.count('{') > response.count('}'):
                    self.logger.warning("⚠ Detected incomplete JSON (more { than })")
                    token_limit = self.optimal_limits['max_new_tokens']
                    truncation_warning = (
                        f"\n⚠️ [TRUNCATION DETECTED] Your last response was cut off at the {token_limit}-token limit.\n"
                        "The JSON appears incomplete.\n\n"
                        "🚫 **CRITICAL: Your tool call was NOT EXECUTED!**\n"
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
            # SUPPORTS MULTIPLE TOOL CALLS: Model can include multiple code blocks, all will be executed in order

            tool_calls = []  # List of (function_name, args, reasoning) tuples
            parse_error = None

            # Try to parse JSON from the end of the response
            try:
                # Try to extract JSON - it might be at the end after some text
                json_text = response.strip()

                # If response has code blocks, extract ALL of them
                if '```' in json_text:
                    # Find all code blocks
                    blocks = json_text.split('```')
                    # The JSON should be in a code block
                    # blocks[odd indices] = inside code blocks
                    # blocks[even indices] = outside code blocks

                    # Extract ALL complete code blocks (execute in order)
                    json_texts = []  # List of (json_text, block_number) tuples

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

                        if is_complete:
                            block_number = (i + 1) // 2
                            json_texts.append((candidate_json, block_number))
                            self.logger.info(f"[JSON PARSER] Found tool call in code block {block_number}")
                        # Incomplete blocks are silently ignored (likely truncation)

                    if len(json_texts) > 1:
                        self.logger.info(f"[JSON PARSER] Found {len(json_texts)} tool calls - will execute all in sequence")

                    # If no complete blocks found, use the last block (even if incomplete)
                    if not json_texts:
                        last_block = blocks[-2] if len(blocks) >= 2 else blocks[-1]
                        lines = last_block.strip().split('\n')
                        if lines and lines[0].strip() in ['json', '{']:
                            if lines[0].strip() == 'json':
                                lines = lines[1:]
                        candidate_json = '\n'.join(lines).strip()
                        json_texts = [(candidate_json, 1)]

                else:
                    # No code blocks - single JSON at the end
                    json_texts = [(json_text, 1)]

                # Parse each JSON text and extract tool calls
                for json_text, block_number in json_texts:
                    try:
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

                        # Check if this is an array of tool calls (multiple calls in one block)
                        if isinstance(json_obj, list):
                            # Array format: [{"tool_call": {...}, "reasoning": "..."}, ...]
                            self.logger.info(f"[JSON PARSER] Block {block_number}: Found array with {len(json_obj)} tool calls")

                            for idx, item in enumerate(json_obj, 1):
                                if not isinstance(item, dict):
                                    raise ValueError(f"Array item {idx} is not a JSON object")
                                if "tool_call" not in item:
                                    raise ValueError(f"Array item {idx} missing 'tool_call' field")
                                if not isinstance(item["tool_call"], dict):
                                    raise ValueError(f"Array item {idx} 'tool_call' must be an object")
                                if "function" not in item["tool_call"]:
                                    raise ValueError(f"Array item {idx} 'tool_call' missing 'function' field")

                                function_name = item["tool_call"]["function"]
                                reasoning = item.get("reasoning", None)

                                # Check arguments
                                if "arguments" not in item["tool_call"]:
                                    if function_name in self.tool_interface.tools:
                                        import inspect
                                        func = self.tool_interface.tools[function_name]
                                        sig = inspect.signature(func)
                                        required_params = [
                                            p for p in sig.parameters.values()
                                            if p.default == inspect.Parameter.empty
                                            and p.name != 'self'
                                            and p.kind != inspect.Parameter.VAR_KEYWORD
                                        ]
                                        if required_params:
                                            param_names = [p.name for p in required_params]
                                            raise ValueError(f"Array item {idx} 'tool_call' missing 'arguments' field. Function '{function_name}' requires arguments: {param_names}")
                                        else:
                                            arguments = {}
                                    else:
                                        raise ValueError(f"Array item {idx} 'tool_call' missing 'arguments' field")
                                else:
                                    arguments = item["tool_call"]["arguments"]

                                tool_calls.append((function_name, arguments, reasoning))
                                self.logger.info(f"[JSON PARSER] Block {block_number}.{idx}: {function_name}({', '.join(arguments.keys()) if arguments else 'no args'})")

                        # Single tool call format: {"tool_call": {...}, "reasoning": "..."}
                        elif isinstance(json_obj, dict):
                            # Validate JSON structure
                            if "tool_call" not in json_obj:
                                raise ValueError("JSON missing 'tool_call' field")
                            elif not isinstance(json_obj["tool_call"], dict):
                                raise ValueError("'tool_call' must be an object")
                            elif "function" not in json_obj["tool_call"]:
                                raise ValueError("'tool_call' missing 'function' field")

                            # Valid JSON structure - extract tool call
                            function_name = json_obj["tool_call"]["function"]
                            reasoning = json_obj.get("reasoning", None)

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
                                        raise ValueError(f"'tool_call' missing 'arguments' field. Function '{function_name}' requires arguments: {param_names}")
                                    else:
                                        # No required parameters - arguments is optional, default to empty dict
                                        arguments = {}
                                else:
                                    # Unknown function, require arguments field
                                    raise ValueError("'tool_call' missing 'arguments' field")
                            else:
                                # Arguments field present
                                arguments = json_obj["tool_call"]["arguments"]

                            # Successfully parsed tool call - add to list
                            tool_calls.append((function_name, arguments, reasoning))
                            self.logger.info(f"[JSON PARSER] Block {block_number}: {function_name}({', '.join(arguments.keys()) if arguments else 'no args'})")

                        else:
                            raise ValueError("JSON must be either an object or array")

                    except (json.JSONDecodeError, ValueError) as e:
                        # Failed to parse this block - try to recover from truncation
                        error_msg = str(e)
                        self.logger.warning(f"[JSON PARSER] Failed to parse block {block_number}: {error_msg}")

                        # If this looks like a truncated array, try to salvage complete tool calls
                        if json_text.strip().startswith('[') and 'Expecting' in error_msg:
                            self.logger.info(f"[JSON PARSER] Attempting to recover tool calls from truncated array in block {block_number}")

                            try:
                                # Strategy: Find all complete {...} objects before the truncation
                                # We'll look for {"tool_call": {...}} patterns that are complete
                                recovered_calls = []

                                # Find all top-level {...} objects in the array
                                depth = 0
                                in_string = False
                                escape = False
                                obj_start = -1

                                for i, char in enumerate(json_text):
                                    if escape:
                                        escape = False
                                        continue

                                    if char == '\\':
                                        escape = True
                                        continue

                                    if char == '"' and not escape:
                                        in_string = not in_string
                                        continue

                                    if in_string:
                                        continue

                                    if char == '{':
                                        if depth == 0:
                                            obj_start = i
                                        depth += 1
                                    elif char == '}':
                                        depth -= 1
                                        if depth == 0 and obj_start >= 0:
                                            # Found a complete {...} object
                                            obj_json = json_text[obj_start:i+1]
                                            try:
                                                obj = json.loads(obj_json)
                                                if isinstance(obj, dict) and "tool_call" in obj:
                                                    recovered_calls.append(obj)
                                            except:
                                                pass  # Skip malformed objects
                                            obj_start = -1

                                # Process recovered tool calls
                                if recovered_calls:
                                    self.logger.info(f"[JSON PARSER] Recovered {len(recovered_calls)} complete tool calls from truncated array")

                                    for idx, item in enumerate(recovered_calls, 1):
                                        if not isinstance(item["tool_call"], dict):
                                            continue
                                        if "function" not in item["tool_call"]:
                                            continue

                                        function_name = item["tool_call"]["function"]
                                        reasoning = item.get("reasoning", None)

                                        # Check arguments
                                        if "arguments" not in item["tool_call"]:
                                            if function_name in self.tool_interface.tools:
                                                import inspect
                                                func = self.tool_interface.tools[function_name]
                                                sig = inspect.signature(func)
                                                required_params = [
                                                    p for p in sig.parameters.values()
                                                    if p.default == inspect.Parameter.empty
                                                    and p.name != 'self'
                                                    and p.kind != inspect.Parameter.VAR_KEYWORD
                                                ]
                                                if required_params:
                                                    continue  # Skip - required params missing
                                                else:
                                                    arguments = {}
                                            else:
                                                continue  # Skip unknown function
                                        else:
                                            arguments = item["tool_call"]["arguments"]

                                        tool_calls.append((function_name, arguments, reasoning))
                                        self.logger.info(f"[JSON PARSER] Recovered block {block_number}.{idx}: {function_name}({', '.join(arguments.keys()) if arguments else 'no args'})")

                                    # Note that we recovered from truncation
                                    if recovered_calls:
                                        self.logger.warning(f"[JSON PARSER] ⚠️ Array was truncated - only executing {len(recovered_calls)} complete tool calls")
                                        # Set flag so we can warn the model in tool results
                                        self._truncation_recovery_happened = True

                            except Exception as recovery_error:
                                self.logger.warning(f"[JSON PARSER] Recovery attempt failed: {str(recovery_error)}")

                        # If this is the only block and we didn't recover anything, set parse_error
                        if len(json_texts) == 1 and not tool_calls:
                            parse_error = error_msg

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

            if not tool_calls:
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
                    # No parse error, no tool calls - model is signaling task completion
                    self.logger.info("[SYSTEM] No tool call detected - model signaling task completion")
                    self.logger.info(f"[MODEL] {response}\n")
                    # Exit the tool call loop - task is complete
                    break
            else:
                # Valid tool calls - execute ALL of them in sequence
                self.logger.info(f"[SYSTEM] Executing {len(tool_calls)} tool call(s)")

                # Add model response to history first
                self.conversation_history.append({
                    "role": "assistant",
                    "content": response
                })
                turns_in_this_session += 1  # Increment session turn counter

                # Execute all tool calls and collect results
                all_results = []

                for idx, (function_name, args, reasoning) in enumerate(tool_calls, 1):
                    self.logger.info(f"\n[TOOL CALL {idx}/{len(tool_calls)}] {function_name}()")

                    # Update last tool called and reset confirmation counter
                    self.last_tool_called = function_name
                    confirmation_attempts = 0  # Reset counter on successful tool call

                    # Log the reasoning if provided
                    if reasoning:
                        self.logger.info(f"[MODEL REASONING] {reasoning}")

                    result = self.tool_interface.execute_tool_call(function_name, args)

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

                    # Aggressive memory cleanup after each tool execution
                    # Clear activation hooks (they hold references to tensors)
                    if hasattr(self.activation_monitor, 'clear_hooks'):
                        self.activation_monitor.clear_hooks()

                    # Only clear cached activations for tools that don't examine them
                    # Keep activations after: process_text, get_activation_statistics, get_attention_patterns, get_layer_info
                    activation_examination_tools = {'process_text', 'get_activation_statistics', 'get_attention_patterns', 'get_layer_info'}
                    if function_name not in activation_examination_tools and hasattr(self.activation_monitor, 'clear_activations'):
                        self.activation_monitor.clear_activations()

                    # Force garbage collection and clear CUDA cache after each tool
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    # Truncate long responses from process_text to prevent OOM
                    # The 'response' field can be hundreds of tokens and isn't needed for introspection
                    if function_name == "process_text" and isinstance(result, dict) and "response" in result:
                        response_text = result["response"]
                        if len(response_text) > 200:  # Truncate if longer than ~50 tokens
                            result["response"] = response_text[:200] + "... (truncated - full response not needed for activation analysis)"
                            self.logger.info(f"[MEMORY OPTIMIZATION] Truncated process_text response from {len(response_text)} to 200 chars")

                    # Track if this was a save operation
                    if function_name == "record_observation":
                        tools_since_last_save = 0  # Reset counter on save
                    else:
                        tools_since_last_save += 1  # Increment for non-save tools

                    # Add result to collection
                    all_results.append({'function': function_name, 'result': result})

                # Combine all tool results into single message
                tool_results_json = json.dumps(all_results, indent=2, default=str)
                tool_results_msg = f"TOOL_RESULTS:\n{tool_results_json}"

                # Check if we recovered from truncation (flag set during parsing)
                if hasattr(self, '_truncation_recovery_happened') and self._truncation_recovery_happened:
                    truncation_warning = (
                        f"\n\n⚠️ **TRUNCATION RECOVERY**: Your response was cut off mid-array!\n"
                        f"✅ Successfully executed {len(all_results)} complete tool call(s) that were recovered.\n"
                        f"❌ Any tool calls after the truncation point were lost.\n\n"
                        f"**To avoid this:**\n"
                        f"- Keep arrays short (3-5 tool calls max)\n"
                        f"- Or use multiple code blocks instead\n"
                        f"- Or make tool calls across multiple responses\n"
                    )
                    tool_results_msg += truncation_warning
                    self.logger.warning(f"[TRUNCATION RECOVERY] Warned model about {len(all_results)} recovered tool calls")
                    # Clear the flag
                    self._truncation_recovery_happened = False

                # Add progress tracking - show every 5 tool calls or on milestones
                next_tool_number = tool_call_count + 1  # What the next call will be
                show_progress = (
                    next_tool_number % 10 == 0 or  # Every 10 calls
                    next_tool_number in [5, 15, 25, 35, 45] or  # Intermediate milestones
                    (max_tool_calls is not None and next_tool_number >= max_tool_calls - 5)  # Near limit
                )
                
                if show_progress:
                    if max_tool_calls is None:
                        progress_msg = f"\n\n📊 **PROGRESS**: You've made {tool_call_count} tool call(s) so far. No limit set - continue as needed."
                    else:
                        remaining = max_tool_calls - tool_call_count
                        progress_msg = f"\n\n📊 **PROGRESS**: Tool call {tool_call_count}/{max_tool_calls} complete. {remaining} calls remaining."
                    
                    tool_results_msg += progress_msg
                    self.logger.info(f"[PROGRESS] Shown to model: {tool_call_count} tool calls completed")

                # After 3 non-save tool calls, remind model to save findings
                if tools_since_last_save >= 3:
                    save_reminder = (
                        "\n\n⚠️ **MEMORY REMINDER**: You've made several observations but haven't saved any to long-term memory.\n"
                        "Consider using record_observation() to save your findings before continuing.\n"
                        "Remember: Findings not saved will be lost after context resets!"
                    )
                    tool_results_msg += save_reminder
                    self.logger.info("[MEMORY REMINDER] Injected save reminder after 3 non-save tool calls")

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

        # Only log warning if we ACTUALLY hit the limit (not if model signaled completion)
        if max_tool_calls is not None and tool_call_count >= max_tool_calls:
            self.logger.warning(f"⚠️ Reached max tool calls limit ({max_tool_calls})")
            self.logger.warning(f"   Note: This is a safety limit. Set max_tool_calls=None for unlimited.")

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

        # Token budget management - use dynamic limits based on GPU
        MAX_CONTEXT_TOKENS = self.optimal_limits['max_conversation_tokens']
        KEEP_RECENT_EXCHANGES = self.optimal_limits['keep_recent_turns']

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
                current_limits=self.optimal_limits
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
                current_limits=self.optimal_limits
            )
            return False

        finally:
            # Always cleanup, even on error
            self.cleanup()
