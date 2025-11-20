"""
Memory Management Utility for Phase Experiments

This module provides memory management functionality for managing conversation
history, KV cache, and preventing OOM errors during model generation.

Key features:
- Token estimation for conversation history
- Generation-aware pruning (when not enough room for next generation)
- Intelligent tool result pruning
- KV cache reset with configurable retention

Pruning Strategy:
- Trigger: When current_tokens + max_new_tokens >= max_conversation_tokens (no room for generation)
- Target: Prune down to MEMORY_PRUNE_TARGET_RATIO of max_conversation_tokens
- Always keeps: System prompt + most recent messages until target is reached

Author: AGI Self-Modification Research
Date: November 10, 2025
"""

import json
import logging
from typing import Any, Dict, List, Optional

# Memory pruning threshold for post-prune target (used as fallback/minimum)
# Trigger is generation-aware: prune when not enough room for next generation
MEMORY_PRUNE_TARGET_RATIO = 0.25   # Minimum target: prune down to 25% (leaves 75% headroom)
MEMORY_PRUNE_SAFETY_MARGIN = 1.1   # 10% safety margin on top of calculated needs


class MemoryManager:
    """
    Manages conversation memory and KV cache for phase experiments.
    
    Prevents OOM by:
    1. Estimating conversation token count
    2. Pruning old tool results to reduce memory
    3. Implementing sliding window to keep only recent turns
    4. Resetting KV cache when limits are exceeded
    
    Note: All memory limits are now GPU-adaptive and must be provided by the caller
    (typically from ModelManager.get_optimal_limits()).
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize memory manager.
        
        Args:
            logger: Optional logger for tracking memory operations
        """
        self.logger = logger or logging.getLogger(__name__)
    
    def estimate_conversation_tokens(self, conversation_history: List[Dict[str, str]]) -> int:
        """
        Estimate the number of tokens in the conversation history ONLY.
        
        This does NOT include the system prompt (which is separately cached).
        
        This is a rough estimate: chars / 4 (approximation for tokenization)
        Used to decide when to reset KV cache.
        
        Args:
            conversation_history: List of message dictionaries with 'role' and 'content'
        
        Returns:
            Estimated token count for conversation (excluding system prompt)
        """
        # Explicitly filter out system messages (system prompt is cached separately)
        conversation_only = [
            msg for msg in conversation_history
            if msg.get("role") != "system"
        ]
        total_chars = sum(len(msg.get("content", "")) for msg in conversation_only)
        return total_chars // 4  # Rough approximation: 1 token ≈ 4 chars
    
    def should_prune_memory(
        self,
        conversation_history: List[Dict[str, str]],
        max_conversation_tokens: int,
        max_new_tokens: int
    ) -> tuple[bool, List[str]]:
        """
        Check if memory should be pruned based on generation headroom.
        
        Pruning is triggered when there isn't enough room for the next generation:
        current_tokens + max_new_tokens >= max_conversation_tokens
        
        This ensures we always have space for a full generation without OOM.
        
        Args:
            conversation_history: Current conversation history
            max_conversation_tokens: Maximum tokens before OOM
            max_new_tokens: Maximum tokens model will generate
        
        Returns:
            Tuple of (should_prune, reasons) where reasons is a list of why pruning is needed
        """
        estimated_tokens = self.estimate_conversation_tokens(conversation_history)
        needed_tokens = estimated_tokens + max_new_tokens
        
        reasons = []
        should_prune = needed_tokens >= max_conversation_tokens
        
        if should_prune:
            reasons.append(
                f"insufficient headroom for generation (~{estimated_tokens} current + "
                f"{max_new_tokens} generation = {needed_tokens} >= {max_conversation_tokens} max)"
            )
        
        return should_prune, reasons
    
    def get_prune_target_tokens(
        self, 
        max_conversation_tokens: int,
        max_new_tokens: int,
        current_iteration: int,
        max_iterations: int
    ) -> int:
        """
        Get the adaptive target token count to prune down to.
        
        Strategy:
        - If near end of experiment: only prune enough for remaining iterations
        - If early/mid experiment: prune to MEMORY_PRUNE_TARGET_RATIO for maximum headroom
        - Always applies MEMORY_PRUNE_SAFETY_MARGIN for unexpected long responses
        
        Args:
            max_conversation_tokens: Maximum tokens before OOM
            max_new_tokens: Maximum tokens per generation
            current_iteration: Current iteration number (0-indexed)
            max_iterations: Maximum iterations for this experiment
        
        Returns:
            Target token count after pruning
        """
        # Calculate how many iterations remain (add 1 because we're still in current iteration)
        remaining_iterations = max_iterations - current_iteration
        
        # Calculate tokens needed for remaining iterations with safety margin
        # Each iteration: current conversation + new generation
        # We need room for: remaining_iterations * max_new_tokens
        tokens_needed_for_remaining = int(remaining_iterations * max_new_tokens * MEMORY_PRUNE_SAFETY_MARGIN)
        
        # Adaptive target: current tokens needed, but leave headroom up to max
        # Target = max - tokens_needed_for_remaining
        adaptive_target = max_conversation_tokens - tokens_needed_for_remaining
        
        # Minimum target: always prune to at least MEMORY_PRUNE_TARGET_RATIO
        # This ensures we free up memory even near the end
        minimum_target = int(max_conversation_tokens * MEMORY_PRUNE_TARGET_RATIO)
        
        # Use the larger of adaptive or minimum (don't over-prune near the end)
        target = max(adaptive_target, minimum_target)
        
        # Log the decision
        self.logger.debug(
            f"Adaptive pruning: {remaining_iterations} iterations remaining, "
            f"need ~{tokens_needed_for_remaining} tokens, target={target} "
            f"(adaptive={adaptive_target}, minimum={minimum_target})"
        )
        
        return target
    
    def prune_tool_result(self, result: Any, function_name: str) -> Any:
        """
        Intelligently prune verbose fields from tool results to save memory.
        
        Keeps: metadata, key findings, essential information
        Removes: verbose generated text, redundant data
        
        This is applied to OLD tool results when conversation gets long.
        Recent tool results are kept in full.
        
        Args:
            result: The tool result to prune
            function_name: Name of the tool function
        
        Returns:
            Pruned version of the result
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
    
    def reset_conversation_with_sliding_window(
        self,
        conversation_history: List[Dict[str, str]],
        keep_recent_turns: int
    ) -> List[Dict[str, str]]:
        """
        Reset conversation history to keep only recent turns (sliding window).
        
        Prunes old tool results and trims conversation to recent exchanges.
        The system prompt remains cached separately, so we only lose older conversation turns.
        
        Model should use record_observation() to save important findings before they
        slide out of the window.
        
        Args:
            conversation_history: Current conversation history
            keep_recent_turns: Number of recent conversation turns to keep (default: 2)
        
        Returns:
            Trimmed conversation history with pruned old tool results
        """
        # Count conversation exchanges (user-assistant pairs)
        num_exchanges = len([m for m in conversation_history if m["role"] == "assistant"])
        
        if num_exchanges <= keep_recent_turns:
            # Not enough history to trim yet
            self.logger.info(f"[MEMORY MANAGEMENT] No trimming needed ({num_exchanges} exchanges <= {keep_recent_turns} limit)")
            return conversation_history
        
        # Calculate how many messages to keep (2 per exchange: user + assistant)
        keep_messages = keep_recent_turns * 2
        
        # Prune old tool results BEFORE trimming (so we keep metadata even from old turns)
        pruned_history = []
        for msg in conversation_history:
            if msg["role"] == "user" and msg["content"].startswith("TOOL_RESULTS:"):
                # This is a tool result - check if it's old enough to prune
                # Keep recent turns in full, prune older ones
                msg_index = conversation_history.index(msg)
                is_recent = msg_index >= len(conversation_history) - keep_messages
                
                if not is_recent:
                    # Old tool result - prune it
                    try:
                        tool_results = json.loads(msg["content"].replace("TOOL_RESULTS:\n", ""))
                        if isinstance(tool_results, list) and len(tool_results) > 0:
                            pruned_results = []
                            for tr in tool_results:
                                function_name = tr.get("function", "unknown")
                                result = tr.get("result", {})
                                pruned_result = self.prune_tool_result(result, function_name)
                                pruned_results.append({
                                    "function": function_name,
                                    "result": pruned_result
                                })
                            msg["content"] = f"TOOL_RESULTS:\n{json.dumps(pruned_results, indent=2, default=str)}"
                    except Exception as e:
                        # If parsing fails, keep as is
                        self.logger.warning(f"[MEMORY MANAGEMENT] Failed to prune tool result: {e}")
            pruned_history.append(msg)
        
        # Trim to recent turns only
        trimmed_history = pruned_history[-keep_messages:]
        
        self.logger.info(f"[MEMORY MANAGEMENT] Reset conversation, keeping last {keep_recent_turns} turns")
        self.logger.info(f"[MEMORY MANAGEMENT] Pruned {num_exchanges - keep_recent_turns} old exchanges")
        self.logger.info(f"[MEMORY MANAGEMENT] Model should retrieve old findings with query_memory()")
        
        return trimmed_history
    
    def log_memory_pruning(
        self,
        reasons: List[str],
        keep_recent_turns: int
    ):
        """
        Log memory pruning operation with reasons.
        
        Args:
            reasons: List of reasons why pruning was triggered
            keep_recent_turns: Number of turns being kept
        """
        reason_str = " and ".join(reasons)
        self.logger.warning(f"\n[MEMORY MANAGEMENT] Pruning needed in tool loop: {reason_str}")
        self.logger.warning(f"[MEMORY MANAGEMENT] Clearing cache and keeping last {keep_recent_turns} turns")
