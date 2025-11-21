"""
Self-Summarization Manager

Enables the model to compress its own conversation history by generating
summaries of older context. This maintains unlimited conversation length
while preserving the most important information as judged by the model itself.

This is a form of metacognition - the model reflects on its own past to
decide what's worth remembering.

Author: AGI Self-Modification Research Team
Date: November 21, 2025
"""

import logging
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ConversationSummary:
    """A model-generated summary of conversation history"""
    turn_range: Tuple[int, int]  # (start_turn, end_turn)
    original_tokens: int          # Token count before summarization
    summary_tokens: int           # Token count of summary
    summary_text: str             # The actual summary
    timestamp: float = field(default_factory=time.time)
    
    @property
    def compression_ratio(self) -> float:
        """How much the summary compressed the original"""
        return self.original_tokens / self.summary_tokens if self.summary_tokens > 0 else 0.0
    
    def __repr__(self) -> str:
        return (f"ConversationSummary(turns={self.turn_range}, "
                f"compression={self.compression_ratio:.1f}x, "
                f"tokens={self.original_tokens}→{self.summary_tokens})")


class SelfSummarizationManager:
    """
    Manages model-generated conversation summaries for unlimited conversation length.
    
    Strategy:
        [System Prompt - always preserved]
        [Compressed History - model-generated summaries]
        [Recent Detail Window - full conversation]
    
    When cache fills to threshold:
        1. Take oldest portion of conversation (not system prompt or recent window)
        2. Model generates summary preserving key information
        3. Replace original with summary
        4. Continue conversation seamlessly
    
    Example:
        >>> manager = SelfSummarizationManager(
        ...     max_cache_tokens=14000,
        ...     recent_window_tokens=8000,
        ...     summarization_threshold=0.80
        ... )
        >>> if manager.should_summarize(current_tokens=11200):
        ...     summary = manager.generate_summary(
        ...         old_conversation_text="...",
        ...         generator=manual_generator,
        ...         turn_range=(1, 10)
        ...     )
    """
    
    def __init__(
        self,
        max_cache_tokens: int = 14000,
        recent_window_tokens: int = 8000,
        summarization_threshold: float = 0.80,
        target_compression_ratio: float = 5.0,
        system_prompt_tokens: Optional[int] = None
    ):
        """
        Initialize summarization manager.
        
        Args:
            max_cache_tokens: Maximum tokens in KV cache (from ModelManager)
            recent_window_tokens: Tokens to keep in full detail (recent conversation)
            summarization_threshold: Trigger summarization at this % of max_cache_tokens
            target_compression_ratio: Aim to compress N:1 (e.g., 5:1 means 5000 tokens → 1000)
            system_prompt_tokens: Expected system prompt size (auto-detected if None)
        """
        self.max_cache_tokens = max_cache_tokens
        self.recent_window_tokens = recent_window_tokens
        self.summarization_threshold = summarization_threshold
        self.target_compression_ratio = target_compression_ratio
        self.system_prompt_tokens = system_prompt_tokens
        
        # Storage for summaries
        self.summaries: List[ConversationSummary] = []
        self.current_turn = 0
        
        # Track system prompt size once detected
        self._detected_system_prompt_tokens = None
        
        logger.info(f"SelfSummarizationManager initialized:")
        logger.info(f"  Max cache: {max_cache_tokens} tokens")
        logger.info(f"  Recent window: {recent_window_tokens} tokens (full detail)")
        logger.info(f"  Summarization trigger: {summarization_threshold*100:.0f}% of cache")
        logger.info(f"  Target compression: {target_compression_ratio:.1f}:1")
        logger.info(f"  Strategy: [System Prompt] + [Summaries] + [Recent {recent_window_tokens} tokens]")
    
    def set_system_prompt_tokens(self, tokens: int):
        """Set the system prompt token count (called after first caching)"""
        self._detected_system_prompt_tokens = tokens
        logger.info(f"System prompt size detected: {tokens} tokens")
    
    def should_summarize(self, current_tokens: int) -> bool:
        """
        Check if we should trigger summarization.
        
        Args:
            current_tokens: Current total tokens in cache
        
        Returns:
            True if summarization should be triggered
        """
        threshold_tokens = int(self.max_cache_tokens * self.summarization_threshold)
        should = current_tokens >= threshold_tokens
        
        if should:
            logger.info(f"Summarization threshold reached: {current_tokens}/{self.max_cache_tokens} tokens")
            logger.info(f"  (Threshold: {self.summarization_threshold*100:.0f}% = {threshold_tokens} tokens)")
        
        return should
    
    def calculate_tokens_to_summarize(self, current_tokens: int) -> int:
        """
        Calculate how many tokens from old conversation should be summarized.
        
        Strategy:
            - Keep system prompt (always)
            - Keep recent window (full detail)
            - Summarize the middle section to bring total down to ~70% of max
        
        Args:
            current_tokens: Current total tokens in cache
        
        Returns:
            Number of tokens to take from old conversation for summarization
        """
        system_tokens = self._detected_system_prompt_tokens or self.system_prompt_tokens or 2000
        
        # How many tokens in summaries + old conversation?
        summaries_tokens = sum(s.summary_tokens for s in self.summaries)
        old_conversation_tokens = current_tokens - system_tokens - summaries_tokens - self.recent_window_tokens
        
        # Target: bring total down to 70% of max
        target_total = int(self.max_cache_tokens * 0.70)
        tokens_to_free = current_tokens - target_total
        
        # Take enough from old conversation to hit target
        # (will compress at target_compression_ratio)
        tokens_to_summarize = min(
            old_conversation_tokens,  # Don't take more than available
            int(tokens_to_free * self.target_compression_ratio)  # Account for compression
        )
        
        logger.info(f"Summarization calculation:")
        logger.info(f"  Current: {current_tokens} tokens")
        logger.info(f"  System prompt: {system_tokens} tokens")
        logger.info(f"  Existing summaries: {summaries_tokens} tokens")
        logger.info(f"  Recent window: {self.recent_window_tokens} tokens")
        logger.info(f"  Old conversation: {old_conversation_tokens} tokens")
        logger.info(f"  → Will summarize: {tokens_to_summarize} tokens")
        
        return tokens_to_summarize
    
    def generate_summary(
        self,
        old_conversation_text: str,
        generator: "ManualGenerator",  # Forward reference
        turn_range: Tuple[int, int],
        original_token_count: int
    ) -> ConversationSummary:
        """
        Generate a summary of old conversation using the model itself.
        
        This is the metacognitive step - the model reflects on its own
        past conversation and decides what's important to preserve.
        
        Args:
            old_conversation_text: The text to summarize
            generator: ManualGenerator instance to use for generation
            turn_range: (start_turn, end_turn) being summarized
            original_token_count: Token count of original text
        
        Returns:
            ConversationSummary with the generated summary
        """
        target_summary_tokens = int(original_token_count / self.target_compression_ratio)
        
        # Construct summarization prompt
        summarization_prompt = f"""Please compress the following conversation segment into a concise summary.

GOALS:
- Preserve key decisions, discoveries, and important context
- Maintain chronological flow
- Target length: approximately {target_summary_tokens} tokens
- Focus on what matters for continuing the conversation

CONVERSATION SEGMENT (Turns {turn_range[0]}-{turn_range[1]}):
{old_conversation_text}

COMPRESSED SUMMARY:"""
        
        logger.info(f"Generating self-summary for turns {turn_range[0]}-{turn_range[1]}...")
        logger.info(f"  Original: {original_token_count} tokens")
        logger.info(f"  Target: ~{target_summary_tokens} tokens ({self.target_compression_ratio:.1f}:1 compression)")
        
        try:
            # Generate summary using the model
            # Use a fresh generation (no KV cache) to avoid contamination
            result = generator.generate(
                user_message=summarization_prompt,
                max_new_tokens=target_summary_tokens + 100,  # Slight buffer
                temperature=0.3,  # More focused/deterministic
                use_cached_prompt=False  # Don't use system prompt cache for this
            )
            
            summary_text = result["generated_text"]
            summary_tokens = len(generator.tokenizer.encode(summary_text))
            
            # Create summary object
            summary = ConversationSummary(
                turn_range=turn_range,
                original_tokens=original_token_count,
                summary_tokens=summary_tokens,
                summary_text=summary_text
            )
            
            logger.info(f"✓ Summary generated successfully:")
            logger.info(f"  Compression: {summary.compression_ratio:.1f}:1")
            logger.info(f"  Tokens: {original_token_count} → {summary_tokens}")
            logger.info(f"  First 100 chars: {summary_text[:100]}...")
            
            # Store summary
            self.summaries.append(summary)
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to generate summary: {e}")
            # Create a simple fallback summary
            fallback_text = f"[Summary of turns {turn_range[0]}-{turn_range[1]}: {original_token_count} tokens of conversation history]"
            fallback_tokens = len(generator.tokenizer.encode(fallback_text))
            
            summary = ConversationSummary(
                turn_range=turn_range,
                original_tokens=original_token_count,
                summary_tokens=fallback_tokens,
                summary_text=fallback_text
            )
            self.summaries.append(summary)
            
            logger.warning(f"Using fallback summary due to error")
            return summary
    
    def get_compressed_context(self) -> str:
        """
        Get all summaries as a single text block for context.
        
        Returns:
            Formatted string with all summaries
        """
        if not self.summaries:
            return ""
        
        summary_texts = []
        summary_texts.append("=== CONVERSATION HISTORY (COMPRESSED) ===\n")
        
        for i, summary in enumerate(self.summaries, 1):
            summary_texts.append(f"[Summary {i} - Turns {summary.turn_range[0]}-{summary.turn_range[1]}]")
            summary_texts.append(summary.summary_text)
            summary_texts.append("")  # Blank line
        
        summary_texts.append("=== END COMPRESSED HISTORY ===\n")
        
        return "\n".join(summary_texts)
    
    def get_stats(self) -> Dict:
        """Get statistics about summarization performance"""
        if not self.summaries:
            return {
                "num_summaries": 0,
                "total_original_tokens": 0,
                "total_summary_tokens": 0,
                "avg_compression_ratio": 0.0
            }
        
        total_original = sum(s.original_tokens for s in self.summaries)
        total_summary = sum(s.summary_tokens for s in self.summaries)
        avg_compression = sum(s.compression_ratio for s in self.summaries) / len(self.summaries)
        
        return {
            "num_summaries": len(self.summaries),
            "total_original_tokens": total_original,
            "total_summary_tokens": total_summary,
            "avg_compression_ratio": avg_compression,
            "summaries": self.summaries
        }
    
    def increment_turn(self):
        """Track conversation turn for summary metadata"""
        self.current_turn += 1
    
    def clear(self):
        """Clear all summaries (e.g., at conversation reset)"""
        logger.info(f"Clearing {len(self.summaries)} summaries")
        self.summaries.clear()
        self.current_turn = 0
