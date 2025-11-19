"""
GPU Memory Monitoring Utility

This module provides lightweight GPU memory tracking for long-running experiments.
Helps determine safe limits for token counts, conversation history, and KV cache sizes.

Key features:
- Negligible overhead (~100 bytes per snapshot)
- Tracks allocated/reserved memory at key events
- Provides smart recommendations based on available headroom
- Works with any PyTorch CUDA-enabled model

Author: AGI Self-Modification Research
Date: November 11, 2025
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class GPUMonitor:
    """
    Lightweight GPU memory monitor for tracking memory usage during experiments.
    
    Usage:
        monitor = GPUMonitor(logger)
        monitor.snapshot("model_loaded")
        # ... run experiments ...
        monitor.snapshot("generation_complete", {"tokens": 500})
        monitor.print_summary()
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None, gpu_total_gb: float = 15.0):
        """
        Initialize GPU monitor.
        
        Args:
            logger: Optional logger for output (creates default if None)
            gpu_total_gb: Total GPU memory in GB (default: 15.0 for T4)
        """
        self.logger = logger or logging.getLogger(__name__)
        self.gpu_total_gb = gpu_total_gb
        self.gpu_usable_gb = gpu_total_gb - 2.0  # Reserve ~2GB for system
        self.snapshots: List[Dict[str, Any]] = []
        
        # Check if CUDA is available
        self.cuda_available = TORCH_AVAILABLE and torch.cuda.is_available()
        
        if not self.cuda_available:
            if not TORCH_AVAILABLE:
                self.logger.warning("[GPU MONITOR] PyTorch not available - monitoring disabled")
            else:
                self.logger.warning("[GPU MONITOR] CUDA not available - monitoring disabled")
        else:
            self.logger.info(f"[GPU MONITOR] Initialized - CUDA available, {self.gpu_total_gb:.1f} GB total")
    
    def snapshot(self, event: str, details: Optional[Dict[str, Any]] = None):
        """
        Capture a lightweight GPU memory snapshot.
        
        Args:
            event: Description of the event (e.g., "generation_start", "after_tool_call")
            details: Optional additional details (conversation length, token count, etc.)
        """
        if not self.cuda_available:
            self.logger.debug(f"[GPU MONITOR] Snapshot '{event}' skipped - CUDA not available")
            return
        
        snapshot = {
            "event": event,
            "timestamp": datetime.now().isoformat(),
            "allocated_gb": torch.cuda.memory_allocated() / 1024**3,
            "reserved_gb": torch.cuda.memory_reserved() / 1024**3,
            "max_allocated_gb": torch.cuda.max_memory_allocated() / 1024**3,
            "max_reserved_gb": torch.cuda.max_memory_reserved() / 1024**3,
        }
        
        if details:
            snapshot.update(details)
        
        self.snapshots.append(snapshot)
        self.logger.debug(f"[GPU MONITOR] Snapshot '{event}': {snapshot['allocated_gb']:.2f} GB allocated, {snapshot['reserved_gb']:.2f} GB reserved")
    
    def get_current_memory(self) -> Dict[str, float]:
        """
        Get current GPU memory usage.
        
        Returns:
            Dict with allocated_gb and reserved_gb (or zeros if CUDA unavailable)
        """
        if not self.cuda_available:
            return {"allocated_gb": 0.0, "reserved_gb": 0.0}
        
        return {
            "allocated_gb": torch.cuda.memory_allocated() / 1024**3,
            "reserved_gb": torch.cuda.memory_reserved() / 1024**3,
        }
    
    def get_peak_memory(self) -> Dict[str, float]:
        """
        Get peak GPU memory usage across all snapshots.
        
        Returns:
            Dict with peak_allocated_gb and peak_reserved_gb
        """
        if not self.snapshots:
            return {"peak_allocated_gb": 0.0, "peak_reserved_gb": 0.0}
        
        allocated_values = [s["allocated_gb"] for s in self.snapshots]
        reserved_values = [s["reserved_gb"] for s in self.snapshots]
        
        return {
            "peak_allocated_gb": max(allocated_values),
            "peak_reserved_gb": max(reserved_values),
        }
    
    def get_average_memory(self) -> Dict[str, float]:
        """
        Get average GPU memory usage across all snapshots.
        
        Returns:
            Dict with avg_allocated_gb and avg_reserved_gb
        """
        if not self.snapshots:
            return {"avg_allocated_gb": 0.0, "avg_reserved_gb": 0.0}
        
        allocated_values = [s["allocated_gb"] for s in self.snapshots]
        reserved_values = [s["reserved_gb"] for s in self.snapshots]
        
        return {
            "avg_allocated_gb": sum(allocated_values) / len(allocated_values),
            "avg_reserved_gb": sum(reserved_values) / len(reserved_values),
        }
    
    def get_headroom(self) -> float:
        """
        Calculate available memory headroom based on peak usage.
        
        Returns:
            Available headroom in GB (negative if over budget)
        """
        if not self.snapshots:
            return self.gpu_usable_gb
        
        peak = self.get_peak_memory()
        return self.gpu_usable_gb - peak["peak_reserved_gb"]
    
    def print_summary(
        self,
        current_limits: Optional[Dict[str, Any]] = None,
        include_recommendations: bool = True
    ):
        """
        Print a comprehensive summary of GPU memory usage.
        
        Args:
            current_limits: Optional dict with current limits for recommendations
                           (e.g., {"max_new_tokens": 500, "max_conversation_tokens": 2000})
            include_recommendations: Whether to include limit increase recommendations
        """
        if not self.cuda_available:
            self.logger.info("[GPU MONITOR] CUDA not available - no summary to print")
            return
        
        if not self.snapshots:
            self.logger.info("[GPU MONITOR] No snapshots collected")
            return
        
        self.logger.info("\n" + "=" * 80)
        self.logger.info("GPU MEMORY USAGE SUMMARY")
        self.logger.info("=" * 80)
        
        # Overall statistics
        peak = self.get_peak_memory()
        avg = self.get_average_memory()
        
        self.logger.info(f"\nOverall Statistics:")
        self.logger.info(f"  GPU Total: {self.gpu_total_gb:.1f} GB")
        self.logger.info(f"  GPU Usable (after system): {self.gpu_usable_gb:.1f} GB")
        self.logger.info(f"  Peak allocated: {peak['peak_allocated_gb']:.2f} GB")
        self.logger.info(f"  Peak reserved: {peak['peak_reserved_gb']:.2f} GB")
        self.logger.info(f"  Avg allocated: {avg['avg_allocated_gb']:.2f} GB")
        self.logger.info(f"  Avg reserved: {avg['avg_reserved_gb']:.2f} GB")
        
        # Key events
        self.logger.info(f"\nKey Memory Events:")
        important_events = [
            "session_start", "after_model_load", "generation_start", 
            "generation_end", "after_pruning", "session_end"
        ]
        for event_name in important_events:
            matching = [s for s in self.snapshots if s["event"] == event_name]
            if matching:
                snapshot = matching[-1]  # Most recent
                self.logger.info(
                    f"  {event_name:20s}: {snapshot['allocated_gb']:.2f} GB allocated, "
                    f"{snapshot['reserved_gb']:.2f} GB reserved"
                )
        
        # Memory growth during generation
        gen_start = [s for s in self.snapshots if s["event"] == "generation_start"]
        gen_end = [s for s in self.snapshots if s["event"] == "generation_end"]
        if gen_start and gen_end:
            avg_growth = sum(
                gen_end[i]["allocated_gb"] - gen_start[i]["allocated_gb"] 
                for i in range(min(len(gen_start), len(gen_end)))
            ) / min(len(gen_start), len(gen_end))
            self.logger.info(f"\nAverage memory growth per generation: {avg_growth:.3f} GB")
        
        # Conversation length vs memory
        conv_lengths = [
            s.get("conversation_turns", 0) 
            for s in self.snapshots 
            if "conversation_turns" in s
        ]
        if conv_lengths:
            self.logger.info(
                f"Conversation length tracked: {min(conv_lengths)} to {max(conv_lengths)} turns"
            )
        
        # Recommendations
        if include_recommendations:
            self._print_recommendations(peak['peak_reserved_gb'], current_limits)
        
        self.logger.info("=" * 80 + "\n")
    
    def _print_recommendations(
        self, 
        peak_reserved: float, 
        current_limits: Optional[Dict[str, Any]] = None
    ):
        """Print recommendations for increasing limits based on headroom"""
        self.logger.info(f"\n" + "-" * 80)
        self.logger.info("RECOMMENDATIONS:")
        
        headroom = self.gpu_usable_gb - peak_reserved
        
        if headroom > 3.0:
            self.logger.info(f"  ✓ Plenty of headroom ({headroom:.1f} GB available)")
            self.logger.info(f"  ✓ Can safely INCREASE limits significantly")
            
            if current_limits:
                self._print_limit_suggestions(current_limits, "aggressive")
            else:
                self.logger.info(f"  ✓ Suggested increases:")
                self.logger.info(f"      - max_new_tokens: +33-78% increase recommended")
                self.logger.info(f"      - max_conversation_tokens: +50-100% increase recommended")
                self.logger.info(f"      - keep_recent_turns: +50-100% increase recommended")
        
        elif headroom > 1.5:
            self.logger.info(f"  ⚠ Moderate headroom ({headroom:.1f} GB available)")
            self.logger.info(f"  ⚠ Can SLIGHTLY increase limits")
            
            if current_limits:
                self._print_limit_suggestions(current_limits, "moderate")
            else:
                self.logger.info(f"  ⚠ Suggested increases:")
                self.logger.info(f"      - max_new_tokens: +20-25% increase recommended")
                self.logger.info(f"      - max_conversation_tokens: +25% increase recommended")
                self.logger.info(f"      - keep_recent_turns: +50% increase recommended")
        
        else:
            self.logger.info(f"  🔴 Limited headroom ({headroom:.1f} GB available)")
            self.logger.info(f"  🔴 DO NOT increase limits - risk of OOM")
            self.logger.info(f"  🔴 Current limits are appropriate for this hardware")
    
    def _print_limit_suggestions(self, current_limits: Dict[str, Any], mode: str):
        """Print specific limit increase suggestions"""
        self.logger.info(f"  Suggested limit increases:")
        
        if "max_new_tokens" in current_limits:
            current = current_limits["max_new_tokens"]
            if mode == "aggressive":
                suggested = f"{current} → {current + 150}-{current + 350}"
            else:  # moderate
                suggested = f"{current} → {current + 100}"
            self.logger.info(f"      - max_new_tokens: {suggested}")
        
        if "max_conversation_tokens" in current_limits:
            current = current_limits["max_conversation_tokens"]
            if mode == "aggressive":
                suggested = f"{current} → {int(current * 1.5)}-{current * 2}"
            else:  # moderate
                suggested = f"{current} → {int(current * 1.25)}"
            self.logger.info(f"      - max_conversation_tokens: {suggested}")
        
        if "keep_recent_turns" in current_limits:
            current = current_limits["keep_recent_turns"]
            if mode == "aggressive":
                suggested = f"{current} → {current + 1}-{current + 2}"
            else:  # moderate
                suggested = f"{current} → {current + 1}"
            self.logger.info(f"      - keep_recent_turns: {suggested}")
    
    def reset_peak_stats(self):
        """Reset peak memory statistics (useful for testing different configurations)"""
        if self.cuda_available:
            torch.cuda.reset_peak_memory_stats()
    
    def clear_snapshots(self):
        """Clear all collected snapshots"""
        self.snapshots.clear()
