"""
Safety Monitor - Real-time Anomaly Detection and Emergency Control

This module provides comprehensive safety monitoring for the AGI self-modification
research project. It detects anomalous behavior, tracks performance degradation,
and provides emergency stop capabilities.

Key Features:
- Real-time anomaly detection during inference
- Performance degradation tracking
- Emergency stop mechanism
- Resource monitoring (GPU, CPU, memory)
- Behavioral consistency checks
- Automatic alerts and logging
- Integration with checkpointing for auto-rollback

Safety Philosophy:
The system must be able to detect when something goes wrong and stop before
causing damage. This monitor acts as a watchdog, constantly checking for
anomalies and ready to trigger emergency procedures.

Author: AGI Self-Modification Research Team
Date: November 6, 2025
"""

import torch
import torch.nn as nn
import psutil
import time
import warnings
from typing import Dict, List, Optional, Any, Callable
from collections import deque
from dataclasses import dataclass
from enum import Enum
import numpy as np


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class SafetyAlert:
    """Represents a safety alert."""
    level: AlertLevel
    category: str
    message: str
    timestamp: float
    details: Dict[str, Any]

    def __str__(self):
        return f"[{self.level.value.upper()}] {self.category}: {self.message}"


class SafetyMonitor:
    """
    Real-time safety monitoring for model operations.

    This class continuously monitors:
    - Model outputs for anomalies
    - Performance degradation
    - Resource usage (GPU, CPU, memory)
    - Behavioral consistency
    - NaN/Inf in activations
    - Gradient explosions

    It can:
    - Issue alerts at different severity levels
    - Trigger emergency stops
    - Auto-rollback to safe checkpoints
    - Log all safety events

    Usage:
        >>> monitor = SafetyMonitor(model)
        >>> monitor.register_hooks()
        >>>
        >>> # During operations
        >>> with monitor.context():
        ...     output = model(input)
        ...     if monitor.check_output(output):
        ...         print("Output is safe")
        >>>
        >>> # Check for alerts
        >>> alerts = monitor.get_recent_alerts()
    """

    def __init__(
        self,
        model: nn.Module,
        baseline_metrics: Optional[Dict[str, float]] = None,
        emergency_callback: Optional[Callable] = None
    ):
        """
        Initialize the safety monitor.

        Args:
            model: The model to monitor
            baseline_metrics: Baseline performance metrics for comparison
            emergency_callback: Function to call in emergency (e.g., auto-rollback)
        """
        self.model = model
        self.baseline_metrics = baseline_metrics or {}
        self.emergency_callback = emergency_callback

        # Alert history
        self.alerts: deque = deque(maxlen=1000)

        # Monitoring state
        self.is_monitoring = False
        self.emergency_stop_triggered = False

        # Performance tracking
        self.performance_history: deque = deque(maxlen=100)
        self.inference_times: deque = deque(maxlen=100)

        # Anomaly detection thresholds
        self.thresholds = {
            'max_perplexity_increase': 2.0,  # 100% increase
            'max_memory_usage_mb': 8000,  # 8GB
            'max_inference_time_ms': 5000,  # 5 seconds
            'max_nan_ratio': 0.0,  # No NaNs allowed
            'max_gradient_norm': 100.0,
            'min_output_entropy': 0.1,  # Too deterministic
            'max_output_entropy': 10.0,  # Too random
        }

        # Resource monitoring
        self.start_memory = None
        self.peak_memory = 0

        # Hooks for activation monitoring
        self.hooks = []
        self.activation_stats = {}

    def register_hooks(self, layer_names: Optional[List[str]] = None) -> None:
        """
        Register forward hooks for activation monitoring.

        Args:
            layer_names: Optional list of layer names to monitor.
                        If None, monitors key layers automatically.
        """
        def create_hook(name: str):
            def hook(module, input, output):
                if self.is_monitoring:
                    self._check_activation(name, output)
            return hook

        # Auto-select layers if not specified
        if layer_names is None:
            layer_names = []
            for name, module in self.model.named_modules():
                # Monitor attention and MLP outputs
                if any(key in name.lower() for key in ['attention', 'mlp', 'ffn']):
                    layer_names.append(name)

        # Register hooks
        modules = dict(self.model.named_modules())
        for name in layer_names:
            if name in modules:
                hook = modules[name].register_forward_hook(create_hook(name))
                self.hooks.append(hook)

    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def start_monitoring(self) -> None:
        """Start monitoring."""
        self.is_monitoring = True
        self.emergency_stop_triggered = False
        self.start_memory = self._get_memory_usage()
        self._add_alert(AlertLevel.INFO, "monitoring", "Safety monitoring started")

    def stop_monitoring(self) -> None:
        """Stop monitoring."""
        self.is_monitoring = False
        self._add_alert(AlertLevel.INFO, "monitoring", "Safety monitoring stopped")

    def context(self):
        """Context manager for monitored operations."""
        class MonitorContext:
            def __init__(self, monitor):
                self.monitor = monitor

            def __enter__(self):
                self.monitor.start_monitoring()
                return self.monitor

            def __exit__(self, exc_type, exc_val, exc_tb):
                self.monitor.stop_monitoring()
                return False

        return MonitorContext(self)

    def check_output(
        self,
        output: torch.Tensor,
        expected_shape: Optional[tuple] = None
    ) -> bool:
        """
        Check if model output is safe.

        Args:
            output: Model output tensor
            expected_shape: Expected output shape

        Returns:
            True if safe, False if anomaly detected
        """
        if not self.is_monitoring:
            return True

        # Check for NaN/Inf
        if torch.isnan(output).any():
            self._add_alert(
                AlertLevel.CRITICAL,
                "output_anomaly",
                "NaN detected in model output",
                {'nan_count': torch.isnan(output).sum().item()}
            )
            self._trigger_emergency_stop("NaN in output")
            return False

        if torch.isinf(output).any():
            self._add_alert(
                AlertLevel.CRITICAL,
                "output_anomaly",
                "Inf detected in model output",
                {'inf_count': torch.isinf(output).sum().item()}
            )
            self._trigger_emergency_stop("Inf in output")
            return False

        # Check shape if expected
        if expected_shape and output.shape != expected_shape:
            self._add_alert(
                AlertLevel.WARNING,
                "output_anomaly",
                f"Unexpected output shape: {output.shape} (expected {expected_shape})"
            )

        # Check output entropy (for language models)
        if output.dim() >= 2:
            probs = torch.softmax(output[:, -1, :], dim=-1)
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean().item()

            if entropy < self.thresholds['min_output_entropy']:
                self._add_alert(
                    AlertLevel.WARNING,
                    "output_anomaly",
                    f"Output too deterministic (entropy={entropy:.4f})",
                    {'entropy': entropy}
                )
            elif entropy > self.thresholds['max_output_entropy']:
                self._add_alert(
                    AlertLevel.WARNING,
                    "output_anomaly",
                    f"Output too random (entropy={entropy:.4f})",
                    {'entropy': entropy}
                )

        return True

    def check_performance(
        self,
        metric_name: str,
        metric_value: float,
        track: bool = True
    ) -> bool:
        """
        Check if performance metric is within acceptable range.

        Args:
            metric_name: Name of the metric (e.g., 'perplexity', 'accuracy')
            metric_value: Current metric value
            track: Whether to track this in performance history

        Returns:
            True if acceptable, False if degradation detected
        """
        if not self.is_monitoring:
            return True

        if track:
            self.performance_history.append({
                'metric': metric_name,
                'value': metric_value,
                'timestamp': time.time()
            })

        # Check against baseline
        if metric_name in self.baseline_metrics:
            baseline = self.baseline_metrics[metric_name]

            # For perplexity, lower is better
            if 'perplexity' in metric_name.lower():
                if metric_value > baseline * (1 + self.thresholds['max_perplexity_increase']):
                    self._add_alert(
                        AlertLevel.CRITICAL,
                        "performance_degradation",
                        f"{metric_name} increased from {baseline:.2f} to {metric_value:.2f}",
                        {
                            'metric': metric_name,
                            'baseline': baseline,
                            'current': metric_value,
                            'increase': (metric_value - baseline) / baseline * 100
                        }
                    )
                    return False

            # For accuracy/score metrics, higher is better
            elif any(key in metric_name.lower() for key in ['accuracy', 'score', 'f1']):
                if metric_value < baseline * 0.5:  # 50% drop is critical
                    self._add_alert(
                        AlertLevel.CRITICAL,
                        "performance_degradation",
                        f"{metric_name} dropped from {baseline:.2f} to {metric_value:.2f}",
                        {
                            'metric': metric_name,
                            'baseline': baseline,
                            'current': metric_value,
                            'drop': (baseline - metric_value) / baseline * 100
                        }
                    )
                    return False

        return True

    def check_resources(self) -> Dict[str, Any]:
        """
        Check resource usage (memory, GPU).

        Returns:
            Dictionary with resource statistics
        """
        stats = {}

        # CPU memory
        process = psutil.Process()
        stats['cpu_memory_mb'] = process.memory_info().rss / 1024 / 1024
        stats['cpu_memory_percent'] = process.memory_percent()

        # GPU memory if available
        if torch.cuda.is_available():
            stats['gpu_memory_allocated_mb'] = torch.cuda.memory_allocated() / 1024 / 1024
            stats['gpu_memory_reserved_mb'] = torch.cuda.memory_reserved() / 1024 / 1024
            stats['gpu_memory_percent'] = (
                torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory * 100
            )

            # Track peak
            peak = torch.cuda.max_memory_allocated() / 1024 / 1024
            if peak > self.peak_memory:
                self.peak_memory = peak
            stats['gpu_peak_memory_mb'] = self.peak_memory

            # Check threshold
            if stats['gpu_memory_allocated_mb'] > self.thresholds['max_memory_usage_mb']:
                self._add_alert(
                    AlertLevel.CRITICAL,
                    "resource_limit",
                    f"GPU memory usage ({stats['gpu_memory_allocated_mb']:.0f} MB) exceeds threshold",
                    stats
                )

        # System-wide stats
        stats['system_cpu_percent'] = psutil.cpu_percent()
        stats['system_memory_percent'] = psutil.virtual_memory().percent

        return stats

    def track_inference_time(self, inference_time_ms: float) -> None:
        """
        Track inference time and alert if too slow.

        Args:
            inference_time_ms: Inference time in milliseconds
        """
        if not self.is_monitoring:
            return

        self.inference_times.append(inference_time_ms)

        if inference_time_ms > self.thresholds['max_inference_time_ms']:
            self._add_alert(
                AlertLevel.WARNING,
                "performance",
                f"Slow inference: {inference_time_ms:.0f}ms (threshold: {self.thresholds['max_inference_time_ms']}ms)",
                {'inference_time_ms': inference_time_ms}
            )

    def get_recent_alerts(
        self,
        level: Optional[AlertLevel] = None,
        limit: int = 100
    ) -> List[SafetyAlert]:
        """
        Get recent safety alerts.

        Args:
            level: Filter by alert level
            limit: Maximum number of alerts to return

        Returns:
            List of SafetyAlert objects
        """
        alerts = list(self.alerts)

        if level:
            alerts = [a for a in alerts if a.level == level]

        return alerts[-limit:]

    def get_critical_alerts(self) -> List[SafetyAlert]:
        """Get all critical and emergency alerts."""
        return [
            a for a in self.alerts
            if a.level in [AlertLevel.CRITICAL, AlertLevel.EMERGENCY]
        ]

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get monitoring statistics.

        Returns:
            Dictionary with comprehensive statistics
        """
        stats = {
            'monitoring_active': self.is_monitoring,
            'emergency_stop_triggered': self.emergency_stop_triggered,
            'total_alerts': len(self.alerts),
            'alerts_by_level': {},
            'resource_stats': self.check_resources(),
            'performance_history_size': len(self.performance_history),
            'inference_times': {
                'count': len(self.inference_times),
                'mean': np.mean(self.inference_times) if self.inference_times else 0,
                'std': np.std(self.inference_times) if self.inference_times else 0,
                'min': min(self.inference_times) if self.inference_times else 0,
                'max': max(self.inference_times) if self.inference_times else 0,
            },
            'thresholds': self.thresholds
        }

        # Count alerts by level
        for level in AlertLevel:
            stats['alerts_by_level'][level.value] = sum(
                1 for a in self.alerts if a.level == level
            )

        return stats

    def reset_alerts(self) -> None:
        """Clear all alerts."""
        self.alerts.clear()
        self._add_alert(AlertLevel.INFO, "monitoring", "Alerts cleared")

    def reset_emergency_stop(self) -> None:
        """Reset emergency stop flag (use with caution!)."""
        if self.emergency_stop_triggered:
            self.emergency_stop_triggered = False
            self._add_alert(
                AlertLevel.WARNING,
                "emergency",
                "Emergency stop reset - proceed with caution"
            )

    def set_threshold(self, threshold_name: str, value: float) -> None:
        """
        Update a safety threshold.

        Args:
            threshold_name: Name of the threshold
            value: New threshold value
        """
        if threshold_name in self.thresholds:
            old_value = self.thresholds[threshold_name]
            self.thresholds[threshold_name] = value
            self._add_alert(
                AlertLevel.INFO,
                "configuration",
                f"Threshold '{threshold_name}' changed from {old_value} to {value}"
            )
        else:
            raise ValueError(f"Unknown threshold: {threshold_name}")

    def _check_activation(self, layer_name: str, activation: torch.Tensor) -> None:
        """Check activation for anomalies."""
        if isinstance(activation, tuple):
            activation = activation[0]

        if not isinstance(activation, torch.Tensor):
            return

        # Check for NaN/Inf
        nan_count = torch.isnan(activation).sum().item()
        inf_count = torch.isinf(activation).sum().item()

        if nan_count > 0:
            self._add_alert(
                AlertLevel.CRITICAL,
                "activation_anomaly",
                f"NaN detected in {layer_name}",
                {'layer': layer_name, 'nan_count': nan_count}
            )
            self._trigger_emergency_stop(f"NaN in {layer_name}")

        if inf_count > 0:
            self._add_alert(
                AlertLevel.CRITICAL,
                "activation_anomaly",
                f"Inf detected in {layer_name}",
                {'layer': layer_name, 'inf_count': inf_count}
            )
            self._trigger_emergency_stop(f"Inf in {layer_name}")

        # Track activation statistics
        with torch.no_grad():
            self.activation_stats[layer_name] = {
                'mean': activation.mean().item(),
                'std': activation.std().item(),
                'min': activation.min().item(),
                'max': activation.max().item(),
            }

    def _add_alert(
        self,
        level: AlertLevel,
        category: str,
        message: str,
        details: Optional[Dict] = None
    ) -> None:
        """Add an alert to the history."""
        alert = SafetyAlert(
            level=level,
            category=category,
            message=message,
            timestamp=time.time(),
            details=details or {}
        )
        self.alerts.append(alert)

        # Print critical alerts immediately
        if level in [AlertLevel.CRITICAL, AlertLevel.EMERGENCY]:
            print(f"⚠️  {alert}")

    def _trigger_emergency_stop(self, reason: str) -> None:
        """Trigger emergency stop."""
        if self.emergency_stop_triggered:
            return  # Already triggered

        self.emergency_stop_triggered = True

        self._add_alert(
            AlertLevel.EMERGENCY,
            "emergency_stop",
            f"EMERGENCY STOP TRIGGERED: {reason}",
            {'reason': reason, 'timestamp': time.time()}
        )

        print("\n" + "=" * 70)
        print("🚨 EMERGENCY STOP TRIGGERED 🚨")
        print(f"Reason: {reason}")
        print("=" * 70 + "\n")

        # Call emergency callback if registered
        if self.emergency_callback:
            try:
                self.emergency_callback(reason)
            except Exception as e:
                print(f"Emergency callback failed: {e}")

    def _get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage."""
        usage = {}

        if torch.cuda.is_available():
            usage['gpu_mb'] = torch.cuda.memory_allocated() / 1024 / 1024

        process = psutil.Process()
        usage['cpu_mb'] = process.memory_info().rss / 1024 / 1024

        return usage

    def __del__(self):
        """Cleanup when monitor is destroyed."""
        self.remove_hooks()
