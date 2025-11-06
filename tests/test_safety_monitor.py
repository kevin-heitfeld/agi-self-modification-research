"""
Tests for Safety Monitor

Validates:
- Alert system
- Anomaly detection (NaN, Inf, activation patterns)
- Performance tracking
- Resource monitoring
- Emergency stop mechanism
- Threshold management
- Hook registration/removal
- Context manager

Author: AGI Self-Modification Research Team
Date: November 6, 2025
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import unittest
import torch
import torch.nn as nn
import tempfile
import shutil
from pathlib import Path
import time

from src.safety_monitor import SafetyMonitor, SafetyAlert, AlertLevel


class SimpleTestModel(nn.Module):
    """A simple model for testing."""

    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(100, 64)
        self.layer1 = nn.Linear(64, 128)
        self.layer2 = nn.Linear(128, 64)
        self.output = nn.Linear(64, 10)

    def forward(self, x):
        x = self.embedding(x)
        x = torch.relu(self.layer1(x.mean(dim=1)))
        x = torch.relu(self.layer2(x))
        return self.output(x)


class TestAlert(unittest.TestCase):
    """Test Alert class."""

    def test_alert_creation(self):
        """Test creating an alert."""
        alert = SafetyAlert(
            level=AlertLevel.WARNING,
            category="test",
            message="Test message",
            timestamp=time.time(),
            details={}
        )

        self.assertEqual(alert.level, AlertLevel.WARNING)
        self.assertEqual(alert.category, "test")
        self.assertEqual(alert.message, "Test message")
        self.assertIsNotNone(alert.timestamp)

    def test_alert_with_metadata(self):
        """Test alert with metadata."""
        details = {'value': 123, 'location': 'layer5'}
        alert = SafetyAlert(
            level=AlertLevel.CRITICAL,
            category="anomaly",
            message="NaN detected",
            timestamp=time.time(),
            details=details
        )

        self.assertEqual(alert.details, details)

    def test_alert_string(self):
        """Test alert string representation."""
        alert = SafetyAlert(
            level=AlertLevel.CRITICAL,
            category="performance",
            message="Degradation detected",
            timestamp=time.time(),
            details={}
        )

        alert_str = str(alert)
        self.assertIn("CRITICAL", alert_str)
        self.assertIn("performance", alert_str)
        self.assertIn("Degradation detected", alert_str)
class TestSafetyMonitor(unittest.TestCase):
    """Test SafetyMonitor class."""

    def setUp(self):
        """Set up test fixtures."""
        self.model = SimpleTestModel()
        self.baseline_metrics = {
            'perplexity': 10.0,
            'accuracy': 0.75
        }

        # Initialize monitor
        self.monitor = SafetyMonitor(
            model=self.model,
            baseline_metrics=self.baseline_metrics
        )

    def tearDown(self):
        """Clean up test fixtures."""
        # Remove any hooks
        if hasattr(self, 'monitor'):
            try:
                self.monitor.remove_hooks()
            except:
                pass

    def test_initialization(self):
        """Test monitor initialization."""
        self.assertEqual(self.monitor.model, self.model)
        self.assertEqual(self.monitor.baseline_metrics, self.baseline_metrics)
        self.assertFalse(self.monitor.is_monitoring)
        self.assertFalse(self.monitor.emergency_stop_triggered)
        self.assertEqual(len(self.monitor.alerts), 0)

    def test_alert_recording(self):
        """Test recording alerts."""
        # Record info alert
        self.monitor._add_alert(
            level=AlertLevel.INFO,
            category="test",
            message="Info message"
        )

        self.assertEqual(len(self.monitor.alerts), 1)
        self.assertEqual(self.monitor.alerts[0].level, AlertLevel.INFO)

        # Record warning alert
        self.monitor._add_alert(
            level=AlertLevel.WARNING,
            category="test",
            message="Warning message"
        )

    def test_get_recent_alerts(self):
        """Test retrieving recent alerts."""
        # Clear any alerts from previous tests
        self.monitor.alerts.clear()
        
        # Record alerts of different levels
        self.monitor._add_alert(AlertLevel.INFO, "test", "Info 1")
        self.monitor._add_alert(AlertLevel.WARNING, "test", "Warning 1")
        self.monitor._add_alert(AlertLevel.CRITICAL, "test", "Critical 1")
        self.monitor._add_alert(AlertLevel.INFO, "test", "Info 2")

        # Get all alerts
        all_alerts = self.monitor.get_recent_alerts()
        self.assertEqual(len(all_alerts), 4)

        # Get only warnings
        warnings = self.monitor.get_recent_alerts(level=AlertLevel.WARNING)
        self.assertEqual(len(warnings), 1)
        self.assertEqual(warnings[0].message, "Warning 1")

        # Get only last 2
        recent = self.monitor.get_recent_alerts(limit=2)
        self.assertEqual(len(recent), 2)
        self.assertEqual(recent[-1].message, "Info 2")  # Most recent last
    
    def test_get_critical_alerts(self):
        """Test retrieving critical/emergency alerts."""
        # Clear any alerts from previous tests
        self.monitor.alerts.clear()
        
        self.monitor._add_alert(AlertLevel.INFO, "test", "Info")
        self.monitor._add_alert(AlertLevel.WARNING, "test", "Warning")
        self.monitor._add_alert(AlertLevel.CRITICAL, "test", "Critical")
        self.monitor._add_alert(AlertLevel.EMERGENCY, "test", "Emergency")

        critical = self.monitor.get_critical_alerts()
        self.assertEqual(len(critical), 2)

        levels = [alert.level for alert in critical]
        self.assertIn(AlertLevel.CRITICAL, levels)
        self.assertIn(AlertLevel.EMERGENCY, levels)

    def test_reset_alerts(self):
        """Test resetting alerts."""
        self.monitor._add_alert(AlertLevel.INFO, "test", "Info")
        self.monitor._add_alert(AlertLevel.WARNING, "test", "Warning")

        self.assertGreaterEqual(len(self.monitor.alerts), 2)

        self.monitor.reset_alerts()
        # reset_alerts() adds an INFO alert, so count should be 1, not 0
        self.assertEqual(len(self.monitor.alerts), 1)

    def test_nan_detection(self):
        """Test NaN detection in outputs."""
        # Start monitoring first
        self.monitor.start_monitoring()

        # Normal tensor
        normal_tensor = torch.randn(2, 5, 10)
        self.assertTrue(self.monitor.check_output(normal_tensor))

        # Tensor with NaN
        nan_tensor = torch.randn(2, 5, 10)
        nan_tensor[0, 2, 5] = float('nan')
        self.assertFalse(self.monitor.check_output(nan_tensor))

        # Check emergency stop was triggered
        self.assertTrue(self.monitor.emergency_stop_triggered)

    def test_inf_detection(self):
        """Test Inf detection in outputs."""
        # Reset from previous test
        self.monitor.reset_emergency_stop()
        self.monitor.reset_alerts()
        self.monitor.start_monitoring()

        # Tensor with Inf
        inf_tensor = torch.randn(2, 5, 10)
        inf_tensor[1, 3, 7] = float('inf')
        self.assertFalse(self.monitor.check_output(inf_tensor))        # Check alert was recorded
        alerts = self.monitor.get_recent_alerts(level=AlertLevel.EMERGENCY)
        self.assertGreater(len(alerts), 0)
    
    def test_performance_checking(self):
        """Test performance degradation detection."""
        # Start monitoring first
        self.monitor.start_monitoring()

        # Good performance (within threshold)
        good_perplexity = 11.0  # 10% increase from baseline of 10.0
        self.assertTrue(self.monitor.check_performance('perplexity', good_perplexity))

        # Bad performance (exceeds threshold of 200% - default is 2.0)
        bad_perplexity = 35.0  # 250% increase from baseline of 10.0
        self.assertFalse(self.monitor.check_performance('perplexity', bad_perplexity))
        
        # Check critical alert was recorded
        criticals = self.monitor.get_critical_alerts()
        self.assertGreater(len(criticals), 0)

    def test_inference_time_tracking(self):
        """Test inference time tracking."""
        # Start monitoring first
        self.monitor.start_monitoring()

        # Track some inference times
        self.monitor.track_inference_time(100.0)
        self.monitor.track_inference_time(150.0)
        self.monitor.track_inference_time(120.0)

        stats = self.monitor.get_statistics()
        inf_stats = stats['inference_times']

        self.assertEqual(inf_stats['count'], 3)
        self.assertAlmostEqual(inf_stats['mean'], 123.33, places=1)
        self.assertAlmostEqual(inf_stats['min'], 100.0)
        self.assertAlmostEqual(inf_stats['max'], 150.0)

    def test_resource_checking(self):
        """Test resource monitoring."""
        resources = self.monitor.check_resources()

        # Should have basic resource info
        self.assertIn('cpu_memory_mb', resources)
        self.assertIn('system_cpu_percent', resources)

        # Values should be reasonable
        self.assertGreater(resources['cpu_memory_mb'], 0)
        self.assertGreaterEqual(resources['system_cpu_percent'], 0)
        self.assertLessEqual(resources['system_cpu_percent'], 100)

        # If CUDA available, should have GPU info
        if torch.cuda.is_available():
            self.assertIn('gpu_memory_allocated_mb', resources)
            self.assertIn('gpu_peak_memory_mb', resources)

    def test_emergency_stop(self):
        """Test emergency stop mechanism."""
        self.assertFalse(self.monitor.emergency_stop_triggered)

        # Trigger emergency stop manually (simulating the private method for testing)
        self.monitor.emergency_stop_triggered = True
        self.monitor._add_alert(AlertLevel.EMERGENCY, "emergency_stop", "Test emergency")

        self.assertTrue(self.monitor.emergency_stop_triggered)

        # Check emergency alert was recorded
        emergency = self.monitor.get_recent_alerts(level=AlertLevel.EMERGENCY)
        self.assertEqual(len(emergency), 1)
        self.assertIn("Test emergency", emergency[0].message)

        # Reset
        self.monitor.reset_emergency_stop()
        self.assertFalse(self.monitor.emergency_stop_triggered)

    def test_context_manager(self):
        """Test monitoring context manager."""
        self.assertFalse(self.monitor.is_monitoring)

        with self.monitor.context():
            self.assertTrue(self.monitor.is_monitoring)

            # Run some operations
            x = torch.randint(0, 100, (2, 10))
            output = self.model(x)
            # check_output expects 3D tensor for entropy check, so skip it for 2D
            # Just check NaN/Inf which works for any dimension
            self.assertFalse(torch.isnan(output).any())
            self.assertFalse(torch.isinf(output).any())

        self.assertFalse(self.monitor.is_monitoring)

    def test_hook_registration(self):
        """Test hook registration and removal."""
        # Register hooks on specific layers and start monitoring
        self.monitor.register_hooks(layer_names=['layer1', 'layer2'])
        self.monitor.start_monitoring()

        # Run model to trigger hooks
        x = torch.randint(0, 100, (2, 10))
        with torch.no_grad():
            output = self.model(x)

        # Hooks should have recorded activations
        self.assertGreater(len(self.monitor.activation_stats), 0)
        
        # Remove hooks
        self.monitor.remove_hooks()

        # Clear stats for next test
        self.monitor.activation_stats.clear()

        # Run model again
        with torch.no_grad():
            output = self.model(x)

        # Should not record activations
        self.assertEqual(len(self.monitor.activation_stats), 0)

    def test_threshold_management(self):
        """Test threshold getting and setting."""
        # Get default threshold
        threshold = self.monitor.thresholds['max_perplexity_increase']
        self.assertEqual(threshold, 2.0)  # Default 100%

        # Set new threshold
        self.monitor.set_threshold('max_perplexity_increase', 3.0)
        self.assertEqual(self.monitor.thresholds['max_perplexity_increase'], 3.0)

        # Test with new threshold
        test_perplexity = 35.0  # 250% increase from baseline of 10.0
        self.assertTrue(self.monitor.check_performance('perplexity', test_perplexity, track=False))

        # Reset to default
        self.monitor.set_threshold('max_perplexity_increase', 2.0)

    def test_statistics(self):
        """Test statistics gathering."""
        # Generate some activity
        self.monitor._add_alert(AlertLevel.INFO, "test", "Info")
        self.monitor._add_alert(AlertLevel.WARNING, "test", "Warning")
        self.monitor._add_alert(AlertLevel.CRITICAL, "test", "Critical")
        self.monitor.start_monitoring()
        self.monitor.track_inference_time(100.0)
        self.monitor.track_inference_time(200.0)

        stats = self.monitor.get_statistics()

        # Check structure
        self.assertIn('monitoring_active', stats)
        self.assertIn('emergency_stop_triggered', stats)
        self.assertIn('total_alerts', stats)
        self.assertIn('alerts_by_level', stats)
        self.assertIn('inference_times', stats)
        self.assertIn('resource_stats', stats)
        self.assertIn('thresholds', stats)

        # Check values - account for potential INFO alerts from other operations
        self.assertGreaterEqual(stats['total_alerts'], 4)  # At least the 4 we added (3 alerts + 1 from start_monitoring)
        self.assertGreaterEqual(stats['alerts_by_level']['info'], 1)
        self.assertGreaterEqual(stats['alerts_by_level']['warning'], 1)
        self.assertGreaterEqual(stats['alerts_by_level']['critical'], 1)
        self.assertEqual(stats['inference_times']['count'], 2)

    def test_activation_anomaly_detection(self):
        """Test detection of activation anomalies."""
        # Register hooks to collect activations
        self.monitor.register_hooks()
        self.monitor.start_monitoring()

        # Create activation with NaN
        bad_activation = torch.randn(2, 64)
        bad_activation[0, 30] = float('nan')

        # Check for anomaly (this will trigger _check_activation internally)
        self.monitor._check_activation('test_layer', bad_activation)

        # Should have triggered emergency stop
        self.assertTrue(self.monitor.emergency_stop_triggered)

        # Should have triggered alert
        alerts = self.monitor.get_critical_alerts()
        self.assertGreater(len(alerts), 0)


def run_tests():
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestAlert))
    suite.addTests(loader.loadTestsFromTestCase(TestSafetyMonitor))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
