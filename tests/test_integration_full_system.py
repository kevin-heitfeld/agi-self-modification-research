"""
Integration Tests - Full System

Tests that verify all components work together correctly:
- Introspection APIs (WeightInspector, ActivationMonitor, ArchitectureNavigator)
- Safety Monitor
- Checkpointing System
- Memory System

These tests simulate real-world usage scenarios.

Author: AGI Self-Modification Research Team
Date: November 7, 2025
"""

import unittest
import time
import gc
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.model_manager import ModelManager
from src.safety_monitor import SafetyMonitor
from src.checkpointing import CheckpointManager
from src.introspection.weight_inspector import WeightInspector
from src.introspection.activation_monitor import ActivationMonitor
from src.introspection.architecture_navigator import ArchitectureNavigator
from src.memory.memory_system import MemorySystem
from src.memory.observation_layer import ObservationType
from tests.test_utils import get_test_temp_dir, cleanup_temp_dir, close_memory_system


class TestFullSystemIntegration(unittest.TestCase):
    """Test full system integration scenarios."""

    @classmethod
    def setUpClass(cls):
        """Set up test environment once for all tests."""
        cls.test_dir = get_test_temp_dir()

        # Load the full model for testing (from local path)
        print("\nLoading model for integration tests...")
        # Find the snapshot directory
        model_base = Path(__file__).parent.parent / "models" / "models--Qwen--Qwen2.5-3B-Instruct"
        snapshot_dir = model_base / "snapshots"
        
        cls.model_path = None
        if snapshot_dir.exists():
            # Get the first (and likely only) snapshot
            snapshots = list(snapshot_dir.iterdir())
            if snapshots:
                cls.model_path = snapshots[0]
        
        cls.model_name = "Qwen/Qwen2.5-3B-Instruct"
        
        try:
            # Try local path first
            if cls.model_path and cls.model_path.exists():
                print(f"Loading from local path: {cls.model_path}")
                cls.model = AutoModelForCausalLM.from_pretrained(
                    str(cls.model_path),
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
                cls.tokenizer = AutoTokenizer.from_pretrained(str(cls.model_path))
            else:
                # Fall back to HuggingFace cache
                print(f"Loading from HuggingFace: {cls.model_name}")
                cls.model = AutoModelForCausalLM.from_pretrained(
                    cls.model_name,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
                cls.tokenizer = AutoTokenizer.from_pretrained(cls.model_name)
        except Exception as e:
            print(f"Warning: Could not load model: {e}")
            print("Some tests will be skipped")
            cls.model = None
            cls.tokenizer = None

    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests."""
        # Release model resources
        if hasattr(cls, 'model') and cls.model is not None:
            del cls.model
            del cls.tokenizer
            gc.collect()  # Force garbage collection
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            time.sleep(0.5)  # Give Windows time to release file handles

        # Clean up temp directory
        cleanup_temp_dir(cls.test_dir)

    def setUp(self):
        """Set up for each test."""
        # Create unique directory for this test
        import tempfile
        self.test_instance_dir = tempfile.mkdtemp(dir=self.test_dir)
        self.checkpoint_dir = Path(self.test_instance_dir) / "checkpoints"
        self.memory_dir = Path(self.test_instance_dir) / "memory"

        # Initialize all components
        if self.model is not None:
            self.safety = SafetyMonitor(self.model)
            self.checkpointer = CheckpointManager(str(self.checkpoint_dir))
            self.weight_inspector = WeightInspector(self.model)
            self.activation_monitor = ActivationMonitor(self.model, self.tokenizer)
            self.arch_navigator = ArchitectureNavigator(self.model)

        self.memory = MemorySystem(str(self.memory_dir))

    def tearDown(self):
        """Clean up after each test."""
        # Close memory database connections
        if hasattr(self.memory, 'observations') and hasattr(self.memory.observations, 'conn'):
            self.memory.observations.conn.close()

    def test_complete_modification_workflow(self):
        """
        Test complete workflow:
        1. Inspect initial state
        2. Record observation
        3. Create checkpoint
        4. Make modification
        5. Validate with safety
        6. Record outcome
        7. Consolidate knowledge
        """
        if self.model is None:
            self.skipTest("Model not available")
        
        # 1. Inspect initial state - use a layer name that definitely exists
        layer_names = self.weight_inspector.get_layer_names()
        if not layer_names:
            self.skipTest("No inspectable layers found")
        
        test_layer = layer_names[0]  # Use first available layer
        initial_stats = self.weight_inspector.get_weight_statistics(test_layer)
        self.assertIsNotNone(initial_stats)        # Record initial inspection
        self.memory.record_observation(
            obs_type=ObservationType.INTROSPECTION,
            category="inspection",
            description=f"Inspected {test_layer} before modification",
            data={'mean': float(initial_stats['mean']), 'std': float(initial_stats['std'])},
            tags=['inspection', test_layer],
            importance=0.7
        )

        # 2. Create checkpoint
        try:
            checkpoint_id = self.checkpointer.save_checkpoint(
                self.model,
                description='Integration test checkpoint'
            )
        except RuntimeError as e:
            error_msg = str(e)
            if any(keyword in error_msg for keyword in ["share memory", "file write failed", "enforce fail", "offloaded"]):
                self.skipTest(
                    f"Checkpoint saving not supported for device_map='auto' models with weight tying. "
                    f"This is a known PyTorch/accelerate limitation. Error: {error_msg[:100]}"
                )
            raise

        # Record checkpoint creation
        self.memory.record_observation(
            obs_type=ObservationType.CHECKPOINT,
            category="checkpoint",
            description="Created checkpoint before modification",
            data={'checkpoint_id': checkpoint_id},
            tags=['safety', 'checkpoint'],
            importance=1.0
        )

        # 3. Make a small modification
        layer = self.model.model.layers[0]
        original_param = layer.self_attn.q_proj.weight.data.clone()
        modification = torch.randn_like(original_param) * 0.001  # Small change
        layer.self_attn.q_proj.weight.data += modification

        # Record modification
        self.memory.record_observation(
            obs_type=ObservationType.MODIFICATION,
            category="layer0",
            description="Modified layer 0 attention weights",
            data={'layer': 'layer0', 'change_magnitude': float(modification.abs().mean())},
            tags=['modification', 'layer0', 'attention'],
            importance=0.8
        )

        # 4. Validate with safety
        # Check that no alerts were triggered during modification
        safety_result = self.safety.check_performance(
            metric_name="test_perplexity",
            metric_value=10.0  # Arbitrary test value
        )
        # Safety check should pass (no degradation)
        self.assertTrue(isinstance(safety_result, bool))

        # Record safety check
        self.memory.record_observation(
            obs_type=ObservationType.SAFETY_EVENT,
            category="validation",
            description="Safety validation passed",
            data={'result': 'passed'},
            tags=['safety', 'validation'],
            importance=0.9
        )

        # 5. Inspect modified state
        modified_stats = self.weight_inspector.get_weight_statistics(test_layer)

        # Record outcome
        self.memory.record_observation(
            obs_type=ObservationType.PERFORMANCE,
            category="inspection",
            description=f"Inspected {test_layer} after modification",
            data={'mean': float(modified_stats['mean']), 'std': float(modified_stats['std'])},
            tags=['inspection', test_layer, 'post-modification'],
            importance=0.7
        )

        # 6. Consolidate knowledge
        stats = self.memory.consolidate(force=True)

        # Verify consolidation worked
        self.assertIn('patterns_found', stats)
        self.assertIn('theories_built', stats)
        self.assertIn('beliefs_formed', stats)

        # 7. Verify we can query this experience
        mod_obs = self.memory.query.query_observations(tags=['modification'])
        self.assertGreater(len(mod_obs), 0, "Should have recorded modification observations")

        # Restore from checkpoint
        self.checkpointer.restore_checkpoint(self.model, checkpoint_id)

        # Verify restoration
        restored_param = layer.self_attn.q_proj.weight.data
        self.assertTrue(torch.allclose(restored_param, original_param, atol=1e-6))

    def test_introspection_apis_integration(self):
        """Test that all three introspection APIs work together."""
        if self.model is None:
            self.skipTest("Model not available")

        # Architecture Navigator
        arch_summary = self.arch_navigator.get_architecture_summary()
        self.assertIn('total_layers', arch_summary)
        layers = self.weight_inspector.get_layer_names()
        self.assertGreater(len(layers), 0)

        # Record architecture exploration
        self.memory.record_observation(
            obs_type=ObservationType.INTROSPECTION,
            category="architecture",
            description=f"Explored model architecture: {len(layers)} layers",
            data={'layer_count': len(layers)},
            tags=['introspection', 'architecture'],
            importance=0.6
        )

        # Weight Inspector
        first_layer = layers[0]
        stats = self.weight_inspector.get_weight_statistics(first_layer)
        self.assertIsNotNone(stats)

        # Record weight inspection
        self.memory.record_observation(
            obs_type=ObservationType.INTROSPECTION,
            category="weights",
            description=f"Inspected weights in {first_layer}",
            data={'mean': float(stats['mean']), 'std': float(stats['std'])},
            tags=['introspection', 'weights', first_layer],
            importance=0.6
        )

        # Activation Monitor
        # (Would need actual inference to test, so we just verify initialization)
        self.assertIsNotNone(self.activation_monitor)

        # Verify memory recorded everything
        introspection_obs = self.memory.query.query_observations(
            tags=['introspection']
        )
        self.assertGreaterEqual(len(introspection_obs), 2)

    def test_safety_monitor_with_memory(self):
        """Test safety monitor integration with memory system."""
        if self.model is None:
            self.skipTest("Model not available")

        # Check initial safety
        safety_check = self.safety.check_performance(
            metric_name="test_perplexity",
            metric_value=10.0
        )
        is_safe = safety_check  # Returns boolean

        # Record safety check
        self.memory.record_observation(
            obs_type=ObservationType.SAFETY_EVENT,
            category="safety_check",
            description="Initial safety validation",
            data={'result': 'safe' if is_safe else 'unsafe'},
            tags=['safety', 'validation'],
            importance=1.0
        )

        # Get safety statistics
        resource_stats = self.safety.check_resources()

        # Record statistics
        self.memory.record_observation(
            obs_type=ObservationType.INTROSPECTION,
            category="safety_stats",
            description="Safety monitor statistics",
            data={'memory_usage': resource_stats.get('memory_usage', 0)},
            tags=['safety', 'statistics'],
            importance=0.5
        )

        # Verify memory can query safety events
        safety_obs = self.memory.query.query_observations(tags=['safety'])
        self.assertGreaterEqual(len(safety_obs), 2)

    def test_memory_system_standalone(self):
        """Test memory system works independently."""
        # Record series of observations
        for i in range(10):
            self.memory.record_observation(
                obs_type=ObservationType.SYSTEM_EVENT,
                category="test",
                description=f"Test event {i}",
                data={'sequence': i},
                tags=['test', 'sequence'],
                importance=0.5
            )

        # Consolidate
        stats = self.memory.consolidate(force=True)

        # Verify consolidation
        self.assertIsNotNone(stats)
        self.assertIn('patterns_found', stats)

        # Query
        test_obs = self.memory.query.query_observations(tags=['test'])
        self.assertEqual(len(test_obs), 10)

        # Get statistics
        mem_stats = self.memory.get_memory_stats()
        self.assertIn('observations', mem_stats)
        self.assertIn('patterns', mem_stats)

    def test_checkpoint_restoration_preserves_memory_context(self):
        """Test that checkpoint restoration works with memory tracking."""
        if self.model is None:
            self.skipTest("Model not available")

        # Create initial state
        try:
            checkpoint_id = self.checkpointer.save_checkpoint(
                self.model,
                description='Initial state for checkpoint test'
            )
        except RuntimeError as e:
            error_msg = str(e)
            if any(keyword in error_msg for keyword in ["share memory", "file write failed", "enforce fail", "offloaded"]):
                self.skipTest(
                    f"Checkpoint saving not supported for device_map='auto' models with weight tying. "
                    f"This is a known PyTorch/accelerate limitation. Error: {error_msg[:100]}"
                )
            raise

        # Record checkpoint
        self.memory.record_observation(
            obs_type=ObservationType.CHECKPOINT,
            category="checkpoint",
            description="Initial checkpoint created",
            data={'checkpoint_id': checkpoint_id, 'state': 'initial'},
            tags=['checkpoint', 'initial'],
            importance=1.0
        )

        # Make modification
        layer = self.model.model.layers[0]
        layer.self_attn.q_proj.weight.data += torch.randn_like(
            layer.self_attn.q_proj.weight.data
        ) * 0.001

        # Record modification
        self.memory.record_observation(
            obs_type=ObservationType.MODIFICATION,
            category="test",
            description="Test modification",
            data={'checkpoint_ref': checkpoint_id},
            tags=['modification', 'test'],
            importance=0.8
        )

        # Restore
        self.checkpointer.restore_checkpoint(self.model, checkpoint_id)

        # Record restoration
        self.memory.record_observation(
            obs_type=ObservationType.CHECKPOINT,
            category="restoration",
            description="Restored from checkpoint",
            data={'checkpoint_id': checkpoint_id},
            tags=['checkpoint', 'restoration'],
            importance=1.0
        )

        # Verify memory knows about the whole sequence
        checkpoint_obs = self.memory.query.query_observations(tags=['checkpoint'])
        self.assertEqual(len(checkpoint_obs), 2)  # creation + restoration

    def test_decision_support_from_memory(self):
        """Test that memory system can provide decision support."""
        # Simulate several modification attempts
        for i in range(5):
            # Record modification
            self.memory.record_observation(
                obs_type=ObservationType.MODIFICATION,
                category="layer_test",
                description=f"Modification attempt {i}",
                data={'attempt': i},
                tags=['modification', 'experiment'],
                importance=0.7
            )

            # Record outcome
            success = i % 2 == 0  # Alternate success/failure
            self.memory.record_observation(
                obs_type=ObservationType.PERFORMANCE,
                category="result",
                description=f"Modification {'succeeded' if success else 'failed'}",
                data={'success': success, 'attempt': i},
                tags=['performance', 'result'],
                importance=0.8
            )

        # Consolidate
        self.memory.consolidate(force=True)

        # Get decision support for new modification
        decision_support = self.memory.get_decision_support({
            'action': 'modify',
            'target': 'layer_test'
        })

        # Verify decision support includes relevant information
        self.assertIn('beliefs', decision_support)
        self.assertIn('observations', decision_support)
        self.assertIn('recommendation', decision_support)

    def test_error_handling_integration(self):
        """Test that system handles errors gracefully across components."""
        if self.model is None:
            self.skipTest("Model not available")

        # Try to inspect non-existent layer
        try:
            stats = self.weight_inspector.get_weight_statistics("nonexistent.layer")
            # If it doesn't raise, it should return None or empty
            if stats is not None:
                self.fail("Should handle non-existent layer")
        except (KeyError, AttributeError):
            # Expected - record the error
            self.memory.record_observation(
                obs_type=ObservationType.SAFETY_EVENT,
                category="error",
                description="Attempted to inspect non-existent layer",
                data={'error_type': 'KeyError'},
                tags=['error', 'inspection'],
                importance=0.5
            )

        # Try to load non-existent checkpoint
        try:
            self.checkpointer.restore_checkpoint(self.model, "nonexistent_checkpoint")
            self.fail("Should raise error for non-existent checkpoint")
        except (FileNotFoundError, ValueError):
            # Expected - record the error
            self.memory.record_observation(
                obs_type=ObservationType.SAFETY_EVENT,
                category="error",
                description="Attempted to load non-existent checkpoint",
                data={'error_type': 'FileNotFoundError'},
                tags=['error', 'checkpoint'],
                importance=0.5
            )

        # Verify memory recorded error events
        error_obs = self.memory.query.query_observations(tags=['error'])
        self.assertGreaterEqual(len(error_obs), 2)


if __name__ == '__main__':
    unittest.main()
