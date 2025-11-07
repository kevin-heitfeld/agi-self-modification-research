"""
Tests for Weight Sharing Detection

Tests the WeightInspector's ability to detect and report shared weights,
which is critical for safe self-modification.
"""

import unittest
import torch
import torch.nn as nn
from src.introspection.weight_inspector import WeightInspector


class SimpleModelWithSharing(nn.Module):
    """A simple model with weight sharing for testing."""
    def __init__(self, vocab_size=100, hidden_dim=64):
        super().__init__()
        # Create embedding layer
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        # Create output layer that shares weights with embeddings
        self.output = nn.Linear(hidden_dim, vocab_size, bias=False)
        # Tie the weights (common pattern in language models)
        self.output.weight = self.embed.weight

    def forward(self, x):
        return self.output(self.embed(x))


class SimpleModelWithoutSharing(nn.Module):
    """A simple model without weight sharing for comparison."""
    def __init__(self, vocab_size=100, hidden_dim=64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.output = nn.Linear(hidden_dim, vocab_size, bias=False)
        # No weight tying - independent weights

    def forward(self, x):
        return self.output(self.embed(x))


class TestWeightSharingDetection(unittest.TestCase):
    """Test weight sharing detection capabilities."""

    def setUp(self):
        """Set up test models."""
        self.model_with_sharing = SimpleModelWithSharing()
        self.model_without_sharing = SimpleModelWithoutSharing()

    def test_detect_shared_weights_positive(self):
        """Test that shared weights are correctly detected."""
        inspector = WeightInspector(self.model_with_sharing, "shared_model")

        shared = inspector.get_shared_weights()

        # Should detect weight sharing
        self.assertGreater(len(shared), 0, "Should detect shared weights")

        # Should identify the specific shared layers
        self.assertTrue(
            any('embed.weight' in names and 'output.weight' in names
                for names in shared.values()),
            "Should detect embed.weight and output.weight sharing"
        )

    def test_detect_shared_weights_negative(self):
        """Test that independent weights are not flagged as shared."""
        inspector = WeightInspector(self.model_without_sharing, "independent_model")

        shared = inspector.get_shared_weights()

        # Should not detect weight sharing
        self.assertEqual(len(shared), 0, "Should not detect shared weights in independent model")

    def test_get_shared_layers(self):
        """Test getting list of layers sharing with a specific layer."""
        inspector = WeightInspector(self.model_with_sharing, "shared_model")

        # Check embed.weight (this is in named_parameters)
        shared_with_embed = inspector.get_shared_layers('embed.weight')
        self.assertIn('output.weight', shared_with_embed)

        # Check output.weight (this is NOT in named_parameters but we detect it)
        shared_with_output = inspector.get_shared_layers('output.weight')
        self.assertIn('embed.weight', shared_with_output)

        # For independent model, check a layer that exists
        inspector_full = WeightInspector(self.model_without_sharing, "independent_model")
        # embed.weight should have no sharing
        shared_with_embed_ind = inspector_full.get_shared_layers('embed.weight')
        self.assertEqual(len(shared_with_embed_ind), 0)

    def test_shared_weight_warning_in_statistics(self):
        """Test that statistics include warning for shared weights."""
        inspector = WeightInspector(self.model_with_sharing, "shared_model")

        stats = inspector.get_weight_statistics('embed.weight')

        # Should have shared_with field
        self.assertIn('shared_with', stats)
        self.assertIn('output.weight', stats['shared_with'])

        # Should have warning message
        self.assertIn('warning', stats)
        self.assertIn('WEIGHT SHARING', stats['warning'])
        self.assertIn('output.weight', stats['warning'])

    def test_shared_weight_statistics_identical(self):
        """Test that shared weights produce identical statistics (for the one in named_parameters)."""
        inspector = WeightInspector(self.model_with_sharing, "shared_model")

        # Get statistics for embed.weight (which is in named_parameters)
        stats_embed = inspector.get_weight_statistics('embed.weight', use_cache=False)

        # Key statistics should exist
        self.assertIn('mean', stats_embed)
        self.assertIn('std', stats_embed)
        self.assertIn('l2_norm', stats_embed)

        # Should have shared_with field indicating output.weight
        self.assertIn('shared_with', stats_embed)
        self.assertIn('output.weight', stats_embed['shared_with'])

    def test_modification_affects_both_layers(self):
        """Test that modifying the shared layer affects both references."""
        inspector = WeightInspector(self.model_with_sharing, "shared_model")

        # Get initial statistics for embed.weight (in named_parameters)
        initial_embed = inspector.get_weight_statistics('embed.weight', use_cache=False)

        # Modify embed.weight by a significant amount
        with torch.no_grad():
            self.model_with_sharing.embed.weight.data += 0.5  # Large change

        # Clear cache and get new statistics
        inspector._stats_cache.clear()
        modified_embed = inspector.get_weight_statistics('embed.weight', use_cache=False)

        # Should have changed significantly
        self.assertGreater(
            abs(modified_embed['mean'] - initial_embed['mean']), 0.1,
            msg="embed.weight should have changed significantly"
        )

        # Verify the output layer also changed (same tensor)
        self.assertEqual(
            self.model_with_sharing.embed.weight.data_ptr(),
            self.model_with_sharing.output.weight.data_ptr(),
            msg="Should still be sharing memory after modification"
        )

    def test_data_ptr_equality(self):
        """Test that shared weights have the same data_ptr."""
        inspector = WeightInspector(self.model_with_sharing, "shared_model")

        # Get the parameter that's in named_parameters
        embed_param = inspector.layers['embed.weight']

        # Get the output layer's weight directly from the model
        output_param = self.model_with_sharing.output.weight

        # Should point to the same memory
        self.assertEqual(
            embed_param.data_ptr(), output_param.data_ptr(),
            "Shared weights should have the same data_ptr"
        )

    def test_get_shared_layers_nonexistent(self):
        """Test that get_shared_layers raises error for nonexistent layer."""
        inspector = WeightInspector(self.model_with_sharing, "shared_model")

        with self.assertRaises(KeyError):
            inspector.get_shared_layers('nonexistent.layer')


class TestWeightSharingWithQwen(unittest.TestCase):
    """Test weight sharing detection with actual Qwen model (if available)."""

    def setUp(self):
        """Try to load Qwen model."""
        try:
            from transformers import AutoModelForCausalLM
            from pathlib import Path

            # Try to load from local path
            model_path = Path(__file__).parent.parent / "models" / "models--Qwen--Qwen2.5-3B-Instruct"
            snapshot_dir = model_path / "snapshots"

            if snapshot_dir.exists():
                snapshots = list(snapshot_dir.iterdir())
                if snapshots:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        str(snapshots[0]),
                        torch_dtype=torch.float16,
                        device_map="cpu"
                    )
                    self.model_available = True
                else:
                    self.model_available = False
            else:
                self.model_available = False
        except Exception:
            self.model_available = False

    def test_qwen_weight_sharing_detection(self):
        """Test that Qwen's weight tying is detected."""
        if not self.model_available:
            self.skipTest("Qwen model not available")

        inspector = WeightInspector(self.model, "Qwen2.5-3B")

        # Qwen should have weight sharing
        shared = inspector.get_shared_weights()
        self.assertGreater(len(shared), 0, "Qwen should have weight sharing")

        # Should detect lm_head and embed_tokens sharing
        self.assertTrue(
            any('lm_head' in str(names) and 'embed_tokens' in str(names)
                for names in shared.values()),
            "Should detect lm_head and embed_tokens sharing in Qwen"
        )

    def test_qwen_shared_layers_query(self):
        """Test querying shared layers for Qwen model."""
        if not self.model_available:
            self.skipTest("Qwen model not available")

        inspector = WeightInspector(self.model, "Qwen2.5-3B")

        # Find embed_tokens layer (this is in named_parameters)
        layer_names = inspector.get_layer_names()
        embed_name = [n for n in layer_names if 'embed_tokens' in n][0]

        shared_layers = inspector.get_shared_layers(embed_name)

        # Should find lm_head
        self.assertTrue(
            any('lm_head' in name for name in shared_layers),
            f"embed_tokens should share with lm_head, found: {shared_layers}"
        )
    def test_qwen_statistics_with_warning(self):
        """Test that statistics include warning for Qwen's shared weights."""
        if not self.model_available:
            self.skipTest("Qwen model not available")

        inspector = WeightInspector(self.model, "Qwen2.5-3B")

        # Find embed_tokens layer (this is in named_parameters)
        layer_names = inspector.get_layer_names()
        embed_name = [n for n in layer_names if 'embed_tokens' in n][0]

        stats = inspector.get_weight_statistics(embed_name)

        # Should have warning
        self.assertIn('shared_with', stats)
        self.assertIn('warning', stats)
        self.assertIn('WEIGHT SHARING', stats['warning'])
        # Should mention lm_head
        self.assertTrue(
            any('lm_head' in layer for layer in stats['shared_with']),
            f"Warning should mention lm_head, shared_with: {stats['shared_with']}"
        )

if __name__ == '__main__':
    unittest.main()
