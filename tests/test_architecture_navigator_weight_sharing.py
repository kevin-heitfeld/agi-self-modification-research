"""
Tests for weight sharing detection in ArchitectureNavigator.

This tests the integration between WeightInspector and ArchitectureNavigator
to ensure weight sharing information is correctly exposed in architectural
summaries and queries.
"""

import unittest
import torch
import torch.nn as nn

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from introspection import WeightInspector, ArchitectureNavigator


class SimpleModelWithSharing(nn.Module):
    """Test model with intentional weight sharing."""
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(100, 32)
        self.hidden = nn.Linear(32, 16)
        self.output = nn.Linear(32, 100)
        # Tie weights (common in language models)
        self.output.weight = self.embed.weight
    
    def forward(self, x):
        x = self.embed(x)
        x = self.hidden(x)
        x = self.output(x)
        return x


class SimpleModelWithoutSharing(nn.Module):
    """Test model without weight sharing."""
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(100, 32)
        self.hidden = nn.Linear(32, 16)
        self.output = nn.Linear(32, 100)
    
    def forward(self, x):
        x = self.embed(x)
        x = self.hidden(x)
        x = self.output(x)
        return x


class TestArchitectureNavigatorWeightSharing(unittest.TestCase):
    """Test weight sharing detection in ArchitectureNavigator."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create models
        self.model_with_sharing = SimpleModelWithSharing()
        self.model_without_sharing = SimpleModelWithoutSharing()
        
        # Create inspectors
        self.inspector_with_sharing = WeightInspector(
            self.model_with_sharing, 
            "shared_model"
        )
        self.inspector_without_sharing = WeightInspector(
            self.model_without_sharing,
            "independent_model"
        )
        
        # Create navigators
        self.nav_with_sharing = ArchitectureNavigator(self.model_with_sharing)
        self.nav_without_sharing = ArchitectureNavigator(self.model_without_sharing)
    
    def test_set_weight_inspector(self):
        """Test setting WeightInspector on navigator."""
        self.assertIsNone(self.nav_with_sharing._weight_inspector)
        
        self.nav_with_sharing.set_weight_inspector(self.inspector_with_sharing)
        
        self.assertIsNotNone(self.nav_with_sharing._weight_inspector)
        self.assertEqual(
            self.nav_with_sharing._weight_inspector,
            self.inspector_with_sharing
        )
    
    def test_summary_without_inspector(self):
        """Test that summary works without inspector (no weight sharing info)."""
        summary = self.nav_with_sharing.get_architecture_summary()
        
        # Should have standard fields
        self.assertIn('model_type', summary)
        self.assertIn('total_parameters', summary)
        
        # Should NOT have weight sharing info
        self.assertNotIn('weight_sharing', summary)
    
    def test_summary_with_inspector_no_sharing(self):
        """Test summary with inspector but no weight sharing."""
        self.nav_without_sharing.set_weight_inspector(
            self.inspector_without_sharing
        )
        
        summary = self.nav_without_sharing.get_architecture_summary()
        
        # Should have standard fields
        self.assertIn('model_type', summary)
        
        # Should NOT have weight sharing info (no sharing detected)
        self.assertNotIn('weight_sharing', summary)
    
    def test_summary_with_sharing_detected(self):
        """Test summary includes weight sharing when detected."""
        self.nav_with_sharing.set_weight_inspector(self.inspector_with_sharing)
        
        summary = self.nav_with_sharing.get_architecture_summary()
        
        # Should have standard fields
        self.assertIn('model_type', summary)
        self.assertIn('total_parameters', summary)
        
        # Should have weight sharing info
        self.assertIn('weight_sharing', summary)
        
        sharing = summary['weight_sharing']
        self.assertTrue(sharing['detected'])
        self.assertGreater(sharing['num_groups'], 0)
        self.assertIn('shared_groups', sharing)
        self.assertIn('summary', sharing)
        self.assertIn('warning', sharing)
    
    def test_sharing_group_structure(self):
        """Test structure of shared groups in summary."""
        self.nav_with_sharing.set_weight_inspector(self.inspector_with_sharing)
        
        summary = self.nav_with_sharing.get_architecture_summary()
        sharing = summary['weight_sharing']
        
        # Should have at least one group
        self.assertGreater(len(sharing['shared_groups']), 0)
        
        # Check first group structure
        group = sharing['shared_groups'][0]
        self.assertIn('layers', group)
        self.assertIn('tensor_shape', group)
        self.assertIn('tensor_size', group)
        self.assertIn('implications', group)
        
        # Should have multiple layers in the group
        self.assertGreaterEqual(len(group['layers']), 2)
        
        # Layers should include embed and output
        layer_names = group['layers']
        has_embed = any('embed' in name.lower() for name in layer_names)
        has_output = any('output' in name.lower() for name in layer_names)
        self.assertTrue(has_embed or has_output)
    
    def test_sharing_summary_text(self):
        """Test that summary text is informative."""
        self.nav_with_sharing.set_weight_inspector(self.inspector_with_sharing)
        
        summary = self.nav_with_sharing.get_architecture_summary()
        sharing = summary['weight_sharing']
        
        summary_text = sharing['summary']
        
        # Should mention number of groups
        self.assertIn('group', summary_text.lower())
        
        # Should mention weight sharing or parameter efficiency
        self.assertTrue(
            'sharing' in summary_text.lower() or 
            'efficiency' in summary_text.lower()
        )
    
    def test_sharing_warning_present(self):
        """Test that warning about modifications is present."""
        self.nav_with_sharing.set_weight_inspector(self.inspector_with_sharing)
        
        summary = self.nav_with_sharing.get_architecture_summary()
        sharing = summary['weight_sharing']
        
        warning = sharing['warning']
        
        # Should have warning emoji
        self.assertIn('⚠️', warning)
        
        # Should mention modification affects all layers
        self.assertIn('modifying', warning.lower())
        self.assertIn('affect', warning.lower())
    
    def test_get_weight_sharing_info_no_inspector(self):
        """Test querying weight sharing without inspector."""
        info = self.nav_with_sharing.get_weight_sharing_info()
        
        self.assertFalse(info['has_sharing'])
        self.assertIn('error', info)
        self.assertIn('WeightInspector', info['error'])
    
    def test_get_weight_sharing_info_all(self):
        """Test querying all weight sharing."""
        self.nav_with_sharing.set_weight_inspector(self.inspector_with_sharing)
        
        info = self.nav_with_sharing.get_weight_sharing_info()
        
        self.assertTrue(info['has_sharing'])
        self.assertIn('num_groups', info)
        self.assertIn('groups', info)
        self.assertIn('summary', info)
        self.assertGreater(info['num_groups'], 0)
    
    def test_get_weight_sharing_info_specific_layer_coupled(self):
        """Test querying weight sharing for a specific coupled layer."""
        self.nav_with_sharing.set_weight_inspector(self.inspector_with_sharing)
        
        info = self.nav_with_sharing.get_weight_sharing_info('embed.weight')
        
        self.assertTrue(info['has_sharing'])
        self.assertEqual(info['layer'], 'embed.weight')
        self.assertIn('coupled_with', info)
        self.assertGreater(len(info['coupled_with']), 0)
        self.assertIn('output.weight', info['coupled_with'])
        self.assertIn('implications', info)
        self.assertIn('warning', info)
    
    def test_get_weight_sharing_info_specific_layer_independent(self):
        """Test querying weight sharing for an independent layer."""
        self.nav_with_sharing.set_weight_inspector(self.inspector_with_sharing)
        
        info = self.nav_with_sharing.get_weight_sharing_info('hidden.weight')
        
        self.assertFalse(info['has_sharing'])
        self.assertEqual(info['layer'], 'hidden.weight')
        self.assertIn('coupled_with', info)
        self.assertEqual(len(info['coupled_with']), 0)
    
    def test_get_weight_sharing_info_no_sharing_model(self):
        """Test querying on model without weight sharing."""
        self.nav_without_sharing.set_weight_inspector(
            self.inspector_without_sharing
        )
        
        info = self.nav_without_sharing.get_weight_sharing_info()
        
        self.assertFalse(info['has_sharing'])
        self.assertIn('summary', info)
        self.assertIn('no', info['summary'].lower())
    
    def test_implications_embed_lm_head_pattern(self):
        """Test that implications correctly identify embed/lm_head pattern."""
        self.nav_with_sharing.set_weight_inspector(self.inspector_with_sharing)
        
        summary = self.nav_with_sharing.get_architecture_summary()
        sharing = summary['weight_sharing']
        
        # Find the shared group
        group = sharing['shared_groups'][0]
        implications = group['implications']
        
        # Should mention embeddings or tied weights
        self.assertTrue(
            'embed' in implications.lower() or
            'tied' in implications.lower() or
            'sharing' in implications.lower()
        )
        
        # Should mention effect on both layers
        self.assertIn('modifying', implications.lower())
        self.assertIn('affect', implications.lower())
    
    def test_query_specific_layer_summary(self):
        """Test that layer-specific query has informative summary."""
        self.nav_with_sharing.set_weight_inspector(self.inspector_with_sharing)
        
        info = self.nav_with_sharing.get_weight_sharing_info('embed.weight')
        
        summary = info['summary']
        
        # Should mention the layer name
        self.assertIn('embed.weight', summary)
        
        # Should mention sharing
        self.assertIn('share', summary.lower())
        
        # Should mention the coupled layer
        self.assertIn('output.weight', summary)
    
    def test_query_returns_correct_coupled_count(self):
        """Test that coupled layer count is correct."""
        self.nav_with_sharing.set_weight_inspector(self.inspector_with_sharing)
        
        info = self.nav_with_sharing.get_weight_sharing_info('embed.weight')
        
        self.assertIn('num_coupled_layers', info)
        # Should be coupled with output.weight (1 layer)
        self.assertEqual(info['num_coupled_layers'], 1)
        self.assertEqual(len(info['coupled_with']), 1)
    
    def test_sharing_info_has_tensor_details(self):
        """Test that shared groups include tensor shape and size."""
        self.nav_with_sharing.set_weight_inspector(self.inspector_with_sharing)
        
        info = self.nav_with_sharing.get_weight_sharing_info()
        
        group = info['groups'][0]
        
        # Should have tensor details
        self.assertIsNotNone(group['tensor_shape'])
        self.assertIsNotNone(group['tensor_size'])
        
        # Shape should be tuple
        self.assertIsInstance(group['tensor_shape'], (tuple, list))
        
        # Size should be positive integer
        self.assertIsInstance(group['tensor_size'], int)
        self.assertGreater(group['tensor_size'], 0)


if __name__ == '__main__':
    unittest.main()
