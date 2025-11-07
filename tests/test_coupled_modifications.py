"""
Tests for coupled modification tracking in Memory System.

This tests the integration between WeightInspector and MemorySystem
to ensure coupled modifications (shared weights) are correctly detected
and recorded.
"""

import unittest
import tempfile
import shutil
import torch
import torch.nn as nn
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from memory import MemorySystem, ObservationType
from introspection import WeightInspector


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


class TestCoupledModifications(unittest.TestCase):
    """Test coupled modification detection and recording."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.memory = MemorySystem(self.temp_dir)
        
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
    
    def tearDown(self):
        """Clean up test files."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_set_weight_inspector(self):
        """Test setting WeightInspector on MemorySystem."""
        self.assertIsNone(self.memory._weight_inspector)
        
        self.memory.set_weight_inspector(self.inspector_with_sharing)
        
        self.assertIsNotNone(self.memory._weight_inspector)
        self.assertEqual(
            self.memory._weight_inspector, 
            self.inspector_with_sharing
        )
    
    def test_record_modification_without_inspector(self):
        """Test recording modification without WeightInspector (no coupling detection)."""
        # Don't set inspector
        obs_id = self.memory.record_modification(
            layer_name="embed.weight",
            modification_data={'change': 0.1, 'method': 'gradient'}
        )
        
        # Should record as standard modification
        obs = self.memory.observations.get(obs_id)
        self.assertIsNotNone(obs)
        self.assertEqual(obs.type, ObservationType.MODIFICATION)
        self.assertEqual(obs.category, "embed.weight")
        self.assertIn('modification', obs.tags)
        self.assertIn('embed.weight', obs.tags)
        self.assertNotIn('coupled', obs.tags)
        self.assertNotIn('coupled_layers', obs.data)
    
    def test_record_coupled_modification(self):
        """Test recording modification that affects coupled layers."""
        # Set inspector with weight sharing
        self.memory.set_weight_inspector(self.inspector_with_sharing)
        
        # Record modification to embed.weight (coupled with output.weight)
        obs_id = self.memory.record_modification(
            layer_name="embed.weight",
            modification_data={'change': 0.1, 'method': 'gradient'}
        )
        
        # Should record as coupled modification
        obs = self.memory.observations.get(obs_id)
        self.assertIsNotNone(obs)
        self.assertEqual(obs.type, ObservationType.MODIFICATION)
        self.assertEqual(obs.category, "coupled_modification")
        
        # Check tags include both layers and 'coupled'
        self.assertIn('modification', obs.tags)
        self.assertIn('coupled', obs.tags)
        self.assertIn('embed.weight', obs.tags)
        self.assertIn('output.weight', obs.tags)
        
        # Check data includes coupling information
        self.assertIn('coupled_layers', obs.data)
        self.assertIn('primary_layer', obs.data)
        self.assertEqual(obs.data['primary_layer'], 'embed.weight')
        self.assertIn('output.weight', obs.data['coupled_layers'])
        
        # Check higher importance
        self.assertGreaterEqual(obs.importance, 0.9)
        
        # Check description mentions coupling
        self.assertIn('coupled', obs.description.lower())
        self.assertIn('output.weight', obs.description)
    
    def test_record_independent_modification(self):
        """Test recording modification to independent layer (no coupling)."""
        # Set inspector without weight sharing
        self.memory.set_weight_inspector(self.inspector_without_sharing)
        
        # Record modification to hidden.weight (not coupled)
        obs_id = self.memory.record_modification(
            layer_name="hidden.weight",
            modification_data={'change': 0.05, 'method': 'random'}
        )
        
        # Should record as standard modification
        obs = self.memory.observations.get(obs_id)
        self.assertIsNotNone(obs)
        self.assertEqual(obs.type, ObservationType.MODIFICATION)
        self.assertEqual(obs.category, "hidden.weight")
        self.assertIn('modification', obs.tags)
        self.assertIn('hidden.weight', obs.tags)
        self.assertNotIn('coupled', obs.tags)
        self.assertNotIn('coupled_layers', obs.data)
    
    def test_custom_description_preserved(self):
        """Test that custom descriptions are preserved for coupled modifications."""
        self.memory.set_weight_inspector(self.inspector_with_sharing)
        
        custom_desc = "Experimental gradient update on embedding layer"
        obs_id = self.memory.record_modification(
            layer_name="embed.weight",
            modification_data={'change': 0.1},
            description=custom_desc
        )
        
        obs = self.memory.observations.get(obs_id)
        self.assertEqual(obs.description, custom_desc)
    
    def test_custom_tags_preserved(self):
        """Test that custom tags are added to auto-generated tags."""
        self.memory.set_weight_inspector(self.inspector_with_sharing)
        
        custom_tags = ['experimental', 'phase1', 'gradient_update']
        obs_id = self.memory.record_modification(
            layer_name="embed.weight",
            modification_data={'change': 0.1},
            tags=custom_tags
        )
        
        obs = self.memory.observations.get(obs_id)
        
        # Should have both custom and auto-generated tags
        for tag in custom_tags:
            self.assertIn(tag, obs.tags)
        
        # Should also have auto-generated tags
        self.assertIn('modification', obs.tags)
        self.assertIn('coupled', obs.tags)
        self.assertIn('embed.weight', obs.tags)
    
    def test_modification_data_preserved(self):
        """Test that modification data is preserved and augmented."""
        self.memory.set_weight_inspector(self.inspector_with_sharing)
        
        mod_data = {
            'change': 0.1,
            'method': 'gradient',
            'learning_rate': 0.001,
            'step': 42
        }
        
        obs_id = self.memory.record_modification(
            layer_name="embed.weight",
            modification_data=mod_data
        )
        
        obs = self.memory.observations.get(obs_id)
        
        # Original data should be preserved
        self.assertEqual(obs.data['change'], 0.1)
        self.assertEqual(obs.data['method'], 'gradient')
        self.assertEqual(obs.data['learning_rate'], 0.001)
        self.assertEqual(obs.data['step'], 42)
        
        # Coupling data should be added
        self.assertIn('layer', obs.data)
        self.assertIn('coupled_layers', obs.data)
        self.assertIn('primary_layer', obs.data)
    
    def test_custom_importance_increased_for_coupled(self):
        """Test that custom importance is increased for coupled modifications."""
        self.memory.set_weight_inspector(self.inspector_with_sharing)
        
        # Try to set low importance
        obs_id = self.memory.record_modification(
            layer_name="embed.weight",
            modification_data={'change': 0.1},
            importance=0.5  # Low importance
        )
        
        obs = self.memory.observations.get(obs_id)
        
        # Should be increased to at least 0.9 for coupled modifications
        self.assertGreaterEqual(obs.importance, 0.9)
    
    def test_query_coupled_modifications(self):
        """Test querying for coupled modifications."""
        self.memory.set_weight_inspector(self.inspector_with_sharing)
        
        # Record several modifications
        self.memory.record_modification(
            layer_name="embed.weight",
            modification_data={'change': 0.1}
        )
        
        self.memory.record_modification(
            layer_name="hidden.weight",
            modification_data={'change': 0.05}
        )
        
        self.memory.record_modification(
            layer_name="embed.weight",
            modification_data={'change': 0.2}
        )
        
        # Query for coupled modifications
        coupled_mods = self.memory.observations.query(
            category="coupled_modification"
        )
        
        # Should find 2 coupled modifications (both embed.weight)
        self.assertEqual(len(coupled_mods), 2)
        
        for obs in coupled_mods:
            self.assertEqual(obs.category, "coupled_modification")
            self.assertIn('coupled', obs.tags)
            self.assertIn('coupled_layers', obs.data)
    
    def test_query_by_coupled_tag(self):
        """Test querying modifications by 'coupled' tag."""
        self.memory.set_weight_inspector(self.inspector_with_sharing)
        
        # Record mixed modifications
        self.memory.record_modification(
            layer_name="embed.weight",
            modification_data={'change': 0.1}
        )
        
        self.memory.record_modification(
            layer_name="hidden.weight",
            modification_data={'change': 0.05}
        )
        
        # Query by 'coupled' tag
        coupled_obs = self.memory.observations.query(tags=['coupled'])
        
        self.assertEqual(len(coupled_obs), 1)
        self.assertEqual(coupled_obs[0].data['primary_layer'], 'embed.weight')
    
    def test_inspector_error_handling(self):
        """Test graceful handling when inspector fails."""
        self.memory.set_weight_inspector(self.inspector_with_sharing)
        
        # Try to record modification with non-existent layer
        # Should not crash, just log warning
        obs_id = self.memory.record_modification(
            layer_name="nonexistent.layer",
            modification_data={'change': 0.1}
        )
        
        # Should still record (as standard modification)
        obs = self.memory.observations.get(obs_id)
        self.assertIsNotNone(obs)
        self.assertEqual(obs.type, ObservationType.MODIFICATION)


if __name__ == '__main__':
    unittest.main()
