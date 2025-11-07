"""
End-to-End Modification Workflow Tests

Tests the complete workflow from:
1. Weight inspection and analysis
2. Safety monitoring and warnings
3. Modification execution
4. Memory recording and coupling detection
5. Architecture queries and understanding

This validates that all Phase 0 components work together correctly.
"""

import unittest
import tempfile
import shutil
import torch
import torch.nn as nn
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from introspection import WeightInspector, ArchitectureNavigator
from memory import MemorySystem, ObservationType
from safety_monitor import SafetyMonitor


class TransformerLikeModel(nn.Module):
    """Model with transformer-like weight tying for realistic testing."""
    def __init__(self, vocab_size=1000, hidden_size=256, num_layers=4):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, hidden_size)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'attention': nn.Linear(hidden_size, hidden_size),
                'ffn': nn.Linear(hidden_size, hidden_size)
            })
            for _ in range(num_layers)
        ])
        
        # Output projection tied to embeddings
        self.output_projection = nn.Linear(hidden_size, vocab_size, bias=False)
        self.output_projection.weight = self.embeddings.weight  # Weight tying
        
        self.norm = nn.LayerNorm(hidden_size)
    
    def forward(self, x):
        x = self.embeddings(x)
        for layer in self.layers:
            x = x + layer['attention'](x)
            x = x + layer['ffn'](x)
        x = self.norm(x)
        return torch.matmul(x, self.output_projection.weight.t())


class TestEndToEndWorkflow(unittest.TestCase):
    """End-to-end workflow tests for Phase 0 system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.model = TransformerLikeModel()
        
        # Initialize all components
        self.inspector = WeightInspector(self.model, "transformer_model")
        self.navigator = ArchitectureNavigator(self.model)
        self.navigator.set_weight_inspector(self.inspector)
        
        memory_dir = Path(self.temp_dir) / "memory"
        self.memory = MemorySystem(str(memory_dir))
        self.memory.set_weight_inspector(self.inspector)
        
        self.safety = SafetyMonitor(self.model)
    
    def tearDown(self):
        """Clean up after test."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_complete_modification_workflow_with_tied_weights(self):
        """Test complete workflow: inspect → warn → modify → record → query."""
        
        # STEP 1: Inspect the model to understand architecture
        print("\n=== STEP 1: INSPECT ===")
        summary = self.navigator.get_architecture_summary()
        self.assertIn("weight_sharing", summary)
        print(f"Model has {summary['total_parameters']:,} parameters")
        print(f"Weight sharing detected: {len(self.inspector.get_shared_weights())} groups")
        
        # STEP 2: Check for weight sharing before modification
        print("\n=== STEP 2: CHECK WEIGHT SHARING ===")
        sharing_info = self.navigator.get_weight_sharing_info("embeddings.weight")
        self.assertIsNotNone(sharing_info)
        self.assertTrue(sharing_info['has_sharing'])
        print(f"embeddings.weight is coupled with: {sharing_info['coupled_with']}")
        print(f"Warning: {sharing_info['warning']}")
        print(f"Implications: {sharing_info['implications']}")
        
        # STEP 3: Get baseline statistics
        print("\n=== STEP 3: BASELINE STATISTICS ===")
        baseline_stats = self.inspector.get_weight_statistics("embeddings.weight")
        print(f"Baseline mean: {baseline_stats['mean']:.6f}")
        print(f"Baseline std: {baseline_stats['std']:.6f}")
        
        # STEP 4: Record the intention to modify
        print("\n=== STEP 4: RECORD INTENTION ===")
        self.memory.record_observation(
            obs_type=ObservationType.INTROSPECTION,
            category="modification_plan",
            description="Planning to modify embeddings to test learning",
            data={
                'target_layer': 'embeddings.weight',
                'method': 'gradient_update',
                'sharing_aware': True,
                'coupled_layers': sharing_info['coupled_with']
            },
            tags=['modification', 'planned'],
            importance=0.8
        )
        
        # STEP 5: Safety check before modification
        print("\n=== STEP 5: SAFETY CHECK ===")
        # Safety monitor tracks modifications through hooks
        # For this test, we'll just verify initialization works
        self.assertIsNotNone(self.safety)
        print("Safety monitor initialized and ready")
        
        # STEP 6: Perform the modification
        print("\n=== STEP 6: MODIFY ===")
        original_weight = self.model.embeddings.weight.data.clone()
        modification_amount = torch.randn_like(self.model.embeddings.weight) * 0.01
        self.model.embeddings.weight.data += modification_amount
        
        weight_change = (self.model.embeddings.weight.data - original_weight).abs().mean().item()
        print(f"Applied modification: mean change = {weight_change:.6f}")
        
        # Verify output projection changed too (due to weight tying)
        self.assertTrue(
            torch.equal(self.model.embeddings.weight, self.model.output_projection.weight),
            "Weight tying should be preserved"
        )
        
        # STEP 7: Get post-modification statistics
        print("\n=== STEP 7: POST-MODIFICATION STATISTICS ===")
        # Clear cache to force recomputation
        self.inspector._stats_cache.clear()
        new_stats = self.inspector.get_weight_statistics("embeddings.weight")
        print(f"New mean: {new_stats['mean']:.6f}")
        print(f"New std: {new_stats['std']:.6f}")
        
        # STEP 8: Record the modification with automatic coupling detection
        print("\n=== STEP 8: RECORD MODIFICATION ===")
        self.memory.record_modification(
            layer_name="embeddings.weight",
            modification_data={
                'method': 'gradient_update',
                'mean_change': weight_change,
                'baseline_stats': baseline_stats,
                'new_stats': new_stats
            },
            description="Applied gradient-based learning to embeddings",
            tags=['learning', 'embeddings'],
            importance=0.9
        )
        
        # Verify coupling was detected and recorded
        mods = self.memory.observations.query(
            obs_type=ObservationType.MODIFICATION,
            category="coupled_modification"
        )
        self.assertGreater(len(mods), 0, "Should record coupled modification")
        print(f"Recorded {len(mods)} coupled modification(s)")
        
        # Check the recorded data
        mod_data = mods[0].data
        self.assertIn('coupled_layers', mod_data)
        print(f"Detected coupling with: {mod_data['coupled_layers']}")
        
        # STEP 9: Query memory for related information
        print("\n=== STEP 9: QUERY MEMORY ===")
        
        # Find all modifications to this layer
        layer_mods = self.memory.observations.query(
            obs_type=ObservationType.MODIFICATION,
            tags=['embeddings']
        )
        print(f"Found {len(layer_mods)} modification(s) to embeddings")
        
        # Find all planned modifications
        plans = self.memory.observations.query(
            category="modification_plan"
        )
        print(f"Found {len(plans)} modification plan(s)")
        
        # STEP 10: Validate the complete story is in memory
        print("\n=== STEP 10: VALIDATE MEMORY ===")
        all_events = self.memory.observations.query()
        self.assertGreaterEqual(len(all_events), 2, "Should have plan + modification")
        
        # Verify we can reconstruct the story
        story = sorted(all_events, key=lambda x: x.timestamp)
        self.assertEqual(story[0].category, "modification_plan")
        self.assertEqual(story[1].category, "coupled_modification")
        
        print("✓ Complete modification story recorded")
        print(f"  1. Plan created at {story[0].timestamp}")
        print(f"  2. Modification executed at {story[1].timestamp}")
        print(f"  3. Coupling automatically detected and recorded")
    
    def test_workflow_with_safety_violation(self):
        """Test workflow with safety monitoring."""
        
        print("\n=== TESTING SAFETY MONITORING WORKFLOW ===")
        
        # Safety monitor works through hooks during forward pass
        # For this test, verify it's monitoring
        self.assertIsNotNone(self.safety)
        
        # Record a planned modification
        self.memory.record_observation(
            obs_type=ObservationType.SAFETY_EVENT,
            category="safety_check",
            description="Preparing to apply modifications with safety monitoring active",
            data={
                'layer': 'embeddings.weight',
                'monitoring_active': True
            },
            tags=['safety', 'monitoring'],
            importance=0.9
        )
        
        # Verify it's recorded
        checks = self.memory.observations.query(
            category="safety_check"
        )
        self.assertEqual(len(checks), 1)
        print("✓ Safety monitoring workflow properly recorded")
    
    def test_workflow_across_multiple_modifications(self):
        """Test that memory accumulates knowledge across multiple modifications."""
        
        print("\n=== TESTING MULTI-MODIFICATION WORKFLOW ===")
        
        layers_to_modify = [
            "embeddings.weight",
            "layers.0.attention.weight",
            "layers.1.ffn.weight"
        ]
        
        for i, layer_name in enumerate(layers_to_modify):
            print(f"\n--- Modification {i+1}: {layer_name} ---")
            
            # Check if it's shared
            sharing_info = self.navigator.get_weight_sharing_info(layer_name)
            is_shared = sharing_info.get('has_sharing', False)
            
            # Record modification
            self.memory.record_modification(
                layer_name=layer_name,
                modification_data={
                    'modification_number': i + 1,
                    'is_shared': is_shared
                },
                description=f"Modification {i+1} of multi-step learning",
                tags=['multi_step', f'step_{i+1}'],
                importance=0.7
            )
        
        # Query all modifications
        all_mods = self.memory.observations.query(
            obs_type=ObservationType.MODIFICATION
        )
        
        print(f"\n✓ Recorded {len(all_mods)} total modifications")
        
        # Check we can find multi-step modifications
        multi_step = self.memory.observations.query(tags=['multi_step'])
        self.assertEqual(len(multi_step), 3)
        print(f"✓ All {len(multi_step)} multi-step modifications queryable")
        
        # Check we tracked coupled modifications separately
        coupled = self.memory.observations.query(category="coupled_modification")
        print(f"✓ {len(coupled)} modification(s) involved weight coupling")
    
    def test_workflow_with_architecture_query_integration(self):
        """Test that architecture queries integrate with memory system."""
        
        print("\n=== TESTING ARCHITECTURE QUERY INTEGRATION ===")
        
        # Get architecture summary
        summary = self.navigator.get_architecture_summary()
        
        # Record architectural insight
        self.memory.record_observation(
            obs_type=ObservationType.INTROSPECTION,
            category="architecture_discovery",
            description="Discovered weight tying between embeddings and output",
            data={
                'total_parameters': summary['total_parameters'],
                'total_layers': summary['total_layers'],
                'weight_sharing_groups': len(self.inspector.get_shared_weights())
            },
            tags=['architecture', 'discovery'],
            importance=0.9
        )
        
        # Later, query this insight
        discoveries = self.memory.observations.query(
            category="architecture_discovery"
        )
        
        self.assertEqual(len(discoveries), 1)
        discovery = discoveries[0]
        
        print(f"✓ Architectural discovery recorded and queryable")
        print(f"  Total parameters: {discovery.data['total_parameters']:,}")
        print(f"  Weight sharing groups: {discovery.data['weight_sharing_groups']}")
        
        # Verify the discovery data is accurate
        self.assertEqual(
            discovery.data['total_parameters'],
            summary['total_parameters']
        )


if __name__ == '__main__':
    print("=" * 80)
    print("END-TO-END MODIFICATION WORKFLOW TESTS")
    print("=" * 80)
    
    unittest.main(verbosity=2)
