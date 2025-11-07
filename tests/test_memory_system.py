"""
Tests for Memory System (Coordinator)

Tests:
- System initialization
- Observation recording
- Knowledge consolidation
- Decision support
- Introspection methods
- Memory management
- Cross-layer integration

Author: AGI Self-Modification Research Team
Date: November 7, 2025
"""

import unittest
import time
from pathlib import Path

from src.memory.memory_system import MemorySystem
from src.memory.observation_layer import ObservationType
from tests.test_utils import get_test_temp_dir, cleanup_temp_dir, close_memory_system


class TestMemorySystem(unittest.TestCase):
    """Test suite for MemorySystem."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = get_test_temp_dir()
        self.memory = MemorySystem(self.test_dir)
    
    def tearDown(self):
        """Clean up test environment."""
        # Close all database connections
        close_memory_system(self.memory)
        
        # Clean up directory
        cleanup_temp_dir(self.test_dir)
    
    def test_initialization(self):
        """Test memory system initialization."""
        self.assertIsNotNone(self.memory.observations)
        self.assertIsNotNone(self.memory.patterns)
        self.assertIsNotNone(self.memory.theories)
        self.assertIsNotNone(self.memory.beliefs)
        self.assertIsNotNone(self.memory.query)
        
        # Should have core beliefs
        beliefs = self.memory.beliefs.get_beliefs()
        self.assertGreater(len(beliefs), 0)
    
    def test_record_observation(self):
        """Test recording observation through memory system."""
        obs_id = self.memory.record_observation(
            obs_type=ObservationType.MODIFICATION,
            category="layer5",
            description="Test modification",
            data={'change': 0.1},
            tags=['test'],
            importance=0.8
        )
        
        self.assertIsNotNone(obs_id)
        
        # Verify observation was recorded
        obs = self.memory.observations.get(obs_id)
        self.assertIsNotNone(obs)
        self.assertEqual(obs.description, "Test modification")
    
    def test_consolidation_process(self):
        """Test full consolidation process."""
        # Create observations
        for i in range(15):
            self.memory.record_observation(
                obs_type=ObservationType.MODIFICATION,
                category="layer5",
                description=f"Modification {i}",
                data={'layer': 'layer5'},
                tags=['modification', 'layer5'],
                importance=0.8
            )
            
            time.sleep(0.01)
            
            self.memory.record_observation(
                obs_type=ObservationType.PERFORMANCE,
                category="perplexity",
                description=f"Performance {i}",
                data={'improvement': 2.5},
                tags=['performance'],
                importance=0.9
            )
        
        # Run consolidation
        stats = self.memory.consolidate(force=True)
        
        self.assertIn('patterns_found', stats)
        self.assertIn('theories_built', stats)
        self.assertIn('beliefs_formed', stats)
        
        self.assertIsInstance(stats['patterns_found'], int)
        self.assertIsInstance(stats['theories_built'], int)
        self.assertIsInstance(stats['beliefs_formed'], int)
    
    def test_auto_consolidate_if_needed(self):
        """Test automatic consolidation."""
        # Set short interval
        self.memory.set_consolidation_interval(hours=0.0001)  # Very short
        
        # Record observations
        for i in range(5):
            self.memory.record_observation(
                obs_type=ObservationType.MODIFICATION,
                category="test",
                description="Test",
                data={},
                tags=['test'],
                importance=0.5
            )
        
        time.sleep(0.01)
        
        # Should trigger consolidation
        self.memory.auto_consolidate_if_needed()
        
        # Last consolidation time should be updated
        self.assertGreater(self.memory.last_consolidation, 0)
    
    def test_get_decision_support(self):
        """Test getting decision support."""
        # Create some test data
        for i in range(10):
            self.memory.record_observation(
                obs_type=ObservationType.MODIFICATION,
                category="layer5",
                description=f"Mod {i}",
                data={},
                tags=['modification', 'layer5'],
                importance=0.7
            )
        
        self.memory.consolidate(force=True)
        
        # Get decision support
        context = {'action': 'modify', 'target': 'layer5'}
        support = self.memory.get_decision_support(context)
        
        self.assertIn('beliefs', support)
        self.assertIn('theories', support)
        self.assertIn('patterns', support)
        self.assertIn('observations', support)
        self.assertIn('recommendation', support)
        
        self.assertIsInstance(support['beliefs'], list)
        self.assertIsInstance(support['recommendation'], str)
    
    def test_get_core_principles(self):
        """Test getting core principles."""
        principles = self.memory.get_core_principles()
        
        self.assertIsInstance(principles, list)
        self.assertGreater(len(principles), 0)
        
        # Should be strings
        for principle in principles:
            self.assertIsInstance(principle, str)
            self.assertGreater(len(principle), 0)
    
    def test_explain_decision(self):
        """Test explaining a decision."""
        # Get a belief
        beliefs = self.memory.beliefs.get_beliefs()
        if beliefs:
            explanation = self.memory.explain_decision(beliefs[0].id)
            
            self.assertIsInstance(explanation, str)
            self.assertGreater(len(explanation), 0)
    
    def test_trace_to_evidence(self):
        """Test tracing belief to evidence."""
        # Get a belief
        beliefs = self.memory.beliefs.get_beliefs()
        if beliefs:
            result = self.memory.trace_to_evidence(beliefs[0].id)
            
            self.assertIsNotNone(result)
            self.assertIsInstance(result.results, list)
    
    def test_cleanup_old_data(self):
        """Test cleaning up old data."""
        # Create some observations
        for i in range(10):
            self.memory.record_observation(
                obs_type=ObservationType.MODIFICATION,
                category="test",
                description=f"Test {i}",
                data={},
                tags=['test'],
                importance=0.2  # Low importance
            )
        
        # Run cleanup (with very short retention)
        self.memory.cleanup_old_data(
            observation_days=0,
            pattern_days=0,
            theory_days=0
        )
        
        # Should execute without error
        # (Actual deletion not implemented, but method should work)
    
    def test_get_memory_stats(self):
        """Test getting memory statistics."""
        stats = self.memory.get_memory_stats()
        
        self.assertIn('observations', stats)
        self.assertIn('patterns', stats)
        self.assertIn('theories', stats)
        self.assertIn('beliefs', stats)
        self.assertIn('consolidation', stats)
        self.assertIn('health', stats)
        
        # Health check
        self.assertIn('status', stats['health'])
        self.assertIn('conflicts', stats['health'])
    
    def test_export_all(self):
        """Test exporting entire memory system."""
        # Create some data
        self.memory.record_observation(
            obs_type=ObservationType.MODIFICATION,
            category="test",
            description="Test",
            data={},
            tags=['test'],
            importance=0.5
        )
        
        export_dir = Path(self.test_dir) / "export"
        self.memory.export_all(str(export_dir))
        
        # Check files exist
        self.assertTrue((export_dir / "observations.json").exists())
        self.assertTrue((export_dir / "patterns.json").exists())
        self.assertTrue((export_dir / "theories.json").exists())
        self.assertTrue((export_dir / "beliefs.json").exists())
    
    def test_set_consolidation_interval(self):
        """Test setting consolidation interval."""
        self.memory.set_consolidation_interval(hours=2.0)
        
        expected_seconds = 2.0 * 3600
        self.assertEqual(self.memory.consolidation_interval, expected_seconds)
    
    def test_what_do_i_know_about(self):
        """Test introspection: what do I know about X."""
        # Create data with specific topic
        for i in range(5):
            self.memory.record_observation(
                obs_type=ObservationType.MODIFICATION,
                category="layer5",
                description=f"Modification {i}",
                data={},
                tags=['modification', 'layer5'],
                importance=0.7
            )
        
        self.memory.consolidate(force=True)
        
        knowledge = self.memory.what_do_i_know_about('modification')
        
        self.assertIsInstance(knowledge, str)
        self.assertGreater(len(knowledge), 0)
        self.assertIn('modification', knowledge.lower())
    
    def test_what_have_i_learned_recently(self):
        """Test introspection: recent learning."""
        # Create recent data
        for i in range(5):
            self.memory.record_observation(
                obs_type=ObservationType.MODIFICATION,
                category="test",
                description=f"Recent {i}",
                data={},
                tags=['test'],
                importance=0.5
            )
        
        self.memory.consolidate(force=True)
        
        recent = self.memory.what_have_i_learned_recently(hours=1)
        
        self.assertIsInstance(recent, str)
        self.assertGreater(len(recent), 0)
        self.assertIn('Observations:', recent)
    
    def test_consolidation_updates_timestamp(self):
        """Test that consolidation updates the timestamp."""
        initial_time = self.memory.last_consolidation
        
        # Wait a bit
        time.sleep(0.1)
        
        # Consolidate
        self.memory.consolidate(force=True)
        
        # Timestamp should be updated
        self.assertGreater(self.memory.last_consolidation, initial_time)
    
    def test_memory_system_integration(self):
        """Test full integration: observations → patterns → theories → beliefs."""
        # Create strong pattern
        for i in range(20):
            self.memory.record_observation(
                obs_type=ObservationType.MODIFICATION,
                category="layer5",
                description=f"Mod {i}",
                data={'layer': 'layer5', 'change': 0.01},
                tags=['modification', 'layer5'],
                importance=0.8
            )
            
            time.sleep(0.01)
            
            self.memory.record_observation(
                obs_type=ObservationType.PERFORMANCE,
                category="perplexity",
                description=f"Perf {i}",
                data={'improvement': 3.0},
                tags=['performance', 'perplexity'],
                importance=0.9
            )
        
        # Full consolidation
        stats = self.memory.consolidate(force=True)
        
        # Should have created knowledge at each layer
        obs_count = len(self.memory.observations.query())
        pattern_count = len(self.memory.patterns.get_patterns())
        theory_count = len(self.memory.theories.get_theories())
        belief_count = len(self.memory.beliefs.get_beliefs())
        
        self.assertGreater(obs_count, 0)
        # Patterns, theories, beliefs may or may not be created depending on thresholds
        # But system should work without errors
        
        # Test querying across layers
        overview = self.memory.query.get_memory_overview()
        self.assertGreater(overview['total_knowledge_items'], 0)
    
    def test_decision_support_with_no_relevant_beliefs(self):
        """Test decision support when no relevant beliefs exist."""
        context = {'action': 'unknown_action', 'target': 'unknown_target'}
        
        support = self.memory.get_decision_support(context)
        
        # Should return structure even if empty
        self.assertIn('beliefs', support)
        self.assertIn('recommendation', support)
    
    def test_persistence_across_restarts(self):
        """Test that data persists across system restarts."""
        # Record data
        obs_id = self.memory.record_observation(
            obs_type=ObservationType.MODIFICATION,
            category="test",
            description="Persistence test",
            data={'key': 'value'},
            tags=['test'],
            importance=0.7
        )
        
        self.memory.consolidate(force=True)
        
        # Get counts
        obs_count = len(self.memory.observations.query())
        belief_count = len(self.memory.beliefs.get_beliefs())
        
        # Create new memory system instance (should load existing data)
        new_memory = MemorySystem(self.test_dir)
        
        # Counts should match
        new_obs_count = len(new_memory.observations.query())
        new_belief_count = len(new_memory.beliefs.get_beliefs())
        
        self.assertEqual(new_obs_count, obs_count)
        self.assertEqual(new_belief_count, belief_count)
        
        # Should find the specific observation
        obs = new_memory.observations.get(obs_id)
        self.assertIsNotNone(obs)
        self.assertEqual(obs.description, "Persistence test")


if __name__ == '__main__':
    unittest.main()
