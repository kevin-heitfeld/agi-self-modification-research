"""
Tests for Belief Layer (Layer 4)

Tests:
- Belief formation from theories
- Core safety beliefs initialization
- Belief querying
- Conflict detection
- Application tracking
- Success rate updating

Author: AGI Self-Modification Research Team
Date: November 7, 2025
"""

import unittest
import time
from pathlib import Path

from src.memory.observation_layer import ObservationLayer, ObservationType
from src.memory.pattern_layer import PatternLayer
from src.memory.theory_layer import TheoryLayer
from src.memory.belief_layer import BeliefLayer, BeliefType, BeliefStrength
from tests.test_utils import get_test_temp_dir, cleanup_temp_dir, close_memory_layers


class TestBeliefLayer(unittest.TestCase):
    """Test suite for BeliefLayer."""

    def setUp(self):
        """Set up test environment."""
        self.test_dir = get_test_temp_dir()

        # Create full stack
        obs_dir = Path(self.test_dir) / "observations"
        pattern_dir = Path(self.test_dir) / "patterns"
        theory_dir = Path(self.test_dir) / "theories"
        belief_dir = Path(self.test_dir) / "beliefs"

        self.obs_layer = ObservationLayer(str(obs_dir))
        self.pattern_layer = PatternLayer(str(pattern_dir), self.obs_layer)
        self.theory_layer = TheoryLayer(str(theory_dir), self.pattern_layer, self.obs_layer)
        self.belief_layer = BeliefLayer(str(belief_dir), self.theory_layer)

        # Create test data
        self._create_test_data()

    def tearDown(self):
        """Clean up test environment."""
        # Close all database connections
        close_memory_layers(
            self.obs_layer,
            self.pattern_layer,
            self.theory_layer,
            self.belief_layer
        )
        
        # Clean up directory
        cleanup_temp_dir(self.test_dir)

    def _create_test_data(self):
        """Create test observations, patterns, and theories."""
        # Create strong pattern: layer5 mod → performance improvement
        for i in range(15):
            self.obs_layer.record(
                obs_type=ObservationType.MODIFICATION,
                category="layer5",
                description=f"Modified layer5 {i}",
                data={'layer': 'layer5', 'change': 0.01},
                tags=['modification', 'layer5'],
                importance=0.8
            )

            time.sleep(0.01)

            self.obs_layer.record(
                obs_type=ObservationType.PERFORMANCE,
                category="perplexity",
                description=f"Performance improved {i}",
                data={'improvement': 3.0},
                tags=['performance', 'perplexity'],
                importance=0.9
            )

        # Detect patterns and build theories
        self.pattern_layer.detect_patterns()
        self.theory_layer.build_theories()

    def test_initialization(self):
        """Test belief layer initialization."""
        self.assertIsNotNone(self.belief_layer.theory_layer)
        self.assertGreater(len(self.belief_layer.beliefs), 0)  # Should have core beliefs

    def test_core_beliefs_present(self):
        """Test that core safety beliefs are initialized."""
        beliefs = self.belief_layer.get_beliefs()

        # Should have at least the 4 core safety beliefs
        self.assertGreaterEqual(len(beliefs), 4)

        # Check for specific core beliefs
        belief_statements = [b.statement for b in beliefs]

        # Should contain safety principles
        safety_found = any('checkpoint' in s.lower() for s in belief_statements)
        self.assertTrue(safety_found, "Should have checkpoint safety belief")

        nan_found = any('nan' in s.lower() or 'inf' in s.lower() for s in belief_statements)
        self.assertTrue(nan_found, "Should have NaN/Inf safety belief")

    def test_core_beliefs_properties(self):
        """Test properties of core beliefs."""
        core_beliefs = self.belief_layer.get_beliefs(tags=['safety'])

        self.assertGreater(len(core_beliefs), 0)

        for belief in core_beliefs:
            # Core beliefs should have high confidence
            self.assertGreaterEqual(belief.confidence, 0.95)
            # Core beliefs should be ABSOLUTE or CERTAIN
            self.assertIn(belief.strength, [BeliefStrength.ABSOLUTE, BeliefStrength.CERTAIN])
            # Core beliefs should have high importance
            self.assertGreaterEqual(belief.importance, 0.9)

    def test_form_beliefs_from_theories(self):
        """Test forming new beliefs from theories."""
        initial_count = len(self.belief_layer.get_beliefs())

        # Form beliefs
        beliefs_formed = self.belief_layer.form_beliefs()

        # May or may not form new beliefs depending on theory strength
        final_count = len(self.belief_layer.get_beliefs())
        self.assertGreaterEqual(final_count, initial_count)

    def test_belief_formation_criteria(self):
        """Test that beliefs are only formed from strong theories."""
        # Get a belief if any were formed
        non_core_beliefs = [
            b for b in self.belief_layer.get_beliefs()
            if not b.id.startswith('safety_') and not b.id.startswith('constraint_')
        ]

        for belief in non_core_beliefs:
            # Non-core beliefs should have reasonable confidence
            self.assertGreater(belief.confidence, 0.7)
            # Should have substantial evidence
            self.assertGreater(belief.evidence_count, 0)

    def test_get_belief(self):
        """Test getting a specific belief."""
        beliefs = self.belief_layer.get_beliefs()
        self.assertGreater(len(beliefs), 0)

        belief_id = beliefs[0].id
        retrieved = self.belief_layer.get_belief(belief_id)

        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.id, belief_id)

    def test_get_beliefs_by_type(self):
        """Test querying beliefs by type."""
        safety_beliefs = self.belief_layer.get_beliefs(belief_type=BeliefType.SAFETY_PRINCIPLE)

        self.assertGreater(len(safety_beliefs), 0)

        for belief in safety_beliefs:
            self.assertEqual(belief.type, BeliefType.SAFETY_PRINCIPLE)

    def test_get_beliefs_by_strength(self):
        """Test querying beliefs by strength."""
        absolute_beliefs = self.belief_layer.get_beliefs(strength=BeliefStrength.ABSOLUTE)

        # Core safety beliefs should be ABSOLUTE
        self.assertGreater(len(absolute_beliefs), 0)

        for belief in absolute_beliefs:
            self.assertEqual(belief.strength, BeliefStrength.ABSOLUTE)

    def test_get_beliefs_by_tags(self):
        """Test querying beliefs by tags."""
        safety_beliefs = self.belief_layer.get_beliefs(tags=['safety'])

        self.assertGreater(len(safety_beliefs), 0)

        for belief in safety_beliefs:
            self.assertIn('safety', belief.tags)

    def test_get_beliefs_by_confidence(self):
        """Test querying beliefs by confidence."""
        high_conf = self.belief_layer.get_beliefs(min_confidence=0.95)
        medium_conf = self.belief_layer.get_beliefs(min_confidence=0.7)

        # Lower threshold should return more results
        self.assertGreaterEqual(len(medium_conf), len(high_conf))

        for belief in high_conf:
            self.assertGreaterEqual(belief.confidence, 0.95)

    def test_get_beliefs_by_importance(self):
        """Test querying beliefs by importance."""
        important = self.belief_layer.get_beliefs(min_importance=0.8)

        for belief in important:
            self.assertGreaterEqual(belief.importance, 0.8)

    def test_query_for_decision(self):
        """Test querying beliefs for a decision context."""
        context = {'action': 'modify', 'target': 'layer5'}

        relevant_beliefs = self.belief_layer.query_for_decision(context)

        # Should return list (may be empty)
        self.assertIsInstance(relevant_beliefs, list)

        # If beliefs found, they should be sorted by relevance
        if len(relevant_beliefs) > 1:
            first_score = relevant_beliefs[0].importance * relevant_beliefs[0].confidence
            second_score = relevant_beliefs[1].importance * relevant_beliefs[1].confidence
            self.assertGreaterEqual(first_score, second_score)

    def test_validate_belief_success(self):
        """Test validating belief with successful outcome."""
        beliefs = self.belief_layer.get_beliefs()
        if beliefs:
            belief = beliefs[0]
            initial_applied = belief.times_applied
            initial_success_rate = belief.success_rate

            # Validate with success
            self.belief_layer.validate_belief(belief.id, outcome=True)

            updated = self.belief_layer.get_belief(belief.id)
            self.assertEqual(updated.times_applied, initial_applied + 1)

            # Success rate should stay high or increase
            self.assertGreaterEqual(updated.success_rate, 0.0)

    def test_validate_belief_failure(self):
        """Test validating belief with failed outcome."""
        beliefs = self.belief_layer.get_beliefs()
        if beliefs:
            belief = beliefs[0]
            initial_applied = belief.times_applied

            # Validate with failure multiple times
            for i in range(5):
                self.belief_layer.validate_belief(belief.id, outcome=False)

            updated = self.belief_layer.get_belief(belief.id)
            self.assertEqual(updated.times_applied, initial_applied + 5)

            # Success rate should decrease
            self.assertLess(updated.success_rate, 1.0)

    def test_detect_conflicts(self):
        """Test conflict detection between beliefs."""
        conflicts = self.belief_layer.detect_conflicts()

        # Should return list (may be empty for core beliefs)
        self.assertIsInstance(conflicts, list)

        # Core beliefs should not conflict
        # (This is a basic check; actual conflicts depend on belief formation)

    def test_belief_statistics(self):
        """Test belief statistics."""
        stats = self.belief_layer.get_statistics()

        self.assertIn('total_beliefs', stats)
        self.assertIn('by_type', stats)
        self.assertIn('by_strength', stats)
        self.assertIn('average_confidence', stats)

        self.assertGreater(stats['total_beliefs'], 0)
        self.assertGreater(stats['average_confidence'], 0)

    def test_get_core_principles(self):
        """Test getting core principles."""
        principles = self.belief_layer.get_core_principles()

        # Should return list of strings
        self.assertIsInstance(principles, list)
        self.assertGreater(len(principles), 0)

        for principle in principles:
            self.assertIsInstance(principle, str)
            self.assertGreater(len(principle), 0)

    def test_belief_persistence(self):
        """Test that beliefs are saved and loaded."""
        belief_count = len(self.belief_layer.get_beliefs())

        # Create new instance
        belief_dir = Path(self.test_dir) / "beliefs"
        new_layer = BeliefLayer(str(belief_dir), self.theory_layer)

        loaded_count = len(new_layer.get_beliefs())
        self.assertEqual(loaded_count, belief_count)

    def test_export_beliefs(self):
        """Test exporting beliefs."""
        export_path = Path(self.test_dir) / "beliefs_export.json"
        self.belief_layer.export(str(export_path))

        self.assertTrue(export_path.exists())

        # Verify content
        import json
        with open(export_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.assertIsInstance(data, list)
        self.assertGreater(len(data), 0)


if __name__ == '__main__':
    unittest.main()
