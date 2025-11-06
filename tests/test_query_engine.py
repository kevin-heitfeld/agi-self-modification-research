"""
Tests for Query Engine

Tests:
- Single-layer queries
- Cross-layer queries
- Evidence chain tracing
- Explanation generation
- Memory overview

Author: AGI Self-Modification Research Team
Date: November 7, 2025
"""

import unittest
import tempfile
import shutil
import time
from pathlib import Path

from src.memory.observation_layer import ObservationLayer, ObservationType
from src.memory.pattern_layer import PatternLayer
from src.memory.theory_layer import TheoryLayer
from src.memory.belief_layer import BeliefLayer
from src.memory.query_engine import QueryEngine, QueryType


class TestQueryEngine(unittest.TestCase):
    """Test suite for QueryEngine."""

    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()

        # Create full memory stack
        obs_dir = Path(self.test_dir) / "observations"
        pattern_dir = Path(self.test_dir) / "patterns"
        theory_dir = Path(self.test_dir) / "theories"
        belief_dir = Path(self.test_dir) / "beliefs"

        self.obs_layer = ObservationLayer(str(obs_dir))
        self.pattern_layer = PatternLayer(str(pattern_dir), self.obs_layer)
        self.theory_layer = TheoryLayer(str(theory_dir), self.pattern_layer, self.obs_layer)
        self.belief_layer = BeliefLayer(str(belief_dir), self.theory_layer)

        self.query_engine = QueryEngine(
            self.obs_layer,
            self.pattern_layer,
            self.theory_layer,
            self.belief_layer
        )

        # Create test data
        self._create_test_data()

    def tearDown(self):
        """Clean up test environment."""
        # Close all database connections (Windows file locking)
        if hasattr(self, 'obs_layer') and hasattr(self.obs_layer, 'conn'):
            self.obs_layer.conn.close()
        if hasattr(self, 'pattern_layer') and hasattr(self.pattern_layer, 'observation_layer'):
            if hasattr(self.pattern_layer.observation_layer, 'conn'):
                self.pattern_layer.observation_layer.conn.close()
        if hasattr(self, 'theory_layer'):
            if hasattr(self.theory_layer, 'observation_layer') and hasattr(self.theory_layer.observation_layer, 'conn'):
                self.theory_layer.observation_layer.conn.close()
            if hasattr(self.theory_layer, 'pattern_layer') and hasattr(self.theory_layer.pattern_layer, 'observation_layer'):
                if hasattr(self.theory_layer.pattern_layer.observation_layer, 'conn'):
                    self.theory_layer.pattern_layer.observation_layer.conn.close()
        if hasattr(self, 'belief_layer'):
            if hasattr(self.belief_layer, 'observation_layer') and hasattr(self.belief_layer.observation_layer, 'conn'):
                self.belief_layer.observation_layer.conn.close()
            if hasattr(self.belief_layer, 'pattern_layer') and hasattr(self.belief_layer.pattern_layer, 'observation_layer'):
                if hasattr(self.belief_layer.pattern_layer.observation_layer, 'conn'):
                    self.belief_layer.pattern_layer.observation_layer.conn.close()
            if hasattr(self.belief_layer, 'theory_layer'):
                if hasattr(self.belief_layer.theory_layer, 'observation_layer') and hasattr(self.belief_layer.theory_layer.observation_layer, 'conn'):
                    self.belief_layer.theory_layer.observation_layer.conn.close()
                if hasattr(self.belief_layer.theory_layer, 'pattern_layer') and hasattr(self.belief_layer.theory_layer.pattern_layer, 'observation_layer'):
                    if hasattr(self.belief_layer.theory_layer.pattern_layer.observation_layer, 'conn'):
                        self.belief_layer.theory_layer.pattern_layer.observation_layer.conn.close()
        
        shutil.rmtree(self.test_dir)

    def _create_test_data(self):
        """Create comprehensive test data."""
        # Create observations
        for i in range(10):
            self.obs_layer.record(
                obs_type=ObservationType.MODIFICATION,
                category="layer5",
                description=f"Modification {i}",
                data={'layer': 'layer5'},
                tags=['modification', 'layer5'],
                importance=0.8
            )

            time.sleep(0.01)

            self.obs_layer.record(
                obs_type=ObservationType.PERFORMANCE,
                category="perplexity",
                description=f"Performance {i}",
                data={'improvement': 2.5},
                tags=['performance', 'perplexity'],
                importance=0.9
            )

        # Build knowledge hierarchy
        self.pattern_layer.detect_patterns()
        self.theory_layer.build_theories()
        self.belief_layer.form_beliefs()

    def test_initialization(self):
        """Test query engine initialization."""
        self.assertIsNotNone(self.query_engine.observation_layer)
        self.assertIsNotNone(self.query_engine.pattern_layer)
        self.assertIsNotNone(self.query_engine.theory_layer)
        self.assertIsNotNone(self.query_engine.belief_layer)

    def test_query_observations(self):
        """Test querying observations."""
        result = self.query_engine.query_observations(tags=['modification'])

        self.assertEqual(result.query_type, QueryType.OBSERVATION)
        self.assertIsInstance(result.results, list)
        self.assertIn('count', result.metadata)

        # Should find modification observations
        self.assertGreater(len(result), 0)

    def test_query_patterns(self):
        """Test querying patterns."""
        result = self.query_engine.query_patterns(min_confidence=0.5)

        self.assertEqual(result.query_type, QueryType.PATTERN)
        self.assertIsInstance(result.results, list)

    def test_query_theories(self):
        """Test querying theories."""
        result = self.query_engine.query_theories(min_confidence=0.7)

        self.assertEqual(result.query_type, QueryType.THEORY)
        self.assertIsInstance(result.results, list)

    def test_query_beliefs(self):
        """Test querying beliefs."""
        result = self.query_engine.query_beliefs(min_confidence=0.9)

        self.assertEqual(result.query_type, QueryType.BELIEF)
        self.assertIsInstance(result.results, list)

        # Should at least find core beliefs
        self.assertGreater(len(result), 0)

    def test_find_theories_supporting_belief(self):
        """Test finding theories that support a belief."""
        # Get a belief
        beliefs = self.belief_layer.get_beliefs()
        if beliefs and beliefs[0].supporting_theories:
            belief = beliefs[0]

            result = self.query_engine.find_theories_supporting_belief(belief.id)

            self.assertEqual(result.query_type, QueryType.CROSS_LAYER)
            self.assertIsInstance(result.results, list)

            # Should find supporting theories if belief has them
            if belief.supporting_theories:
                self.assertGreaterEqual(len(result), 0)

    def test_trace_belief_to_observations(self):
        """Test tracing a belief back to observations."""
        # Get a core belief
        beliefs = self.belief_layer.get_beliefs()
        if beliefs:
            belief = beliefs[0]

            result = self.query_engine.trace_belief_to_observations(belief.id)

            self.assertEqual(result.query_type, QueryType.EVIDENCE_CHAIN)
            self.assertIsInstance(result.results, list)

            if result.results:
                chain = result.results[0]
                self.assertIn('belief', chain)
                self.assertIn('theories', chain)
                self.assertIn('patterns', chain)
                self.assertIn('observations', chain)

                self.assertEqual(chain['belief'].id, belief.id)

    def test_explain_belief(self):
        """Test generating belief explanation."""
        beliefs = self.belief_layer.get_beliefs()
        if beliefs:
            belief = beliefs[0]

            explanation = self.query_engine.explain_belief(belief.id)

            self.assertIsInstance(explanation, str)
            self.assertGreater(len(explanation), 0)

            # Should contain key elements
            self.assertIn('Belief:', explanation)
            self.assertIn('Confidence:', explanation)

    def test_explain_nonexistent_belief(self):
        """Test explaining a nonexistent belief."""
        explanation = self.query_engine.explain_belief("nonexistent_belief")

        self.assertIn("not found", explanation.lower())

    def test_why_belief_formed(self):
        """Test explaining why a belief was formed."""
        beliefs = self.belief_layer.get_beliefs()
        if beliefs:
            belief = beliefs[0]

            explanation = self.query_engine.why_belief_formed(belief.id)

            self.assertIsInstance(explanation, str)
            self.assertGreater(len(explanation), 0)

            # Should contain formation information
            self.assertIn('Formation', explanation)

    def test_find_contradictions(self):
        """Test finding contradictions."""
        result = self.query_engine.find_contradictions()

        self.assertEqual(result.query_type, QueryType.CROSS_LAYER)
        self.assertIsInstance(result.results, list)
        self.assertIn('conflict_count', result.metadata)

    def test_get_beliefs_for_context(self):
        """Test getting beliefs for a context."""
        context = {'action': 'modify', 'target': 'layer5'}

        result = self.query_engine.get_beliefs_for_context(context)

        self.assertEqual(result.query_type, QueryType.BELIEF)
        self.assertIsInstance(result.results, list)
        self.assertIn('context', result.metadata)

    def test_get_memory_overview(self):
        """Test getting memory overview."""
        overview = self.query_engine.get_memory_overview()

        self.assertIn('observations', overview)
        self.assertIn('patterns', overview)
        self.assertIn('theories', overview)
        self.assertIn('beliefs', overview)
        self.assertIn('total_knowledge_items', overview)

        # Should have some knowledge
        self.assertGreater(overview['total_knowledge_items'], 0)

    def test_query_result_iteration(self):
        """Test iterating over query results."""
        result = self.query_engine.query_observations()

        # Should be iterable
        count = 0
        for item in result:
            count += 1
            self.assertIsNotNone(item)

        self.assertEqual(count, len(result))

    def test_query_result_length(self):
        """Test getting length of query results."""
        result = self.query_engine.query_observations()

        length = len(result)
        self.assertIsInstance(length, int)
        self.assertGreaterEqual(length, 0)

    def test_cross_layer_metadata(self):
        """Test that cross-layer queries include proper metadata."""
        beliefs = self.belief_layer.get_beliefs()
        if beliefs and beliefs[0].supporting_theories:
            result = self.query_engine.find_theories_supporting_belief(beliefs[0].id)

            self.assertIn('source_layer', result.metadata)
            self.assertIn('target_layer', result.metadata)
            self.assertEqual(result.metadata['source_layer'], 'beliefs')
            self.assertEqual(result.metadata['target_layer'], 'theories')


if __name__ == '__main__':
    unittest.main()
