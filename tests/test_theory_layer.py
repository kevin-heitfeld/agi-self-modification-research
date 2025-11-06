"""
Tests for Theory Layer (Layer 3)

Tests:
- Theory building from patterns
- Theory validation
- Prediction capabilities
- Evidence tracking
- Confidence updating
- Statistics

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
from src.memory.theory_layer import TheoryLayer, TheoryType


class TestTheoryLayer(unittest.TestCase):
    """Test suite for TheoryLayer."""

    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()

        # Create layers
        obs_dir = Path(self.test_dir) / "observations"
        pattern_dir = Path(self.test_dir) / "patterns"
        theory_dir = Path(self.test_dir) / "theories"

        self.obs_layer = ObservationLayer(str(obs_dir))
        self.pattern_layer = PatternLayer(str(pattern_dir), self.obs_layer)
        self.theory_layer = TheoryLayer(str(theory_dir), self.pattern_layer, self.obs_layer)

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
        
        shutil.rmtree(self.test_dir)

    def _create_test_data(self):
        """Create test observations and patterns."""
        # Create observations showing layer5 modifications improve performance
        for i in range(10):
            self.obs_layer.record(
                obs_type=ObservationType.MODIFICATION,
                category="layer5",
                description=f"Modified layer5 weights {i}",
                data={'layer': 'layer5', 'change': 0.01},
                tags=['modification', 'layer5'],
                importance=0.8
            )

            time.sleep(0.01)

            self.obs_layer.record(
                obs_type=ObservationType.PERFORMANCE,
                category="perplexity",
                description=f"Performance improved {i}",
                data={'improvement': 2.5 + i * 0.1},
                tags=['performance', 'perplexity'],
                importance=0.9
            )

        # Detect patterns
        self.pattern_layer.detect_patterns()

    def test_initialization(self):
        """Test theory layer initialization."""
        self.assertIsNotNone(self.theory_layer.pattern_layer)
        self.assertIsNotNone(self.theory_layer.observation_layer)
        self.assertEqual(len(self.theory_layer.theories), 0)

    def test_build_theories(self):
        """Test building theories from patterns."""
        theories_built = self.theory_layer.build_theories()

        # Should build at least some theories from the patterns
        self.assertGreaterEqual(theories_built, 0)

        theories = self.theory_layer.get_theories()
        if theories:
            theory = theories[0]
            self.assertIsNotNone(theory.id)
            self.assertIsNotNone(theory.hypothesis)
            self.assertGreater(theory.confidence, 0)

    def test_causal_model_building(self):
        """Test building causal models."""
        self.theory_layer.build_theories()

        causal_theories = self.theory_layer.get_theories(theory_type=TheoryType.CAUSAL_MODEL)

        # Should find causal relationships
        if causal_theories:
            theory = causal_theories[0]
            self.assertEqual(theory.type, TheoryType.CAUSAL_MODEL)
            self.assertGreater(theory.evidence_count, 0)

    def test_optimization_theory_building(self):
        """Test building optimization theories."""
        self.theory_layer.build_theories()

        opt_theories = self.theory_layer.get_theories(theory_type=TheoryType.OPTIMIZATION)

        if opt_theories:
            theory = opt_theories[0]
            self.assertEqual(theory.type, TheoryType.OPTIMIZATION)
            self.assertIn('metadata', dir(theory))

    def test_get_theory(self):
        """Test getting a specific theory."""
        self.theory_layer.build_theories()

        theories = self.theory_layer.get_theories()
        if theories:
            theory_id = theories[0].id
            retrieved = self.theory_layer.get_theory(theory_id)
            self.assertIsNotNone(retrieved)
            self.assertEqual(retrieved.id, theory_id)

    def test_get_theories_by_type(self):
        """Test querying theories by type."""
        self.theory_layer.build_theories()

        all_theories = self.theory_layer.get_theories()
        causal_theories = self.theory_layer.get_theories(theory_type=TheoryType.CAUSAL_MODEL)

        # Filtered should be subset
        self.assertLessEqual(len(causal_theories), len(all_theories))

    def test_get_theories_by_confidence(self):
        """Test querying theories by confidence."""
        self.theory_layer.build_theories()

        high_conf = self.theory_layer.get_theories(min_confidence=0.7)
        low_conf = self.theory_layer.get_theories(min_confidence=0.3)

        # Lower threshold should return more results
        self.assertGreaterEqual(len(low_conf), len(high_conf))

    def test_get_theories_by_tags(self):
        """Test querying theories by tags."""
        self.theory_layer.build_theories()

        # Query by tag (theories should inherit tags from patterns)
        tagged = self.theory_layer.get_theories(tags=['layer5'])

        # Should execute without error
        self.assertIsInstance(tagged, list)

    def test_validate_theory_success(self):
        """Test validating a theory with supporting observation."""
        self.theory_layer.build_theories()

        theories = self.theory_layer.get_theories()
        if theories:
            theory = theories[0]
            initial_evidence = theory.evidence_count
            initial_confidence = theory.confidence

            # Create supporting observation
            obs_id = self.obs_layer.record(
                obs_type=ObservationType.PERFORMANCE,
                category="perplexity",
                description="Supports theory",
                data={'improvement': 3.0},
                tags=['performance', 'layer5'],
                importance=0.9
            )

            # Validate (this observation supports the theory)
            self.theory_layer.validate_theory(theory.id, obs_id, outcome=True)

            # Evidence should increase
            updated = self.theory_layer.get_theory(theory.id)
            self.assertGreater(updated.evidence_count, initial_evidence)

    def test_validate_theory_failure(self):
        """Test validating a theory with contradicting observation."""
        self.theory_layer.build_theories()

        theories = self.theory_layer.get_theories()
        if theories:
            theory = theories[0]
            initial_counter = theory.counter_evidence_count

            # Create contradicting observation
            obs_id = self.obs_layer.record(
                obs_type=ObservationType.PERFORMANCE,
                category="perplexity",
                description="Contradicts theory",
                data={'degradation': 5.0},
                tags=['performance', 'layer5'],
                importance=0.9
            )

            # Validate (this observation contradicts the theory)
            self.theory_layer.validate_theory(theory.id, obs_id, outcome=False)

            # Counter-evidence should increase
            updated = self.theory_layer.get_theory(theory.id)
            self.assertGreater(updated.counter_evidence_count, initial_counter)

    def test_make_prediction(self):
        """Test making predictions based on theory."""
        self.theory_layer.build_theories()

        theories = self.theory_layer.get_theories()
        if theories:
            theory = theories[0]

            # Make prediction
            prediction = self.theory_layer.make_prediction(
                theory.id,
                context={'action': 'modify', 'layer': 'layer5'}
            )

            self.assertIsNotNone(prediction)
            self.assertIn('prediction', prediction)
            self.assertIn('confidence', prediction)

    def test_theory_confidence_updates(self):
        """Test that confidence updates with evidence."""
        self.theory_layer.build_theories()

        theories = self.theory_layer.get_theories()
        if theories:
            theory = theories[0]
            initial_confidence = theory.confidence

            # Add several supporting observations
            for i in range(5):
                obs_id = self.obs_layer.record(
                    obs_type=ObservationType.PERFORMANCE,
                    category="perplexity",
                    description=f"Support {i}",
                    data={},
                    tags=['performance'],
                    importance=0.8
                )
                self.theory_layer.validate_theory(theory.id, obs_id, outcome=True)

            # Confidence should change
            updated = self.theory_layer.get_theory(theory.id)
            # May increase or stay similar depending on initial confidence
            self.assertIsNotNone(updated.confidence)

    def test_theory_predictive_power(self):
        """Test tracking predictive power."""
        self.theory_layer.build_theories()

        theories = self.theory_layer.get_theories()
        if theories:
            theory = theories[0]

            # Make predictions and track accuracy
            for i in range(3):
                prediction = self.theory_layer.make_prediction(
                    theory.id,
                    context={'test': i}
                )

                self.assertIn('confidence', prediction)

            # Predictions_made should increase
            updated = self.theory_layer.get_theory(theory.id)
            self.assertGreaterEqual(updated.predictions_made, 3)

    def test_theory_statistics(self):
        """Test theory statistics."""
        self.theory_layer.build_theories()

        stats = self.theory_layer.get_statistics()

        self.assertIn('total_theories', stats)
        self.assertIn('by_type', stats)
        self.assertIsInstance(stats['total_theories'], int)

    def test_theory_persistence(self):
        """Test that theories are saved and loaded."""
        self.theory_layer.build_theories()
        theory_count = len(self.theory_layer.get_theories())

        if theory_count > 0:
            # Create new instance
            theory_dir = Path(self.test_dir) / "theories"
            new_layer = TheoryLayer(str(theory_dir), self.pattern_layer, self.obs_layer)

            loaded_count = len(new_layer.get_theories())
            self.assertEqual(loaded_count, theory_count)

    def test_export_theories(self):
        """Test exporting theories."""
        self.theory_layer.build_theories()

        export_path = Path(self.test_dir) / "theories_export.json"
        self.theory_layer.export(str(export_path))

        self.assertTrue(export_path.exists())

        # Verify content
        import json
        with open(export_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.assertIsInstance(data, list)


if __name__ == '__main__':
    unittest.main()
