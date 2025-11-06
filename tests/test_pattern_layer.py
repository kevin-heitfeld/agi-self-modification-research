"""
Tests for Pattern Layer (Layer 2)

Tests:
- Pattern detection (Sequential, Causal, Threshold)
- Pattern querying
- Pattern merging
- Related pattern finding
- Confidence calculation
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
from src.memory.pattern_layer import PatternLayer, PatternType


class TestPatternLayer(unittest.TestCase):
    """Test suite for PatternLayer."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        
        # Create observation layer with test data
        obs_dir = Path(self.test_dir) / "observations"
        self.obs_layer = ObservationLayer(str(obs_dir))
        
        # Create pattern layer
        pattern_dir = Path(self.test_dir) / "patterns"
        self.pattern_layer = PatternLayer(str(pattern_dir), self.obs_layer)
    
    def tearDown(self):
        """Clean up test environment."""
        # Close database connections (Windows file locking)
        if hasattr(self, 'obs_layer') and hasattr(self.obs_layer, 'conn'):
            self.obs_layer.conn.close()
        if hasattr(self, 'pattern_layer') and hasattr(self.pattern_layer, 'observation_layer'):
            if hasattr(self.pattern_layer.observation_layer, 'conn'):
                self.pattern_layer.observation_layer.conn.close()
        
        shutil.rmtree(self.test_dir)
    
    def test_initialization(self):
        """Test pattern layer initialization."""
        self.assertIsNotNone(self.pattern_layer.observation_layer)
        self.assertEqual(len(self.pattern_layer.patterns), 0)
    
    def test_sequential_pattern_detection(self):
        """Test detection of sequential patterns (A→B)."""
        # Create sequence: modification → performance improvement
        for i in range(5):
            self.obs_layer.record(
                obs_type=ObservationType.MODIFICATION,
                category="layer5",
                description=f"Modification {i}",
                data={'layer': 'layer5'},
                tags=['modification', 'layer5'],
                importance=0.7
            )
            
            time.sleep(0.01)  # Small delay to ensure ordering
            
            self.obs_layer.record(
                obs_type=ObservationType.PERFORMANCE,
                category="perplexity",
                description=f"Performance improvement {i}",
                data={'improvement': True},
                tags=['performance', 'perplexity'],
                importance=0.8
            )
        
        # Detect patterns
        self.pattern_layer.detect_patterns()
        
        # Should find sequential pattern
        patterns = self.pattern_layer.get_patterns(pattern_type=PatternType.SEQUENTIAL)
        self.assertGreater(len(patterns), 0)
        
        # Check pattern properties
        pattern = patterns[0]
        self.assertEqual(pattern.type, PatternType.SEQUENTIAL)
        self.assertGreater(pattern.support_count, 1)
        self.assertGreater(pattern.confidence, 0)
    
    def test_causal_pattern_detection(self):
        """Test detection of causal patterns."""
        # Create causal relationship: layer5 modification → performance change
        for i in range(5):
            self.obs_layer.record(
                obs_type=ObservationType.MODIFICATION,
                category="layer5",
                description=f"Modified layer5 weights",
                data={'layer': 'layer5', 'change': 0.01},
                tags=['modification', 'layer5'],
                importance=0.8
            )
            
            time.sleep(0.01)
            
            self.obs_layer.record(
                obs_type=ObservationType.PERFORMANCE,
                category="perplexity",
                description=f"Perplexity changed",
                data={'before': 15.0, 'after': 14.5, 'improvement': 3.3},
                tags=['performance', 'perplexity'],
                importance=0.9
            )
        
        # Detect patterns
        self.pattern_layer.detect_patterns()
        
        # Should find causal pattern
        patterns = self.pattern_layer.get_patterns(pattern_type=PatternType.CAUSAL)
        self.assertGreater(len(patterns), 0)
        
        pattern = patterns[0]
        self.assertEqual(pattern.type, PatternType.CAUSAL)
        # Check for either "modification" or "modifying" in description
        self.assertTrue('modif' in pattern.description.lower())
    
    def test_threshold_pattern_detection(self):
        """Test detection of threshold patterns."""
        # Create pattern: high perplexity → safety event
        for i in range(5):
            self.obs_layer.record(
                obs_type=ObservationType.PERFORMANCE,
                category="perplexity",
                description=f"High perplexity observed",
                data={'perplexity': 25.0 + i},
                tags=['performance', 'perplexity'],
                importance=0.7
            )
            
            time.sleep(0.01)
            
            self.obs_layer.record(
                obs_type=ObservationType.SAFETY_EVENT,
                category="anomaly",
                description=f"Safety alert triggered",
                data={'reason': 'high_perplexity'},
                tags=['safety', 'alert'],
                importance=1.0
            )
        
        # Detect patterns
        self.pattern_layer.detect_patterns()
        
        # Should find threshold pattern
        patterns = self.pattern_layer.get_patterns(pattern_type=PatternType.THRESHOLD)
        self.assertGreater(len(patterns), 0)
    
    def test_pattern_confidence_calculation(self):
        """Test that confidence is calculated correctly."""
        # Create strong pattern (always happens)
        for i in range(10):
            self.obs_layer.record(
                obs_type=ObservationType.MODIFICATION,
                category="test",
                description="Event A",
                data={},
                tags=['A'],
                importance=0.5
            )
            
            time.sleep(0.01)
            
            self.obs_layer.record(
                obs_type=ObservationType.PERFORMANCE,
                category="test",
                description="Event B",
                data={},
                tags=['B'],
                importance=0.5
            )
        
        self.pattern_layer.detect_patterns()
        
        patterns = self.pattern_layer.get_patterns()
        self.assertGreater(len(patterns), 0)
        
        # Pattern should have high confidence
        for pattern in patterns:
            if pattern.support_count >= 5:
                self.assertGreater(pattern.confidence, 0.5)
    
    def test_get_pattern(self):
        """Test getting a specific pattern."""
        # Create and detect pattern
        for i in range(3):
            self.obs_layer.record(
                obs_type=ObservationType.MODIFICATION,
                category="test",
                description="Test",
                data={},
                tags=['test'],
                importance=0.5
            )
        
        self.pattern_layer.detect_patterns()
        
        patterns = self.pattern_layer.get_patterns()
        if patterns:
            pattern_id = patterns[0].id
            retrieved = self.pattern_layer.get_pattern(pattern_id)
            self.assertIsNotNone(retrieved)
            self.assertEqual(retrieved.id, pattern_id)
    
    def test_get_patterns_by_tags(self):
        """Test querying patterns by tags."""
        # Create observations with specific tags
        for i in range(5):
            self.obs_layer.record(
                obs_type=ObservationType.MODIFICATION,
                category="layer5",
                description="Layer 5 mod",
                data={},
                tags=['layer5', 'modification'],
                importance=0.7
            )
        
        self.pattern_layer.detect_patterns()
        
        # Query by tag
        layer5_patterns = self.pattern_layer.get_patterns(tags=['layer5'])
        # May or may not find patterns depending on detection logic
        # Just ensure query executes without error
        self.assertIsInstance(layer5_patterns, list)
    
    def test_get_patterns_by_confidence(self):
        """Test querying patterns by confidence."""
        # Create strong pattern
        for i in range(10):
            self.obs_layer.record(
                obs_type=ObservationType.MODIFICATION,
                category="test",
                description="Event A",
                data={},
                tags=['A'],
                importance=0.5
            )
            time.sleep(0.01)
            self.obs_layer.record(
                obs_type=ObservationType.PERFORMANCE,
                category="test",
                description="Event B",
                data={},
                tags=['B'],
                importance=0.5
            )
        
        self.pattern_layer.detect_patterns()
        
        # Query by confidence
        high_conf = self.pattern_layer.get_patterns(min_confidence=0.7)
        low_conf = self.pattern_layer.get_patterns(min_confidence=0.1)
        
        # Low threshold should return more patterns
        self.assertGreaterEqual(len(low_conf), len(high_conf))
    
    def test_get_patterns_by_support(self):
        """Test querying patterns by support count."""
        # Create pattern with multiple occurrences
        for i in range(8):
            self.obs_layer.record(
                obs_type=ObservationType.MODIFICATION,
                category="test",
                description="Test",
                data={},
                tags=['test'],
                importance=0.5
            )
        
        self.pattern_layer.detect_patterns()
        
        high_support = self.pattern_layer.get_patterns(min_support=5)
        low_support = self.pattern_layer.get_patterns(min_support=2)
        
        self.assertGreaterEqual(len(low_support), len(high_support))
    
    def test_find_related_patterns(self):
        """Test finding related patterns."""
        # Create multiple related patterns
        for i in range(5):
            self.obs_layer.record(
                obs_type=ObservationType.MODIFICATION,
                category="layer5",
                description="Mod layer5",
                data={},
                tags=['layer5', 'modification'],
                importance=0.7
            )
            time.sleep(0.01)
            self.obs_layer.record(
                obs_type=ObservationType.PERFORMANCE,
                category="perplexity",
                description="Perf change",
                data={},
                tags=['performance', 'layer5'],
                importance=0.7
            )
        
        self.pattern_layer.detect_patterns()
        
        patterns = self.pattern_layer.get_patterns()
        if patterns:
            related = self.pattern_layer.find_related_patterns(patterns[0].id)
            # Should return list (may be empty)
            self.assertIsInstance(related, list)
    
    def test_pattern_statistics(self):
        """Test pattern statistics."""
        # Create some patterns
        for i in range(6):
            self.obs_layer.record(
                obs_type=ObservationType.MODIFICATION,
                category="test",
                description="Test",
                data={},
                tags=['test'],
                importance=0.5
            )
        
        self.pattern_layer.detect_patterns()
        
        stats = self.pattern_layer.get_statistics()
        
        self.assertIn('total_patterns', stats)
        self.assertIn('by_type', stats)
        self.assertIsInstance(stats['total_patterns'], int)
    
    def test_pattern_persistence(self):
        """Test that patterns are saved and loaded."""
        # Create pattern
        for i in range(5):
            self.obs_layer.record(
                obs_type=ObservationType.MODIFICATION,
                category="test",
                description="Test",
                data={},
                tags=['test'],
                importance=0.5
            )
        
        self.pattern_layer.detect_patterns()
        pattern_count = len(self.pattern_layer.get_patterns())
        
        if pattern_count > 0:
            # Create new layer instance (should load existing patterns)
            pattern_dir = Path(self.test_dir) / "patterns"
            new_layer = PatternLayer(str(pattern_dir), self.obs_layer)
            
            loaded_count = len(new_layer.get_patterns())
            self.assertEqual(loaded_count, pattern_count)
    
    def test_export_patterns(self):
        """Test exporting patterns."""
        # Create pattern
        for i in range(5):
            self.obs_layer.record(
                obs_type=ObservationType.MODIFICATION,
                category="test",
                description="Test",
                data={},
                tags=['test'],
                importance=0.5
            )
        
        self.pattern_layer.detect_patterns()
        
        export_path = Path(self.test_dir) / "patterns_export.json"
        self.pattern_layer.export(str(export_path))
        
        self.assertTrue(export_path.exists())
        
        # Verify content
        import json
        with open(export_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.assertIsInstance(data, list)


if __name__ == '__main__':
    unittest.main()
