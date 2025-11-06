"""
Tests for Observation Layer (Layer 1)

Tests:
- Observation recording
- Querying with various filters
- Statistics calculation
- Cache behavior
- Export functionality
- Tag management

Author: AGI Self-Modification Research Team
Date: November 7, 2025
"""

import unittest
import tempfile
import shutil
import time
from pathlib import Path

from src.memory.observation_layer import (
    ObservationLayer,
    Observation,
    ObservationType
)


class TestObservationLayer(unittest.TestCase):
    """Test suite for ObservationLayer."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.layer = ObservationLayer(self.test_dir)
    
    def tearDown(self):
        """Clean up test environment."""
        # Explicitly close database connection before cleanup (Windows file locking)
        if hasattr(self, 'layer') and hasattr(self.layer, 'conn'):
            self.layer.conn.close()
        
        shutil.rmtree(self.test_dir)
    
    def test_initialization(self):
        """Test layer initialization."""
        self.assertIsNotNone(self.layer.conn)
        self.assertEqual(len(self.layer.cache), 0)
        self.assertTrue(Path(self.test_dir).exists())
    
    def test_record_observation(self):
        """Test recording an observation."""
        obs_id = self.layer.record(
            obs_type=ObservationType.MODIFICATION,
            category="layer5",
            description="Test modification",
            data={'change': 0.1},
            tags=['test', 'modification'],
            importance=0.8
        )
        
        self.assertIsNotNone(obs_id)
        self.assertTrue(obs_id.startswith('obs_'))
        
        # Verify observation was recorded
        obs = self.layer.get(obs_id)
        self.assertIsNotNone(obs)
        self.assertEqual(obs.type, ObservationType.MODIFICATION)
        self.assertEqual(obs.category, "layer5")
        self.assertEqual(obs.description, "Test modification")
        self.assertEqual(obs.data['change'], 0.1)
        self.assertIn('test', obs.tags)
        self.assertEqual(obs.importance, 0.8)
    
    def test_record_multiple_observations(self):
        """Test recording multiple observations."""
        obs_ids = []
        for i in range(5):
            obs_id = self.layer.record(
                obs_type=ObservationType.PERFORMANCE,
                category="perplexity",
                description=f"Performance test {i}",
                data={'value': 10.0 + i},
                tags=['test', 'performance'],
                importance=0.5
            )
            obs_ids.append(obs_id)
        
        self.assertEqual(len(obs_ids), 5)
        
        # Verify all were recorded
        for obs_id in obs_ids:
            obs = self.layer.get(obs_id)
            self.assertIsNotNone(obs)
    
    def test_get_nonexistent_observation(self):
        """Test getting a nonexistent observation."""
        obs = self.layer.get("nonexistent_id")
        self.assertIsNone(obs)
    
    def test_query_by_type(self):
        """Test querying by observation type."""
        # Record different types
        self.layer.record(
            obs_type=ObservationType.MODIFICATION,
            category="test",
            description="Mod 1",
            data={},
            tags=[],
            importance=0.5
        )
        self.layer.record(
            obs_type=ObservationType.PERFORMANCE,
            category="test",
            description="Perf 1",
            data={},
            tags=[],
            importance=0.5
        )
        self.layer.record(
            obs_type=ObservationType.MODIFICATION,
            category="test",
            description="Mod 2",
            data={},
            tags=[],
            importance=0.5
        )
        
        # Query by type
        mods = self.layer.query(obs_type=ObservationType.MODIFICATION)
        perfs = self.layer.query(obs_type=ObservationType.PERFORMANCE)
        
        self.assertEqual(len(mods), 2)
        self.assertEqual(len(perfs), 1)
        
        for obs in mods:
            self.assertEqual(obs.type, ObservationType.MODIFICATION)
    
    def test_query_by_category(self):
        """Test querying by category."""
        self.layer.record(
            obs_type=ObservationType.MODIFICATION,
            category="layer5",
            description="Test 1",
            data={},
            tags=[],
            importance=0.5
        )
        self.layer.record(
            obs_type=ObservationType.MODIFICATION,
            category="layer10",
            description="Test 2",
            data={},
            tags=[],
            importance=0.5
        )
        
        layer5_obs = self.layer.query(category="layer5")
        self.assertEqual(len(layer5_obs), 1)
        self.assertEqual(layer5_obs[0].category, "layer5")
    
    def test_query_by_tags(self):
        """Test querying by tags."""
        self.layer.record(
            obs_type=ObservationType.MODIFICATION,
            category="test",
            description="Test 1",
            data={},
            tags=['alpha', 'beta'],
            importance=0.5
        )
        self.layer.record(
            obs_type=ObservationType.MODIFICATION,
            category="test",
            description="Test 2",
            data={},
            tags=['alpha', 'gamma'],
            importance=0.5
        )
        self.layer.record(
            obs_type=ObservationType.MODIFICATION,
            category="test",
            description="Test 3",
            data={},
            tags=['delta'],
            importance=0.5
        )
        
        # Query with tags
        alpha_obs = self.layer.query(tags=['alpha'])
        self.assertEqual(len(alpha_obs), 2)
        
        beta_obs = self.layer.query(tags=['beta'])
        self.assertEqual(len(beta_obs), 1)
    
    def test_query_by_importance(self):
        """Test querying by importance."""
        self.layer.record(
            obs_type=ObservationType.MODIFICATION,
            category="test",
            description="Low importance",
            data={},
            tags=[],
            importance=0.3
        )
        self.layer.record(
            obs_type=ObservationType.MODIFICATION,
            category="test",
            description="High importance",
            data={},
            tags=[],
            importance=0.9
        )
        
        high_importance = self.layer.query(min_importance=0.7)
        self.assertEqual(len(high_importance), 1)
        self.assertGreaterEqual(high_importance[0].importance, 0.7)
    
    def test_query_by_time_range(self):
        """Test querying by time range."""
        start_time = time.time()
        
        self.layer.record(
            obs_type=ObservationType.MODIFICATION,
            category="test",
            description="Test 1",
            data={},
            tags=[],
            importance=0.5
        )
        
        time.sleep(0.1)
        mid_time = time.time()
        time.sleep(0.1)
        
        self.layer.record(
            obs_type=ObservationType.MODIFICATION,
            category="test",
            description="Test 2",
            data={},
            tags=[],
            importance=0.5
        )
        
        end_time = time.time()
        
        # Query before mid_time
        early = self.layer.query(end_time=mid_time)
        self.assertEqual(len(early), 1)
        
        # Query after mid_time
        late = self.layer.query(start_time=mid_time)
        self.assertEqual(len(late), 1)
        
        # Query full range
        all_obs = self.layer.query(start_time=start_time, end_time=end_time)
        self.assertEqual(len(all_obs), 2)
    
    def test_query_with_limit(self):
        """Test querying with limit."""
        for i in range(10):
            self.layer.record(
                obs_type=ObservationType.MODIFICATION,
                category="test",
                description=f"Test {i}",
                data={},
                tags=[],
                importance=0.5
            )
        
        limited = self.layer.query(limit=5)
        self.assertEqual(len(limited), 5)
    
    def test_get_recent(self):
        """Test getting recent observations."""
        # Record some observations
        for i in range(3):
            self.layer.record(
                obs_type=ObservationType.MODIFICATION,
                category="test",
                description=f"Test {i}",
                data={},
                tags=[],
                importance=0.5
            )
        
        recent = self.layer.get_recent(limit=2)
        self.assertEqual(len(recent), 2)
        
        # Should be ordered newest first
        self.assertGreater(recent[0].timestamp, recent[1].timestamp)
    
    def test_cache_behavior(self):
        """Test cache stores recent observations."""
        obs_id = self.layer.record(
            obs_type=ObservationType.MODIFICATION,
            category="test",
            description="Test cache",
            data={},
            tags=[],
            importance=0.5
        )
        
        # Should be in cache
        self.assertIn(obs_id, self.layer.cache)
        
        # Getting should hit cache
        obs = self.layer.get(obs_id)
        self.assertIsNotNone(obs)
        self.assertEqual(obs.id, obs_id)
    
    def test_statistics(self):
        """Test statistics calculation."""
        # Record different types
        self.layer.record(
            obs_type=ObservationType.MODIFICATION,
            category="test",
            description="Mod",
            data={},
            tags=[],
            importance=0.5
        )
        self.layer.record(
            obs_type=ObservationType.PERFORMANCE,
            category="test",
            description="Perf",
            data={},
            tags=[],
            importance=0.8
        )
        self.layer.record(
            obs_type=ObservationType.SAFETY_EVENT,
            category="test",
            description="Safety",
            data={},
            tags=[],
            importance=1.0
        )
        
        stats = self.layer.get_statistics()
        
        self.assertEqual(stats['total'], 3)
        self.assertIn('modification', stats['by_type'])
        self.assertIn('performance', stats['by_type'])
        self.assertIn('safety_event', stats['by_type'])
        self.assertGreater(stats['average_importance'], 0)
    
    def test_export_json(self):
        """Test exporting to JSON."""
        self.layer.record(
            obs_type=ObservationType.MODIFICATION,
            category="test",
            description="Export test",
            data={'key': 'value'},
            tags=['export'],
            importance=0.7
        )
        
        export_path = Path(self.test_dir) / "export.json"
        self.layer.export(str(export_path), export_format='json')
        
        self.assertTrue(export_path.exists())
        
        # Verify content
        import json
        with open(export_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]['description'], "Export test")
    
    def test_export_csv(self):
        """Test exporting to CSV."""
        self.layer.record(
            obs_type=ObservationType.MODIFICATION,
            category="test",
            description="CSV test",
            data={},
            tags=[],
            importance=0.5
        )
        
        export_path = Path(self.test_dir) / "export.csv"
        self.layer.export(str(export_path), export_format='csv')
        
        self.assertTrue(export_path.exists())
        
        # Verify content
        import csv
        with open(export_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            rows = list(reader)
        
        self.assertEqual(len(rows), 2)  # Header + 1 data row
        self.assertIn('id', rows[0])
        self.assertIn('description', rows[0])


if __name__ == '__main__':
    unittest.main()
