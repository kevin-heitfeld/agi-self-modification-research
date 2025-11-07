"""
Performance Benchmarks for Phase 0 Components

Measures performance of key introspection and memory components to ensure
they meet performance requirements for real-time use during self-modification.
"""

import time
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


class SimpleTestModel(nn.Module):
    """Simple model for benchmarking."""
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(512, 512) for _ in range(12)
        ])
        self.norm = nn.LayerNorm(512)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class TestPerformanceBenchmarks(unittest.TestCase):
    """Benchmark tests for Phase 0 components."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures once for all tests."""
        cls.model = SimpleTestModel()
        cls.temp_dir = tempfile.mkdtemp()
    
    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests."""
        shutil.rmtree(cls.temp_dir, ignore_errors=True)
    
    def test_weight_inspector_initialization_performance(self):
        """Test WeightInspector initialization time."""
        start = time.time()
        inspector = WeightInspector(self.model, "test_model")
        elapsed = time.time() - start
        
        print(f"\n[BENCHMARK] WeightInspector initialization: {elapsed*1000:.2f}ms")
        
        # Should initialize in under 100ms
        self.assertLess(elapsed, 0.1, 
                       f"WeightInspector init took {elapsed*1000:.2f}ms (target: <100ms)")
    
    def test_weight_statistics_performance(self):
        """Test weight statistics computation time."""
        inspector = WeightInspector(self.model, "test_model")
        layer_names = inspector.get_layer_names()
        
        # Benchmark first call (no cache)
        start = time.time()
        stats = inspector.get_weight_statistics(layer_names[0], use_cache=False)
        elapsed_no_cache = time.time() - start
        
        print(f"\n[BENCHMARK] Weight statistics (no cache): {elapsed_no_cache*1000:.2f}ms")
        
        # Benchmark cached call
        start = time.time()
        stats = inspector.get_weight_statistics(layer_names[0], use_cache=True)
        elapsed_cached = time.time() - start
        
        print(f"[BENCHMARK] Weight statistics (cached): {elapsed_cached*1000:.2f}ms")
        
        # No cache should be under 50ms, cached under 1ms
        self.assertLess(elapsed_no_cache, 0.05,
                       f"Statistics computation took {elapsed_no_cache*1000:.2f}ms (target: <50ms)")
        self.assertLess(elapsed_cached, 0.001,
                       f"Cached statistics took {elapsed_cached*1000:.2f}ms (target: <1ms)")
    
    def test_weight_sharing_detection_performance(self):
        """Test weight sharing detection performance."""
        # Create model with sharing
        model_with_sharing = nn.Module()
        model_with_sharing.embed = nn.Embedding(1000, 512)
        model_with_sharing.output = nn.Linear(512, 1000)
        model_with_sharing.output.weight = model_with_sharing.embed.weight
        
        start = time.time()
        inspector = WeightInspector(model_with_sharing, "shared_model")
        elapsed = time.time() - start
        
        print(f"\n[BENCHMARK] Weight sharing detection: {elapsed*1000:.2f}ms")
        
        # Should detect in under 150ms
        self.assertLess(elapsed, 0.15,
                       f"Sharing detection took {elapsed*1000:.2f}ms (target: <150ms)")
        
        # Verify detection worked
        shared = inspector.get_shared_weights()
        self.assertGreater(len(shared), 0, "Should detect weight sharing")
    
    def test_architecture_navigator_initialization_performance(self):
        """Test ArchitectureNavigator initialization time."""
        start = time.time()
        navigator = ArchitectureNavigator(self.model)
        elapsed = time.time() - start
        
        print(f"\n[BENCHMARK] ArchitectureNavigator initialization: {elapsed*1000:.2f}ms")
        
        # Should initialize in under 50ms
        self.assertLess(elapsed, 0.05,
                       f"Navigator init took {elapsed*1000:.2f}ms (target: <50ms)")
    
    def test_architecture_summary_performance(self):
        """Test architecture summary generation time."""
        navigator = ArchitectureNavigator(self.model)
        
        # First call (no cache)
        start = time.time()
        summary = navigator.get_architecture_summary()
        elapsed_no_cache = time.time() - start
        
        print(f"\n[BENCHMARK] Architecture summary (no cache): {elapsed_no_cache*1000:.2f}ms")
        
        # Second call (with cache)
        start = time.time()
        summary = navigator.get_architecture_summary()
        elapsed_cached = time.time() - start
        
        print(f"[BENCHMARK] Architecture summary (cached): {elapsed_cached*1000:.2f}ms")
        
        # No cache should be under 100ms
        self.assertLess(elapsed_no_cache, 0.1,
                       f"Summary generation took {elapsed_no_cache*1000:.2f}ms (target: <100ms)")
    
    def test_memory_system_initialization_performance(self):
        """Test MemorySystem initialization time."""
        memory_dir = Path(self.temp_dir) / "memory_bench_init"
        
        start = time.time()
        memory = MemorySystem(str(memory_dir))
        elapsed = time.time() - start
        
        print(f"\n[BENCHMARK] MemorySystem initialization: {elapsed*1000:.2f}ms")
        
        # Should initialize in under 200ms
        self.assertLess(elapsed, 0.2,
                       f"Memory init took {elapsed*1000:.2f}ms (target: <200ms)")
    
    def test_observation_recording_performance(self):
        """Test observation recording speed."""
        memory_dir = Path(self.temp_dir) / "memory_bench_obs_recording"
        memory = MemorySystem(str(memory_dir))
        
        # Record 100 observations and measure total time
        num_observations = 100
        start = time.time()
        
        for i in range(num_observations):
            memory.record_observation(
                obs_type=ObservationType.MODIFICATION,
                category="test",
                description=f"Test observation {i}",
                data={'index': i},
                tags=['benchmark'],
                importance=0.5
            )
        
        elapsed = time.time() - start
        avg_per_obs = (elapsed / num_observations) * 1000
        
        print(f"\n[BENCHMARK] Recording {num_observations} observations: {elapsed*1000:.2f}ms")
        print(f"[BENCHMARK] Average per observation: {avg_per_obs:.2f}ms")
        
        # Average should be under 10ms per observation
        self.assertLess(avg_per_obs, 10,
                       f"Avg observation recording took {avg_per_obs:.2f}ms (target: <10ms)")
    
    def test_coupled_modification_recording_performance(self):
        """Test coupled modification recording with inspector."""
        memory_dir = Path(self.temp_dir) / "memory_bench_coupled_mods"
        memory = MemorySystem(str(memory_dir))
        
        # Create model with sharing
        model_with_sharing = nn.Module()
        model_with_sharing.embed = nn.Embedding(100, 32)
        model_with_sharing.output = nn.Linear(32, 100)
        model_with_sharing.output.weight = model_with_sharing.embed.weight
        
        inspector = WeightInspector(model_with_sharing, "shared")
        memory.set_weight_inspector(inspector)
        
        # Record 50 coupled modifications
        num_mods = 50
        start = time.time()
        
        for i in range(num_mods):
            memory.record_modification(
                layer_name="embed.weight",
                modification_data={'change': 0.01 * i}
            )
        
        elapsed = time.time() - start
        avg_per_mod = (elapsed / num_mods) * 1000
        
        print(f"\n[BENCHMARK] Recording {num_mods} coupled modifications: {elapsed*1000:.2f}ms")
        print(f"[BENCHMARK] Average per modification: {avg_per_mod:.2f}ms")
        
        # Average should be under 15ms per modification (includes coupling check)
        self.assertLess(avg_per_mod, 15,
                       f"Avg coupled mod recording took {avg_per_mod:.2f}ms (target: <15ms)")
    
    def test_memory_query_performance(self):
        """Test memory query speed."""
        memory_dir = Path(self.temp_dir) / "memory_bench_queries"
        memory = MemorySystem(str(memory_dir))
        
        # Add 200 observations first
        for i in range(200):
            memory.record_observation(
                obs_type=ObservationType.MODIFICATION if i % 2 == 0 else ObservationType.PERFORMANCE,
                category=f"category_{i % 10}",
                description=f"Test {i}",
                data={'value': i},
                tags=[f"tag_{i % 5}", 'benchmark'],
                importance=0.5
            )
        
        # Benchmark different query types
        
        # Query by type
        start = time.time()
        results = memory.observations.query(obs_type=ObservationType.MODIFICATION)
        elapsed_type = time.time() - start
        print(f"\n[BENCHMARK] Query by type (200 obs): {elapsed_type*1000:.2f}ms ({len(results)} results)")
        
        # Query by tags
        start = time.time()
        results = memory.observations.query(tags=['tag_1'])
        elapsed_tags = time.time() - start
        print(f"[BENCHMARK] Query by tags (200 obs): {elapsed_tags*1000:.2f}ms ({len(results)} results)")
        
        # Query by category
        start = time.time()
        results = memory.observations.query(category="category_5")
        elapsed_category = time.time() - start
        print(f"[BENCHMARK] Query by category (200 obs): {elapsed_category*1000:.2f}ms ({len(results)} results)")
        
        # All queries should be under 50ms
        self.assertLess(elapsed_type, 0.05, "Query by type too slow")
        self.assertLess(elapsed_tags, 0.05, "Query by tags too slow")
        self.assertLess(elapsed_category, 0.05, "Query by category too slow")
    
    def test_end_to_end_workflow_performance(self):
        """Test complete workflow from inspection to memory recording."""
        memory_dir = Path(self.temp_dir) / "memory_bench_e2e_workflow"
        
        start_total = time.time()
        
        # 1. Create model
        model = SimpleTestModel()
        
        # 2. Initialize introspection
        inspector = WeightInspector(model, "benchmark_model")
        navigator = ArchitectureNavigator(model)
        navigator.set_weight_inspector(inspector)
        
        # 3. Initialize memory
        memory = MemorySystem(str(memory_dir))
        memory.set_weight_inspector(inspector)
        
        # 4. Inspect a layer
        layers = inspector.get_layer_names()
        stats = inspector.get_weight_statistics(layers[0])
        
        # 5. Query architecture
        summary = navigator.get_architecture_summary()
        
        # 6. Record modification
        memory.record_modification(
            layer_name=layers[0],
            modification_data={'method': 'test', 'change': 0.01}
        )
        
        # 7. Query memory
        mods = memory.observations.query(obs_type=ObservationType.MODIFICATION)
        
        elapsed_total = time.time() - start_total
        
        print(f"\n[BENCHMARK] End-to-end workflow: {elapsed_total*1000:.2f}ms")
        
        # Complete workflow should be under 500ms
        self.assertLess(elapsed_total, 0.5,
                       f"E2E workflow took {elapsed_total*1000:.2f}ms (target: <500ms)")
        
        # Verify all steps worked
        self.assertIsNotNone(stats)
        self.assertIsNotNone(summary)
        self.assertGreater(len(mods), 0)


if __name__ == '__main__':
    # Run benchmarks
    print("=" * 80)
    print("PHASE 0 PERFORMANCE BENCHMARKS")
    print("=" * 80)
    
    unittest.main(verbosity=2)
