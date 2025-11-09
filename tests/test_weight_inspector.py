"""
Tests for WeightInspector

Basic validation of the introspection API
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from model_manager import ModelManager
from introspection import WeightInspector


def test_weight_inspector_basic():
    """Test basic WeightInspector functionality"""
    print("Loading model...")
    manager = ModelManager(model_name="Qwen/Qwen2.5-3B-Instruct")
    assert manager.load_model(), "Model should load successfully"
    
    print("Creating WeightInspector...")
    inspector = WeightInspector(manager.model, model_name="Qwen2.5-3B-Instruct")
    
    # Test 1: Layer discovery
    print("\n[TEST 1] Layer discovery")
    layers = inspector.get_layer_names()
    assert len(layers) > 0, "Should find layers"
    print(f"✓ Found {len(layers)} layers")
    
    # Test 2: Query layers
    print("\n[TEST 2] Query layers")
    attention_layers = inspector.query_weights("attention")
    assert len(attention_layers) > 0, "Should find attention layers"
    print(f"✓ Found {len(attention_layers)} attention layers")
    
    # Test 3: Get weight information
    print("\n[TEST 3] Get weight information")
    first_layer = layers[0]
    weight_info = inspector.get_layer_weights(first_layer)
    assert "shape" in weight_info, "Should have shape info"
    assert "data" in weight_info, "Should have weight data"
    print(f"✓ Retrieved info for {first_layer}")
    print(f"  Shape: {weight_info['shape']}, Params: {weight_info['num_parameters']:,}")
    
    # Test 4: Get statistics
    print("\n[TEST 4] Get statistics")
    stats = inspector.get_weight_statistics(first_layer)
    assert "mean" in stats, "Should have mean"
    assert "std" in stats, "Should have std"
    assert "l2_norm" in stats, "Should have L2 norm"
    print(f"✓ Computed statistics for {first_layer}")
    print(f"  Mean: {stats['mean']:.6f}, Std: {stats['std']:.6f}")
    
    # Test 5: Model summary
    print("\n[TEST 5] Model summary")
    summary = inspector.get_weight_summary()
    assert "total_parameters" in summary, "Should have total parameters"
    assert "overall_statistics" in summary, "Should have overall statistics"
    print(f"✓ Generated model summary")
    print(f"  Total parameters: {summary['total_parameters']:,}")
    print(f"  Total layers: {summary['total_layers']}")
    
    # Test 6: Compare layers
    print("\n[TEST 6] Compare layers")
    if len(attention_layers) >= 2:
        comparison = inspector.compare_weights(attention_layers[0], attention_layers[1])
        assert "mean_difference" in comparison, "Should have comparison metrics"
        print(f"✓ Compared two layers")
        print(f"  Mean difference: {comparison['mean_difference']:.6f}")
    else:
        print("⊘ Skipped (need at least 2 attention layers)")
    
    print("\n" + "="*70)
    print("ALL TESTS PASSED ✓")
    print("="*70)
    print("\nWeightInspector is fully operational and ready for use!")


def test_multi_layer_weight_statistics():
    """Test multi-layer support for get_weight_statistics"""
    print("\n" + "="*70)
    print("Testing Multi-Layer Weight Statistics")
    print("="*70)
    
    print("Loading model...")
    manager = ModelManager(model_name="Qwen/Qwen2.5-3B-Instruct")
    assert manager.load_model(), "Model should load successfully"
    
    print("Creating WeightInspector...")
    inspector = WeightInspector(manager.model, model_name="Qwen2.5-3B-Instruct")
    
    # Get some test layers
    layers = inspector.get_layer_names()
    test_layers = [layers[0], layers[1], layers[2]]
    
    # Test 1: Single layer (baseline)
    print("\n[TEST 1] Single layer statistics")
    single_result = inspector.get_weight_statistics(test_layers[0])
    assert isinstance(single_result, dict), "Single layer should return dict"
    assert "mean" in single_result, "Should have mean"
    assert "std" in single_result, "Should have std"
    print(f"✓ Single layer: {test_layers[0]}")
    print(f"  Mean: {single_result['mean']:.6f}, Std: {single_result['std']:.6f}")
    
    # Test 2: Multiple layers
    print("\n[TEST 2] Multiple layers statistics")
    multi_result = inspector.get_weight_statistics(test_layers)
    assert isinstance(multi_result, list), "Multiple layers should return list"
    assert len(multi_result) == 3, "Should return 3 results"
    for i, result in enumerate(multi_result):
        assert isinstance(result, dict), f"Result {i} should be dict"
        assert "mean" in result, f"Result {i} should have mean"
        assert "name" in result, f"Result {i} should have name"
    print(f"✓ Examined {len(multi_result)} layers in one call:")
    for result in multi_result:
        print(f"  - {result['name']}: mean={result['mean']:.6f}, std={result['std']:.6f}")
    
    # Test 3: Mixed valid/invalid layers
    print("\n[TEST 3] Mixed valid/invalid layers")
    mixed_layers = [test_layers[0], "invalid.layer.name", test_layers[1]]
    mixed_result = inspector.get_weight_statistics(mixed_layers)
    assert isinstance(mixed_result, list), "Should return list"
    assert len(mixed_result) == 3, "Should return 3 results"
    assert "error" not in mixed_result[0], "First should be valid"
    assert "error" in mixed_result[1], "Second should have error"
    assert "error" not in mixed_result[2], "Third should be valid"
    print(f"✓ Handled mixed valid/invalid layers:")
    print(f"  - Valid: {mixed_result[0]['name']}")
    print(f"  - Invalid: {mixed_result[1].get('layer_name', 'unknown')} (error: {mixed_result[1]['error'][:50]}...)")
    print(f"  - Valid: {mixed_result[2]['name']}")
    
    # Test 4: Empty list
    print("\n[TEST 4] Empty list")
    empty_result = inspector.get_weight_statistics([])
    assert isinstance(empty_result, list), "Should return list"
    assert len(empty_result) == 0, "Should be empty"
    print("✓ Handled empty list correctly")
    
    print("\n" + "="*70)
    print("MULTI-LAYER TESTS PASSED ✓")
    print("="*70)
    print("\nMulti-layer weight statistics work correctly!")
    print("Models can now examine multiple layers in one call!")
    

if __name__ == "__main__":
    test_weight_inspector_basic()
    test_multi_layer_weight_statistics()
