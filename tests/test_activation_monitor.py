"""
Tests for ActivationMonitor

Basic validation of the activation monitoring API
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from model_manager import ModelManager
from introspection import ActivationMonitor


def test_activation_monitor_basic():
    """Test basic ActivationMonitor functionality"""
    print("Loading model...")
    manager = ModelManager(model_name="Qwen/Qwen2.5-3B-Instruct")
    assert manager.load_model(), "Model should load successfully"
    
    print("Creating ActivationMonitor...")
    monitor = ActivationMonitor(
        manager.model, 
        manager.tokenizer,
        model_name="Qwen2.5-3B-Instruct"
    )
    
    # Test 1: Layer discovery
    print("\n[TEST 1] Layer discovery")
    layers = monitor.get_layer_names()
    assert len(layers) > 0, "Should find layers"
    print(f"✓ Found {len(layers)} modules")
    
    # Test 2: Query layers
    print("\n[TEST 2] Query layers")
    attn_layers = monitor.query_layers("self_attn")
    assert len(attn_layers) > 0, "Should find attention layers"
    print(f"✓ Found {len(attn_layers)} attention modules")
    
    # Test 3: Capture activations
    print("\n[TEST 3] Capture activations")
    test_layers = [attn_layers[0], attn_layers[1]]  # Monitor 2 layers
    result = monitor.capture_activations(
        "Hello world!", 
        layer_names=test_layers,
        max_length=10
    )
    assert "activations" in result, "Should have activations"
    assert len(result["activations"]) > 0, "Should capture some activations"
    print(f"✓ Captured activations for {len(result['activations'])} layers")
    print(f"  Input tokens: {result['num_tokens']}")
    
    # Test 4: Activation statistics
    print("\n[TEST 4] Activation statistics")
    layer_name = list(result["activations"].keys())[0]
    stats = monitor.get_activation_statistics(layer_name)
    assert "mean" in stats, "Should have mean"
    assert "std" in stats, "Should have std"
    assert "l2_norm" in stats, "Should have L2 norm"
    print(f"✓ Computed statistics for {layer_name}")
    print(f"  Mean: {stats['mean']:.6f}, Std: {stats['std']:.6f}")
    
    # Test 5: Compare activations
    print("\n[TEST 5] Compare activations")
    comparison = monitor.compare_activations(
        "I am happy",
        "I am sad",
        [test_layers[0]]
    )
    assert "comparisons" in comparison, "Should have comparisons"
    print(f"✓ Compared activations for different inputs")
    if test_layers[0] in comparison["comparisons"]:
        comp_metrics = comparison["comparisons"][test_layers[0]]
        if "cosine_similarity" in comp_metrics:
            print(f"  Cosine similarity: {comp_metrics['cosine_similarity']:.4f}")
    
    # Test 6: Attention patterns
    print("\n[TEST 6] Attention patterns")
    if monitor.attention_weights:
        attn_layer = list(monitor.attention_weights.keys())[0]
        patterns = monitor.get_attention_patterns(attn_layer)
        assert "shape" in patterns, "Should have attention shape"
        print(f"✓ Retrieved attention patterns")
        print(f"  Num heads: {patterns['num_heads']}")
    else:
        print("⊘ No attention weights captured (expected for some layer types)")
    
    # Cleanup
    monitor.clear_hooks()
    monitor.clear_activations()
    
    print("\n" + "="*70)
    print("ALL TESTS PASSED ✓")
    print("="*70)
    print("\nActivationMonitor is fully operational and ready for use!")
    

if __name__ == "__main__":
    test_activation_monitor_basic()
