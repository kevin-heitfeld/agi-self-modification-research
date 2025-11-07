"""
Example: Using WeightInspector to examine the Qwen2.5 model

This demonstrates how the system can introspect its own weights.
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from model_manager import ModelManager
from introspection import WeightInspector
import json

def main():
    print("=" * 80)
    print("WEIGHT INSPECTOR DEMONSTRATION")
    print("Phase 0 - Week 3 - Introspection API")
    print("=" * 80)
    print()
    
    # Load the model
    print("Loading Qwen2.5-3B-Instruct model...")
    manager = ModelManager(model_name="Qwen/Qwen2.5-3B-Instruct")
    success = manager.load_model()
    
    if not success:
        print("✗ Failed to load model")
        return
    
    print("✓ Model loaded")
    print()
    
    # Create WeightInspector
    print("Initializing WeightInspector...")
    inspector = WeightInspector(manager.model, model_name="Qwen2.5-3B-Instruct")
    print(f"✓ {inspector}")
    print()
    
    # Demonstrate capabilities
    
    # 1. Get all layer names
    print("=" * 80)
    print("1. LAYER DISCOVERY")
    print("=" * 80)
    all_layers = inspector.get_layer_names()
    print(f"Total layers: {len(all_layers)}")
    print(f"First 10 layers:")
    for i, name in enumerate(all_layers[:10], 1):
        print(f"  {i}. {name}")
    print(f"  ... and {len(all_layers) - 10} more")
    print()
    
    # 2. Query for specific layers
    print("=" * 80)
    print("2. NATURAL LANGUAGE QUERIES")
    print("=" * 80)
    
    queries = ["attention", "mlp", "embed", "layer.0"]
    for query in queries:
        matches = inspector.query_weights(query)
        print(f"Query: '{query}' → {len(matches)} matches")
        if matches:
            print(f"  Example: {matches[0]}")
    print()
    
    # 3. Examine specific layer weights
    print("=" * 80)
    print("3. EXAMINE SPECIFIC LAYER")
    print("=" * 80)
    
    # Pick an interesting layer (first attention layer)
    target_layer = inspector.get_layer_names(filter_pattern="layers.0.self_attn.q_proj.weight")
    if target_layer:
        layer_name = target_layer[0]
        print(f"Examining: {layer_name}")
        print()
        
        weights = inspector.get_layer_weights(layer_name)
        print(f"  Shape: {weights['shape']}")
        print(f"  Parameters: {weights['num_parameters']:,}")
        print(f"  Data type: {weights['dtype']}")
        print(f"  Device: {weights['device']}")
        print()
        
        # Get detailed statistics
        stats = inspector.get_weight_statistics(layer_name)
        print(f"  Statistics:")
        print(f"    Mean: {stats['mean']:.6f}")
        print(f"    Std: {stats['std']:.6f}")
        print(f"    Min: {stats['min']:.6f}")
        print(f"    Max: {stats['max']:.6f}")
        print(f"    Median: {stats['median']:.6f}")
        print(f"    L2 Norm: {stats['l2_norm']:.2f}")
        print(f"    Zeros: {stats['zeros_percentage']:.2f}%")
        print(f"    Near-zero: {stats['near_zero_percentage']:.2f}%")
        print()
    
    # 4. Compare layers
    print("=" * 80)
    print("4. COMPARE LAYERS")
    print("=" * 80)
    
    # Compare Q and K projection matrices in first attention layer
    q_layers = inspector.query_weights("layers.0.self_attn.q_proj.weight")
    k_layers = inspector.query_weights("layers.0.self_attn.k_proj.weight")
    
    if q_layers and k_layers:
        print(f"Comparing Query vs Key projections in layer 0:")
        comparison = inspector.compare_weights(q_layers[0], k_layers[0])
        print(f"  Layer 1: {comparison['layer1']}")
        print(f"  Layer 2: {comparison['layer2']}")
        print(f"  Mean difference: {comparison['mean_difference']:.6f}")
        print(f"  Std difference: {comparison['std_difference']:.6f}")
        if comparison['shapes_match']:
            print(f"  Correlation: {comparison['correlation']:.4f}")
            print(f"  Cosine similarity: {comparison['cosine_similarity']:.4f}")
        else:
            print(f"  Note: {comparison['note']}")
    print()
    
    # 5. Find similar weight patterns
    print("=" * 80)
    print("5. FIND SIMILAR WEIGHT PATTERNS")
    print("=" * 80)
    
    if q_layers:
        reference = q_layers[0]
        print(f"Finding layers similar to: {reference}")
        similar = inspector.find_similar_weights(reference, top_k=5, metric="correlation")
        
        print(f"\nTop 5 most similar layers:")
        for i, (layer_name, score) in enumerate(similar, 1):
            short_name = layer_name.split('.')[-3:]  # Last 3 parts
            print(f"  {i}. {'.'.join(short_name)}: {score:.4f}")
    print()
    
    # 6. Model-wide summary
    print("=" * 80)
    print("6. MODEL-WIDE SUMMARY")
    print("=" * 80)
    
    summary = inspector.get_weight_summary()
    print(f"Model: {summary['model_name']}")
    print(f"Total parameters: {summary['total_parameters']:,}")
    print(f"Total layers: {summary['total_layers']}")
    print()
    print(f"Overall statistics across all weights:")
    overall = summary['overall_statistics']
    print(f"  Mean: {overall['mean']:.6f}")
    print(f"  Std: {overall['std']:.6f}")
    print(f"  Min: {overall['min']:.6f}")
    print(f"  Max: {overall['max']:.6f}")
    print(f"  L2 Norm²: {overall['l2_norm_squared']:.2e}")
    print(f"  Sparsity (zeros): {overall['zeros_percentage']:.4f}%")
    print()
    print(f"Layer groups:")
    for group, count in sorted(summary['layer_groups'].items()):
        print(f"  {group}: {count} layers")
    print()
    
    # Save detailed summary to JSON
    output_file = Path("data/introspection/weight_summary.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✓ Detailed summary saved to: {output_file}")
    print()
    
    # Conclusion
    print("=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("=" * 80)
    print()
    print("The WeightInspector allows the system to:")
    print("  ✓ Discover all layers in its architecture")
    print("  ✓ Query for specific components using natural language")
    print("  ✓ Examine weight values and distributions")
    print("  ✓ Compute statistical properties")
    print("  ✓ Compare weights across layers")
    print("  ✓ Find similar weight patterns")
    print("  ✓ Generate model-wide summaries")
    print()
    print("This is the first step toward self-examination and autonomous modification.")
    print()
    print("Next steps:")
    print("  • Build ActivationMonitor (observe activations during inference)")
    print("  • Build ArchitectureNavigator (understand computational graph)")
    print("  • Integrate with natural language reasoning")
    print()


if __name__ == "__main__":
    main()
