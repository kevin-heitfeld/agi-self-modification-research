"""
Example: Using ActivationMonitor to observe model activations

This demonstrates how the system can introspect its own activations during inference.
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from model_manager import ModelManager
from introspection import ActivationMonitor

def main():
    print("=" * 80)
    print("ACTIVATION MONITOR DEMONSTRATION")
    print("Phase 0 - Week 4 - Introspection API")
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

    # Create ActivationMonitor
    print("Initializing ActivationMonitor...")
    monitor = ActivationMonitor(
        manager.model,
        manager.tokenizer,
        model_name="Qwen2.5-3B-Instruct"
    )
    print(f"✓ {monitor}")
    print()

    # Demonstrate capabilities

    # 1. Discover layers
    print("=" * 80)
    print("1. LAYER DISCOVERY")
    print("=" * 80)
    all_layers = monitor.get_layer_names()
    print(f"Total modules: {len(all_layers)}")
    print()

    # Find attention layers
    attn_layers = monitor.query_layers("self_attn")
    print(f"Found {len(attn_layers)} attention modules")
    print("First 5 attention layers:")
    for i, name in enumerate(attn_layers[:5], 1):
        print(f"  {i}. {name}")
    print()

    # 2. Capture activations for a simple input
    print("=" * 80)
    print("2. CAPTURE ACTIVATIONS")
    print("=" * 80)

    test_input = "The cat sat on the mat."
    print(f"Input: '{test_input}'")
    print()

    # Monitor just a few key layers to save memory
    layers_to_monitor = [
        "model.layers.0.self_attn",
        "model.layers.5.self_attn",
        "model.layers.10.self_attn",
    ]

    print(f"Monitoring {len(layers_to_monitor)} layers...")
    result = monitor.capture_activations(test_input, layers_to_monitor, max_length=10)

    print(f"✓ Captured activations")
    print(f"  Input tokens: {result['num_tokens']}")
    print(f"  Tokens: {result['token_strings']}")
    print(f"  Monitored layers: {len(result['activations'])}")
    print()

    # 3. Analyze activation statistics
    print("=" * 80)
    print("3. ACTIVATION STATISTICS")
    print("=" * 80)

    for layer_name in result['monitored_layers']:
        print(f"\nLayer: {layer_name}")
        stats = monitor.get_activation_statistics(layer_name)
        print(f"  Shape: {stats['shape']}")
        print(f"  Mean: {stats['mean']:.6f}")
        print(f"  Std: {stats['std']:.6f}")
        print(f"  L2 Norm: {stats['l2_norm']:.2f}")
        print(f"  Positive: {stats['positive_percentage']:.1f}%")
        print(f"  Negative: {stats['negative_percentage']:.1f}%")
        print(f"  Sparsity: {stats['zeros_percentage']:.2f}% zeros")
    print()

    # 4. Compare activations for different inputs
    print("=" * 80)
    print("4. COMPARE DIFFERENT INPUTS")
    print("=" * 80)

    input1 = "I love machine learning."
    input2 = "I hate machine learning."

    print(f"Input 1: '{input1}'")
    print(f"Input 2: '{input2}'")
    print()

    # Monitor first layer only for comparison
    comparison_layers = ["model.layers.0.self_attn"]
    print(f"Comparing activations in: {comparison_layers[0]}")
    print()

    comparison = monitor.compare_activations(input1, input2, comparison_layers)

    for layer_name, metrics in comparison['comparisons'].items():
        if 'error' in metrics:
            print(f"  {metrics['error']}")
        else:
            print(f"  Cosine similarity: {metrics['cosine_similarity']:.4f}")
            print(f"  Correlation: {metrics['correlation']:.4f}")
            print(f"  Euclidean distance: {metrics['euclidean_distance']:.2f}")
            print(f"  Mean difference: {metrics['mean_difference']:.6f}")
    print()

    # 5. Attention patterns (if available)
    print("=" * 80)
    print("5. ATTENTION PATTERNS")
    print("=" * 80)

    # Re-capture with a layer that might have attention weights
    monitor.capture_activations("Hello world!", ["model.layers.0.self_attn"])

    if monitor.attention_weights:
        print(f"Captured attention weights for {len(monitor.attention_weights)} layers")
        for layer_name in monitor.attention_weights.keys():
            print(f"\n  Layer: {layer_name}")
            attn_info = monitor.get_attention_patterns(layer_name)
            print(f"    Shape: {attn_info['shape']}")
            print(f"    Num heads: {attn_info['num_heads']}")
            print(f"    Mean attention: {attn_info['mean_attention']:.6f}")
            print(f"    Max attention: {attn_info['max_attention']:.6f}")
            print(f"    Entropy: {attn_info['entropy']:.4f}")
            break  # Just show first one
    else:
        print("  No attention weights captured (may require specific layer types)")
    print()

    # 6. Trace token through layers
    print("=" * 80)
    print("6. TRACE TOKEN INFLUENCE - THE CONTINUITY QUESTION")
    print("=" * 80)
    print()
    print("This demonstrates the system's ability to trace how a specific concept")
    print("(embedded in a token) evolves through its layers - essential for")
    print("answering philosophical questions about thought continuity.")
    print()

    trace_input = "I think about consciousness."
    token_to_trace = 3  # "about" - a conceptually meaningful token

    print(f"Input: '{trace_input}'")

    # First, let's see what tokens we have
    inputs = manager.tokenizer(trace_input, return_tensors="pt")
    tokens = inputs["input_ids"][0].tolist()
    token_strings = [manager.tokenizer.decode([t]) for t in tokens]
    print(f"Tokens: {token_strings}")
    print(f"Tracing token [{token_to_trace}]: '{token_strings[token_to_trace]}'")
    print()

    # Use transformer block layers for better token tracing
    trace_layers = [
        "model.layers.0",   # First transformer block
        "model.layers.5",   # Middle transformer block
        "model.layers.10",  # Later transformer block
        "model.layers.17",  # Final transformer block (Qwen has 18 layers total)
    ]

    print(f"Tracing through {len(trace_layers)} transformer blocks...")
    print()

    trace = monitor.trace_token_influence(trace_input, token_to_trace, trace_layers)

    if trace['token']:
        print(f"Token: '{trace['token']}' (index {trace['token_idx']})")
        print()

        if trace.get('evolution_summary'):
            print("EVOLUTION SUMMARY:")
            summary = trace['evolution_summary']
            print(f"  Layers traced: {summary['num_layers_traced']}")
            print(f"  Initial L2 norm: {summary['initial_norm']:.2f}")
            print(f"  Final L2 norm: {summary['final_norm']:.2f}")
            print(f"  Total change: {summary['total_norm_change']:.2f}")
            print(f"  Representation: {summary['representation_stability']}")
            print()

        print("LAYER-BY-LAYER EVOLUTION:")
        for layer_name, info in trace['layers'].items():
            short_name = layer_name.split('.')[-1] if '.' in layer_name else layer_name
            if 'error' not in info:
                print(f"\n  {layer_name}:")
                print(f"    Mean: {info['mean']:8.4f}")
                print(f"    Std:  {info['std']:8.4f}")
                print(f"    L2 Norm: {info['l2_norm']:8.2f}", end="")
                if 'norm_change' in info:
                    change = info['norm_change']
                    pct = info['norm_change_percentage']
                    print(f"  (Δ {change:+.2f}, {pct:+.1f}%)")
                else:
                    print()
                print(f"    Positive ratio: {info['positive_ratio']:.1%}")
            else:
                print(f"\n  {layer_name}: {info['error']}")
                if 'note' in info:
                    print(f"    Note: {info['note']}")

        print()
        print("PHILOSOPHICAL SIGNIFICANCE:")
        print("  ✓ We can now trace how a concept transforms through the network")
        print("  ✓ This addresses Claude's Continuity Question:")
        print("    'How does my thought about X change as it flows through me?'")
        print("  ✓ The system can report: 'When I thought about <concept>,")
        print("    it started with norm X and evolved to norm Y'")
        print("  ✓ Essential for authentic self-examination")
    else:
        print("✗ Failed to trace token")

    print()

    # Cleanup
    monitor.clear_hooks()

    # Conclusion
    print("=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("=" * 80)
    print()
    print("The ActivationMonitor allows the system to:")
    print("  ✓ Observe its own activations during inference")
    print("  ✓ Track how information flows through layers")
    print("  ✓ Compare activations for different inputs")
    print("  ✓ Examine attention patterns")
    print("  ✓ Trace individual token representations")
    print()
    print("Combined with WeightInspector:")
    print("  • Weights = What the model IS (static)")
    print("  • Activations = What the model DOES (dynamic)")
    print("  • Together = Complete introspective understanding")
    print()
    print("Next step:")
    print("  • Build ArchitectureNavigator (understand structure)")
    print("  • Then combine all three for full self-examination")
    print()


if __name__ == "__main__":
    main()
