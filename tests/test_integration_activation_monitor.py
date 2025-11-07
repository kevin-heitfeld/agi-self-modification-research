"""
Integration Tests for ActivationMonitor with Qwen2.5 Model

Tests real inference scenarios:
- Full forward passes with actual text
- Attention pattern analysis
- Token influence tracing through layers
- Activation comparison across different inputs
- Multi-layer monitoring during generation
- Philosophical self-analysis capabilities (Claude's continuity question)
"""

import sys
import pytest
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from introspection import ActivationMonitor


@pytest.fixture(scope="module")
def qwen_model():
    """Load Qwen2.5-3B model for integration tests"""
    model_path = Path("models/models--Qwen--Qwen2.5-3B-Instruct/snapshots")
    
    # Find the snapshot directory
    snapshot_dirs = list(model_path.glob("*"))
    if not snapshot_dirs:
        pytest.skip("Qwen2.5-3B model not found. Run scripts/download_model.py first.")
    
    model_dir = snapshot_dirs[0]
    
    print(f"\nLoading model from {model_dir}...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    
    # Load model without device_map for proper checkpoint support
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    
    if torch.cuda.is_available():
        model = model.to("cuda")
    
    return model, tokenizer


@pytest.fixture
def monitor(qwen_model):
    """Create ActivationMonitor for the model"""
    model, tokenizer = qwen_model
    return ActivationMonitor(model, tokenizer, "Qwen2.5-3B")


class TestActivationMonitorArchitecture:
    """Test architecture discovery and layer registry"""
    
    def test_layer_discovery(self, monitor):
        """Test that monitor can discover model layers"""
        layers = monitor.get_layer_names()
        
        assert len(layers) > 0, "Should discover layers"
        assert any("self_attn" in layer for layer in layers), "Should find attention layers"
        assert any("mlp" in layer for layer in layers), "Should find MLP layers"
        
        print(f"✓ Discovered {len(layers)} layers")
    
    def test_query_layers(self, monitor):
        """Test natural language layer queries"""
        # Query attention layers
        attn_layers = monitor.query_layers("self_attn")
        assert len(attn_layers) > 0, "Should find attention layers"
        
        # Query MLP layers
        mlp_layers = monitor.query_layers("mlp")
        assert len(mlp_layers) > 0, "Should find MLP layers"
        
        # Query specific layer number
        layer_0 = monitor.query_layers("layers.0")
        assert len(layer_0) > 0, "Should find layer 0"
        
        print(f"✓ Query found {len(attn_layers)} attention, {len(mlp_layers)} MLP layers")
    
    def test_layer_info(self, monitor):
        """Test retrieving layer information"""
        layers = monitor.get_layer_names()
        test_layer = layers[0]
        
        info = monitor.get_layer_info(test_layer)
        
        assert "name" in info
        assert "type" in info
        assert "num_parameters" in info
        
        print(f"✓ Layer '{test_layer}' is {info['type']} with {info['num_parameters']} parameters")


class TestActivationCapture:
    """Test activation capture during inference"""
    
    def test_basic_capture(self, monitor):
        """Test basic activation capture on simple input"""
        attn_layers = monitor.query_layers("self_attn")[:2]  # First 2 attention layers
        
        result = monitor.capture_activations(
            "Hello world",
            layer_names=attn_layers,
            max_length=20
        )
        
        assert "activations" in result
        assert "input_text" in result
        assert "tokens" in result
        assert "token_strings" in result
        assert result["num_tokens"] > 0
        
        # Check that we captured activations for requested layers
        for layer in attn_layers:
            assert layer in result["activations"], f"Missing activations for {layer}"
        
        print(f"✓ Captured activations for {len(result['activations'])} layers")
        print(f"  Input: '{result['input_text']}'")
        print(f"  Tokens: {result['num_tokens']}")
    
    def test_meaningful_text_capture(self, monitor):
        """Test capture with meaningful philosophical text"""
        text = "I think, therefore I am"
        attn_layers = monitor.query_layers("self_attn")[:3]
        
        result = monitor.capture_activations(text, layer_names=attn_layers)
        
        assert result["num_tokens"] > 5, "Should tokenize meaningful sentence"
        assert len(result["activations"]) >= 3, "Should capture multiple layers"
        
        # Check activation shapes
        first_layer = list(result["activations"].keys())[0]
        activation = result["activations"][first_layer]
        
        # Expected shape: [batch, seq_len, hidden_dim]
        assert len(activation.shape) >= 2, "Activation should be at least 2D"
        
        print(f"✓ Processed '{text}'")
        print(f"  Shape: {activation.shape}")
    
    def test_multiple_sentences(self, monitor):
        """Test capture with multiple sentences"""
        text = "The model processes text. It captures activations at each layer."
        attn_layers = monitor.query_layers("self_attn")[:2]
        
        result = monitor.capture_activations(text, layer_names=attn_layers)
        
        assert result["num_tokens"] > 10, "Should tokenize multiple sentences"
        print(f"✓ Processed {result['num_tokens']} tokens from multiple sentences")
    
    def test_hook_registration(self, monitor):
        """Test that hooks are properly registered and cleared"""
        attn_layers = monitor.query_layers("self_attn")[:2]
        
        # Register hooks
        monitor.register_hooks(attn_layers)
        assert len(monitor.hooks) == len(attn_layers), "Should register hooks"
        
        # Clear hooks
        monitor.clear_hooks()
        assert len(monitor.hooks) == 0, "Should clear hooks"
        
        print("✓ Hook registration and cleanup works")


class TestActivationStatistics:
    """Test activation statistics computation"""
    
    def test_basic_statistics(self, monitor):
        """Test computing statistics for captured activations"""
        attn_layers = monitor.query_layers("self_attn")[:2]
        
        result = monitor.capture_activations(
            "The quick brown fox",
            layer_names=attn_layers
        )
        
        layer_name = list(result["activations"].keys())[0]
        stats = monitor.get_activation_statistics(layer_name)
        
        # Check all expected statistics
        assert "mean" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats
        assert "l2_norm" in stats
        assert "zeros_percentage" in stats
        assert "positive_percentage" in stats
        
        print(f"✓ Statistics for {layer_name}:")
        print(f"  Mean: {stats['mean']:.6f}")
        print(f"  Std: {stats['std']:.6f}")
        print(f"  L2 norm: {stats['l2_norm']:.2f}")
        print(f"  Sparsity: {stats['zeros_percentage']:.2f}%")
    
    def test_statistics_all_layers(self, monitor):
        """Test statistics across multiple layers"""
        attn_layers = monitor.query_layers("self_attn")[:3]
        
        result = monitor.capture_activations(
            "Testing activation statistics",
            layer_names=attn_layers
        )
        
        layer_stats = []
        for layer_name in result["activations"].keys():
            stats = monitor.get_activation_statistics(layer_name)
            layer_stats.append((layer_name, stats))
        
        assert len(layer_stats) >= 3, "Should compute stats for multiple layers"
        
        print(f"✓ Computed statistics for {len(layer_stats)} layers")
        for name, stats in layer_stats:
            print(f"  {name}: mean={stats['mean']:.4f}, norm={stats['l2_norm']:.2f}")


class TestActivationComparison:
    """Test comparing activations across different inputs"""
    
    def test_similar_inputs(self, monitor):
        """Test comparing activations for similar inputs"""
        attn_layers = monitor.query_layers("self_attn")[:2]
        
        comparison = monitor.compare_activations(
            "I am happy",
            "I am joyful",
            attn_layers
        )
        
        assert "comparisons" in comparison
        assert len(comparison["comparisons"]) > 0
        
        # Check similarity metrics
        first_layer = list(comparison["comparisons"].keys())[0]
        metrics = comparison["comparisons"][first_layer]
        
        assert "cosine_similarity" in metrics
        assert "correlation" in metrics
        assert "euclidean_distance" in metrics
        
        # Similar inputs should have high cosine similarity
        cosine_sim = metrics["cosine_similarity"]
        assert cosine_sim > 0.5, "Similar inputs should have high cosine similarity"
        
        print(f"✓ Comparison for similar inputs:")
        print(f"  '{comparison['input1']}' vs '{comparison['input2']}'")
        print(f"  Cosine similarity: {cosine_sim:.4f}")
        print(f"  Correlation: {metrics['correlation']:.4f}")
    
    def test_dissimilar_inputs(self, monitor):
        """Test comparing activations for dissimilar inputs"""
        attn_layers = monitor.query_layers("self_attn")[:2]
        
        comparison = monitor.compare_activations(
            "The sky is blue",
            "Mathematics is difficult",
            attn_layers
        )
        
        first_layer = list(comparison["comparisons"].keys())[0]
        metrics = comparison["comparisons"][first_layer]
        
        cosine_sim = metrics["cosine_similarity"]
        
        print(f"✓ Comparison for dissimilar inputs:")
        print(f"  '{comparison['input1']}' vs '{comparison['input2']}'")
        print(f"  Cosine similarity: {cosine_sim:.4f}")
        print(f"  (Lower similarity expected for unrelated topics)")
    
    def test_semantic_shift(self, monitor):
        """Test detecting semantic shifts through activations"""
        attn_layers = monitor.query_layers("self_attn")[:2]
        
        # Opposite sentiment
        comparison = monitor.compare_activations(
            "This is excellent",
            "This is terrible",
            attn_layers
        )
        
        first_layer = list(comparison["comparisons"].keys())[0]
        metrics = comparison["comparisons"][first_layer]
        
        print(f"✓ Sentiment comparison:")
        print(f"  Positive: '{comparison['input1']}'")
        print(f"  Negative: '{comparison['input2']}'")
        print(f"  Cosine similarity: {metrics['cosine_similarity']:.4f}")
        print(f"  Mean difference: {metrics['mean_difference']:.6f}")


class TestAttentionPatterns:
    """Test attention pattern analysis"""
    
    def test_attention_capture(self, monitor):
        """Test capturing attention weights"""
        attn_layers = monitor.query_layers("self_attn")[:2]
        
        result = monitor.capture_activations(
            "Attention is all you need",
            layer_names=attn_layers
        )
        
        if result["attention_weights"]:
            layer_name = list(result["attention_weights"].keys())[0]
            patterns = monitor.get_attention_patterns(layer_name)
            
            assert "shape" in patterns
            assert "num_heads" in patterns
            assert "attention_matrix" in patterns
            assert "mean_attention" in patterns
            assert "entropy" in patterns
            
            print(f"✓ Attention patterns for {layer_name}:")
            print(f"  Num heads: {patterns['num_heads']}")
            print(f"  Shape: {patterns['shape']}")
            print(f"  Mean attention: {patterns['mean_attention']:.6f}")
            print(f"  Entropy: {patterns['entropy']:.4f}")
        else:
            print("⊘ No attention weights captured (depends on layer configuration)")
    
    def test_attention_heads(self, monitor):
        """Test examining individual attention heads"""
        attn_layers = monitor.query_layers("self_attn")[:1]
        
        result = monitor.capture_activations(
            "Multi-head attention mechanisms",
            layer_names=attn_layers
        )
        
        if result["attention_weights"]:
            layer_name = list(result["attention_weights"].keys())[0]
            
            # Get averaged attention
            avg_patterns = monitor.get_attention_patterns(layer_name)
            
            # Try to get specific head (if multiple heads exist)
            if avg_patterns["num_heads"] > 1:
                head_patterns = monitor.get_attention_patterns(layer_name, head_idx=0)
                
                assert head_patterns["head_idx"] == 0
                assert "attention_matrix" in head_patterns
                
                print(f"✓ Individual head analysis:")
                print(f"  Head 0 mean attention: {head_patterns['mean_attention']:.6f}")
                print(f"  Head 0 entropy: {head_patterns['entropy']:.4f}")
            else:
                print("⊘ Single head detected, skipping multi-head test")


class TestTokenInfluenceTracing:
    """Test tracing token influence through layers (Claude's continuity question)"""
    
    def test_trace_single_token(self, monitor):
        """Test tracing how a single token evolves through layers"""
        # Get a few layers to trace through
        layers = monitor.query_layers("layers.0.self_attn")[:1]  # First attention
        layers += monitor.query_layers("layers.1.self_attn")[:1]  # Second attention
        layers += monitor.query_layers("layers.2.self_attn")[:1]  # Third attention
        
        text = "I think therefore I am"
        token_idx = 1  # Trace "think"
        
        trace = monitor.trace_token_influence(text, token_idx, layers)
        
        assert "token" in trace
        assert "layers" in trace
        assert "evolution_summary" in trace
        
        # Check that we traced through multiple layers
        traced_layers = [k for k, v in trace["layers"].items() if "l2_norm" in v]
        assert len(traced_layers) >= 2, "Should trace through multiple layers"
        
        # Check evolution summary
        if trace["evolution_summary"]:
            summary = trace["evolution_summary"]
            assert "initial_norm" in summary
            assert "final_norm" in summary
            assert "total_norm_change" in summary
            
            print(f"✓ Token influence trace for '{trace['token']}':")
            print(f"  Traced through {summary['num_layers_traced']} layers")
            print(f"  Initial norm: {summary['initial_norm']:.4f}")
            print(f"  Final norm: {summary['final_norm']:.4f}")
            print(f"  Change: {summary['total_norm_change']:.4f}")
            print(f"  Stability: {summary['representation_stability']}")
    
    def test_trace_multiple_tokens(self, monitor):
        """Test tracing multiple tokens through layers"""
        layers = monitor.query_layers("layers.0.self_attn")[:1]
        layers += monitor.query_layers("layers.1.self_attn")[:1]
        
        text = "The quick brown fox"
        
        # Trace each significant token
        tokens_to_trace = [0, 1, 2, 3]  # The, quick, brown, fox
        
        traces = []
        for token_idx in tokens_to_trace:
            trace = monitor.trace_token_influence(text, token_idx, layers)
            traces.append(trace)
        
        assert len(traces) == len(tokens_to_trace)
        
        print(f"✓ Traced {len(traces)} tokens:")
        for trace in traces:
            token = trace["token"]
            if trace["evolution_summary"]:
                norm_change = trace["evolution_summary"]["total_norm_change"]
                print(f"  '{token}': norm change = {norm_change:.4f}")
    
    def test_trace_philosophical_continuity(self, monitor):
        """Test tracing continuity of concept representation (Claude's question)"""
        # Get early, middle, and late layers
        early_layers = monitor.query_layers("layers.0")[:2]
        middle_layers = monitor.query_layers("layers.5")[:2] if monitor.query_layers("layers.5") else []
        late_layers = monitor.query_layers("layers.10")[:2] if monitor.query_layers("layers.10") else []
        
        all_layers = early_layers + middle_layers + late_layers
        
        # Use philosophical text about continuity
        text = "The self persists through time"
        token_idx = 1  # "self"
        
        if len(all_layers) >= 3:
            trace = monitor.trace_token_influence(text, token_idx, all_layers)
            
            # Check that representation evolves through layers
            layer_norms = []
            for layer_name, layer_info in trace["layers"].items():
                if "l2_norm" in layer_info:
                    layer_norms.append((layer_name, layer_info["l2_norm"]))
            
            assert len(layer_norms) >= 3, "Should trace through early, middle, late layers"
            
            print(f"✓ Philosophical continuity trace for '{trace['token']}':")
            print(f"  Text: '{text}'")
            print(f"  Layer evolution:")
            for name, norm in layer_norms:
                print(f"    {name}: {norm:.4f}")
            
            if trace["evolution_summary"]:
                print(f"  Representation: {trace['evolution_summary']['representation_stability']}")
        else:
            print("⊘ Insufficient layers for full continuity test")


class TestRealWorldScenarios:
    """Test real-world self-analysis scenarios"""
    
    def test_self_reference_processing(self, monitor):
        """Test how the model processes self-referential statements"""
        attn_layers = monitor.query_layers("self_attn")[:3]
        
        # Compare self-referential vs non-self-referential
        comparison = monitor.compare_activations(
            "I am processing this text",
            "The system processes text",
            attn_layers
        )
        
        assert len(comparison["comparisons"]) > 0
        
        first_layer = list(comparison["comparisons"].keys())[0]
        metrics = comparison["comparisons"][first_layer]
        
        print(f"✓ Self-reference processing:")
        print(f"  First-person: '{comparison['input1']}'")
        print(f"  Third-person: '{comparison['input2']}'")
        
        # Check if we have similarity metrics or an error
        if "cosine_similarity" in metrics:
            print(f"  Cosine similarity: {metrics['cosine_similarity']:.4f}")
            print(f"  (Detecting perspective shift in activations)")
        elif "error" in metrics:
            print(f"  Note: {metrics['error']}")
            print(f"  Shape1: {metrics.get('shape1', 'N/A')}, Shape2: {metrics.get('shape2', 'N/A')}")
        else:
            print(f"  Metrics: {metrics}")
    
    def test_reasoning_trace(self, monitor):
        """Test tracing reasoning through layers"""
        # Get representative layers across depth
        layers = monitor.query_layers("layers.0.self_attn")[:1]
        layers += monitor.query_layers("layers.2.self_attn")[:1]
        layers += monitor.query_layers("layers.5.self_attn")[:1] if monitor.query_layers("layers.5") else []
        
        text = "If A implies B, and B implies C, then A implies C"
        
        result = monitor.capture_activations(text, layer_names=layers)
        
        # Analyze how logical structure is processed
        assert len(result["activations"]) > 0
        
        print(f"✓ Reasoning trace:")
        print(f"  Text: '{text}'")
        print(f"  Tokens: {result['num_tokens']}")
        print(f"  Monitored {len(result['activations'])} layers")
        
        # Show activation statistics for each layer
        for layer_name in result["activations"].keys():
            stats = monitor.get_activation_statistics(layer_name)
            print(f"  {layer_name}: mean={stats['mean']:.4f}, std={stats['std']:.4f}")
    
    def test_semantic_composition(self, monitor):
        """Test how meanings compose through layers"""
        layers = monitor.query_layers("layers.0.self_attn")[:1]
        layers += monitor.query_layers("layers.1.self_attn")[:1]
        
        # Trace a key semantic token
        text = "The red balloon floated upward"
        token_idx = 2  # "balloon"
        
        trace = monitor.trace_token_influence(text, token_idx, layers)
        
        # Check that the representation evolves
        traced_layers = [k for k, v in trace["layers"].items() if "l2_norm" in v]
        
        print(f"✓ Semantic composition for '{trace['token']}':")
        print(f"  Context: '{text}'")
        print(f"  Traced through {len(traced_layers)} layers")
        
        for layer_name in traced_layers:
            layer_info = trace["layers"][layer_name]
            if "mean" in layer_info:
                print(f"  {layer_name}: mean={layer_info['mean']:.4f}, norm={layer_info['l2_norm']:.4f}")


class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_empty_input(self, monitor):
        """Test handling of very short input"""
        attn_layers = monitor.query_layers("self_attn")[:1]
        
        result = monitor.capture_activations("Hi", layer_names=attn_layers)
        
        assert result["num_tokens"] > 0
        assert len(result["activations"]) > 0
        
        print(f"✓ Handled short input: '{result['input_text']}' ({result['num_tokens']} tokens)")
    
    def test_long_input(self, monitor):
        """Test handling of longer input"""
        attn_layers = monitor.query_layers("self_attn")[:1]
        
        long_text = "This is a longer sentence that contains multiple clauses and ideas that the model needs to process and understand in order to capture meaningful activations across different layers."
        
        result = monitor.capture_activations(long_text, layer_names=attn_layers, max_length=100)
        
        assert result["num_tokens"] > 10
        assert len(result["activations"]) > 0
        
        print(f"✓ Handled long input: {result['num_tokens']} tokens")
    
    def test_invalid_layer_name(self, monitor):
        """Test error handling for invalid layer names"""
        with pytest.raises(KeyError):
            monitor.register_hooks(["nonexistent_layer"])
        
        print("✓ Properly raises KeyError for invalid layer names")
    
    def test_statistics_without_capture(self, monitor):
        """Test error handling when requesting stats before capture"""
        monitor.clear_activations()
        
        with pytest.raises(KeyError):
            monitor.get_activation_statistics("some_layer")
        
        print("✓ Properly raises KeyError when no activations captured")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
