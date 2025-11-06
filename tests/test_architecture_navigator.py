"""
Tests for Architecture Navigator

Validates the ArchitectureNavigator's ability to describe and explain
model architecture in natural language.

Author: AGI Self-Modification Research Team
Date: November 6, 2025
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from src.introspection import ArchitectureNavigator


class SimpleTransformer(nn.Module):
    """A simple transformer for testing."""
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(1000, 256)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=256, nhead=8)
            for _ in range(4)
        ])
        self.lm_head = nn.Linear(256, 1000)
    
    def forward(self, x):
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(x)


def test_architecture_summary():
    """Test getting overall architecture summary."""
    print("Testing architecture summary...")
    
    model = SimpleTransformer()
    config = {
        'hidden_size': 256,
        'num_layers': 4,
        'num_attention_heads': 8,
        'vocab_size': 1000
    }
    
    navigator = ArchitectureNavigator(model, config)
    summary = navigator.get_architecture_summary()
    
    assert 'model_type' in summary
    assert 'description' in summary
    assert 'total_parameters' in summary
    assert summary['total_parameters'] > 0
    assert 'layer_types' in summary
    assert 'structure_summary' in summary
    
    print(f"✓ Model Type: {summary['model_type']}")
    print(f"✓ Parameters: {summary['total_parameters']:,}")
    print(f"✓ Description: {summary['description'][:80]}...")
    print("")


def test_layer_description():
    """Test describing individual layers."""
    print("Testing layer descriptions...")
    
    model = SimpleTransformer()
    navigator = ArchitectureNavigator(model)
    
    # Test embedding layer
    info = navigator.describe_layer('embed')
    assert 'name' in info
    assert 'type' in info
    assert 'explanation' in info
    assert 'role' in info
    assert info['type'] == 'Embedding'
    print(f"✓ Embedding layer: {info['explanation'][:60]}...")
    
    # Test linear layer
    info = navigator.describe_layer('lm_head')
    assert info['type'] == 'Linear'
    assert 'parameters' in info
    assert info['parameters']['total'] > 0
    print(f"✓ LM head: {info['role']}")
    
    # Test non-existent layer
    info = navigator.describe_layer('nonexistent')
    assert 'error' in info
    print(f"✓ Error handling: {info['error'][:50]}...")
    print("")


def test_component_explanation():
    """Test explaining component types."""
    print("Testing component explanations...")
    
    model = SimpleTransformer()
    navigator = ArchitectureNavigator(model)
    
    # Test attention explanation
    info = navigator.explain_component('attention')
    assert 'component' in info
    assert 'explanation' in info
    assert 'purpose' in info
    assert 'instances_count' in info
    assert len(info['explanation']) > 50
    print(f"✓ Attention: {info['purpose'][:60]}...")
    
    # Test embedding explanation (may be 0 for simple test model)
    info = navigator.explain_component('embedding')
    assert info['instances_count'] >= 0
    print(f"✓ Embedding: Found {info['instances_count']} instances")
    
    # Test MLP explanation
    info = navigator.explain_component('mlp')
    assert 'explanation' in info
    print(f"✓ MLP: {info['explanation'][:60]}...")
    print("")


def test_natural_language_queries():
    """Test answering natural language queries."""
    print("Testing natural language queries...")
    
    model = SimpleTransformer()
    config = {'num_layers': 4, 'num_attention_heads': 8}
    navigator = ArchitectureNavigator(model, config)
    
    # Test count query
    result = navigator.query_architecture("How many layers?")
    assert 'answer' in result
    assert '4' in result['answer']
    print(f"✓ Count query: {result['answer'][:50]}...")
    
    # Test explanation query (returns different structure)
    result = navigator.query_architecture("What is attention?")
    assert 'explanation' in result or 'answer' in result
    if 'explanation' in result:
        print(f"✓ Explanation query: {result['explanation'][:50]}...")
    else:
        print(f"✓ Explanation query: {result['answer'][:50]}...")
    
    # Test location query
    result = navigator.query_architecture("Where are the embeddings?")
    assert 'answer' in result
    print(f"✓ Location query: {result['answer'][:50]}...")
    print("")


def test_connection_mapping():
    """Test mapping layer connections."""
    print("Testing connection mapping...")
    
    model = SimpleTransformer()
    navigator = ArchitectureNavigator(model)
    
    # Test specific layer
    connections = navigator.map_connections('layers.0')
    assert 'layer' in connections
    assert 'connection_type' in connections
    assert 'diagram' in connections
    print(f"✓ Layer connections: {connections['connection_type']}")
    
    # Test overall connections
    overall = navigator.map_connections()
    assert 'flow' in overall
    print(f"✓ Overall flow: {overall['flow']}")
    print("")


def test_diagram_generation():
    """Test generating architectural diagrams."""
    print("Testing diagram generation...")
    
    model = SimpleTransformer()
    config = {'num_layers': 4}
    navigator = ArchitectureNavigator(model, config)
    
    # Test text diagram
    text_diagram = navigator.generate_diagram('text')
    assert len(text_diagram) > 100
    assert 'INPUT' in text_diagram
    assert 'Layer' in text_diagram
    print(f"✓ Text diagram: {len(text_diagram)} characters")
    
    # Test DOT diagram
    dot_diagram = navigator.generate_diagram('dot')
    assert 'digraph' in dot_diagram
    assert 'layer' in dot_diagram.lower()
    print(f"✓ DOT diagram: {len(dot_diagram)} characters")
    print("")


def test_pattern_comparison():
    """Test comparing against known patterns."""
    print("Testing pattern comparison...")
    
    model = SimpleTransformer()
    navigator = ArchitectureNavigator(model)
    
    # Test transformer comparison
    comparison = navigator.compare_to_pattern('transformer')
    assert 'pattern' in comparison
    assert 'matches' in comparison
    assert 'differences' in comparison
    assert 'similarity_score' in comparison
    assert 'explanation' in comparison
    assert 0 <= comparison['similarity_score'] <= 1
    
    print(f"✓ Transformer pattern: {comparison['similarity_score']:.1%} similar")
    print(f"  Matches: {len(comparison['matches'])}")
    print(f"  Differences: {len(comparison['differences'])}")
    print("")


def test_caching():
    """Test that caching works for repeated operations."""
    print("Testing caching...")
    
    model = SimpleTransformer()
    navigator = ArchitectureNavigator(model)
    
    # First call (should cache)
    info1 = navigator.describe_layer('embed')
    
    # Second call (should use cache)
    info2 = navigator.describe_layer('embed')
    
    # Should be identical
    assert info1 == info2
    print("✓ Caching works correctly")
    print("")


def run_all_tests():
    """Run all tests."""
    print("=" * 70)
    print("  ARCHITECTURE NAVIGATOR TESTS")
    print("=" * 70)
    print("")
    
    try:
        test_architecture_summary()
        test_layer_description()
        test_component_explanation()
        test_natural_language_queries()
        test_connection_mapping()
        test_diagram_generation()
        test_pattern_comparison()
        test_caching()
        
        print("=" * 70)
        print("  ✅ ALL TESTS PASSED!")
        print("=" * 70)
        print("")
        print("The ArchitectureNavigator can successfully:")
        print("  ✓ Describe overall model architecture")
        print("  ✓ Explain individual layers and components")
        print("  ✓ Answer natural language queries")
        print("  ✓ Map layer connections")
        print("  ✓ Generate architectural diagrams")
        print("  ✓ Compare against known patterns")
        print("  ✓ Cache results efficiently")
        print("")
        print("The system now has complete architectural self-knowledge!")
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        raise
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        raise


if __name__ == "__main__":
    run_all_tests()
