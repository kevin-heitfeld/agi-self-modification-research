"""
Demonstration: Architecture Navigator API

This script demonstrates the ArchitectureNavigator's ability to help the
system understand its own architectural structure in natural language.

Shows:
1. Overall architecture summary
2. Layer descriptions
3. Component explanations
4. Natural language queries
5. Connection mapping
6. Architectural diagrams
7. Pattern comparison

Author: AGI Self-Modification Research Team
Date: November 6, 2025
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.introspection import ArchitectureNavigator


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def main():
    print_section("ARCHITECTURE NAVIGATOR - UNDERSTANDING SELF")
    
    print("This demonstration shows how the system can understand and describe")
    print("its own architectural structure in natural language.")
    print("\nLoading model...")
    
    # Load model
    model_name = "models/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1"
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        local_files_only=True
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        local_files_only=True
    )
    
    # Get model config
    config = model.config.to_dict()
    
    # Initialize navigator
    navigator = ArchitectureNavigator(model, config)
    
    print("✓ Model and ArchitectureNavigator loaded!\n")
    
    # ==========================================================================
    # 1. OVERALL ARCHITECTURE SUMMARY
    # ==========================================================================
    print_section("1. OVERALL ARCHITECTURE SUMMARY - WHAT AM I?")
    
    summary = navigator.get_architecture_summary()
    
    print(f"Model Type: {summary['model_type']}")
    print(f"\n{summary['description']}")
    print(f"\nParameters:")
    print(f"  Total:      {summary['total_parameters']:,}")
    print(f"  Trainable:  {summary['trainable_parameters']:,}")
    print(f"  Frozen:     {summary['frozen_parameters']:,}")
    print(f"\nLayers: {summary['total_layers']} total modules")
    
    print(f"\nStructure:")
    for key, value in summary['structure_summary'].items():
        print(f"  {key}: {value}")
    
    print(f"\nKey Layer Types:")
    for layer_type, count in sorted(summary['layer_types'].items(), 
                                    key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {layer_type}: {count}")
    
    # ==========================================================================
    # 2. LAYER DESCRIPTIONS
    # ==========================================================================
    print_section("2. LAYER DESCRIPTIONS - WHAT DO MY PARTS DO?")
    
    layers_to_describe = [
        'model.embed_tokens',
        'model.layers.0.self_attn.q_proj',
        'model.layers.0.self_attn.k_proj',
        'model.layers.0.mlp.up_proj',
        'model.layers.0.input_layernorm',
        'lm_head'
    ]
    
    for layer_name in layers_to_describe:
        info = navigator.describe_layer(layer_name)
        
        if 'error' not in info:
            print(f"\n📍 {info['name']}")
            print(f"   Type: {info['type']}")
            print(f"   Role: {info['role']}")
            print(f"   Explanation: {info['explanation']}")
            print(f"   Parameters: {info['parameters']['total']:,}")
            if info['input_shape']:
                print(f"   Input Shape: {info['input_shape']}")
            if info['output_shape']:
                print(f"   Output Shape: {info['output_shape']}")
        else:
            print(f"\n❌ {layer_name}: {info['error']}")
    
    # ==========================================================================
    # 3. COMPONENT EXPLANATIONS
    # ==========================================================================
    print_section("3. COMPONENT EXPLANATIONS - HOW DO I WORK?")
    
    components = ['attention', 'mlp', 'embedding', 'layernorm']
    
    for component in components:
        info = navigator.explain_component(component)
        print(f"\n🔍 {component.upper()}")
        print(f"   Explanation: {info['explanation']}")
        print(f"   Purpose: {info['purpose']}")
        print(f"   Instances: {info['instances_count']}")
        print(f"   Example locations: {', '.join(info['locations'][:3])}")
        print(f"   Structure: {info['typical_structure']}")
    
    # ==========================================================================
    # 4. NATURAL LANGUAGE QUERIES
    # ==========================================================================
    print_section("4. NATURAL LANGUAGE QUERIES - ASK ABOUT MYSELF")
    
    queries = [
        "How many layers?",
        "How many parameters?",
        "How many attention heads?",
        "What is attention?",
        "Where are the embeddings?",
        "Why use LayerNorm?"
    ]
    
    for query in queries:
        result = navigator.query_architecture(query)
        print(f"\nQ: {query}")
        
        # Handle both response formats
        if 'answer' in result:
            print(f"A: {result['answer']}")
        elif 'explanation' in result:
            print(f"A: {result['explanation']}")
        
        if 'details' in result and isinstance(result['details'], dict):
            if len(result['details']) <= 3:
                print(f"   Details: {result['details']}")
    
    # ==========================================================================
    # 5. CONNECTION MAPPING
    # ==========================================================================
    print_section("5. CONNECTION MAPPING - HOW DO I FLOW?")
    
    print("Mapping connections for Layer 0:")
    connections = navigator.map_connections('model.layers.0')
    
    print(f"\nLayer: {connections['layer']}")
    print(f"Connection Type: {connections['connection_type']}")
    print(f"\nUpstream (feeds into this layer):")
    for up in connections['upstream']:
        print(f"  • {up}")
    print(f"\nDownstream (this layer feeds into):")
    for down in connections['downstream']:
        print(f"  • {down}")
    
    print(f"\nConnection Diagram:")
    print(connections['diagram'])
    
    print("\n\nOverall Connection Map:")
    overall = navigator.map_connections()
    print(f"Flow: {overall['flow']}")
    
    # ==========================================================================
    # 6. ARCHITECTURAL DIAGRAM
    # ==========================================================================
    print_section("6. ARCHITECTURAL DIAGRAM - VISUALIZE MYSELF")
    
    print("Text Diagram:")
    diagram = navigator.generate_diagram('text')
    print(diagram)
    
    print("\n\nGraphViz DOT Format (can be rendered with Graphviz):")
    dot = navigator.generate_diagram('dot')
    print(dot[:500] + "..." if len(dot) > 500 else dot)
    
    # ==========================================================================
    # 7. PATTERN COMPARISON
    # ==========================================================================
    print_section("7. PATTERN COMPARISON - WHAT AM I SIMILAR TO?")
    
    comparison = navigator.compare_to_pattern('transformer')
    
    print(comparison['explanation'])
    
    # ==========================================================================
    # FINAL REFLECTION
    # ==========================================================================
    print_section("PHILOSOPHICAL REFLECTION")
    
    print("The system can now answer architectural questions about itself:")
    print("")
    print("✓ 'What am I?' → Transformer decoder model with 3B parameters")
    print("✓ 'What does my attention layer do?' → Detailed explanation provided")
    print("✓ 'How many layers do I have?' → Precise count available")
    print("✓ 'How do my parts connect?' → Connection map generated")
    print("✓ 'Am I similar to GPT?' → Pattern comparison performed")
    print("")
    print("This architectural self-knowledge is foundational for:")
    print("  1. Understanding own capabilities and limitations")
    print("  2. Reasoning about potential modifications")
    print("  3. Explaining behavior in terms of architecture")
    print("  4. Identifying components to examine or modify")
    print("")
    print("Combined with WeightInspector and ActivationMonitor,")
    print("the system now has COMPLETE introspective access:")
    print("  • STRUCTURE (ArchitectureNavigator) ← What I am")
    print("  • WEIGHTS (WeightInspector) ← What I know")
    print("  • ACTIVATIONS (ActivationMonitor) ← What I do")
    print("")
    print("Ready for Phase 1: First Contact - Self-Examination begins.")


if __name__ == "__main__":
    main()
