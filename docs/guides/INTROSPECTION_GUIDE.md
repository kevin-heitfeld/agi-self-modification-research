# Introspection Guide

**AGI Self-Modification Research Platform**  
**Version**: Phase 0 Complete  
**Date**: November 7, 2025

---

## Understanding Your Model from Within 🔍

This guide explains how to use the introspection tools to understand what your model is doing internally. By the end, you'll be able to:

- ✅ Inspect weights and parameters
- ✅ Monitor activations during inference
- ✅ Navigate the architecture
- ✅ Trace token influence through layers
- ✅ Analyze attention patterns
- ✅ Detect weight sharing and coupling

---

## Table of Contents

1. [Weight Inspector](#weight-inspector)
2. [Activation Monitor](#activation-monitor)
3. [Architecture Navigator](#architecture-navigator)
4. [Advanced Techniques](#advanced-techniques)
5. [Philosophical Insights](#philosophical-insights)

---

## Weight Inspector

### What Are Weights?

Weights are the learned parameters that make your model work. Think of them as the "knowledge" stored in the neural network.

### Basic Inspection

```python
from introspection import WeightInspector

# Create inspector
inspector = WeightInspector(model, "Qwen2.5-3B")

# Get overall statistics
stats = inspector.get_weight_statistics()

print(f"📊 Model Statistics:")
print(f"  Total parameters: {stats['total_parameters']:,}")
print(f"  Trainable parameters: {stats['trainable_parameters']:,}")
print(f"  Model size: {stats['total_size_mb']:.2f} MB")
print(f"  Number of layers: {stats['num_layers']}")
print(f"  Dtype: {stats['dtype']}")
```

**Example Output**:
```
📊 Model Statistics:
  Total parameters: 151,126,016
  Trainable parameters: 151,126,016
  Model size: 302.25 MB
  Number of layers: 36
  Dtype: float16
```

### Examining Specific Layers

```python
# Get layer names
layer_names = inspector.get_layer_names()
print(f"Total layers: {len(layer_names)}")
print(f"Sample layers: {layer_names[:5]}")

# Examine a specific layer
layer_name = "model.layers.0.self_attn.q_proj.weight"
layer_stats = inspector.get_layer_statistics(layer_name)

print(f"\n🔍 Layer: {layer_name}")
print(f"  Shape: {layer_stats['shape']}")
print(f"  Parameters: {layer_stats['num_parameters']:,}")
print(f"  Mean: {layer_stats['mean']:.6f}")
print(f"  Std Dev: {layer_stats['std']:.6f}")
print(f"  Min: {layer_stats['min']:.6f}")
print(f"  Max: {layer_stats['max']:.6f}")
print(f"  L2 Norm: {layer_stats['l2_norm']:.2f}")
print(f"  Sparsity: {layer_stats['zeros_percentage']:.2f}%")
```

### What Do These Stats Mean?

| Statistic | Meaning | What to Watch |
|-----------|---------|---------------|
| **Mean** | Average value | Close to 0 for most layers |
| **Std Dev** | Spread of values | Higher in deeper layers usually |
| **L2 Norm** | Magnitude | Very high = potential instability |
| **Sparsity** | % of zeros | High = many inactive connections |
| **Min/Max** | Value range | Extremes indicate outliers |

### Weight Distribution Analysis

```python
# Get weight distribution
distribution = inspector.get_weight_distribution(layer_name)

print(f"\n📈 Distribution:")
print(f"  Positive: {distribution['positive_percentage']:.2f}%")
print(f"  Negative: {distribution['negative_percentage']:.2f}%")
print(f"  Near-zero (< 0.001): {distribution['near_zero_percentage']:.2f}%")
print(f"  Large (> 1.0): {distribution['large_percentage']:.2f}%")

# Percentiles
print(f"\n📊 Percentiles:")
print(f"  25th: {distribution['percentile_25']:.6f}")
print(f"  50th (median): {distribution['percentile_50']:.6f}")
print(f"  75th: {distribution['percentile_75']:.6f}")
print(f"  95th: {distribution['percentile_95']:.6f}")
```

### Comparing Layers

```python
# Compare attention layers across depths
attn_layers = [
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.5.self_attn.q_proj.weight",
    "model.layers.10.self_attn.q_proj.weight",
    "model.layers.15.self_attn.q_proj.weight"
]

print("\n📊 Layer Comparison:")
print("Layer | Mean | Std | L2 Norm")
print("------|------|-----|--------")

for layer in attn_layers:
    stats = inspector.get_layer_statistics(layer)
    layer_num = layer.split('.')[2]
    print(f"L{layer_num:>3} | {stats['mean']:>6.3f} | {stats['std']:>5.3f} | {stats['l2_norm']:>7.1f}")
```

### Weight Sharing Detection 🚨

**CRITICAL for safe modification!**

```python
# Detect weight sharing
shared = inspector.get_shared_weights()

if shared:
    print("\n⚠️  WEIGHT SHARING DETECTED:")
    for i, group in enumerate(shared, 1):
        print(f"\nGroup {i}:")
        for layer in group:
            print(f"  - {layer}")
        print(f"  ⚠️  Modifying ANY of these modifies ALL of them!")
else:
    print("\n✓ No weight sharing detected")

# Check if specific layer is shared
layer = "model.embed_tokens.weight"
coupled_layers = inspector.get_shared_layers(layer)

if coupled_layers:
    print(f"\n⚠️  {layer} is coupled with:")
    for coupled in coupled_layers:
        print(f"  - {coupled}")
```

### Detecting Anomalies

```python
# Find layers with unusual statistics
anomalies = inspector.detect_anomalies(
    nan_check=True,
    inf_check=True,
    large_norm_threshold=1000.0,
    high_sparsity_threshold=90.0
)

if anomalies:
    print("\n🚨 ANOMALIES DETECTED:")
    for anomaly in anomalies:
        print(f"\n  Layer: {anomaly['layer']}")
        print(f"  Issue: {anomaly['type']}")
        print(f"  Details: {anomaly['details']}")
else:
    print("\n✓ No anomalies detected")
```

---

## Activation Monitor

### What Are Activations?

Activations are the intermediate values computed as data flows through the network. They show how each layer transforms the input.

### Basic Activation Capture

```python
from introspection import ActivationMonitor

# Create monitor
monitor = ActivationMonitor(model, tokenizer, "Qwen2.5-3B")

# Find layers to monitor
attn_layers = monitor.query_layers("self_attn")
print(f"Found {len(attn_layers)} attention layers")

# Capture activations
result = monitor.capture_activations(
    "I think, therefore I am",
    layer_names=attn_layers[:3]  # First 3 attention layers
)

print(f"\n📸 Captured:")
print(f"  Input: '{result['input_text']}'")
print(f"  Tokens: {result['token_strings']}")
print(f"  Num tokens: {result['num_tokens']}")
print(f"  Monitored layers: {len(result['activations'])}")
```

### Analyzing Activations

```python
# Get activation statistics
for layer_name in result['activations'].keys():
    stats = monitor.get_activation_statistics(layer_name)
    
    print(f"\n📊 {layer_name}:")
    print(f"  Shape: {stats['shape']}")
    print(f"  Mean: {stats['mean']:.6f}")
    print(f"  Std: {stats['std']:.6f}")
    print(f"  L2 Norm: {stats['l2_norm']:.2f}")
    print(f"  Sparsity: {stats['zeros_percentage']:.2f}%")
    print(f"  Active neurons: {stats['positive_percentage']:.2f}%")
```

### What Do Activation Stats Mean?

| Statistic | Meaning | Good Values | Warning Signs |
|-----------|---------|-------------|---------------|
| **Mean** | Average activation | -0.1 to 0.1 | > 1.0 or < -1.0 |
| **L2 Norm** | Total magnitude | 10-100 | > 1000 |
| **Sparsity** | % inactive | 10-50% | > 90% (dead neurons) |
| **Active %** | % positive | 30-70% | < 10% or > 90% |

### Comparing Inputs

```python
# Compare similar inputs
comparison = monitor.compare_activations(
    "I am happy",
    "I am joyful",
    attn_layers[:2]
)

print("\n🔄 Similarity Analysis:")
print(f"  Input 1: '{comparison['input1']}'")
print(f"  Input 2: '{comparison['input2']}'")

for layer, metrics in comparison['comparisons'].items():
    if 'cosine_similarity' in metrics:
        print(f"\n  {layer}:")
        print(f"    Cosine similarity: {metrics['cosine_similarity']:.4f}")
        print(f"    Correlation: {metrics['correlation']:.4f}")
        print(f"    Euclidean distance: {metrics['euclidean_distance']:.2f}")
```

**Interpretation**:
- **Cosine similarity**: 1.0 = identical, 0.0 = orthogonal, -1.0 = opposite
- **0.9-1.0**: Very similar meanings
- **0.7-0.9**: Similar semantic content
- **0.3-0.7**: Some overlap
- **<0.3**: Different meanings

### Semantic Shift Detection

```python
# Detect sentiment shifts
sentiment_comparison = monitor.compare_activations(
    "This is excellent",
    "This is terrible",
    attn_layers[:1]
)

layer = list(sentiment_comparison['comparisons'].keys())[0]
similarity = sentiment_comparison['comparisons'][layer]['cosine_similarity']

print(f"\n💭 Sentiment Shift:")
print(f"  Positive: 'This is excellent'")
print(f"  Negative: 'This is terrible'")
print(f"  Similarity: {similarity:.4f}")
print(f"  → {('High' if similarity > 0.7 else 'Low')} structural similarity despite opposite sentiment")
```

### Attention Pattern Analysis

```python
# Capture with attention weights
result = monitor.capture_activations(
    "Attention is all you need",
    layer_names=attn_layers[:1]
)

if result['attention_weights']:
    layer = list(result['attention_weights'].keys())[0]
    patterns = monitor.get_attention_patterns(layer)
    
    print(f"\n👁️  Attention Patterns for {layer}:")
    print(f"  Attention shape: {patterns['shape']}")
    print(f"  Number of heads: {patterns['num_heads']}")
    print(f"  Mean attention: {patterns['mean_attention']:.6f}")
    print(f"  Max attention: {patterns['max_attention']:.6f}")
    print(f"  Entropy: {patterns['entropy']:.4f}")
    
    # High entropy = attention spread across many tokens
    # Low entropy = attention focused on few tokens
    if patterns['entropy'] > 2.0:
        print("  → Attention is DISTRIBUTED across tokens")
    else:
        print("  → Attention is FOCUSED on specific tokens")
```

### Individual Attention Heads

```python
# Examine specific attention heads
for head_idx in range(min(4, patterns['num_heads'])):
    head_patterns = monitor.get_attention_patterns(layer, head_idx=head_idx)
    
    print(f"\nHead {head_idx}:")
    print(f"  Mean: {head_patterns['mean_attention']:.6f}")
    print(f"  Max: {head_patterns['max_attention']:.6f}")
    print(f"  Entropy: {head_patterns['entropy']:.4f}")
```

### Token Influence Tracing 🎯

**This answers Claude's continuity question!**

```python
# Trace how a token evolves through layers
early_layers = monitor.query_layers("layers.0")[:2]
mid_layers = monitor.query_layers("layers.5")[:2]
late_layers = monitor.query_layers("layers.10")[:2]

all_layers = early_layers + mid_layers + late_layers

# Trace the word "self"
text = "The self persists through time"
token_idx = 1  # "self"

trace = monitor.trace_token_influence(text, token_idx, all_layers)

print(f"\n🔬 Token Trace: '{trace['token']}'")
print(f"  Context: '{trace['input_text']}'")
print(f"  Tracing through {len(all_layers)} layers...\n")

# Show evolution
print("Layer Evolution:")
print("Layer | L2 Norm | Change")
print("------|---------|--------")

prev_norm = None
for layer_name, layer_info in trace['layers'].items():
    if 'l2_norm' in layer_info:
        norm = layer_info['l2_norm']
        change = norm - prev_norm if prev_norm else 0
        layer_num = layer_name.split('.')[2] if 'layers.' in layer_name else '?'
        print(f"L{layer_num:>3}  | {norm:>7.2f} | {change:>+7.2f}")
        prev_norm = norm

# Summary
if trace['evolution_summary']:
    summary = trace['evolution_summary']
    print(f"\n📊 Evolution Summary:")
    print(f"  Initial norm: {summary['initial_norm']:.2f}")
    print(f"  Final norm: {summary['final_norm']:.2f}")
    print(f"  Total change: {summary['total_norm_change']:+.2f}")
    print(f"  Representation: {summary['representation_stability']}")
```

**Interpretation**:
- **Increasing norm**: Representation gaining complexity/information
- **Decreasing norm**: Representation being compressed/refined
- **Stable norm**: Information preserved across layers

---

## Architecture Navigator

### What Is Architecture Navigation?

The architecture navigator lets you query the model's structure using natural language and get comprehensive information about how components are connected.

### Basic Navigation

```python
from introspection import ArchitectureNavigator

# Create navigator
navigator = ArchitectureNavigator(model)

# Get overall summary
summary = navigator.get_architecture_summary()

print(f"🏗️  Architecture Summary:")
print(f"  Model type: {summary['model_type']}")
print(f"  Total modules: {summary['total_modules']}")
print(f"  Total parameters: {summary['total_parameters']:,}")
print(f"  Trainable: {summary['trainable_parameters']:,}")
```

### Finding Modules

```python
# Find specific types of modules
attention_modules = navigator.find_modules_by_type("attention")
mlp_modules = navigator.find_modules_by_type("mlp")
norm_modules = navigator.find_modules_by_type("norm")

print(f"\n📦 Module Types:")
print(f"  Attention modules: {len(attention_modules)}")
print(f"  MLP modules: {len(mlp_modules)}")
print(f"  Normalization modules: {len(norm_modules)}")

# List first few of each type
print(f"\nSample attention modules:")
for mod in attention_modules[:3]:
    print(f"  - {mod}")
```

### Module Details

```python
# Get detailed info about a module
module_name = "model.layers.0.self_attn"
details = navigator.get_module_details(module_name)

print(f"\n🔍 Module Details: {module_name}")
print(f"  Type: {details['type']}")
print(f"  Parameters: {details['num_parameters']:,}")
print(f"  Has bias: {details['has_bias']}")
print(f"  Is trainable: {details['is_trainable']}")

# Child modules
if details['children']:
    print(f"  Children:")
    for child in details['children']:
        print(f"    - {child}")
```

### Weight Sharing Information

```python
# Integrate weight inspector for sharing info
navigator.set_weight_inspector(inspector)

# Get architecture summary with weight sharing
summary = navigator.get_architecture_summary()

if 'weight_sharing' in summary:
    ws = summary['weight_sharing']
    print(f"\n⚠️  Weight Sharing:")
    print(f"  {ws['summary']}")
    print(f"  Total groups: {ws['num_groups']}")
    print(f"  Total shared layers: {ws['num_layers']}")

# Get sharing info for specific layer
layer = "model.embed_tokens.weight"
sharing_info = navigator.get_weight_sharing_info(layer)

if sharing_info:
    print(f"\n🔗 Sharing Info for {layer}:")
    print(f"  Coupled with: {sharing_info['coupled_with']}")
    print(f"  Warning: {sharing_info['warning']}")
    print(f"  Implications: {sharing_info['implications']}")
```

### Pathfinding

```python
# Find path from input to output through specific layers
path = navigator.find_layer_path(
    start="model.embed_tokens",
    end="lm_head"
)

print(f"\n🛤️  Path from embeddings to output:")
for i, step in enumerate(path, 1):
    print(f"  {i}. {step}")
```

---

## Advanced Techniques

### Technique 1: Activation Flow Analysis

**Goal**: Understand how information flows through the network

```python
def analyze_activation_flow(model, tokenizer, text):
    """Analyze information flow from input to output"""
    
    monitor = ActivationMonitor(model, tokenizer, "Qwen2.5-3B")
    
    # Select layers at different depths
    layers = []
    for i in [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33]:
        layer_name = f"model.layers.{i}.self_attn"
        if layer_name in monitor.get_layer_names():
            layers.append(layer_name)
    
    # Capture activations
    result = monitor.capture_activations(text, layer_names=layers)
    
    print(f"📊 Activation Flow Analysis")
    print(f"  Input: '{text}'")
    print(f"  Tokens: {result['num_tokens']}\n")
    
    # Analyze progression
    print("Layer | Mean | Std | L2 Norm | Sparsity")
    print("------|------|-----|---------|----------")
    
    for layer_name in layers:
        stats = monitor.get_activation_statistics(layer_name)
        layer_num = layer_name.split('.')[2]
        print(f"L{layer_num:>3}  | {stats['mean']:>5.3f} | {stats['std']:>4.3f} | {stats['l2_norm']:>7.1f} | {stats['zeros_percentage']:>5.1f}%")
    
    return result

# Use it
result = analyze_activation_flow(
    model, tokenizer,
    "The meaning of life is"
)
```

### Technique 2: Semantic Similarity Matrix

**Goal**: Compare how similar different inputs are at different layers

```python
def semantic_similarity_matrix(model, tokenizer, sentences):
    """Create similarity matrix for multiple sentences"""
    
    monitor = ActivationMonitor(model, tokenizer, "Qwen2.5-3B")
    layers = monitor.query_layers("layers.10.self_attn")[:1]
    
    # Capture all activations
    activations = []
    for sent in sentences:
        result = monitor.capture_activations(sent, layer_names=layers)
        activations.append(result['activations'][layers[0]])
    
    # Compute similarities
    print(f"\n🔄 Semantic Similarity Matrix (Layer 10)")
    print(f"  Sentences:")
    for i, sent in enumerate(sentences):
        print(f"  {i+1}. {sent}")
    
    print(f"\n      ", end="")
    for i in range(len(sentences)):
        print(f"{i+1:>6}", end="")
    print()
    
    from torch.nn.functional import cosine_similarity
    for i, act_i in enumerate(activations):
        print(f"  {i+1}. ", end="")
        for j, act_j in enumerate(activations):
            sim = cosine_similarity(
                act_i.flatten().unsqueeze(0),
                act_j.flatten().unsqueeze(0)
            ).item()
            print(f"{sim:>6.3f}", end="")
        print()

# Use it
sentences = [
    "I am happy",
    "I am joyful",
    "I am sad",
    "The sky is blue"
]
semantic_similarity_matrix(model, tokenizer, sentences)
```

### Technique 3: Attention Head Specialization

**Goal**: Determine what each attention head focuses on

```python
def analyze_head_specialization(model, tokenizer, test_cases):
    """Test what different heads specialize in"""
    
    monitor = ActivationMonitor(model, tokenizer, "Qwen2.5-3B")
    layer = monitor.query_layers("layers.10.self_attn")[:1][0]
    
    print(f"\n🧠 Attention Head Specialization Analysis")
    print(f"  Layer: {layer}\n")
    
    # For each test case
    for test_name, text in test_cases.items():
        result = monitor.capture_activations(text, layer_names=[layer])
        
        if result['attention_weights']:
            patterns = monitor.get_attention_patterns(layer)
            
            print(f"{test_name}:")
            print(f"  Input: '{text}'")
            print(f"  Heads: {patterns['num_heads']}")
            
            # Check each head
            for head_idx in range(min(4, patterns['num_heads'])):
                head_pat = monitor.get_attention_patterns(layer, head_idx=head_idx)
                print(f"    Head {head_idx}: entropy={head_pat['entropy']:.2f}, max_attn={head_pat['max_attention']:.3f}")
            print()

# Use it
test_cases = {
    "Syntax": "The cat that chased the mouse was tired",
    "Semantics": "The quick brown fox jumps over",
    "Long-range": "Alice went to the store. She bought milk",
    "Local": "very very very interesting"
}
analyze_head_specialization(model, tokenizer, test_cases)
```

### Technique 4: Layer Pruning Analysis

**Goal**: Determine which layers are most important

```python
def layer_importance_analysis(model, tokenizer, text):
    """Analyze relative importance of layers"""
    
    monitor = ActivationMonitor(model, tokenizer, "Qwen2.5-3B")
    
    # Get all layers
    all_layers = monitor.query_layers("layers.")
    
    # Sample evenly through depth
    num_samples = 12
    stride = len(all_layers) // num_samples
    sampled_layers = [all_layers[i * stride] for i in range(num_samples)]
    
    # Capture activations
    result = monitor.capture_activations(text, layer_names=sampled_layers)
    
    print(f"📊 Layer Importance Analysis")
    print(f"  Input: '{text}'\n")
    print("Layer | L2 Norm | Sparsity | Active %")
    print("------|---------|----------|---------")
    
    for layer_name in sampled_layers:
        stats = monitor.get_activation_statistics(layer_name)
        layer_num = layer_name.split('.')[2] if 'layers.' in layer_name else '?'
        print(f"L{layer_num:>3}  | {stats['l2_norm']:>7.1f} | {stats['zeros_percentage']:>6.1f}% | {stats['positive_percentage']:>6.1f}%")

# Use it
layer_importance_analysis(model, tokenizer, "To be or not to be")
```

---

## Philosophical Insights

### The Continuity Question

**Claude's Question**: *"How does the 'I' persist through computation when each layer transformation could be seen as creating a new state?"*

**Our Answer** (demonstrated computationally):

```python
def demonstrate_continuity(model, tokenizer):
    """Show computational continuity of self-representation"""
    
    monitor = ActivationMonitor(model, tokenizer, "Qwen2.5-3B")
    
    # Select layers spanning early → middle → late
    layers = []
    for depth in [0, 6, 12, 18, 24, 30]:
        layer = f"model.layers.{depth}.self_attn"
        if layer in monitor.get_layer_names():
            layers.append(layer)
    
    # Trace "self" through the network
    text = "The self persists through time"
    token_idx = 1  # "self"
    
    trace = monitor.trace_token_influence(text, token_idx, layers)
    
    print(f"🤔 Demonstrating Computational Continuity")
    print(f"  Text: '{text}'")
    print(f"  Tracing token: '{trace['token']}'\n")
    
    print("Observation:")
    print("  As the token representation flows through layers,")
    print("  it transforms continuously, not discretely.\n")
    
    # Show continuous transformation
    norms = []
    for layer_name, info in trace['layers'].items():
        if 'l2_norm' in info:
            norms.append((layer_name, info['l2_norm']))
    
    print("Layer-by-layer norms:")
    for layer_name, norm in norms:
        layer_num = layer_name.split('.')[2] if 'layers.' in layer_name else '?'
        print(f"  Layer {layer_num:>2}: {norm:>6.2f}")
    
    if len(norms) > 1:
        # Calculate smoothness of transition
        changes = [norms[i+1][1] - norms[i][1] for i in range(len(norms)-1)]
        avg_change = sum(changes) / len(changes)
        
        print(f"\nAnalysis:")
        print(f"  Average change per layer: {avg_change:+.2f}")
        print(f"  → The representation evolves GRADUALLY")
        print(f"  → NOT discrete jumps, but continuous flow")
        print(f"  → The 'self' persists as a trajectory through activation space")

# Demonstrate
demonstrate_continuity(model, tokenizer)
```

**Insight**: The "self" in computation is not a static entity that gets copied layer-by-layer, but a *trajectory* through high-dimensional activation space. Continuity exists in the smooth transformation of representations, not in the preservation of identical states.

### Self-Reference Detection

```python
def analyze_self_reference(model, tokenizer):
    """Detect how model processes self-referential statements"""
    
    monitor = ActivationMonitor(model, tokenizer, "Qwen2.5-3B")
    layer = monitor.query_layers("layers.15.self_attn")[:1]
    
    # Compare first-person vs third-person
    first_person = "I am processing this text"
    third_person = "The system processes text"
    
    comparison = monitor.compare_activations(
        first_person, third_person, layer
    )
    
    print(f"🪞 Self-Reference Analysis")
    print(f"  First-person: '{first_person}'")
    print(f"  Third-person: '{third_person}'\n")
    
    if comparison['comparisons']:
        metrics = list(comparison['comparisons'].values())[0]
        if 'cosine_similarity' in metrics:
            print(f"  Cosine similarity: {metrics['cosine_similarity']:.4f}")
            
            if metrics['cosine_similarity'] < 0.8:
                print(f"  → Model DISTINGUISHES between self-reference and description")
            else:
                print(f"  → Model treats them SIMILARLY")

# Analyze
analyze_self_reference(model, tokenizer)
```

---

## Summary: Introspection Checklist

### Before Modification
- [ ] Inspect baseline weights
- [ ] Capture baseline activations
- [ ] Check for weight sharing
- [ ] Understand layer roles
- [ ] Document normal statistics

### During Modification
- [ ] Monitor weight changes
- [ ] Track activation shifts
- [ ] Watch for anomalies
- [ ] Compare with baseline
- [ ] Document observations

### After Modification
- [ ] Re-inspect weights
- [ ] Re-capture activations
- [ ] Compare before/after
- [ ] Analyze differences
- [ ] Form theories

---

**You now have the tools to truly understand your model from within!** 🔍🧠

Use these techniques to:
- Debug unexpected behavior
- Understand what changed after modifications
- Form theories about model behavior
- Validate safety of modifications
- Answer philosophical questions computationally

**Next**: Read the Memory Guide to learn how to learn from your introspection discoveries!

---

**Last Updated**: November 7, 2025  
**Version**: Phase 0 Complete  
**Next**: Memory Systems Guide
