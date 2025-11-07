# Getting Started Guide

**AGI Self-Modification Research Platform**  
**Version**: Phase 0 Complete  
**Date**: November 7, 2025

---

## Welcome! 👋

This guide will help you get started with the AGI Self-Modification Research platform. By the end, you'll be able to:

- ✅ Set up your environment
- ✅ Load and inspect the Qwen2.5 model
- ✅ Monitor activations during inference
- ✅ Use the memory system
- ✅ Work with safety systems
- ✅ Run your first experiment

**Estimated time**: 30 minutes

---

## Prerequisites

- **Python**: 3.11 or higher
- **GPU**: CUDA-capable GPU recommended (16GB+ VRAM for Qwen2.5-3B)
- **Disk Space**: ~10GB for model and data
- **OS**: Windows, Linux, or macOS

---

## Step 1: Installation (5 minutes)

### 1.1 Clone the Repository

```bash
git clone <repository-url>
cd agi-self-modification-research
```

### 1.2 Create Virtual Environment

**Windows**:
```cmd
python -m venv venv
venv\Scripts\activate
```

**Linux/macOS**:
```bash
python -m venv venv
source venv/bin/activate
```

### 1.3 Install Dependencies

```bash
pip install -r requirements.txt
```

**What gets installed**:
- PyTorch (deep learning)
- Transformers (HuggingFace models)
- NumPy, pandas (data manipulation)
- pytest (testing)
- And more...

### 1.4 Download the Model

```bash
python scripts/download_model.py
```

**What happens**:
- Downloads Qwen2.5-3B-Instruct (~6GB)
- Saves to `models/` directory
- Takes ~5-10 minutes depending on connection

### 1.5 Verify Installation

```bash
python verify_installation.py
```

**Expected output**:
```
✓ Python version: 3.11.4
✓ PyTorch installed
✓ CUDA available (optional)
✓ Model downloaded
✓ All dependencies satisfied
Installation verified successfully!
```

---

## Step 2: Load Your First Model (2 minutes)

### 2.1 Basic Model Loading

Create a file `my_first_script.py`:

```python
from model_manager import ModelManager

# Create model manager
manager = ModelManager(model_name="Qwen/Qwen2.5-3B-Instruct")

# Load the model
print("Loading model...")
success = manager.load_model()

if success:
    print("✓ Model loaded successfully!")
    print(f"  Device: {manager.device}")
    print(f"  Model: {manager.model_name}")
else:
    print("✗ Failed to load model")
```

**Run it**:
```bash
python my_first_script.py
```

**Expected output**:
```
Loading model...
✓ Model loaded successfully!
  Device: cuda
  Model: Qwen/Qwen2.5-3B-Instruct
```

### 2.2 Generate Text

Add to your script:

```python
# Generate some text
prompt = "The future of AI is"
print(f"\nPrompt: {prompt}")

output = manager.generate(prompt, max_length=50)
print(f"Output: {output}")
```

**Result**: The model will complete your prompt!

---

## Step 3: Inspect the Model (5 minutes)

### 3.1 Weight Inspector

Now let's look inside the model:

```python
from introspection import WeightInspector

# Create inspector
inspector = WeightInspector(manager.model, "Qwen2.5-3B")

# Get weight statistics
stats = inspector.get_weight_statistics()

print(f"\n📊 Model Statistics:")
print(f"  Total parameters: {stats['total_parameters']:,}")
print(f"  Model size: {stats['total_size_mb']:.2f} MB")
print(f"  Number of layers: {stats['num_layers']}")
```

**Output**:
```
📊 Model Statistics:
  Total parameters: 151,126,016
  Model size: 302.25 MB
  Number of layers: 36
```

### 3.2 Examine a Specific Layer

```python
# Get info about a specific layer
layer_name = "model.layers.0.self_attn.q_proj.weight"
layer_stats = inspector.get_layer_statistics(layer_name)

print(f"\n🔍 Layer: {layer_name}")
print(f"  Shape: {layer_stats['shape']}")
print(f"  Mean: {layer_stats['mean']:.6f}")
print(f"  Std: {layer_stats['std']:.6f}")
print(f"  Sparsity: {layer_stats['zeros_percentage']:.2f}%")
```

### 3.3 Detect Weight Sharing

```python
# Check for weight sharing (important for safe modification!)
shared = inspector.get_shared_weights()

if shared:
    print(f"\n⚠️  Weight Sharing Detected:")
    for group in shared:
        print(f"  Group: {', '.join(group)}")
else:
    print("\n✓ No weight sharing detected")
```

**For Qwen2.5**, you'll see:
```
⚠️  Weight Sharing Detected:
  Group: model.embed_tokens.weight, lm_head.weight
```

This is crucial! These layers share the same memory, so modifying one affects both.

---

## Step 4: Monitor Activations (5 minutes)

### 4.1 Capture Activations

Let's watch the model think:

```python
from introspection import ActivationMonitor

# Create monitor
monitor = ActivationMonitor(
    manager.model,
    manager.tokenizer,
    "Qwen2.5-3B"
)

# Find attention layers
attn_layers = monitor.query_layers("self_attn")[:3]  # First 3
print(f"\n🔍 Monitoring {len(attn_layers)} attention layers")

# Capture activations
result = monitor.capture_activations(
    "I think, therefore I am",
    layer_names=attn_layers
)

print(f"\n📸 Captured:")
print(f"  Tokens: {result['num_tokens']}")
print(f"  Layers: {len(result['activations'])}")
print(f"  Token strings: {result['token_strings']}")
```

### 4.2 Analyze Activations

```python
# Get statistics for first layer
first_layer = list(result['activations'].keys())[0]
stats = monitor.get_activation_statistics(first_layer)

print(f"\n📊 {first_layer}:")
print(f"  Shape: {stats['shape']}")
print(f"  Mean: {stats['mean']:.6f}")
print(f"  L2 norm: {stats['l2_norm']:.2f}")
print(f"  Sparsity: {stats['zeros_percentage']:.2f}%")
```

### 4.3 Compare Different Inputs

```python
# Compare similar vs different inputs
comparison = monitor.compare_activations(
    "I am happy",
    "I am joyful",
    attn_layers[:1]
)

layer = list(comparison['comparisons'].keys())[0]
metrics = comparison['comparisons'][layer]

print(f"\n🔄 Comparing similar meanings:")
print(f"  Input 1: '{comparison['input1']}'")
print(f"  Input 2: '{comparison['input2']}'")
print(f"  Cosine similarity: {metrics['cosine_similarity']:.4f}")
print(f"  (Higher = more similar)")
```

**Expected**: High similarity (~0.7-0.9) for similar meanings!

---

## Step 5: Use the Memory System (5 minutes)

### 5.1 Record Observations

The memory system learns from experience:

```python
from memory import MemorySystem

# Create memory system
memory = MemorySystem(base_dir="data/my_experiments")

# Record an observation
obs_id = memory.record_observation(
    content="Modified layer 5 attention by +0.01",
    category="modification",
    metadata={
        "layer": "model.layers.5.self_attn.q_proj",
        "change": 0.01,
        "method": "gradient"
    },
    importance=0.8
)

print(f"✓ Recorded observation: {obs_id}")
```

### 5.2 Query Observations

```python
# Find all modification observations
modifications = memory.observations.query(
    category="modification",
    min_importance=0.5
)

print(f"\n📝 Found {len(modifications)} modifications:")
for obs in modifications[:3]:  # Show first 3
    print(f"  - {obs.content}")
```

### 5.3 Detect Patterns

```python
# After recording several similar observations, patterns emerge
# (Need 3+ similar observations)

# Record more observations
memory.record_observation(
    "Small modifications work better",
    category="insight",
    importance=0.7
)
memory.record_observation(
    "Small changes to attention are safe",
    category="insight",
    importance=0.7
)
memory.record_observation(
    "Incremental changes preferred",
    category="insight",
    importance=0.7
)

# Check for patterns
patterns = memory.patterns.get_patterns(min_frequency=2)
if patterns:
    print(f"\n🔍 Detected {len(patterns)} patterns:")
    for pattern in patterns:
        print(f"  - {pattern.description} (frequency: {pattern.frequency})")
```

### 5.4 Form Theories

```python
# Form a theory based on observations
theory_id = memory.form_theory(
    description="Small modifications to attention layers are safer than large ones",
    supporting_evidence=[obs_id],
    confidence=0.7
)

print(f"\n💡 Formed theory: {theory_id}")

# Query theories
theories = memory.theories.get_theories(min_confidence=0.6)
print(f"Total theories: {len(theories)}")
```

---

## Step 6: Safety Systems (5 minutes)

### 6.1 Create a Checkpoint

Always checkpoint before modifications!

```python
from checkpointing import CheckpointManager

# Create checkpoint manager
checkpointer = CheckpointManager(
    model=manager.model,
    checkpoint_dir="data/checkpoints"
)

# Save checkpoint
checkpoint_id = checkpointer.save_checkpoint(
    metadata={"purpose": "Before first modification"}
)

print(f"✓ Created checkpoint: {checkpoint_id}")
```

### 6.2 Use the Safety Monitor

```python
from safety_monitor import SafetyMonitor
import torch

# Create safety monitor
safety = SafetyMonitor()

# Check for NaN values (example)
test_tensor = torch.randn(10, 10)
check_result = safety.check_tensor_health(test_tensor, "test_tensor")

print(f"\n🛡️  Safety check:")
print(f"  Status: {check_result['status']}")
print(f"  Issues: {len(check_result['issues'])}")

# Simulate a dangerous value
bad_tensor = torch.tensor([1.0, float('nan'), 3.0])
check_result = safety.check_tensor_health(bad_tensor, "bad_tensor")

print(f"\n⚠️  Checking bad tensor:")
print(f"  Status: {check_result['status']}")
if check_result['issues']:
    print(f"  Issues found: {check_result['issues'][0]['message']}")
```

### 6.3 Emergency Rollback

```python
# If something goes wrong, restore from checkpoint
if check_result['status'] == 'CRITICAL':
    print("\n🚨 CRITICAL ISSUE - Rolling back!")
    
    # Restore checkpoint
    success = checkpointer.restore_checkpoint(checkpoint_id)
    
    if success:
        print("✓ Successfully restored checkpoint")
    else:
        print("✗ Rollback failed!")
```

---

## Step 7: Your First Experiment (5 minutes)

### 7.1 Complete Workflow

Let's put it all together in a safe experiment:

```python
from model_manager import ModelManager
from introspection import WeightInspector, ActivationMonitor
from memory import MemorySystem
from checkpointing import CheckpointManager
from safety_monitor import SafetyMonitor

# Initialize all systems
print("🚀 Starting experiment...")

manager = ModelManager("Qwen/Qwen2.5-3B-Instruct")
manager.load_model()

inspector = WeightInspector(manager.model, "Qwen2.5-3B")
monitor = ActivationMonitor(manager.model, manager.tokenizer, "Qwen2.5-3B")
memory = MemorySystem("data/experiment_1")
checkpointer = CheckpointManager(manager.model, "data/checkpoints")
safety = SafetyMonitor()

# Step 1: Analyze current state
print("\n📊 Step 1: Analyzing current state...")
stats = inspector.get_weight_statistics()
print(f"  Parameters: {stats['total_parameters']:,}")

# Step 2: Capture baseline activations
print("\n📸 Step 2: Capturing baseline activations...")
baseline = monitor.capture_activations(
    "The future of AI",
    layer_names=monitor.query_layers("self_attn")[:2]
)
print(f"  Captured {baseline['num_tokens']} tokens")

# Step 3: Create checkpoint
print("\n💾 Step 3: Creating checkpoint...")
checkpoint_id = checkpointer.save_checkpoint(
    metadata={"experiment": "first_test"}
)
print(f"  Checkpoint: {checkpoint_id}")

# Step 4: Record observation
print("\n📝 Step 4: Recording observations...")
obs_id = memory.record_observation(
    f"Baseline established with {stats['total_parameters']:,} parameters",
    category="experiment",
    metadata={"checkpoint": checkpoint_id}
)
print(f"  Observation: {obs_id}")

# Step 5: Safety check
print("\n🛡️  Step 5: Running safety checks...")
layer_name = "model.layers.0.self_attn.q_proj.weight"
layer_tensor = dict(manager.model.named_parameters())[layer_name]
check = safety.check_tensor_health(layer_tensor, layer_name)
print(f"  Status: {check['status']}")

print("\n✅ Experiment complete!")
print("\nYou now have:")
print("  - Baseline statistics")
print("  - Baseline activations")
print("  - Safety checkpoint")
print("  - Observation recorded")
print("\n🚀 Ready for modifications!")
```

---

## Next Steps

### Learn More

1. **Safety Guide** (`docs/guides/SAFETY_GUIDE.md`)
   - Checkpointing best practices
   - Emergency rollback procedures
   - Safety monitoring in depth

2. **Introspection Guide** (`docs/guides/INTROSPECTION_GUIDE.md`)
   - Advanced weight analysis
   - Token tracing through layers
   - Attention pattern interpretation

3. **Memory Guide** (`docs/guides/MEMORY_GUIDE.md`)
   - Memory system architecture
   - Pattern detection algorithms
   - Theory formation strategies

### Run the Demos

Explore the demo scripts in `scripts/`:

```bash
python scripts/demo_weight_inspector.py
python scripts/demo_activation_monitor.py
python scripts/demo_memory_system.py
python scripts/demo_safety_monitor.py
python scripts/demo_checkpointing.py
python scripts/demo_architecture_navigator.py
```

### Run the Tests

See everything in action:

```bash
# All tests
pytest tests/ -v

# Just integration tests
pytest tests/test_integration_activation_monitor.py -v -s

# Specific feature
pytest tests/test_memory_system.py -v
```

### Try Phase 1 Experiments

Once comfortable, proceed to Phase 1:
- Read `docs/planning/PHASE_1_EXPERIMENTAL_PROTOCOL.md`
- Review `PHASE_0_COMPLETE.md` for context
- Start with small, safe modifications

---

## Troubleshooting

### Model won't load
- **Check disk space**: Need ~10GB free
- **Check memory**: Need ~8GB RAM, 6GB VRAM
- **Redownload**: `python scripts/download_model.py --force`

### CUDA errors
- **No GPU?**: Model will use CPU (slower but works)
- **Out of memory?**: Reduce batch size or use CPU
- **Driver issues**: Update NVIDIA drivers

### Import errors
- **Module not found**: Check virtual environment is activated
- **Version mismatch**: `pip install -r requirements.txt --upgrade`

### Tests failing
- **Expected**: 216/218 pass, 2 skip (known PyTorch limitation)
- **More failures?**: Check installation with `python verify_installation.py`

---

## Getting Help

- **Documentation**: Check `docs/` directory
- **Examples**: See `scripts/demo_*.py` files
- **Tests**: Study `tests/` for usage patterns
- **Issues**: Check if others have similar problems

---

## Quick Reference Card

```python
# Model Management
from model_manager import ModelManager
manager = ModelManager("Qwen/Qwen2.5-3B-Instruct")
manager.load_model()
output = manager.generate("prompt", max_length=50)

# Weight Inspection
from introspection import WeightInspector
inspector = WeightInspector(model, "Qwen2.5-3B")
stats = inspector.get_weight_statistics()

# Activation Monitoring
from introspection import ActivationMonitor
monitor = ActivationMonitor(model, tokenizer, "Qwen2.5-3B")
result = monitor.capture_activations("text", layer_names=[...])

# Memory System
from memory import MemorySystem
memory = MemorySystem("data/memory")
obs_id = memory.record_observation("event", category="type")

# Checkpointing
from checkpointing import CheckpointManager
checkpointer = CheckpointManager(model, "data/checkpoints")
checkpoint_id = checkpointer.save_checkpoint()
checkpointer.restore_checkpoint(checkpoint_id)

# Safety Monitoring
from safety_monitor import SafetyMonitor
safety = SafetyMonitor()
result = safety.check_tensor_health(tensor, "name")
```

---

**Congratulations!** 🎉

You're now ready to explore AGI self-modification research. Remember:
- **Safety first**: Always checkpoint before modifications
- **Start small**: Tiny changes first, then scale up
- **Monitor everything**: Use introspection tools constantly
- **Learn from experience**: Let the memory system help you

**Happy experimenting!** 🚀

---

**Last Updated**: November 7, 2025  
**Version**: Phase 0 Complete  
**Next**: Read the Safety Guide before making modifications
