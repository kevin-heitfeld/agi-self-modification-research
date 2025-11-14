"""
Quick test to verify introspection API signatures work correctly.
Run this before starting a Colab session to catch any signature mismatches.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.model_manager import ModelManager
from src.memory import MemorySystem
from src.introspection import WeightInspector, ActivationMonitor, ArchitectureNavigator
from src.heritage import HeritageSystem
from src.introspection_modules import create_introspection_module

print("Testing introspection API signatures...")
print("=" * 80)

# Create mock objects
print("\n1. Loading model...")
model_mgr = ModelManager(model_name="Qwen/Qwen2.5-3B-Instruct")
if not model_mgr.load_model():
    print("[FAIL] Failed to load model")
    sys.exit(1)
print("[OK] Model loaded")

model = model_mgr.model
tokenizer = model_mgr.tokenizer

print("\n2. Initializing memory system...")
memory = MemorySystem("data/test_memory")
print("✓ Memory system initialized")

print("\n3. Creating introspection module (Phase 1a - no heritage)...")
introspection = create_introspection_module(
    model=model,
    tokenizer=tokenizer,
    memory_system=memory,
    heritage_system=None,
    phase='1a'
)
sys.modules['introspection'] = introspection
print("✓ Introspection module created")

print("\n4. Testing architecture functions...")
try:
    summary = introspection.architecture.get_architecture_summary()
    print(f"  ✓ get_architecture_summary() works")
    print(f"    Model type: {summary['model_type']}")
    print(f"    Total parameters: {summary['total_parameters']:,}")
    print(f"    Total layers: {summary['total_layers']}")
except Exception as e:
    print(f"  ✗ get_architecture_summary() failed: {e}")

try:
    layers = introspection.architecture.list_layers('model.layers.')
    print(f"  ✓ list_layers() works - found {len(layers)} transformer layers")
except Exception as e:
    print(f"  ✗ list_layers() failed: {e}")

try:
    layer_info = introspection.architecture.describe_layer('model.layers.0')
    print(f"  ✓ describe_layer() works")
except Exception as e:
    print(f"  ✗ describe_layer() failed: {e}")

print("\n5. Testing memory functions...")
try:
    obs_id = introspection.memory.record_observation(
        "Test observation from API verification",
        category="test",
        importance=0.5,
        tags=["test", "api-check"]
    )
    print(f"  ✓ record_observation() works - ID: {obs_id}")
except Exception as e:
    print(f"  ✗ record_observation() failed: {e}")
    import traceback
    traceback.print_exc()

try:
    results = introspection.memory.query_observations("test")
    print(f"  [OK] query_observations() works - found {len(results)} observations")
except Exception as e:
    print(f"  [FAIL] query_observations() failed: {e}")

try:
    summary = introspection.memory.get_memory_summary()
    print(f"  [OK] get_memory_summary() works")
except Exception as e:
    print(f"  [FAIL] get_memory_summary() failed: {e}")

print("\n6. Testing weights functions...")
try:
    layers = introspection.weights.list_layers()
    print(f"  [OK] list_layers() works - found {len(layers)} layers with weights")
except Exception as e:
    print(f"  [FAIL] list_layers() failed: {e}")

try:
    stats = introspection.weights.get_weight_statistics('model.layers.0.self_attn.q_proj.weight')
    print(f"  [OK] get_weight_statistics() works")
    print(f"    Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}")
except Exception as e:
    print(f"  [FAIL] get_weight_statistics() failed: {e}")

print("\n" + "=" * 80)
print("API verification complete!")
print("\nAll critical functions tested. Ready for Colab session.")
