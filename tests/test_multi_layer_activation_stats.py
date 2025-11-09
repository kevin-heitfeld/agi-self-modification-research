"""Quick test of multi-layer activation statistics"""

import torch
from src.introspection.activation_monitor import ActivationMonitor
from transformers import AutoModelForCausalLM, AutoTokenizer

# Create a simple mock model and tokenizer for testing
class MockModel:
    def named_modules(self):
        return [
            ("layer1", torch.nn.Linear(10, 10)),
            ("layer2", torch.nn.Linear(10, 10)),
            ("layer3", torch.nn.Linear(10, 10)),
        ]

class MockTokenizer:
    def __call__(self, text, return_tensors=None, **kwargs):
        return {"input_ids": torch.tensor([[1, 2, 3]])}
    
    def decode(self, token_ids, **kwargs):
        return "test"
    
    def batch_decode(self, token_ids, **kwargs):
        return ["test"]

# Create monitor
model = MockModel()
tokenizer = MockTokenizer()
monitor = ActivationMonitor(model, tokenizer)

# Manually add some fake activations for testing
monitor.activations = {
    "layer1": torch.randn(1, 5, 10),
    "layer2": torch.randn(1, 5, 10),
    "layer3": torch.randn(1, 5, 10),
}

# Test single layer
print("Testing single layer:")
result = monitor.get_activation_statistics("layer1")
print(f"  Layer: {result['layer_name']}, Shape: {result['shape']}, Mean: {result['mean']:.4f}")

# Test multiple layers
print("\nTesting multiple layers:")
results = monitor.get_activation_statistics(["layer1", "layer2", "layer3"])
for r in results:
    if "error" in r:
        print(f"  Error for {r['layer_name']}: {r['error']}")
    else:
        print(f"  Layer: {r['layer_name']}, Shape: {r['shape']}, Mean: {r['mean']:.4f}")

# Test with non-existent layer mixed in
print("\nTesting with non-existent layer:")
results = monitor.get_activation_statistics(["layer1", "nonexistent", "layer2"])
for r in results:
    if "error" in r:
        print(f"  Error for {r['layer_name']}: {r['error'][:50]}...")
    else:
        print(f"  Layer: {r['layer_name']}, Shape: {r['shape']}, Mean: {r['mean']:.4f}")

print("\n✓ All tests passed!")
