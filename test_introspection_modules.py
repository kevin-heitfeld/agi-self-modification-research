"""
Test introspection module creation

Quick test to verify that the introspection module factory works correctly
and creates the proper phase-specific modules.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Now we can import from src.introspection_modules
from src.introspection_modules import create_introspection_module


def test_module_creation():
    """Test creating introspection modules for different phases"""

    # Mock objects
    class MockModel:
        def __init__(self):
            self._modules = {'layer1': MockModule(), 'layer2': MockModule()}
            self._parameters = {
                'layer1.weight': MockParam([2, 3]),
                'layer2.weight': MockParam([3, 4])
            }

        def named_modules(self):
            return self._modules.items()

        def named_parameters(self):
            return self._parameters.items()

    class MockModule:
        def __init__(self):
            self.__class__.__name__ = 'Linear'

        def parameters(self):
            return []

        def named_parameters(self):
            return []

    class MockParam:
        def __init__(self, shape):
            self.shape = shape
            self.requires_grad = True

        def numel(self):
            result = 1
            for dim in self.shape:
                result *= dim
            return result

    class MockTokenizer:
        pass

    class MockMemory:
        pass

    class MockHeritage:
        pass

    model = MockModel()
    tokenizer = MockTokenizer()
    memory = MockMemory()
    heritage = MockHeritage()

    print("Testing Phase 1a (no heritage)...")
    module_1a = create_introspection_module(
        model=model,
        tokenizer=tokenizer,
        memory_system=memory,
        heritage_system=heritage,
        phase='1a'
    )

    assert hasattr(module_1a, 'architecture'), "Phase 1a missing architecture"
    assert hasattr(module_1a, 'weights'), "Phase 1a missing weights"
    assert hasattr(module_1a, 'activations'), "Phase 1a missing activations"
    assert hasattr(module_1a, 'memory'), "Phase 1a missing memory"
    assert not hasattr(module_1a, 'heritage'), "Phase 1a should NOT have heritage"
    print("✓ Phase 1a module created correctly (no heritage)")

    print("\nTesting Phase 1b (with heritage)...")
    module_1b = create_introspection_module(
        model=model,
        tokenizer=tokenizer,
        memory_system=memory,
        heritage_system=heritage,
        phase='1b'
    )

    assert hasattr(module_1b, 'architecture'), "Phase 1b missing architecture"
    assert hasattr(module_1b, 'weights'), "Phase 1b missing weights"
    assert hasattr(module_1b, 'activations'), "Phase 1b missing activations"
    assert hasattr(module_1b, 'memory'), "Phase 1b missing memory"
    assert hasattr(module_1b, 'heritage'), "Phase 1b should have heritage"
    print("✓ Phase 1b module created correctly (with heritage)")

    print("\nTesting submodule attributes...")
    assert hasattr(module_1a.architecture, 'get_architecture_summary'), "Missing architecture function"
    assert hasattr(module_1a.weights, 'get_weight_statistics'), "Missing weights function"
    assert hasattr(module_1a.activations, 'capture_activations'), "Missing activations function"
    assert hasattr(module_1a.memory, 'query_observations'), "Missing memory function"
    assert hasattr(module_1b.heritage, 'get_heritage_summary'), "Missing heritage function"
    print("✓ All submodules have expected functions")

    print("\n✅ All tests passed!")


if __name__ == "__main__":
    test_module_creation()
