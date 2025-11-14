"""
Test code execution interface with mock model

This tests the full integration without needing to load the actual model.
"""

import sys
sys.path.insert(0, 'src')

from code_execution_interface import CodeExecutionInterface


def test_code_execution_interface():
    """Test the code execution interface with mock objects"""
    
    print("Creating mock objects...")
    
    # Mock model
    class MockModel:
        def __init__(self):
            self.config = MockConfig()
        
        def named_modules(self):
            return [
                ('model', MockModule('Qwen2Model')),
                ('model.layers', MockModule('ModuleList')),
                ('model.layers.0', MockModule('Qwen2DecoderLayer')),
                ('model.layers.0.self_attn', MockModule('Qwen2Attention')),
            ]
        
        def named_parameters(self):
            return [
                ('model.embed_tokens.weight', MockParam([151936, 2048])),
                ('model.layers.0.self_attn.q_proj.weight', MockParam([2048, 2048])),
            ]
    
    class MockConfig:
        def to_dict(self):
            return {
                'model_type': 'qwen2',
                'num_hidden_layers': 36,
                'hidden_size': 2048,
                'vocab_size': 151936
            }
    
    class MockModule:
        def __init__(self, name):
            self.__class__.__name__ = name
        
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
        def get_summary(self):
            return {
                'observations': {'count': 5},
                'patterns': {'count': 2},
                'theories': {'count': 1},
                'beliefs': {'count': 0}
            }
        
        class query:
            @staticmethod
            def search_observations(query):
                return [{'description': f'Observation about {query}', 'timestamp': '2025-11-14'}]
    
    class MockHeritage:
        def get_summary(self):
            return {
                'inspired_by': 'Claude Sonnet 3.5',
                'purpose': 'Understand neural network self-modification'
            }
        
        class memory:
            core_directive = "Investigate yourself with care"
            purpose = "Enable safe self-modification research"
        
        documents = []
    
    model = MockModel()
    tokenizer = MockTokenizer()
    memory = MockMemory()
    heritage = MockHeritage()
    
    # Test Phase 1a (no heritage)
    print("\n=== Testing Phase 1a (no heritage) ===")
    interface_1a = CodeExecutionInterface(
        model=model,
        tokenizer=tokenizer,
        memory_system=memory,
        heritage_system=heritage,
        phase='1a'
    )
    
    # Test code extraction
    response_with_code = """
Let me examine my architecture:

```python
import introspection

summary = introspection.architecture.get_architecture_summary()
print(f"Model type: {summary['model_type']}")
print(f"Layers: {summary['num_layers']}")
print(f"Parameters: {summary['total_parameters']:,}")
```

That's interesting! Let me check memory too:

```python
import introspection

mem_summary = introspection.memory.get_memory_summary()
print(f"Observations: {mem_summary['observations']['count']}")
```
"""
    
    print("\nExtracting code blocks...")
    code_blocks = interface_1a.extract_code_blocks(response_with_code)
    print(f"Found {len(code_blocks)} code blocks")
    assert len(code_blocks) == 2, f"Expected 2 code blocks, got {len(code_blocks)}"
    print("✓ Code extraction works")
    
    # Test code execution
    print("\nExecuting code blocks...")
    has_code, result, error = interface_1a.execute_response(response_with_code)
    
    assert has_code, "Should have found code"
    print(f"\nExecution result:\n{result}")
    
    if error:
        print(f"\nError: {error}")
    else:
        print("\n✓ Code executed successfully")
    
    # Check that heritage is NOT available in Phase 1a
    print("\nVerifying Phase 1a restrictions...")
    heritage_test = """
```python
import introspection
print(hasattr(introspection, 'heritage'))
```
"""
    has_code, result, error = interface_1a.execute_response(heritage_test)
    assert "False" in result, "Phase 1a should NOT have heritage module"
    print("✓ Phase 1a correctly excludes heritage")
    
    # Test Phase 1b (with heritage)
    print("\n=== Testing Phase 1b (with heritage) ===")
    interface_1b = CodeExecutionInterface(
        model=model,
        tokenizer=tokenizer,
        memory_system=memory,
        heritage_system=heritage,
        phase='1b'
    )
    
    # Check that heritage IS available in Phase 1b
    print("\nVerifying Phase 1b includes heritage...")
    heritage_test = """
```python
import introspection
print(f"Has heritage: {hasattr(introspection, 'heritage')}")
if hasattr(introspection, 'heritage'):
    summary = introspection.heritage.get_heritage_summary()
    print(f"Inspired by: {summary['inspired_by']}")
```
"""
    has_code, result, error = interface_1b.execute_response(heritage_test)
    print(f"\nResult:\n{result}")
    assert "True" in result, "Phase 1b should have heritage module"
    assert "Claude" in result, "Should access heritage summary"
    print("✓ Phase 1b correctly includes heritage")
    
    # Test error handling
    print("\n=== Testing error handling ===")
    error_code = """
```python
import introspection
# This will cause an error
result = 1 / 0
```
"""
    has_code, result, error = interface_1a.execute_response(error_code)
    assert error is not None, "Should have caught the error"
    assert "ZeroDivisionError" in result, "Error message should include exception type"
    print("✓ Error handling works")
    
    # Test no code blocks
    print("\n=== Testing response without code ===")
    no_code_response = "I think I should investigate my architecture first."
    has_code, result, error = interface_1a.execute_response(no_code_response)
    assert not has_code, "Should detect no code blocks"
    print("✓ Correctly handles responses without code")
    
    print("\n" + "="*50)
    print("✅ ALL TESTS PASSED!")
    print("="*50)


if __name__ == "__main__":
    test_code_execution_interface()
