"""
Test: Variable Persistence Across Code Blocks

This test demonstrates that variables now persist across code blocks
within the same response (Option 1 implementation).
"""

import sys
from pathlib import Path
import tempfile

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.code_execution_interface import CodeExecutionInterface
from src.introspection_modules import create_introspection_module
from src.memory import MemorySystem


def test_variable_persistence():
    """Test that variables persist across code blocks in same response"""
    
    print("="*80)
    print("TEST: Variable Persistence Across Code Blocks")
    print("="*80)
    
    # Create mock components
    model = None  # Not needed for this test
    tokenizer = None
    
    # Create temporary memory system (don't use context manager to avoid cleanup issues)
    tmpdir = tempfile.mkdtemp()
    memory_path = str(Path(tmpdir) / "test_memory")
    
    try:
        memory_system = MemorySystem(storage_dir=memory_path)
        
        # Create code execution interface
        interface = CodeExecutionInterface(
            model=model,
            tokenizer=tokenizer,
            memory_system=memory_system,
            heritage_system=None,
            phase='1a'
        )
        
        # Test 1: Variables persist within same response
        print("\n[Test 1] Variables persist within same response")
        print("-"*80)
        
        response1 = """
Let me test variable persistence.

```python
# Block 1: Define a variable
architecture_summary = {"model_type": "QWEN2", "params": 7615616512}
print(f"Defined: {architecture_summary}")
```

Now let me use it in a second block:

```python
# Block 2: Use the variable from block 1
print(f"Accessing from block 2: {architecture_summary}")
print(f"Model type: {architecture_summary['model_type']}")
```
"""
        
        has_code, result, error = interface.execute_response(response1)
        
        print(f"Has code: {has_code}")
        print(f"Result:\n{result}")
        print(f"Error: {error}")
        
        # Check if it worked
        if has_code and error is None and "QWEN2" in result:
            print("\n✅ Test 1 PASSED: Variables persisted across blocks!")
        else:
            print("\n❌ Test 1 FAILED")
            return False
        
        # Test 2: Variables reset in new response
        print("\n\n[Test 2] Variables reset in new response")
        print("-"*80)
        
        response2 = """
Let me try to access the variable from the previous response:

```python
# This should fail - architecture_summary is from previous response
# Using a generic catch since specific exceptions aren't available in sandbox
error_caught = False
try:
    print(architecture_summary)
    print("❌ Variable incorrectly persisted from previous response!")
except:
    error_caught = True
    print("✓ Variable correctly not available from previous response")
    
print(f"Error caught: {error_caught}")
```
"""
        
        has_code2, result2, error2 = interface.execute_response(response2)
        
        print(f"Has code: {has_code2}")
        print(f"Result:\n{result2}")
        print(f"Error: {error2}")
        
        # Check if it correctly failed
        if has_code2 and error2 is None and "Error caught: True" in result2:
            print("\n✅ Test 2 PASSED: Variables correctly reset between responses!")
        else:
            print("\n❌ Test 2 FAILED")
            return False
        
        # Test 3: Multiple operations in sequence
        print("\n\n[Test 3] Multiple operations building on each other")
        print("-"*80)
        
        response3 = """
Let me compute something step by step:

```python
# Step 1: Initial computation
layers = 28
params_per_layer = 7615616512 // 28
print(f"Params per layer (approx): {params_per_layer:,}")
```

```python
# Step 2: Use previous computation
hidden_size = 3584
computed_params = hidden_size * 1000  # Simplified
ratio = params_per_layer / computed_params
print(f"Using layers={layers} and params_per_layer={params_per_layer:,}")
print(f"Ratio: {ratio:.2f}")
```

```python
# Step 3: Final calculation using all previous variables
total_check = layers * params_per_layer
print(f"Total check using all vars: {total_check:,}")
print(f"Available vars: layers={layers}, params_per_layer={params_per_layer:,}, hidden_size={hidden_size}")
```
"""
        
        has_code3, result3, error3 = interface.execute_response(response3)
        
        print(f"Has code: {has_code3}")
        print(f"Result:\n{result3}")
        print(f"Error: {error3}")
        
        # Check if all blocks executed successfully
        if has_code3 and error3 is None and "Available vars" in result3:
            print("\n✅ Test 3 PASSED: Multiple blocks successfully shared variables!")
        else:
            print("\n❌ Test 3 FAILED")
            return False
        
        print("\n" + "="*80)
        print("ALL TESTS PASSED! ✅")
        print("="*80)
        print("\nSummary:")
        print("- Variables persist across code blocks within same response")
        print("- Variables are cleared at the start of each new response")
        print("- Multiple blocks can build on each other's computations")
        print("\nThis matches Option 1: Jupyter-like notebook cell behavior per response")
        return True
    
    finally:
        # Cleanup on Windows - close DB connections first
        import shutil
        import time
        if memory_system:
            try:
                memory_system.close()  # Close any open connections
            except:
                pass
        time.sleep(0.1)  # Give Windows time to release file handles
        try:
            shutil.rmtree(tmpdir, ignore_errors=True)
        except:
            pass


if __name__ == "__main__":
    try:
        success = test_variable_persistence()
        exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ TEST FAILED WITH EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
