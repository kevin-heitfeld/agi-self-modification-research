"""
Test that Python namespace persists across iterations within an experiment
but is cleared between experiments.

This tests the fix for the issue where the model kept trying to use variables
like `sample_text` that it defined in previous iterations, but those variables
were being cleared at each response.

Author: AGI Self-Modification Research Team
Date: November 15, 2025
"""

import pytest
import sys
from src.code_executor import CodeExecutor


class TestNamespacePersistence:
    """Test namespace persistence behavior using CodeExecutor directly"""

    @pytest.fixture
    def executor(self):
        """Create a code executor with a mock introspection module"""
        # Create a minimal mock introspection module
        class MockIntrospection:
            class architecture:
                @staticmethod
                def get_architecture_summary():
                    return {'model_type': 'TEST_MODEL', 'num_layers': 10}
        
        mock = MockIntrospection()
        
        # Register in sys.modules so imports work
        sys.modules['introspection'] = mock  # type: ignore
        sys.modules['introspection.architecture'] = mock.architecture  # type: ignore
        
        executor = CodeExecutor(introspection_module=mock)
        
        yield executor
        
        # Cleanup
        if 'introspection' in sys.modules:
            del sys.modules['introspection']
        if 'introspection.architecture' in sys.modules:
            del sys.modules['introspection.architecture']

    def test_variables_persist_in_same_namespace(self, executor):
        """Test that variables persist when using the same namespace dict"""
        
        namespace = {}
        
        # First execution: Define a variable
        code1 = """
sample_text = "The quick brown fox"
print(f"Defined: {sample_text}")
"""
        
        success, output, error = executor.execute_with_namespace(code1, namespace)
        assert success
        assert error is None
        assert "Defined: The quick brown fox" in output
        assert 'sample_text' in namespace
        assert namespace['sample_text'] == "The quick brown fox"
        
        # Second execution: Use the variable from first execution
        code2 = """
print(f"Using previous variable: {sample_text}")
print(f"Length: {len(sample_text)}")
"""
        
        success, output, error = executor.execute_with_namespace(code2, namespace)
        assert success
        assert error is None
        assert "Using previous variable: The quick brown fox" in output
        assert "Length: 19" in output
        
        # Third execution: Modify the variable
        code3 = """
sample_text = sample_text.upper()
print(f"Modified: {sample_text}")
"""
        
        success, output, error = executor.execute_with_namespace(code3, namespace)
        assert success
        assert error is None
        assert "Modified: THE QUICK BROWN FOX" in output
        assert namespace['sample_text'] == "THE QUICK BROWN FOX"

    def test_namespace_cleared_between_experiments(self, executor):
        """Test that clearing namespace removes variables"""
        
        namespace = {}
        
        # Define a variable
        code1 = """
experiment_data = {"iteration": 1, "value": 42}
print(f"Defined: {experiment_data}")
"""
        
        success, output, error = executor.execute_with_namespace(code1, namespace)
        assert success
        assert error is None
        assert "Defined:" in output
        assert 'experiment_data' in namespace
        
        # Variable should still exist
        code2 = """
print(f"Still exists: {experiment_data}")
"""
        
        success, output, error = executor.execute_with_namespace(code2, namespace)
        assert success
        assert error is None
        assert "Still exists:" in output
        
        # Clear namespace (simulating end of experiment)
        namespace.clear()
        
        # Variable should now be gone
        code3 = """
print(f"After reset: {experiment_data}")
"""
        
        success, output, error = executor.execute_with_namespace(code3, namespace)
        assert not success
        assert error is not None
        assert "NameError" in error
        assert "experiment_data" in error

    def test_functions_persist_across_executions(self, executor):
        """Test that function definitions persist"""
        
        namespace = {}
        
        # Define a function
        code1 = """
def calculate_stats(data):
    return {
        'mean': sum(data) / len(data),
        'sum': sum(data),
        'count': len(data)
    }

print("Function defined")
"""
        
        success, output, error = executor.execute_with_namespace(code1, namespace)
        assert success
        assert error is None
        assert "Function defined" in output
        assert 'calculate_stats' in namespace
        
        # Use the function in next execution
        code2 = """
data = [1, 2, 3, 4, 5]
stats = calculate_stats(data)
print(f"Mean: {stats['mean']}")
print(f"Sum: {stats['sum']}")
"""
        
        success, output, error = executor.execute_with_namespace(code2, namespace)
        assert success
        assert error is None
        assert "Mean: 3.0" in output
        assert "Sum: 15" in output

    def test_introspection_import_persists(self, executor):
        """Test that introspection module import persists across executions"""
        
        namespace = {}
        
        # Import introspection
        code1 = """
import introspection
print("Imported introspection")
"""
        
        success, output, error = executor.execute_with_namespace(code1, namespace)
        assert success, f"Failed to import: {error}"
        assert error is None
        assert "Imported introspection" in output
        assert 'introspection' in namespace
        
        # Use introspection without re-importing
        code2 = """
# Should work without re-importing
summary = introspection.architecture.get_architecture_summary()
print(f"Model type: {summary['model_type']}")
print(f"Layers: {summary['num_layers']}")
"""
        
        success, output, error = executor.execute_with_namespace(code2, namespace)
        assert success, f"Failed to use introspection: {error}"
        assert error is None
        assert "Model type: TEST_MODEL" in output
        assert "Layers: 10" in output

    def test_classes_persist(self, executor):
        """Test that class definitions persist across executions"""
        
        namespace = {}
        
        # Define a class
        code1 = """
class Counter:
    def __init__(self):
        self.count = 0
    
    def increment(self):
        self.count += 1
        return self.count

counter = Counter()
print(f"Counter created: {counter.count}")
"""
        
        success, output, error = executor.execute_with_namespace(code1, namespace)
        assert success, f"Failed to create class: {error}"
        assert error is None
        assert "Counter created: 0" in output
        
        # Use the instance in next execution
        code2 = """
print(f"Count 1: {counter.increment()}")
print(f"Count 2: {counter.increment()}")
print(f"Count 3: {counter.increment()}")
"""
        
        success, output, error = executor.execute_with_namespace(code2, namespace)
        assert success, f"Failed to use class instance: {error}"
        assert error is None
        assert "Count 1: 1" in output
        assert "Count 2: 2" in output
        assert "Count 3: 3" in output


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
