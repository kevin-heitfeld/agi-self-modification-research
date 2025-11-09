"""
Test the process_text tool for self-prompting and activation capture
"""

import pytest
from src.tool_interface import ToolInterface
from src.model_manager import ModelManager
from src.introspection.weight_inspector import WeightInspector
from src.introspection.activation_monitor import ActivationMonitor
from src.introspection.architecture_navigator import ArchitectureNavigator


def test_process_text_tool_registered():
    """Test that process_text tool is registered when model_manager is provided"""
    model_mgr = ModelManager(model_name="Qwen/Qwen2.5-3B-Instruct")
    model_mgr.load_model()
    
    inspector = WeightInspector(model_mgr.model, "Qwen2.5-3B")
    activation_monitor = ActivationMonitor(model_mgr.model, model_mgr.tokenizer)
    navigator = ArchitectureNavigator(model_mgr.model)
    
    # With model_manager
    interface = ToolInterface(
        inspector=inspector,
        activation_monitor=activation_monitor,
        navigator=navigator,
        model_manager=model_mgr
    )
    
    assert 'process_text' in interface.tools
    
    # Without model_manager
    interface_no_mgr = ToolInterface(
        inspector=inspector,
        activation_monitor=activation_monitor,
        navigator=navigator
    )
    
    assert 'process_text' not in interface_no_mgr.tools


def test_process_text_execution():
    """Test that process_text actually works"""
    model_mgr = ModelManager(model_name="Qwen/Qwen2.5-3B-Instruct")
    model_mgr.load_model()
    
    inspector = WeightInspector(model_mgr.model, "Qwen2.5-3B")
    activation_monitor = ActivationMonitor(model_mgr.model, model_mgr.tokenizer)
    navigator = ArchitectureNavigator(model_mgr.model)
    
    interface = ToolInterface(
        inspector=inspector,
        activation_monitor=activation_monitor,
        navigator=navigator,
        model_manager=model_mgr
    )
    
    # Call process_text
    result = interface.execute_tool_call('process_text', {'text': 'Hello, world!'})
    
    # Check result structure
    assert 'prompt' in result
    assert 'response' in result
    assert 'activations_captured' in result
    assert 'num_layers_with_activations' in result
    assert 'note' in result
    
    assert result['prompt'] == 'Hello, world!'
    assert isinstance(result['response'], str)
    assert len(result['response']) > 0


def test_process_text_in_tool_documentation():
    """Test that process_text appears in available tools documentation"""
    model_mgr = ModelManager(model_name="Qwen/Qwen2.5-3B-Instruct")
    model_mgr.load_model()
    
    inspector = WeightInspector(model_mgr.model, "Qwen2.5-3B")
    activation_monitor = ActivationMonitor(model_mgr.model, model_mgr.tokenizer)
    navigator = ArchitectureNavigator(model_mgr.model)
    
    interface = ToolInterface(
        inspector=inspector,
        activation_monitor=activation_monitor,
        navigator=navigator,
        model_manager=model_mgr
    )
    
    tools_desc = interface.get_available_tools()
    
    assert 'process_text' in tools_desc
    assert 'Process text through yourself' in tools_desc
    assert 'capture activations' in tools_desc


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
