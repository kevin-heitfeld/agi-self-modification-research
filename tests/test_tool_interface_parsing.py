"""
Tests for ToolInterface parsing - validates correct tool call behavior
"""

import pytest
from src.tool_interface import ToolInterface


def test_parse_last_tool_call_single():
    """Test parsing a single tool call that ends the response"""
    interface = ToolInterface(memory=None, heritage_docs=[])

    response = """
    Let me examine the architecture.

    get_architecture_summary()
    """

    result = interface.parse_last_tool_call_if_stopped(response)
    assert result is not None
    assert result[0] == "get_architecture_summary"
    assert result[1] == {}


def test_parse_last_tool_call_with_text_after():
    """Test that tool calls with text after are rejected"""
    interface = ToolInterface(memory=None, heritage_docs=[])

    response = """
    get_architecture_summary()

    Now let me also examine the layers...
    """

    result = interface.parse_last_tool_call_if_stopped(response)
    # Should be None because model didn't stop
    assert result is None


def test_parse_last_tool_call_multiple_uses_last():
    """Test that only the LAST tool call is considered"""
    interface = ToolInterface(memory=None, heritage_docs=[])

    response = """
    get_architecture_summary()

    Now let me look at layers.

    get_layer_names(filter_pattern="attention")
    """

    result = interface.parse_last_tool_call_if_stopped(response)
    assert result is not None
    # Should get the LAST one
    assert result[0] == "get_layer_names"
    assert result[1] == {"filter_pattern": "attention"}


def test_parse_last_tool_call_no_tool_calls():
    """Test response with no tool calls"""
    interface = ToolInterface(memory=None, heritage_docs=[])

    response = "Just some text without any tool calls."

    result = interface.parse_last_tool_call_if_stopped(response)
    assert result is None


def test_parse_last_tool_call_with_args():
    """Test tool call with various argument types"""
    interface = ToolInterface(memory=None, heritage_docs=[])

    response = """
    compare_weights(layer1="model.layers.0", layer2="model.layers.1")
    """

    result = interface.parse_last_tool_call_if_stopped(response)
    assert result is not None
    assert result[0] == "compare_weights"
    assert result[1] == {"layer1": "model.layers.0", "layer2": "model.layers.1"}


def test_parse_last_tool_call_numeric_args():
    """Test tool call with numeric arguments"""
    interface = ToolInterface(memory=None, heritage_docs=[])

    response = """
    get_attention_patterns(layer_name="model.layers.0", head_idx=5)
    """

    result = interface.parse_last_tool_call_if_stopped(response)
    assert result is not None
    assert result[0] == "get_attention_patterns"
    assert result[1] == {"layer_name": "model.layers.0", "head_idx": 5}


def test_parse_all_tool_calls_for_analysis():
    """Test parse_all_tool_calls for analyzing hallucinated responses"""
    interface = ToolInterface(memory=None, heritage_docs=[])

    # This is what the model generated - multiple hallucinated tool calls
    response = """get_architecture_summary()

get_layer_names(filter_pattern="attention")

record_observation(obs_type="INTROSPECTION", category="Architecture", description="test", data={}, tags=[], importance=0.5)"""

    # parse_all_tool_calls can still be used for analysis/debugging
    results = interface.parse_all_tool_calls(response)
    assert len(results) == 3
    assert results[0][0] == "get_architecture_summary"
    assert results[1][0] == "get_layer_names"
    assert results[2][0] == "record_observation"

    # parse_last_tool_call_if_stopped gets the last one (since model stopped after it)
    result = interface.parse_last_tool_call_if_stopped(response)
    assert result is not None
    assert result[0] == "record_observation"
