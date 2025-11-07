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
    
    TOOL_CALL: get_architecture_summary
    ARGS: {}
    """
    
    result = interface.parse_last_tool_call_if_stopped(response)
    assert result is not None
    assert result[0] == "get_architecture_summary"
    assert result[1] == {}


def test_parse_last_tool_call_with_text_after():
    """Test that tool calls with text after are rejected"""
    interface = ToolInterface(memory=None, heritage_docs=[])
def test_parse_last_tool_call_with_text_after():
    """Test that tool calls with text after are rejected"""
    interface = ToolInterface(memory=None, heritage_docs=[])
    
    response = """
    TOOL_CALL: get_architecture_summary
    ARGS: {}
    
    Now let me also examine the layers...
    """
    
    result = interface.parse_last_tool_call_if_stopped(response)
    # Should be None because model didn't stop
    assert result is None


def test_parse_last_tool_call_multiple_uses_last():
    """Test that only the LAST tool call is considered"""
    interface = ToolInterface(memory=None, heritage_docs=[])
    
    response = """
    TOOL_CALL: get_architecture_summary
    ARGS: {}
    
    Now let me look at layers.
    
    TOOL_CALL: get_layer_names
    ARGS: {"filter_pattern": "attention"}
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


def test_parse_last_tool_call_no_args():
    """Test tool call without ARGS"""
    interface = ToolInterface(memory=None, heritage_docs=[])
    
    response = """
    TOOL_CALL: get_architecture_summary
    """
    
    result = interface.parse_last_tool_call_if_stopped(response)
    # Should be None because no ARGS found
    assert result is None


def test_parse_last_tool_call_malformed_json():
    """Test tool call with invalid JSON in args"""
    interface = ToolInterface(memory=None, heritage_docs=[])
    
    response = """
    TOOL_CALL: get_architecture_summary
    ARGS: {invalid json}
    """
    
    result = interface.parse_last_tool_call_if_stopped(response)
    # Should be None because JSON parsing failed
    assert result is None


def test_parse_all_tool_calls_for_analysis():
    """Test parse_all_tool_calls for analyzing hallucinated responses"""
    interface = ToolInterface(memory=None, heritage_docs=[])
    
    # This is what the model generated - multiple hallucinated tool calls
    response = """TOOL_CALL: get_architecture_summary
ARGS: {}

TOOL_CALL: get_layer_names
ARGS: {"filter_pattern": "attention"}

TOOL_CALL: record_observation
ARGS: {"obs_type": "INTROSPECTION", "category": "Architecture", "description": "test", "data": {}, "tags": [], "importance": 0.5}"""
    
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
