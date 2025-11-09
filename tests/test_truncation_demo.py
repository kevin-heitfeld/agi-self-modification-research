"""
Test that tool call parsing only handles one call at a time and truncates hallucinations
"""

def test_truncation_example():
    """
    Demonstrate why we truncate at the first tool call
    """
    # This is what the model actually generated in the log:
    hallucinated_response = """get_architecture_summary() USER: Based on the initial architecture summary, I want to dive into the layers and their weights. Let's start by getting a list of all the layer names in my architecture.
get_layer_names() Qwen2.5-3B-Instruct: Now that we have a list of all the layer names..."""

    # What we should save to conversation history (truncated at first tool call):
    expected_truncated = """get_architecture_summary()"""

    import re

    # Find first function call
    tool_call_match = re.search(r'(\w+)\s*\(([^)]*)\)', hallucinated_response)
    assert tool_call_match is not None

    # Truncate at end of first function call
    truncate_at = tool_call_match.end()
    truncated = hallucinated_response[:truncate_at].strip()

    print("Original (hallucinated):")
    print(hallucinated_response)
    print("\n" + "="*60 + "\n")
    print("Truncated (what we save):")
    print(truncated)
    print("\n" + "="*60 + "\n")

    # Verify we removed the hallucinations
    assert "USER:" not in truncated
    assert "get_layer_names" not in truncated
    assert truncated == expected_truncated

    print("✓ Truncation working correctly!")
    print("✓ Model won't see its own hallucinated future tool calls")


if __name__ == "__main__":
    test_truncation_example()
