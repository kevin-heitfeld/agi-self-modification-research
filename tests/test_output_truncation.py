"""
Tests for output truncation functionality

Tests the smart truncation system that prevents OOM from massive code execution outputs.
"""

import pytest
from src.code_execution_interface import truncate_output, MAX_OUTPUT_CHARS, MAX_LIST_ITEMS


class TestOutputTruncation:
    """Test suite for output truncation."""

    def test_short_output_unchanged(self):
        """Short outputs should pass through unchanged."""
        short_text = "Hello, world!"
        result = truncate_output(short_text)
        assert result == short_text

    def test_empty_output(self):
        """Empty outputs should be handled."""
        assert truncate_output("") == ""

    def test_exact_limit_output(self):
        """Output at exact limit should pass through."""
        text = "x" * MAX_OUTPUT_CHARS
        result = truncate_output(text)
        assert result == text

    def test_long_text_truncation(self):
        """Long text should be truncated with notice."""
        long_text = "a" * 5000
        result = truncate_output(long_text, max_chars=2000)
        
        assert len(result) < len(long_text)
        assert "Output truncated" in result
        assert "5000 characters total" in result
        # Should have beginning and end
        assert result.startswith("a")
        assert result.rstrip().endswith("a")

    def test_list_truncation(self):
        """Large lists should be truncated intelligently."""
        # Create a large list
        large_list = [f"layer_{i}" for i in range(500)]
        output = str(large_list)
        
        result = truncate_output(output)
        
        assert "List with 500 items" in result
        assert "showing first" in result
        assert "items omitted" in result
        # Should show first items
        assert "layer_0" in result
        # Should show last items
        assert "layer_499" in result or "layer_498" in result

    def test_dict_truncation(self):
        """Large dicts should be truncated intelligently."""
        # Create a large dict
        large_dict = {f"key_{i}": f"value_{i}" for i in range(100)}
        output = str(large_dict)
        
        result = truncate_output(output)
        
        assert "Dict with 100 items" in result
        assert "showing first" in result
        assert "items omitted" in result

    def test_small_list_unchanged(self):
        """Small lists should not be truncated."""
        small_list = ["a", "b", "c"]
        output = str(small_list)
        
        result = truncate_output(output)
        
        # Should be unchanged
        assert result == output
        assert "truncated" not in result.lower()

    def test_invalid_list_syntax(self):
        """Invalid list syntax should fall back to character truncation."""
        # Looks like a list but isn't valid
        fake_list = "[" + "x" * 5000 + "]"
        result = truncate_output(fake_list, max_chars=2000)
        
        assert "Output truncated" in result
        assert len(result) < len(fake_list)

    def test_multiline_output(self):
        """Multiline output should be truncated with line count."""
        lines = [f"Line {i}" for i in range(1000)]
        long_text = "\n".join(lines)
        
        result = truncate_output(long_text, max_chars=2000)
        
        assert "Output truncated" in result
        assert "1000 lines" in result

    def test_preserves_structure(self):
        """Truncation should preserve meaningful structure."""
        # List with detailed layer information (large enough to trigger truncation)
        detailed_list = [
            "lm_head",
            "model.embed_tokens",
            "model.layers.0",
            "model.layers.0.input_layernorm",
            "model.layers.0.mlp",
        ] + [f"model.layers.{i}.submodule_{j}" for i in range(1, 200) for j in range(5)]
        
        output = str(detailed_list)
        result = truncate_output(output)
        
        # Should show beginning
        assert "lm_head" in result
        assert "model.embed_tokens" in result
        # Should indicate truncation (list is large enough to trigger it)
        assert "items omitted" in result or "List with" in result or "truncated" in result.lower()


class TestIntegrationScenarios:
    """Test realistic scenarios from actual usage."""

    def test_layer_list_scenario(self):
        """Test the exact scenario that caused OOM."""
        # This is what introspection.architecture.list_layers() returns
        layers = ['lm_head', 'model', 'model.embed_tokens', 'model.layers']
        
        # Add all 28 layers with their submodules (like 7B model)
        for i in range(28):
            layers.extend([
                f'model.layers.{i}',
                f'model.layers.{i}.input_layernorm',
                f'model.layers.{i}.mlp',
                f'model.layers.{i}.mlp.act_fn',
                f'model.layers.{i}.mlp.down_proj',
                f'model.layers.{i}.mlp.gate_proj',
                f'model.layers.{i}.mlp.up_proj',
                f'model.layers.{i}.post_attention_layernorm',
                f'model.layers.{i}.self_attn',
                f'model.layers.{i}.self_attn.k_proj',
                f'model.layers.{i}.self_attn.o_proj',
                f'model.layers.{i}.self_attn.q_proj',
                f'model.layers.{i}.self_attn.v_proj',
            ])
        
        layers.extend(['model.norm', 'model.rotary_emb'])
        
        # This is what gets printed
        output = f"List of Layers:\n{layers}"
        
        # Should be MUCH shorter after truncation
        result = truncate_output(output)
        
        assert len(result) < len(output) * 0.3  # At least 70% reduction
        assert "List with" in result or "truncated" in result.lower()
        
        # Should still show important info
        assert "lm_head" in result  # First layer
        assert "model.norm" in result or "rotary_emb" in result  # Last layers

    def test_activation_stats_scenario(self):
        """Test typical activation statistics output."""
        # Typical activation capture output
        stats = {
            f"model.layers.{i}": {
                "shape": [1, 10, 3584],
                "mean": 0.123,
                "std": 1.456,
                "min": -5.678,
                "max": 6.789
            }
            for i in range(50)
        }
        
        output = str(stats)
        result = truncate_output(output)
        
        # Should be truncated but preserve useful info
        assert len(result) < len(output)
        assert "model.layers.0" in result  # First entry
        assert "Dict with 50 items" in result or "truncated" in result.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
