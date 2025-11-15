"""
Demonstration of Output Truncation System

This script demonstrates how the truncation system prevents OOM errors
while preserving useful information.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.code_execution_interface import truncate_output


def demo_layer_list():
    """Demonstrate truncating a large layer list (the original problem)."""
    print("=" * 80)
    print("DEMO 1: Large Layer List (Original OOM Problem)")
    print("=" * 80)
    
    # Simulate the layer list from a 7B model (28 layers)
    layers = ['lm_head', 'model', 'model.embed_tokens', 'model.layers']
    
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
    
    original = f"List of Layers:\n{layers}"
    truncated = truncate_output(original)
    
    print(f"\nOriginal length: {len(original)} characters")
    print(f"Truncated length: {len(truncated)} characters")
    print(f"Reduction: {100 * (1 - len(truncated)/len(original)):.1f}%")
    print(f"\nTruncated output:\n{truncated[:500]}...")
    print()


def demo_dict_output():
    """Demonstrate truncating a large dictionary."""
    print("=" * 80)
    print("DEMO 2: Large Dictionary")
    print("=" * 80)
    
    # Simulate activation statistics for many layers
    stats = {
        f"model.layers.{i}": {
            "shape": [1, 10, 3584],
            "mean": 0.123 + i * 0.001,
            "std": 1.456,
            "min": -5.678,
            "max": 6.789
        }
        for i in range(100)
    }
    
    original = str(stats)
    truncated = truncate_output(original)
    
    print(f"\nOriginal length: {len(original)} characters")
    print(f"Truncated length: {len(truncated)} characters")
    print(f"Reduction: {100 * (1 - len(truncated)/len(original)):.1f}%")
    print(f"\nTruncated output:\n{truncated[:500]}...")
    print()


def demo_long_text():
    """Demonstrate truncating long text output."""
    print("=" * 80)
    print("DEMO 3: Long Text Output")
    print("=" * 80)
    
    # Simulate long output from weight statistics
    original = "\n".join([
        f"Layer {i}: mean={0.123:.6f}, std={1.456:.6f}, min={-5.678:.6f}, max={6.789:.6f}"
        for i in range(1000)
    ])
    
    truncated = truncate_output(original, max_chars=2000)
    
    print(f"\nOriginal length: {len(original)} characters ({len(original.split(chr(10)))} lines)")
    print(f"Truncated length: {len(truncated)} characters")
    print(f"Reduction: {100 * (1 - len(truncated)/len(original)):.1f}%")
    print(f"\nTruncated output preview:")
    print(truncated[:300])
    print("\n[... middle section ...]")
    print(truncated[-300:])
    print()


def demo_small_output():
    """Demonstrate that small outputs are not affected."""
    print("=" * 80)
    print("DEMO 4: Small Output (No Truncation)")
    print("=" * 80)
    
    original = "Model has 28 layers\nTotal parameters: 7,615,616,512"
    truncated = truncate_output(original)
    
    print(f"\nOriginal length: {len(original)} characters")
    print(f"Truncated length: {len(truncated)} characters")
    print(f"Are they equal? {original == truncated}")
    print(f"\nOutput:\n{truncated}")
    print()


def demo_comparison():
    """Show before/after comparison for the OOM scenario."""
    print("=" * 80)
    print("DEMO 5: Before/After - OOM Prevention")
    print("=" * 80)
    
    # Create a massive layer list (like what caused OOM)
    layers = []
    for i in range(500):
        layers.extend([
            f'model.layers.{i}',
            f'model.layers.{i}.self_attn',
            f'model.layers.{i}.mlp',
        ])
    
    original = str(layers)
    truncated = truncate_output(original)
    
    # Estimate token counts (rough: ~4 chars per token)
    orig_tokens = len(original) // 4
    trunc_tokens = len(truncated) // 4
    
    print(f"\n{'Metric':<30} {'Before':<20} {'After':<20} {'Saved':<20}")
    print("-" * 90)
    print(f"{'Characters':<30} {len(original):<20,} {len(truncated):<20,} {len(original)-len(truncated):<20,}")
    print(f"{'Est. Tokens':<30} {orig_tokens:<20,} {trunc_tokens:<20,} {orig_tokens-trunc_tokens:<20,}")
    print(f"{'Reduction':<30} {'-':<20} {'-':<20} {100 * (1 - len(truncated)/len(original)):.1f}%")
    print()
    
    print("This is how much context space is saved per code execution!")
    print("Over a 20-iteration experiment, this prevents context overflow.")
    print()


if __name__ == "__main__":
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 20 + "OUTPUT TRUNCATION DEMONSTRATION" + " " * 27 + "║")
    print("║" + " " * 15 + "Preventing OOM from Code Execution Output" + " " * 22 + "║")
    print("╚" + "═" * 78 + "╝")
    print()
    
    demo_layer_list()
    demo_dict_output()
    demo_long_text()
    demo_small_output()
    demo_comparison()
    
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print("✓ Large lists/dicts show count + sample items")
    print("✓ Long text shows beginning and end with truncation notice")
    print("✓ Small outputs pass through unchanged")
    print("✓ Typical 70-95% reduction in output size")
    print("✓ Model receives enough info to understand what happened")
    print("✓ Prevents OOM from token explosion in long experiments")
    print()
