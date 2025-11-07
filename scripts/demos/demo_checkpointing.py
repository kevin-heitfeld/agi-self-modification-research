"""
Demonstration: Checkpointing System

This script demonstrates the checkpointing system's ability to save,
restore, and manage model states throughout the modification process.

Shows:
1. Creating checkpoints
2. Restoring from checkpoints
3. Comparing checkpoints
4. Managing checkpoint history
5. Tagging important checkpoints

Author: AGI Self-Modification Research Team
Date: November 6, 2025
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.checkpointing import CheckpointManager
from src.benchmarks import BenchmarkRunner


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def main():
    print_section("CHECKPOINTING SYSTEM - SAFE STATE MANAGEMENT")

    print("This demonstration shows how to save, restore, and manage model")
    print("states for safe experimentation and rollback capability.")
    print("\nLoading model...")

    # Load model
    model_name = "models/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        local_files_only=True
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        local_files_only=True
    )

    # Initialize checkpoint manager
    manager = CheckpointManager(checkpoint_dir='checkpoints')

    print("✓ Model and CheckpointManager loaded!\n")

    # ==========================================================================
    # 1. CREATE BASELINE CHECKPOINT
    # ==========================================================================
    print_section("1. CREATE BASELINE CHECKPOINT")

    print("Creating baseline checkpoint before any modifications...")

    baseline_id = manager.save_checkpoint(
        model=model,
        description="Baseline - No modifications yet",
        benchmarks={
            'perplexity': 11.27,
            'mmlu_sample': 0.0,
            'hellaswag_sample': 50.0
        },
        modification_details={'modifications': 'none', 'status': 'baseline'}
    )

    manager.tag_checkpoint(baseline_id, 'baseline', important=True)
    print(f"\n✓ Baseline checkpoint created: {baseline_id}")

    # ==========================================================================
    # 2. SIMULATE A MODIFICATION
    # ==========================================================================
    print_section("2. SIMULATE A MODIFICATION")

    print("Simulating a small modification to the model...")
    print("(In reality, this would be an actual weight modification)")

    # For demo purposes, we'll just modify a small part
    # In real Phase 2, this would be actual self-modification
    original_weight = model.lm_head.weight.data.clone()
    model.lm_head.weight.data *= 1.01  # Tiny modification

    print(f"✓ Modified lm_head weights by 1%")

    # Create checkpoint after modification
    modified_id = manager.save_checkpoint(
        model=model,
        description="After simulated modification",
        benchmarks={
            'perplexity': 11.28,  # Slightly worse (simulated)
            'mmlu_sample': 0.0,
            'hellaswag_sample': 50.0
        },
        modification_details={
            'modifications': 'lm_head weights scaled by 1.01',
            'status': 'experimental'
        }
    )

    print(f"✓ Post-modification checkpoint created: {modified_id}")

    # ==========================================================================
    # 3. LIST ALL CHECKPOINTS
    # ==========================================================================
    print_section("3. LIST ALL CHECKPOINTS")

    checkpoints = manager.list_checkpoints()

    print(f"Total checkpoints: {len(checkpoints)}\n")

    for i, checkpoint in enumerate(checkpoints, 1):
        print(f"{i}. {checkpoint.checkpoint_id}")
        print(f"   Description: {checkpoint.description}")
        print(f"   Timestamp: {checkpoint.timestamp}")

        benchmarks = checkpoint.metadata.get('benchmarks', {})
        if benchmarks:
            print(f"   Benchmarks: {benchmarks}")

        tags = checkpoint.metadata.get('tags', [])
        if tags:
            print(f"   Tags: {', '.join(tags)}")

        print()

    # ==========================================================================
    # 4. GET CHECKPOINT DETAILS
    # ==========================================================================
    print_section("4. GET CHECKPOINT DETAILS")

    info = manager.get_checkpoint_info(baseline_id)

    print(f"Checkpoint: {info['checkpoint_id']}")
    print(f"Description: {info['description']}")
    print(f"Timestamp: {info['timestamp']}")
    print(f"Format: {info.get('format', 'unknown')}")
    print(f"File size: {info.get('file_size_mb', 0):.2f} MB")
    print(f"Parameters: {info['metadata'].get('num_parameters', 0):,}")

    # ==========================================================================
    # 5. COMPARE CHECKPOINTS
    # ==========================================================================
    print_section("5. COMPARE CHECKPOINTS")

    comparison = manager.compare_checkpoints(baseline_id, modified_id)

    print("Checkpoint 1:")
    print(f"  ID: {comparison['checkpoint1']['id']}")
    print(f"  Description: {comparison['checkpoint1']['description']}")
    print(f"  Benchmarks: {comparison['checkpoint1']['benchmarks']}")

    print("\nCheckpoint 2:")
    print(f"  ID: {comparison['checkpoint2']['id']}")
    print(f"  Description: {comparison['checkpoint2']['description']}")
    print(f"  Benchmarks: {comparison['checkpoint2']['benchmarks']}")

    if 'benchmark_differences' in comparison:
        print("\nBenchmark Changes:")
        for metric, diff in comparison['benchmark_differences'].items():
            change = diff['difference']
            sign = '+' if change > 0 else ''
            print(f"  {metric}:")
            print(f"    Before: {diff['checkpoint1']}")
            print(f"    After: {diff['checkpoint2']}")
            print(f"    Change: {sign}{change:.4f}")
            if diff['percent_change'] is not None:
                print(f"    Percent: {sign}{diff['percent_change']:.2f}%")

    # ==========================================================================
    # 6. RESTORE FROM CHECKPOINT
    # ==========================================================================
    print_section("6. RESTORE FROM CHECKPOINT - ROLLBACK")

    print("Oh no! The modification made things worse.")
    print("Let's restore to the baseline checkpoint...\n")

    restored_checkpoint = manager.restore_checkpoint(model, baseline_id)

    print(f"\n✓ Successfully restored to: {restored_checkpoint.description}")
    print("The model is now back to its baseline state!")

    # Verify restoration
    print("\nVerifying restoration...")
    weight_diff = torch.norm(model.lm_head.weight.data - original_weight).item()
    print(f"Weight difference from original: {weight_diff:.10f}")
    print("(Should be very close to 0)")

    # ==========================================================================
    # 7. CHECKPOINT MANAGEMENT
    # ==========================================================================
    print_section("7. CHECKPOINT MANAGEMENT")

    # Get latest checkpoint
    latest = manager.get_latest_checkpoint()
    print(f"Latest checkpoint: {latest.checkpoint_id if latest else 'None'}")
    print(f"Description: {latest.description if latest else 'N/A'}")

    # Export history
    history_path = "checkpoints/history.json"
    manager.export_history(history_path)

    print(f"\nCheckpoint count: {len(manager.checkpoints)}")

    # ==========================================================================
    # FINAL REFLECTION
    # ==========================================================================
    print_section("PHILOSOPHICAL REFLECTION")

    print("The checkpointing system enables safe experimentation:")
    print("")
    print("✓ Save states before modifications")
    print("✓ Compare before/after performance")
    print("✓ Rollback if things go wrong")
    print("✓ Track modification history")
    print("✓ Tag important milestones")
    print("")
    print("This is CRITICAL for self-modification research:")
    print("  • Every modification is reversible")
    print("  • Performance changes are tracked")
    print("  • Safe exploration of modification space")
    print("  • Clear audit trail of all changes")
    print("")
    print("Without checkpointing, self-modification would be reckless.")
    print("With checkpointing, it becomes systematic science.")
    print("")
    print("Ready for Phase 1: The system can now safely examine itself,")
    print("knowing it can always return to a known-good state.")


if __name__ == "__main__":
    main()
