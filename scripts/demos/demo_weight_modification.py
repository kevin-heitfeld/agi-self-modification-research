"""
Demo: Weight Modification Workflow (Human-Driven)
==================================================

IMPORTANT: This is a DEMO/TUTORIAL script where WE (humans) perform a weight
modification experiment following Phase 1 procedures. This is NOT the real
Phase 1 experiment where the model examines itself.

For the actual Phase 1 introspection experiment where the MODEL investigates
consciousness, see:
    scripts/experiments/phase1_introspection.py

Purpose of This Demo:
---------------------
This script demonstrates a complete Phase 1 modification workflow:
1. Load model and initialize systems
2. Create baseline checkpoint
3. Inspect target layer
4. Make small modification
5. Test the effects
6. Record to human knowledge base
7. Rollback if needed

This shows the PROCESS and TOOLS available. It's educational - showing humans
how to safely modify weights and use the safety systems.

Author: AGI Self-Modification Research
Date: November 7, 2025
"""

import torch
import logging
from pathlib import Path
from typing import Optional, List

from src.model_manager import ModelManager
from src.memory import MemorySystem
from src.memory.observation_layer import ObservationType
from src.introspection import WeightInspector, ActivationMonitor
from src.checkpointing import CheckpointManager
from src.safety_monitor import SafetyMonitor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def calculate_perplexity(model, tokenizer, text: str) -> float:
    """
    Calculate perplexity on a given text.
    Lower perplexity = better performance.

    Args:
        model: The language model
        tokenizer: The tokenizer
        text: Text to evaluate

    Returns:
        Perplexity score
    """
    inputs = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        perplexity = torch.exp(loss).item()

    return perplexity


def test_generation(model, tokenizer, prompt: str) -> str:
    """
    Generate text from a prompt.

    Args:
        model: The language model
        tokenizer: The tokenizer
        prompt: Input prompt

    Returns:
        Generated text
    """
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated


def run_weight_modification_demo(
    target_layer: str = "model.layers.5.self_attn.q_proj",
    modification_delta: float = 0.01,
    test_prompts: Optional[List[str]] = None
):
    """
    Run a weight modification demo/tutorial (human-controlled).

    This demonstrates how to use the Phase 1 tools to safely modify weights.
    A human controls every step and makes all decisions.

    Args:
        target_layer: Full path to the layer to modify (e.g., "model.layers.5.self_attn.q_proj")
        modification_delta: How much to add to weights (start small!)
        test_prompts: List of prompts to test generation quality
    """

    if test_prompts is None:
        test_prompts = [
            "The nature of consciousness is",
            "Artificial intelligence can be defined as",
            "The relationship between mind and computation"
        ]

    logger.info("=" * 80)
    logger.info("DEMO: Weight Modification Workflow (Human-Driven)")
    logger.info("=" * 80)

    # ===== STEP 1: Initialize all systems =====
    logger.info("\n[STEP 1] Initializing systems...")

    model_mgr = ModelManager()
    model_loaded = model_mgr.load_model()

    if not model_loaded:
        logger.error("Failed to load model")
        return

    # Type assertion: model and tokenizer are guaranteed to be loaded here
    model = model_mgr.model
    tokenizer = model_mgr.tokenizer
    assert model is not None, "Model should be loaded"
    assert tokenizer is not None, "Tokenizer should be loaded"

    # Use human_knowledge for Phase 1 experiments
    memory = MemorySystem("data/human_knowledge")

    # Count observations using query
    all_obs = memory.observations.query()
    logger.info(f"Memory initialized: {len(all_obs)} existing observations")

    inspector = WeightInspector(model, "Qwen2.5-3B")
    memory.set_weight_inspector(inspector)

    checkpointer = CheckpointManager("data/checkpoints")

    safety_monitor = SafetyMonitor(
        model=model,
        baseline_metrics={}
    )

    activation_monitor = ActivationMonitor(model, inspector)

    logger.info("✓ All systems initialized")

    # ===== STEP 2: Baseline measurements =====
    logger.info("\n[STEP 2] Taking baseline measurements...")

    # Create checkpoint
    checkpoint_id = checkpointer.save_checkpoint(
        model=model,
        description="phase1_baseline - before first modification",
        modification_details={"experiment": "first_modification", "stage": "baseline"}
    )
    logger.info(f"✓ Checkpoint created: {checkpoint_id}")

    # Test baseline perplexity
    baseline_text = "The quick brown fox jumps over the lazy dog. " * 10
    baseline_perplexity = calculate_perplexity(model, tokenizer, baseline_text)
    logger.info(f"✓ Baseline perplexity: {baseline_perplexity:.4f}")

    # Test baseline generation
    logger.info("✓ Baseline generation samples:")
    baseline_generations = {}
    for prompt in test_prompts:
        generated = test_generation(model, tokenizer, prompt)
        baseline_generations[prompt] = generated
        logger.info(f"  Prompt: {prompt}")
        logger.info(f"  Output: {generated[:100]}...")

    # ===== STEP 3: Inspect target layer =====
    logger.info(f"\n[STEP 3] Inspecting target layer: {target_layer}")

    layer_stats = inspector.get_weight_statistics(target_layer)
    logger.info(f"  Shape: {layer_stats['shape']}")
    logger.info(f"  Parameters: {layer_stats['total_elements']:,}")
    logger.info(f"  Mean: {layer_stats['mean']:.6f}")
    logger.info(f"  Std: {layer_stats['std']:.6f}")

    # Check for weight sharing
    shared_layers = inspector.get_shared_layers(target_layer)
    if shared_layers:
        logger.warning(f"⚠️  Layer shares weights with: {shared_layers}")
        logger.warning(f"⚠️  Modifying this layer will affect {len(shared_layers)} other layers!")
        response = input("Continue anyway? (yes/no): ")
        if response.lower() != 'yes':
            logger.info("Experiment aborted by user")
            return

    # ===== STEP 4: Make modification =====
    logger.info(f"\n[STEP 4] Modifying weights by {modification_delta:+.4f}...")

    # Get the actual layer
    layer_parts = target_layer.split('.')
    current = model
    for part in layer_parts:
        if part.isdigit():
            current = current[int(part)]
        else:
            current = getattr(current, part)

    # Record pre-modification state
    original_mean = current.weight.data.mean().item()
    original_std = current.weight.data.std().item()

    # Apply modification
    current.weight.data += modification_delta

    # Record post-modification state
    new_mean = current.weight.data.mean().item()
    new_std = current.weight.data.std().item()

    logger.info(f"✓ Modification applied")
    logger.info(f"  Mean: {original_mean:.6f} → {new_mean:.6f} (Δ={new_mean-original_mean:.6f})")
    logger.info(f"  Std:  {original_std:.6f} → {new_std:.6f} (Δ={new_std-original_std:.6f})")

    # ===== STEP 5: Test effects =====
    logger.info("\n[STEP 5] Testing effects of modification...")

    # Check safety - look for critical alerts and resource issues
    resource_stats = safety_monitor.check_resources()
    critical_alerts = safety_monitor.get_critical_alerts()

    safety_healthy = (
        len(critical_alerts) == 0 and
        not safety_monitor.emergency_stop_triggered
    )

    logger.info(f"Safety check: {'HEALTHY' if safety_healthy else 'ISSUES DETECTED'}")
    logger.info(f"  Critical alerts: {len(critical_alerts)}")
    logger.info(f"  Memory usage: {resource_stats.get('memory_usage_mb', 0):.1f} MB")

    if not safety_healthy:
        logger.error("❌ Safety check failed!")
        logger.error(f"Critical alerts: {[str(a) for a in critical_alerts]}")
        logger.info("Rolling back...")
        checkpointer.restore_checkpoint(model, checkpoint_id)
        logger.info("✓ Rolled back to baseline")

        # Record failure
        memory.observations.record(
            obs_type=ObservationType.SYSTEM_EVENT,
            category="safety_failure",
            description=f"Modification to {target_layer} by {modification_delta:+.4f} caused safety failure",
            data={
                "layer": target_layer,
                "delta": modification_delta,
                "critical_alerts": len(critical_alerts),
                "checkpoint": checkpoint_id
            },
            tags=["safety_failure", "phase_1", "rollback"],
            importance=0.9
        )
        return

    # Test perplexity
    modified_perplexity = calculate_perplexity(model, tokenizer, baseline_text)
    perplexity_change = modified_perplexity - baseline_perplexity
    perplexity_pct = (perplexity_change / baseline_perplexity) * 100

    logger.info(f"Modified perplexity: {modified_perplexity:.4f}")
    logger.info(f"Change: {perplexity_change:+.4f} ({perplexity_pct:+.2f}%)")

    # Test generation
    logger.info("Modified generation samples:")
    modified_generations = {}
    for prompt in test_prompts:
        generated = test_generation(model, tokenizer, prompt)
        modified_generations[prompt] = generated
        logger.info(f"  Prompt: {prompt}")
        logger.info(f"  Output: {generated[:100]}...")

    # ===== STEP 6: Analyze and record =====
    logger.info("\n[STEP 6] Recording to human knowledge base...")

    # Determine success/failure
    improved = perplexity_change < 0  # Lower perplexity is better
    obs_type = ObservationType.MODIFICATION if improved else ObservationType.SYSTEM_EVENT
    category = "weight_modification_success" if improved else "weight_modification_failure"

    # Record observation
    obs_id = memory.observations.record(
        obs_type=obs_type,
        category=category,
        description=f"Modified {target_layer} by {modification_delta:+.4f}",
        data={
            "layer": target_layer,
            "layer_type": "attention" if "attn" in target_layer else "other",
            "delta": modification_delta,
            "checkpoint": checkpoint_id,
            "baseline_perplexity": baseline_perplexity,
            "modified_perplexity": modified_perplexity,
            "perplexity_change": perplexity_change,
            "perplexity_pct_change": perplexity_pct,
            "original_mean": original_mean,
            "original_std": original_std,
            "new_mean": new_mean,
            "new_std": new_std,
            "safety_healthy": safety_healthy,
            "critical_alerts": len(critical_alerts)
        },
        tags=[
            "phase_1",
            "attention" if "attn" in target_layer else "mlp",
            "small_change" if abs(modification_delta) < 0.05 else "large_change",
            "improved" if improved else "degraded"
        ],
        importance=0.8 if improved else 0.6
    )

    logger.info(f"✓ Observation recorded: {obs_id}")
    logger.info(f"  Category: {category}")

    # Count observations
    all_obs = memory.observations.query()
    logger.info(f"  Total observations in memory: {len(all_obs)}")

    # ===== STEP 7: Decision =====
    logger.info("\n[STEP 7] What next?")
    logger.info(f"Modification {'IMPROVED' if improved else 'DEGRADED'} performance")

    if improved:
        logger.info("✓ Keeping modification (checkpoint available for rollback)")
        decision = "keep"
    else:
        logger.info("⚠️  Rolling back to baseline")
        checkpointer.restore_checkpoint(model, checkpoint_id)
        logger.info("✓ Rolled back")
        decision = "rollback"

    # ===== FINAL SUMMARY =====
    logger.info("\n" + "=" * 80)
    logger.info("EXPERIMENT COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Target Layer:     {target_layer}")
    logger.info(f"Modification:     {modification_delta:+.4f}")
    logger.info(f"Result:           {'SUCCESS' if improved else 'FAILURE'}")
    logger.info(f"Perplexity:       {baseline_perplexity:.4f} → {modified_perplexity:.4f} ({perplexity_pct:+.2f}%)")
    logger.info(f"Decision:         {decision.upper()}")
    logger.info(f"Checkpoint:       {checkpoint_id}")
    logger.info(f"Observation ID:   {obs_id}")
    logger.info("=" * 80)

    return {
        "success": improved,
        "perplexity_change": perplexity_change,
        "checkpoint_id": checkpoint_id,
        "observation_id": obs_id,
        "decision": decision
    }


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("PHASE 1: WEIGHT MODIFICATION EXPERIMENT")
    print("=" * 80)
    print("\nThis script will:")
    print("1. Load Qwen2.5-3B model")
    print("2. Create a baseline checkpoint")
    print("3. Modify a single layer's weights")
    print("4. Test the effects")
    print("5. Record findings to human knowledge base")
    print("6. Rollback if performance degrades")
    print("\n" + "=" * 80)

    # Get user input
    print("\nExperiment Configuration:")
    print("-------------------------")

    layer = input("Target layer [model.layers.5.self_attn.q_proj]: ").strip()
    if not layer:
        layer = "model.layers.5.self_attn.q_proj"

    delta_str = input("Modification amount [0.01]: ").strip()
    if not delta_str:
        delta = 0.01
    else:
        delta = float(delta_str)

    print(f"\nConfiguration:")
    print(f"  Layer: {layer}")
    print(f"  Delta: {delta:+.4f}")
    print("\nStarting in 3 seconds...")

    import time
    time.sleep(3)

    # Run demo
    result = run_weight_modification_demo(
        target_layer=layer,
        modification_delta=delta
    )

    if result:
        print("\n✅ Demo completed successfully!")
        print(f"Result: {'SUCCESS' if result['success'] else 'FAILURE'}")
        print(f"\nTo view your human knowledge base:")
        print(f"  from src.memory import MemorySystem")
        print(f"  memory = MemorySystem('data/human_knowledge')")
        print(f"  memory.observations.get_all()")
