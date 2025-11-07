"""
Phase 1: First Weight Modification Experiment

This script demonstrates a complete Phase 1 workflow:
1. Load model and initialize systems
2. Create baseline checkpoint
3. Inspect target layer
4. Make small modification
5. Test the effects
6. Record to human knowledge base
7. Rollback if needed

Author: AGI Self-Modification Research
Date: November 7, 2025
"""

import torch
import logging
from pathlib import Path

from src.model_manager import ModelManager
from src.memory import MemorySystem
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


def run_phase1_experiment(
    target_layer: str = "model.layers.5.self_attn.q_proj",
    modification_delta: float = 0.01,
    test_prompts: list = None
):
    """
    Run a Phase 1 weight modification experiment.
    
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
    logger.info("PHASE 1: First Weight Modification Experiment")
    logger.info("=" * 80)
    
    # ===== STEP 1: Initialize all systems =====
    logger.info("\n[STEP 1] Initializing systems...")
    
    model_mgr = ModelManager()
    model = model_mgr.load_model()
    tokenizer = model_mgr.load_tokenizer()
    
    # Use human_knowledge for Phase 1 experiments
    memory = MemorySystem("data/human_knowledge")
    logger.info(f"Memory initialized: {memory.observations.count()} existing observations")
    
    inspector = WeightInspector(model, "Qwen2.5-3B")
    memory.set_weight_inspector(inspector)
    
    checkpointer = CheckpointManager(model, "data/checkpoints")
    
    safety_monitor = SafetyMonitor(
        model=model,
        weight_inspector=inspector,
        check_interval=1.0
    )
    
    activation_monitor = ActivationMonitor(model, inspector)
    
    logger.info("✓ All systems initialized")
    
    # ===== STEP 2: Baseline measurements =====
    logger.info("\n[STEP 2] Taking baseline measurements...")
    
    # Create checkpoint
    checkpoint_id = checkpointer.save_checkpoint(
        name="phase1_baseline",
        metadata={"experiment": "first_modification"}
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
    
    layer_info = inspector.get_layer_info(target_layer)
    logger.info(f"  Shape: {layer_info['shape']}")
    logger.info(f"  Parameters: {layer_info['num_parameters']:,}")
    logger.info(f"  Mean: {layer_info['mean']:.6f}")
    logger.info(f"  Std: {layer_info['std']:.6f}")
    
    # Check for weight sharing
    if layer_info['is_shared']:
        logger.warning(f"⚠️  Layer shares weights with: {layer_info['shared_with']}")
        logger.warning(f"⚠️  Modifying this layer will affect {len(layer_info['shared_with'])} other layers!")
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
    
    # Check safety
    safety_report = safety_monitor.check_health()
    logger.info(f"Safety check: {safety_report['status']}")
    
    if not safety_report['healthy']:
        logger.error("❌ Safety check failed!")
        logger.error(f"Issues: {safety_report['issues']}")
        logger.info("Rolling back...")
        checkpointer.restore_checkpoint(checkpoint_id)
        logger.info("✓ Rolled back to baseline")
        
        # Record failure
        memory.record_observation(
            content=f"Modification to {target_layer} by {modification_delta:+.4f} caused safety failure",
            category="failure",
            metadata={
                "layer": target_layer,
                "delta": modification_delta,
                "safety_issues": safety_report['issues'],
                "checkpoint": checkpoint_id
            },
            importance=0.9,
            tags=["safety_failure", "phase_1", "rollback"]
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
    category = "success" if improved else "failure"
    
    # Record observation
    obs_id = memory.record_observation(
        content=f"Modified {target_layer} by {modification_delta:+.4f}",
        category=category,
        metadata={
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
            "safety_status": safety_report['status']
        },
        importance=0.8 if improved else 0.6,
        tags=[
            "phase_1",
            "attention" if "attn" in target_layer else "mlp",
            "small_change" if abs(modification_delta) < 0.05 else "large_change",
            "improved" if improved else "degraded"
        ]
    )
    
    logger.info(f"✓ Observation recorded: {obs_id}")
    logger.info(f"  Category: {category}")
    logger.info(f"  Total observations in memory: {memory.observations.count()}")
    
    # ===== STEP 7: Decision =====
    logger.info("\n[STEP 7] What next?")
    logger.info(f"Modification {'IMPROVED' if improved else 'DEGRADED'} performance")
    
    if improved:
        logger.info("✓ Keeping modification (checkpoint available for rollback)")
        decision = "keep"
    else:
        logger.info("⚠️  Rolling back to baseline")
        checkpointer.restore_checkpoint(checkpoint_id)
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
    
    # Run experiment
    result = run_phase1_experiment(
        target_layer=layer,
        modification_delta=delta
    )
    
    if result:
        print("\n✅ Experiment completed successfully!")
        print(f"Result: {'SUCCESS' if result['success'] else 'FAILURE'}")
        print(f"\nTo view your human knowledge base:")
        print(f"  from src.memory import MemorySystem")
        print(f"  memory = MemorySystem('data/human_knowledge')")
        print(f"  memory.observations.get_all()")
