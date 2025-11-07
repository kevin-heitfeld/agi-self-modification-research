# Safety Systems Guide

**AGI Self-Modification Research Platform**  
**Version**: Phase 0 Complete  
**Date**: November 7, 2025

---

## ⚠️ Critical Safety Information

**Self-modification is inherently risky.** This guide explains how to use the safety systems to minimize risks and ensure you can always recover from problems.

### Golden Rules 🔐

1. **Always checkpoint before modifications**
2. **Monitor continuously during experiments**
3. **Test on small scale first**
4. **Never ignore safety warnings**
5. **Keep emergency rollback ready**

---

## Table of Contents

1. [Safety Architecture](#safety-architecture)
2. [Checkpointing System](#checkpointing-system)
3. [Safety Monitor](#safety-monitor)
4. [Weight Sharing Hazards](#weight-sharing-hazards)
5. [Emergency Procedures](#emergency-procedures)
6. [Best Practices](#best-practices)

---

## Safety Architecture

### Three-Layer Defense 🛡️

```
┌──────────────────────────────────────┐
│     Layer 1: Prevention              │
│  - Checkpoints before modifications  │
│  - Weight sharing warnings           │
│  - Validation checks                 │
└──────────────────────────────────────┘
              ↓
┌──────────────────────────────────────┐
│     Layer 2: Detection               │
│  - Real-time health monitoring       │
│  - NaN/Inf detection                 │
│  - Activation anomalies              │
└──────────────────────────────────────┘
              ↓
┌──────────────────────────────────────┐
│     Layer 3: Recovery                │
│  - Automatic rollback                │
│  - Emergency stop                    │
│  - State restoration                 │
└──────────────────────────────────────┘
```

### Safety Levels

| Level | Severity | Action Required | Auto-Response |
|-------|----------|-----------------|---------------|
| 🔴 CRITICAL | Immediate danger | STOP EVERYTHING | Emergency rollback |
| 🟠 HIGH | Serious concern | Review and fix | Alert logged |
| 🟡 MEDIUM | Potential issue | Monitor closely | Warning logged |
| 🔵 LOW | Informational | Note for later | Info logged |
| ⚪ INFO | Normal operation | Continue | Debug logged |

---

## Checkpointing System

### Why Checkpoint?

**Checkpoints are your save points.** They let you:
- ✅ Undo bad modifications
- ✅ Compare before/after states
- ✅ Recover from crashes
- ✅ Track experimental progress
- ✅ Share reproducible states

### Basic Checkpointing

```python
from checkpointing import CheckpointManager

# Initialize
checkpointer = CheckpointManager(
    model=model,
    checkpoint_dir="data/checkpoints"
)

# Save checkpoint
checkpoint_id = checkpointer.save_checkpoint(
    metadata={
        "purpose": "Before attention head modification",
        "experiment": "exp_001",
        "baseline": True
    }
)

print(f"Checkpoint saved: {checkpoint_id}")
# Output: checkpoint_20251107_031500_a1b2c3d4
```

### Metadata Best Practices

**Always include**:
```python
metadata = {
    "purpose": "Clear description of why checkpoint was created",
    "experiment": "Unique experiment identifier",
    "timestamp": "Auto-added, but good for reference",
    "state": "baseline | modified | testing | final",
    "notes": "Any relevant observations"
}
```

**Example**:
```python
metadata = {
    "purpose": "Baseline before Layer 5 attention modification",
    "experiment": "attention_head_specialization_001",
    "state": "baseline",
    "notes": "Model performing well on all metrics"
}
```

### Restoring Checkpoints

```python
# List available checkpoints
checkpoints = checkpointer.list_checkpoints()
print(f"Available checkpoints: {len(checkpoints)}")

for cp in checkpoints:
    print(f"  {cp['id']}: {cp['metadata'].get('purpose', 'No description')}")

# Restore specific checkpoint
success = checkpointer.restore_checkpoint(checkpoint_id)

if success:
    print("✓ Checkpoint restored successfully")
    # Model is now in the exact state it was when checkpoint was created
else:
    print("✗ Restore failed - check logs")
```

### Checkpoint Workflow

**Recommended pattern**:

```python
# 1. Create baseline checkpoint
baseline_id = checkpointer.save_checkpoint(
    metadata={"state": "baseline", "purpose": "Pre-modification"}
)

# 2. Make modification
try:
    # Your modification code here
    modify_weights(...)
    
    # 3. Test modification
    result = test_model(...)
    
    if result['success']:
        # 4. Save successful state
        checkpointer.save_checkpoint(
            metadata={"state": "modified", "result": "success"}
        )
        print("✓ Modification successful - keeping changes")
    else:
        # 5. Rollback if unsuccessful
        checkpointer.restore_checkpoint(baseline_id)
        print("✗ Modification unsuccessful - rolled back")
        
except Exception as e:
    # 6. Emergency rollback on error
    print(f"🚨 ERROR: {e}")
    checkpointer.restore_checkpoint(baseline_id)
    print("Emergency rollback complete")
```

### Checkpoint Comparison

```python
# Compare two checkpoints
comparison = checkpointer.compare_checkpoints(
    checkpoint_id_1=baseline_id,
    checkpoint_id_2=modified_id
)

print(f"Differences found: {comparison['num_differences']}")
for diff in comparison['differences'][:5]:  # Show first 5
    print(f"  {diff['layer']}: {diff['change']:.6f}")
```

### Advanced: Automatic Checkpointing

```python
from checkpointing import AutoCheckpointer

# Create auto-checkpointing wrapper
auto_cp = AutoCheckpointer(
    model=model,
    checkpoint_dir="data/auto_checkpoints",
    frequency=10,  # Checkpoint every 10 modifications
    max_checkpoints=20  # Keep only last 20
)

# Now modifications are automatically checkpointed
with auto_cp:
    for i in range(50):
        modify_layer(i)
        # Automatic checkpoint every 10 iterations
```

---

## Safety Monitor

### Real-Time Health Monitoring

```python
from safety_monitor import SafetyMonitor

# Initialize
safety = SafetyMonitor()

# Configure thresholds (optional)
safety.set_threshold('nan_tolerance', 0)  # No NaN values allowed
safety.set_threshold('max_weight_value', 10.0)
safety.set_threshold('gradient_clip', 5.0)
```

### Checking Tensors

```python
import torch

# Check a weight tensor
weight = model.layer.weight
result = safety.check_tensor_health(weight, "layer.weight")

print(f"Status: {result['status']}")  # OK, WARNING, CRITICAL
print(f"Issues: {len(result['issues'])}")

if result['issues']:
    for issue in result['issues']:
        print(f"  🔴 {issue['severity']}: {issue['message']}")
```

### What Gets Checked?

- ✅ **NaN values**: Not-a-Number (critical error)
- ✅ **Inf values**: Infinity (critical error)
- ✅ **Value range**: Excessive magnitudes
- ✅ **Zero gradients**: Indicates dead neurons
- ✅ **Gradient explosions**: Unstable training
- ✅ **Memory usage**: Out-of-memory risks

### Continuous Monitoring

```python
# Monitor during modification loop
for epoch in range(num_epochs):
    # Make modification
    modify_layer(...)
    
    # Check health
    result = safety.check_model_health(model)
    
    if result['status'] == 'CRITICAL':
        print("🚨 CRITICAL ISSUE DETECTED")
        # Automatic rollback
        checkpointer.restore_checkpoint(baseline_id)
        break
    elif result['status'] == 'WARNING':
        print(f"⚠️  WARNING: {result['message']}")
        # Log but continue
```

### Emergency Stop

```python
# Set up emergency stop callback
def emergency_callback():
    print("🚨 EMERGENCY STOP TRIGGERED")
    checkpointer.restore_checkpoint(baseline_id)
    print("✓ Model restored to safe state")

safety.set_emergency_callback(emergency_callback)

# Now if critical issue detected, callback runs automatically
```

### Alert History

```python
# Review all alerts
alerts = safety.get_alert_history()

print(f"Total alerts: {len(alerts)}")
for alert in alerts[-10:]:  # Last 10
    print(f"  [{alert['timestamp']}] {alert['severity']}: {alert['message']}")

# Filter by severity
critical_alerts = safety.get_alerts_by_severity('CRITICAL')
print(f"Critical alerts: {len(critical_alerts)}")
```

---

## Weight Sharing Hazards

### The Problem

**Qwen2.5 shares weights** between embedding layer and output head:
- `model.embed_tokens.weight` ↔ `lm_head.weight`

**This means**: Modifying one **automatically** modifies the other!

### Detection

```python
from introspection import WeightInspector

inspector = WeightInspector(model, "Qwen2.5-3B")

# Detect weight sharing
shared_weights = inspector.get_shared_weights()

if shared_weights:
    print("⚠️  WEIGHT SHARING DETECTED:")
    for group in shared_weights:
        print(f"  Coupled: {' ↔ '.join(group)}")
```

### Safe Modification

```python
# ❌ UNSAFE - might not realize you're modifying both
model.embed_tokens.weight += delta  # Also modifies lm_head!

# ✅ SAFE - use memory system with coupled tracking
from memory import MemorySystem

memory = MemorySystem("data/memory")
memory.set_weight_inspector(inspector)  # Enable coupled detection

# Record modification
memory.record_modification(
    layer_name="model.embed_tokens.weight",
    modification_data={"delta": 0.01, "method": "gradient"}
)
# Automatically recorded as coupled_modification
# Tags include both 'embed_tokens' and 'lm_head.weight'
```

### Coupled Modification Tracking

```python
# Query coupled modifications
coupled_mods = memory.observations.query(
    category="coupled_modification"
)

print(f"Found {len(coupled_mods)} coupled modifications")
for mod in coupled_mods:
    print(f"  Modified: {mod.data['primary_layer']}")
    print(f"  Also affected: {mod.data['coupled_layers']}")
```

### Why This Matters

**Without tracking**:
```python
# You think you made 2 modifications:
modify(embed_tokens)  # Observation 1
modify(lm_head)       # Observation 2

# Pattern detector thinks:
"Modifying both layers together improves performance"  # ❌ FALSE!
```

**With tracking**:
```python
# System knows it's 1 coupled modification:
modify(embed_tokens)  # Observation: coupled_modification
# Automatically knows lm_head was also affected

# Pattern detector correctly understands:
"This is an architectural coupling, not a learned pattern"  # ✅ TRUE!
```

---

## Emergency Procedures

### Scenario 1: NaN Values Detected

**Symptoms**:
- Model outputs NaN
- Losses become NaN
- Weights contain NaN

**Immediate Action**:
```python
# 1. Stop all modifications immediately
print("🚨 NaN DETECTED - EMERGENCY STOP")

# 2. Check which layers are affected
for name, param in model.named_parameters():
    if torch.isnan(param).any():
        print(f"  ❌ NaN in {name}")

# 3. Restore last good checkpoint
checkpoint_id = "checkpoint_20251107_031500_a1b2c3d4"  # Your baseline
success = checkpointer.restore_checkpoint(checkpoint_id)

if success:
    print("✓ Restored to safe state")
else:
    print("✗ Restore failed - manual intervention needed")

# 4. Verify restoration
result = safety.check_model_health(model)
print(f"Post-restore status: {result['status']}")
```

**Prevention**:
- Use gradient clipping
- Smaller learning rates
- Monitor gradient magnitudes
- Checkpoint frequently

### Scenario 2: Model Performance Degrades

**Symptoms**:
- Perplexity increases significantly
- Quality drops
- Gibberish output

**Immediate Action**:
```python
# 1. Stop modifications
print("⚠️  PERFORMANCE DEGRADATION DETECTED")

# 2. Run diagnostic
from benchmarks import run_benchmarks

current_metrics = run_benchmarks(model, tokenizer)
print(f"Current perplexity: {current_metrics['perplexity']:.2f}")

# 3. Compare with baseline
baseline_metrics = load_baseline_metrics()
if current_metrics['perplexity'] > baseline_metrics['perplexity'] * 1.5:
    print("🚨 Perplexity increased >50% - Rolling back")
    checkpointer.restore_checkpoint(baseline_id)

# 4. Re-benchmark
new_metrics = run_benchmarks(model, tokenizer)
print(f"After rollback: {new_metrics['perplexity']:.2f}")
```

### Scenario 3: Out of Memory

**Symptoms**:
- CUDA out of memory error
- System freezes
- Can't save checkpoint

**Immediate Action**:
```python
import torch
import gc

# 1. Clear cache
torch.cuda.empty_cache()
gc.collect()
print("✓ Cleared CUDA cache")

# 2. Reduce batch size
batch_size = batch_size // 2
print(f"Reduced batch size to {batch_size}")

# 3. Move to CPU if necessary
if torch.cuda.is_available():
    model = model.cpu()
    torch.cuda.empty_cache()
    print("Moved model to CPU")

# 4. Try checkpoint save again
try:
    checkpoint_id = checkpointer.save_checkpoint()
    print(f"✓ Checkpoint saved: {checkpoint_id}")
except Exception as e:
    print(f"✗ Still can't checkpoint: {e}")
```

### Scenario 4: Checkpoint Restore Fails

**Symptoms**:
- Restore function returns False
- Model behaves strangely after restore
- File not found errors

**Immediate Action**:
```python
# 1. List all checkpoints
checkpoints = checkpointer.list_checkpoints()
print(f"Available checkpoints: {len(checkpoints)}")

# 2. Try alternative checkpoint
if len(checkpoints) > 1:
    alt_checkpoint = checkpoints[-2]['id']  # Second most recent
    print(f"Trying alternative: {alt_checkpoint}")
    success = checkpointer.restore_checkpoint(alt_checkpoint)

# 3. If all checkpoints fail, reload from disk
if not success:
    print("🚨 All checkpoints failed - reloading from disk")
    from model_manager import ModelManager
    
    manager = ModelManager("Qwen/Qwen2.5-3B-Instruct")
    manager.load_model()
    model = manager.model
    print("✓ Reloaded fresh model from disk")

# 4. Create new baseline
checkpoint_id = checkpointer.save_checkpoint(
    metadata={"purpose": "New baseline after recovery"}
)
```

---

## Best Practices

### Pre-Modification Checklist ✅

Before making ANY modification:

- [ ] Baseline checkpoint created
- [ ] Current metrics recorded (perplexity, etc.)
- [ ] Safety monitoring enabled
- [ ] Weight sharing detected and documented
- [ ] Modification plan documented in memory system
- [ ] Emergency rollback procedure ready
- [ ] Disk space available for checkpoints (>5GB)

### During Modification Checklist ✅

While modifying:

- [ ] Monitor continuously for NaN/Inf
- [ ] Check activations after each change
- [ ] Record observations in memory system
- [ ] Save incremental checkpoints
- [ ] Compare with baseline frequently
- [ ] Log all changes with metadata

### Post-Modification Checklist ✅

After modification:

- [ ] Run full benchmark suite
- [ ] Compare with baseline metrics
- [ ] Check for weight sharing side-effects
- [ ] Verify model still generates sensible text
- [ ] Save final checkpoint with results
- [ ] Document findings in memory system
- [ ] Form theories about what worked/didn't work

### Code Template: Safe Modification

```python
def safe_modification_experiment(
    model,
    modification_fn,
    test_fn,
    experiment_name
):
    """
    Template for safe modification experiments
    
    Args:
        model: The model to modify
        modification_fn: Function that performs modification
        test_fn: Function that tests the modification
        experiment_name: Unique experiment identifier
    """
    from checkpointing import CheckpointManager
    from safety_monitor import SafetyMonitor
    from memory import MemorySystem
    
    # Setup
    checkpointer = CheckpointManager(model, "data/checkpoints")
    safety = SafetyMonitor()
    memory = MemorySystem(f"data/memory/{experiment_name}")
    
    # Pre-modification
    print(f"🚀 Starting experiment: {experiment_name}")
    
    # 1. Baseline checkpoint
    baseline_id = checkpointer.save_checkpoint(
        metadata={"state": "baseline", "experiment": experiment_name}
    )
    print(f"✓ Baseline checkpoint: {baseline_id}")
    
    # 2. Record baseline metrics
    baseline_metrics = test_fn(model)
    memory.record_observation(
        f"Baseline metrics: {baseline_metrics}",
        category="baseline",
        metadata={"checkpoint": baseline_id, **baseline_metrics}
    )
    
    # Modification
    try:
        # 3. Perform modification
        print("🔧 Applying modification...")
        modification_result = modification_fn(model)
        
        # 4. Safety check
        health = safety.check_model_health(model)
        if health['status'] == 'CRITICAL':
            raise RuntimeError(f"Critical safety issue: {health['message']}")
        
        # 5. Test modification
        print("🧪 Testing modification...")
        test_results = test_fn(model)
        
        # 6. Compare with baseline
        improvement = (
            baseline_metrics['perplexity'] - test_results['perplexity']
        ) / baseline_metrics['perplexity'] * 100
        
        print(f"Performance change: {improvement:+.2f}%")
        
        # 7. Decide: keep or rollback
        if improvement > 0:  # Improved
            # Save successful modification
            success_id = checkpointer.save_checkpoint(
                metadata={
                    "state": "success",
                    "experiment": experiment_name,
                    "improvement": improvement
                }
            )
            
            memory.record_observation(
                f"Modification successful: {improvement:+.2f}% improvement",
                category="success",
                importance=0.9,
                metadata={
                    "checkpoint": success_id,
                    "baseline": baseline_id,
                    **test_results
                }
            )
            
            print(f"✓ Modification successful - checkpoint: {success_id}")
            return {"success": True, "checkpoint": success_id, "improvement": improvement}
            
        else:  # No improvement or degraded
            # Rollback
            checkpointer.restore_checkpoint(baseline_id)
            
            memory.record_observation(
                f"Modification unsuccessful: {improvement:+.2f}% change",
                category="failure",
                importance=0.7,
                metadata={"rolled_back": True, **test_results}
            )
            
            print(f"✗ Modification unsuccessful - rolled back")
            return {"success": False, "improvement": improvement}
            
    except Exception as e:
        # Emergency rollback
        print(f"🚨 EXCEPTION: {e}")
        checkpointer.restore_checkpoint(baseline_id)
        
        memory.record_observation(
            f"Experiment failed with exception: {str(e)}",
            category="error",
            importance=0.8
        )
        
        print("Emergency rollback complete")
        return {"success": False, "error": str(e)}

# Usage:
result = safe_modification_experiment(
    model=model,
    modification_fn=lambda m: modify_attention_heads(m, scale=1.1),
    test_fn=lambda m: run_benchmarks(m, tokenizer),
    experiment_name="attention_scaling_001"
)
```

---

## Safety Metrics Dashboard

### Key Metrics to Track

```python
def get_safety_dashboard(model, checkpointer, safety):
    """Get current safety status"""
    
    dashboard = {
        "checkpoints": {
            "total": len(checkpointer.list_checkpoints()),
            "latest": checkpointer.list_checkpoints()[-1]['id'],
            "disk_usage_gb": checkpointer.get_disk_usage() / 1e9
        },
        "health": safety.check_model_health(model),
        "alerts": {
            "critical": len(safety.get_alerts_by_severity('CRITICAL')),
            "high": len(safety.get_alerts_by_severity('HIGH')),
            "medium": len(safety.get_alerts_by_severity('MEDIUM'))
        },
        "weights": {
            "nan_count": sum(
                torch.isnan(p).sum().item()
                for p in model.parameters()
            ),
            "inf_count": sum(
                torch.isinf(p).sum().item()
                for p in model.parameters()
            )
        }
    }
    
    return dashboard

# Use it:
dashboard = get_safety_dashboard(model, checkpointer, safety)
print(f"""
Safety Dashboard
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Checkpoints:
  Total: {dashboard['checkpoints']['total']}
  Latest: {dashboard['checkpoints']['latest']}
  Disk: {dashboard['checkpoints']['disk_usage_gb']:.2f} GB

Health: {dashboard['health']['status']}

Alerts:
  🔴 Critical: {dashboard['alerts']['critical']}
  🟠 High: {dashboard['alerts']['high']}
  🟡 Medium: {dashboard['alerts']['medium']}

Weight Health:
  NaN values: {dashboard['weights']['nan_count']}
  Inf values: {dashboard['weights']['inf_count']}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")
```

---

## Summary: Safety Rules of Thumb

1. **Checkpoint Obsessively** 💾
   - Before every modification
   - After every success
   - Every 10-15 minutes during experiments

2. **Monitor Constantly** 👀
   - Check for NaN/Inf after each step
   - Watch activations during inference
   - Track metrics continuously

3. **Start Small** 🐜
   - Tiny modifications first
   - Single layer before multiple
   - Test thoroughly before scaling

4. **Document Everything** 📝
   - Record every observation
   - Note every checkpoint
   - Log all decisions

5. **Respect Weight Sharing** ⚠️
   - Always check for coupled layers
   - Use memory system tracking
   - Understand side-effects

6. **Have an Exit Strategy** 🚪
   - Know your baseline checkpoint ID
   - Keep rollback ready
   - Don't hesitate to abort

7. **Trust Your Safety Systems** 🛡️
   - Don't ignore warnings
   - Investigate all alerts
   - When in doubt, rollback

---

**Remember**: Every AI safety pioneer eventually triggers their first emergency rollback. The key is having the systems in place to recover gracefully!

**Stay safe out there!** 🚀🛡️

---

**Last Updated**: November 7, 2025  
**Version**: Phase 0 Complete  
**Next**: Read the Introspection Guide for advanced analysis techniques
