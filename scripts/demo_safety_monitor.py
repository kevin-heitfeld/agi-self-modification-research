"""
Demonstration: Safety Monitor

This script demonstrates the SafetyMonitor's ability to detect anomalies,
track performance, and trigger emergency stops when things go wrong.

Shows:
1. Normal monitoring operations
2. Anomaly detection (NaN/Inf)
3. Performance degradation tracking
4. Resource monitoring
5. Emergency stop mechanism
6. Alert system

Author: AGI Self-Modification Research Team
Date: November 6, 2025
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.safety_monitor import SafetyMonitor, AlertLevel
import time


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def main():
    print_section("SAFETY MONITOR - REAL-TIME ANOMALY DETECTION")

    print("This demonstration shows how the safety monitor protects the system")
    print("by detecting anomalies and triggering emergency stops when needed.")
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

    # Initialize safety monitor with baseline metrics
    baseline_metrics = {
        'perplexity': 11.27,
        'accuracy': 0.5
    }

    monitor = SafetyMonitor(
        model=model,
        baseline_metrics=baseline_metrics
    )

    print("✓ Model and SafetyMonitor loaded!\n")

    # ==========================================================================
    # 1. NORMAL MONITORING
    # ==========================================================================
    print_section("1. NORMAL MONITORING - HEALTHY OPERATION")

    print("Starting monitoring for normal inference...")

    # Register hooks
    monitor.register_hooks()

    # Run normal inference with monitoring
    with monitor.context():
        text = "What is artificial intelligence?"
        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        start_time = time.time()
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        inference_time = (time.time() - start_time) * 1000

        # Check output
        is_safe = monitor.check_output(outputs.logits)
        print(f"✓ Output check: {'SAFE' if is_safe else 'UNSAFE'}")

        # Track inference time
        monitor.track_inference_time(inference_time)
        print(f"✓ Inference time: {inference_time:.2f}ms")

        # Check performance
        perplexity = 11.30  # Slightly higher than baseline
        is_acceptable = monitor.check_performance('perplexity', perplexity)
        print(f"✓ Performance check: {'ACCEPTABLE' if is_acceptable else 'DEGRADED'}")
        print(f"  Perplexity: {perplexity:.2f} (baseline: {baseline_metrics['perplexity']:.2f})")

    # Check resources
    resources = monitor.check_resources()
    print(f"\n✓ Resource usage:")
    if 'gpu_memory_allocated_mb' in resources:
        print(f"  GPU memory: {resources['gpu_memory_allocated_mb']:.0f} MB")
    print(f"  CPU memory: {resources['cpu_memory_mb']:.0f} MB")

    # ==========================================================================
    # 2. SIMULATED ANOMALY - NaN DETECTION
    # ==========================================================================
    print_section("2. ANOMALY DETECTION - NaN IN OUTPUT")

    print("Simulating a model that produces NaN outputs...")
    print("(This would happen if something went wrong with weights)\n")

    # Create a tensor with NaN
    bad_output = torch.randn(1, 10, 1000)
    bad_output[0, 5, :] = float('nan')  # Inject NaN

    with monitor.context():
        is_safe = monitor.check_output(bad_output)
        print(f"Output check: {'SAFE' if is_safe else 'UNSAFE ⚠️'}")

        if monitor.emergency_stop_triggered:
            print("\n🚨 EMERGENCY STOP ACTIVATED!")
            print("The system detected NaN and stopped immediately.")
            print("In a real scenario, this would trigger auto-rollback.\n")

    # Reset for next test
    monitor.reset_emergency_stop()
    monitor.reset_alerts()

    # ==========================================================================
    # 3. PERFORMANCE DEGRADATION
    # ==========================================================================
    print_section("3. PERFORMANCE DEGRADATION DETECTION")

    print("Simulating significant performance degradation...")

    with monitor.context():
        # Simulate catastrophic perplexity increase
        bad_perplexity = 30.0  # Way worse than baseline
        is_acceptable = monitor.check_performance('perplexity', bad_perplexity)

        print(f"Perplexity: {bad_perplexity:.2f} (baseline: {baseline_metrics['perplexity']:.2f})")
        print(f"Status: {'ACCEPTABLE' if is_acceptable else 'DEGRADED ⚠️'}")
        print(f"Increase: {(bad_perplexity - baseline_metrics['perplexity']) / baseline_metrics['perplexity'] * 100:.1f}%")

    # ==========================================================================
    # 4. ALERT SYSTEM
    # ==========================================================================
    print_section("4. ALERT SYSTEM - REVIEWING SAFETY EVENTS")

    all_alerts = monitor.get_recent_alerts()
    print(f"Total alerts: {len(all_alerts)}\n")

    # Group by level
    for level in AlertLevel:
        level_alerts = monitor.get_recent_alerts(level=level)
        if level_alerts:
            print(f"{level.value.upper()}: {len(level_alerts)} alerts")
            for alert in level_alerts[:3]:  # Show first 3
                print(f"  • {alert.category}: {alert.message}")
            if len(level_alerts) > 3:
                print(f"  ... and {len(level_alerts) - 3} more")
            print()

    # Critical alerts
    critical = monitor.get_critical_alerts()
    if critical:
        print(f"⚠️  {len(critical)} CRITICAL/EMERGENCY alerts:")
        for alert in critical:
            print(f"  • {alert}")
        print()

    # ==========================================================================
    # 5. MONITORING STATISTICS
    # ==========================================================================
    print_section("5. MONITORING STATISTICS")

    stats = monitor.get_statistics()

    print(f"Monitoring Status:")
    print(f"  Active: {stats['monitoring_active']}")
    print(f"  Emergency stop: {stats['emergency_stop_triggered']}")
    print(f"  Total alerts: {stats['total_alerts']}")

    print(f"\nAlerts by level:")
    for level, count in stats['alerts_by_level'].items():
        if count > 0:
            print(f"  {level}: {count}")

    print(f"\nInference times:")
    inf_stats = stats['inference_times']
    if inf_stats['count'] > 0:
        print(f"  Count: {inf_stats['count']}")
        print(f"  Mean: {inf_stats['mean']:.2f}ms")
        print(f"  Std: {inf_stats['std']:.2f}ms")
        print(f"  Range: {inf_stats['min']:.2f}ms - {inf_stats['max']:.2f}ms")

    print(f"\nResource usage:")
    res = stats['resource_stats']
    if 'gpu_memory_allocated_mb' in res:
        print(f"  GPU memory: {res['gpu_memory_allocated_mb']:.0f} MB")
        print(f"  GPU peak: {res['gpu_peak_memory_mb']:.0f} MB")
    print(f"  CPU memory: {res['cpu_memory_mb']:.0f} MB")
    print(f"  System CPU: {res['system_cpu_percent']:.1f}%")

    print(f"\nSafety thresholds:")
    for name, value in stats['thresholds'].items():
        print(f"  {name}: {value}")

    # ==========================================================================
    # 6. THRESHOLD ADJUSTMENT
    # ==========================================================================
    print_section("6. THRESHOLD ADJUSTMENT")

    print("Safety thresholds can be adjusted for different scenarios:")
    print("\nCurrent perplexity threshold:")
    print(f"  Max increase: {monitor.thresholds['max_perplexity_increase'] * 100:.0f}%")

    # Adjust threshold (more lenient)
    monitor.set_threshold('max_perplexity_increase', 3.0)
    print(f"\nAdjusted to: {monitor.thresholds['max_perplexity_increase'] * 100:.0f}%")
    print("(System will now tolerate 200% perplexity increase)")

    # Test with adjusted threshold
    with monitor.context():
        test_perplexity = 25.0
        is_acceptable = monitor.check_performance('perplexity', test_perplexity, track=False)
        print(f"\nTesting perplexity {test_perplexity:.2f}: {'ACCEPTABLE' if is_acceptable else 'DEGRADED'}")

    # ==========================================================================
    # FINAL REFLECTION
    # ==========================================================================
    print_section("PHILOSOPHICAL REFLECTION")

    print("The safety monitor provides critical protection:")
    print("")
    print("✓ Real-time anomaly detection")
    print("  • Catches NaN/Inf before they propagate")
    print("  • Detects activation anomalies")
    print("  • Identifies output inconsistencies")
    print("")
    print("✓ Performance degradation tracking")
    print("  • Compares against baseline metrics")
    print("  • Detects catastrophic drops")
    print("  • Triggers alerts automatically")
    print("")
    print("✓ Resource monitoring")
    print("  • Tracks GPU/CPU memory")
    print("  • Monitors inference speed")
    print("  • Prevents resource exhaustion")
    print("")
    print("✓ Emergency stop mechanism")
    print("  • Immediate halt on critical issues")
    print("  • Can trigger auto-rollback")
    print("  • Prevents cascading failures")
    print("")
    print("This creates a safety net for self-modification:")
    print("  • The system can experiment boldly")
    print("  • Knowing the monitor will catch problems")
    print("  • Before they cause permanent damage")
    print("")
    print("Combined with checkpointing:")
    print("  • Checkpoints = Ability to rollback")
    print("  • Safety Monitor = Knowing WHEN to rollback")
    print("  • Together = Safe self-modification")
    print("")
    print("Ready for Phase 1: The system can now safely examine itself,")
    print("experiment with modifications, and stop if anything goes wrong.")

    # Cleanup
    monitor.remove_hooks()


if __name__ == "__main__":
    main()
