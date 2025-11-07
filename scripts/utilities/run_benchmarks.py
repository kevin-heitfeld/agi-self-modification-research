"""
Run baseline benchmarks on Llama 3.2 3B
Establishes performance baseline before any modifications
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from model_manager import ModelManager
from benchmarks import BenchmarkRunner
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def main():
    print("=" * 70)
    print("BASELINE BENCHMARK SUITE")
    print("Phase 0 - Week 2")
    print("=" * 70)
    print()
    
    print("This will run the following benchmarks:")
    print("  • MMLU Sample - General knowledge")
    print("  • HellaSwag Sample - Commonsense reasoning")
    print("  • GSM8K Sample - Mathematical reasoning")
    print("  • Perplexity Test - Language modeling quality")
    print("  • Generation Test - Text generation capability")
    print()
    print("Note: Using minimal samples. Full datasets can be added later.")
    print()
    
    input("Press Enter to continue...")
    print()
    
    # Load model
    print("Loading model...")
    manager = ModelManager(model_name="Qwen/Qwen2.5-3B-Instruct")
    
    # Try loading (might need token)
    token = None
    import os
    if 'HF_TOKEN' in os.environ:
        token = os.environ['HF_TOKEN']
    
    success = manager.load_model(use_auth_token=token)
    
    if not success:
        print()
        print("✗ Failed to load model")
        print()
        print("If you haven't downloaded the model yet:")
        print("  python scripts/download_model.py")
        print()
        print("If the model is downloaded but needs authentication:")
        print("  Set HF_TOKEN environment variable")
        return
    
    print("✓ Model loaded successfully")
    print()
    
    # Show model info
    info = manager.get_model_info()
    print("Model Information:")
    print(f"  Device: {info['device']}")
    print(f"  Parameters: {info['num_parameters']:,}")
    print(f"  Dtype: {info['dtype']}")
    if 'gpu_memory_allocated' in info:
        print(f"  GPU Memory: {info['gpu_memory_allocated']}")
    print()
    
    # Run benchmarks
    runner = BenchmarkRunner(manager)
    results = runner.run_all_benchmarks()
    
    # Print summary
    print("\nBENCHMARK SUMMARY:")
    print("-" * 70)
    
    benchmarks = results["benchmarks"]
    
    if "mmlu_sample" in benchmarks:
        print(f"MMLU Sample:      {benchmarks['mmlu_sample']['accuracy']:.1%} "
              f"({benchmarks['mmlu_sample']['correct']}/{benchmarks['mmlu_sample']['total']})")
    
    if "hellaswag_sample" in benchmarks:
        print(f"HellaSwag Sample: {benchmarks['hellaswag_sample']['accuracy']:.1%} "
              f"({benchmarks['hellaswag_sample']['correct']}/{benchmarks['hellaswag_sample']['total']})")
    
    if "gsm8k_sample" in benchmarks:
        print(f"GSM8K Sample:     {benchmarks['gsm8k_sample']['accuracy']:.1%} "
              f"({benchmarks['gsm8k_sample']['correct']}/{benchmarks['gsm8k_sample']['total']})")
    
    if "perplexity" in benchmarks:
        print(f"Perplexity:       {benchmarks['perplexity']['perplexity']:.2f}")
    
    print("-" * 70)
    print()
    print("✓ Week 2 benchmarks complete!")
    print()
    print("Next steps:")
    print("  • Review results in data/benchmarks/")
    print("  • Document baseline performance")
    print("  • Begin Month 2: Build Introspection APIs")


if __name__ == "__main__":
    main()
