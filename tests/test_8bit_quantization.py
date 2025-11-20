"""
Test script to verify 8-bit model quantization.

This script:
1. Loads the model in float16 (default)
2. Checks memory usage
3. Unloads the model
4. Loads the model in 8-bit quantization
5. Compares memory usage

Expected result: ~50% memory savings with 8-bit quantization
"""

import logging
import torch
from src.model_manager import ModelManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def get_gpu_memory_mb():
    """Get current GPU memory usage in MB"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024**2)
    return 0

def test_float16_loading():
    """Test loading in float16 (default)"""
    print("\n" + "="*60)
    print("TEST 1: Loading model in float16 (default)")
    print("="*60)
    
    manager = ModelManager()
    
    # Clear GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    initial_memory = get_gpu_memory_mb()
    print(f"Initial GPU memory: {initial_memory:.1f} MB")
    
    # Load model
    success = manager.load_model(use_flash_attention=False)  # Disable flash attn for simplicity
    
    if not success:
        print("❌ Failed to load model in float16")
        return None
    
    after_load_memory = get_gpu_memory_mb()
    model_memory = after_load_memory - initial_memory
    
    print(f"After load GPU memory: {after_load_memory:.1f} MB")
    print(f"Model memory usage: {model_memory:.1f} MB ({model_memory/1024:.2f} GB)")
    
    # Test generation
    try:
        output = manager.generate("Hello", max_length=10)
        print(f"✓ Generation works: '{output[:50]}...'")
    except Exception as e:
        print(f"❌ Generation failed: {e}")
    
    # Unload
    manager.unload_model()
    
    return model_memory

def test_8bit_loading():
    """Test loading with 8-bit quantization"""
    print("\n" + "="*60)
    print("TEST 2: Loading model with 8-bit quantization")
    print("="*60)
    
    manager = ModelManager()
    
    # Clear GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    initial_memory = get_gpu_memory_mb()
    print(f"Initial GPU memory: {initial_memory:.1f} MB")
    
    # Load model with 8-bit quantization
    success = manager.load_model(
        use_flash_attention=False,  # Disable flash attn for simplicity
        quantize_model="8bit"
    )
    
    if not success:
        print("❌ Failed to load model with 8-bit quantization")
        return None
    
    after_load_memory = get_gpu_memory_mb()
    model_memory = after_load_memory - initial_memory
    
    print(f"After load GPU memory: {after_load_memory:.1f} MB")
    print(f"Model memory usage: {model_memory:.1f} MB ({model_memory/1024:.2f} GB)")
    
    # Test generation
    try:
        output = manager.generate("Hello", max_length=10)
        print(f"✓ Generation works: '{output[:50]}...'")
    except Exception as e:
        print(f"❌ Generation failed: {e}")
    
    # Unload
    manager.unload_model()
    
    return model_memory

def main():
    print("="*60)
    print("8-BIT QUANTIZATION TEST")
    print("="*60)
    
    if not torch.cuda.is_available():
        print("❌ No GPU available - quantization requires GPU")
        return
    
    # Test float16
    float16_memory = test_float16_loading()
    
    if float16_memory is None:
        print("\n❌ Float16 test failed, aborting")
        return
    
    # Test 8-bit
    bit8_memory = test_8bit_loading()
    
    if bit8_memory is None:
        print("\n❌ 8-bit test failed")
        return
    
    # Compare
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"Float16 memory:  {float16_memory:.1f} MB ({float16_memory/1024:.2f} GB)")
    print(f"8-bit memory:    {bit8_memory:.1f} MB ({bit8_memory/1024:.2f} GB)")
    print(f"Memory savings:  {float16_memory - bit8_memory:.1f} MB ({(float16_memory - bit8_memory)/1024:.2f} GB)")
    print(f"Reduction:       {(1 - bit8_memory/float16_memory)*100:.1f}%")
    
    expected_savings = 0.4  # Expect ~40-50% savings
    actual_savings = (1 - bit8_memory/float16_memory)
    
    if actual_savings >= expected_savings:
        print(f"\n✅ SUCCESS! Achieved {actual_savings*100:.1f}% memory savings (expected ~50%)")
    else:
        print(f"\n⚠️  WARNING: Only achieved {actual_savings*100:.1f}% savings (expected ~50%)")

if __name__ == "__main__":
    main()
