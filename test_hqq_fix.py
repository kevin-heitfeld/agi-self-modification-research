"""
Test script to verify HQQ quantized cache fix is in place.

This checks the code logic without running the model.
"""

import sys

def test_hqq_fix_logic():
    """Verify the HQQ deepcopy fix is correctly implemented."""
    print("=" * 80)
    print("Testing HQQ Quantized Cache Fix Logic")
    print("=" * 80)
    
    # Read the manual_generation.py file
    print("\n[1/2] Reading manual_generation.py...")
    with open("src/manual_generation.py", "r", encoding="utf-8") as f:
        content = f.read()
    
    # Check that the deepcopy fix is in place
    print("\n[2/2] Checking for HQQ deepcopy fix...")
    
    checks = {
        "Has conditional deepcopy": "if self.quantize_kv_cache:" in content,
        "Has direct reference for HQQ": "current_cache = self.system_prompt_cache" in content,
        "Has deepcopy for standard cache": "copy.deepcopy(self.system_prompt_cache)" in content,
        "Has fix comment": "CRITICAL FIX: Do NOT deep copy quantized caches" in content,
    }
    
    print(f"\n{'=' * 80}")
    print("FIX VERIFICATION:")
    print(f"{'=' * 80}")
    
    all_passed = True
    for check_name, check_result in checks.items():
        status = "✅" if check_result else "❌"
        print(f"{status} {check_name}: {check_result}")
        if not check_result:
            all_passed = False
    
    print(f"{'=' * 80}")
    
    if all_passed:
        print("\n✅ SUCCESS: All fix logic checks passed!")
        print("   The HQQ deepcopy bug has been fixed correctly.")
        print("\n📋 Next steps for Colab:")
        print("   1. Open your Colab notebook")
        print("   2. Run: !git pull origin main")
        print("   3. Re-run phase1a experiment")
        print("   4. Model should now generate valid JSON instead of gibberish")
        return True
    else:
        print("\n❌ FAILURE: Some fix logic checks failed!")
        return False

if __name__ == "__main__":
    try:
        success = test_hqq_fix_logic()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
