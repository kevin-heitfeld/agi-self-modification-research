"""
Script to download Llama 3.2 3B model
Run this before running benchmarks
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from model_manager import ModelManager
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def main():
    print("=" * 70)
    print("PHI-3.5-MINI-INSTRUCT MODEL DOWNLOAD")
    print("=" * 70)
    print()

    # Instructions
    print("No authentication required - fully open model!")
    print()
    print("MODEL INFO:")
    print("- Model: Microsoft Phi-3.5-mini-instruct")
    print("- Size: ~7.5GB download")
    print("- Parameters: 3.82 billion")
    print("- License: MIT License (fully open)")
    print()

    input("Press Enter to start download...")
    print()
    print("Starting download...")
    print("=" * 70)
    print()

    # Create model manager and download
    manager = ModelManager(model_name="Qwen/Qwen2.5-3B-Instruct")
    
    success = manager.download_model()
    
    print()
    print("=" * 70)
    if success:
        print("✓ MODEL DOWNLOAD COMPLETE")
        print()
        print("Model cached in: models/")
        print()
        print("Next steps:")
        print("1. Run baseline benchmarks: python scripts/run_benchmarks.py")
        print("2. Test generation: python -c \"from src.model_manager import ModelManager; m=ModelManager(); m.load_model(); print(m.generate('Hello'))\"")
    else:
        print("✗ MODEL DOWNLOAD FAILED")
        print()
        print("Troubleshooting:")
        print("- Ensure you have ~15GB free disk space")
        print("- Check your internet connection")
        print("- Try running again (downloads can resume)")
    print("=" * 70)


if __name__ == "__main__":
    main()
