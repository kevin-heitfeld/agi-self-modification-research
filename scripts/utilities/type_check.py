"""
Type Checking Script
Run with: python scripts/utilities/type_check.py
"""

import subprocess
import sys
from pathlib import Path

def run_pyright():
    """Run pyright type checker"""
    print("=" * 60)
    print("Running Pyright Type Checker")
    print("=" * 60)
    
    result = subprocess.run(
        ["pyright", "src/", "scripts/experiments/"],
        capture_output=False
    )
    
    return result.returncode == 0

def main():
    """Run type checking"""
    project_root = Path(__file__).parent.parent.parent
    
    print(f"Project root: {project_root}")
    print()
    
    success = run_pyright()
    
    print()
    if success:
        print("✅ Type checking passed!")
        return 0
    else:
        print("❌ Type checking found issues")
        print("\nFix type errors or add type: ignore comments where appropriate")
        return 1

if __name__ == "__main__":
    sys.exit(main())
