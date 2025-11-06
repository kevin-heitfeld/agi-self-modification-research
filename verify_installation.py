#!/usr/bin/env python3
"""
AGI Self-Modification Research - Installation Verification Script
This script verifies that all required packages are installed correctly
and checks system requirements.
"""

import sys
import platform
from typing import List, Tuple

def print_header(text: str):
    """Print a formatted header"""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)

def print_status(name: str, status: bool, details: str = ""):
    """Print a status line with checkmark or X"""
    symbol = "✓" if status else "✗"
    color = "\033[92m" if status else "\033[91m"
    reset = "\033[0m"
    
    # Windows cmd doesn't support ANSI colors well, use simpler format
    if platform.system() == "Windows":
        symbol = "[OK]" if status else "[FAIL]"
        print(f"{symbol} {name}")
    else:
        print(f"{color}{symbol}{reset} {name}")
    
    if details:
        print(f"    {details}")

def check_python_version() -> Tuple[bool, str]:
    """Check if Python version is appropriate"""
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    
    if version.major == 3 and version.minor in [10, 11]:
        return True, f"Python {version_str} (Recommended)"
    elif version.major == 3 and version.minor >= 10:
        return True, f"Python {version_str} (May work, but untested)"
    else:
        return False, f"Python {version_str} (Requires 3.10 or 3.11)"

def check_package(package_name: str, import_name: str = None) -> Tuple[bool, str]:
    """Check if a package is installed and return version"""
    if import_name is None:
        import_name = package_name
    
    try:
        module = __import__(import_name)
        version = getattr(module, '__version__', 'unknown version')
        return True, f"{package_name} {version}"
    except ImportError:
        return False, f"{package_name} not found"

def check_torch_cuda() -> Tuple[bool, str]:
    """Check PyTorch CUDA availability"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            cuda_version = torch.version.cuda
            return True, f"CUDA {cuda_version}, {gpu_name}, {gpu_memory:.1f}GB"
        else:
            return False, "CUDA not available (CPU mode - not recommended)"
    except ImportError:
        return False, "PyTorch not installed"

def check_disk_space() -> Tuple[bool, str]:
    """Check available disk space"""
    try:
        import shutil
        total, used, free = shutil.disk_usage(".")
        free_gb = free / (1024**3)
        
        if free_gb >= 500:
            return True, f"{free_gb:.1f} GB free (Excellent)"
        elif free_gb >= 200:
            return True, f"{free_gb:.1f} GB free (Sufficient)"
        elif free_gb >= 100:
            return True, f"{free_gb:.1f} GB free (Tight - may need cleanup)"
        else:
            return False, f"{free_gb:.1f} GB free (Insufficient - need 200+ GB)"
    except Exception as e:
        return False, f"Could not check disk space: {e}"

def check_memory() -> Tuple[bool, str]:
    """Check system RAM"""
    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / (1024**3)
        
        if ram_gb >= 32:
            return True, f"{ram_gb:.1f} GB RAM (Excellent)"
        elif ram_gb >= 16:
            return True, f"{ram_gb:.1f} GB RAM (Minimum - may struggle)"
        else:
            return False, f"{ram_gb:.1f} GB RAM (Insufficient - need 16+ GB)"
    except ImportError:
        return False, "psutil not installed (cannot check RAM)"
    except Exception as e:
        return False, f"Could not check RAM: {e}"

def main():
    print_header("AGI Self-Modification Research - Installation Verification")
    
    all_passed = True
    critical_failed = False
    
    # System Requirements
    print_header("System Requirements")
    
    status, details = check_python_version()
    print_status("Python Version", status, details)
    if not status:
        critical_failed = True
    all_passed = all_passed and status
    
    status, details = check_memory()
    print_status("System RAM", status, details)
    all_passed = all_passed and status
    
    status, details = check_disk_space()
    print_status("Disk Space", status, details)
    all_passed = all_passed and status
    
    # Core Deep Learning
    print_header("Core Deep Learning Packages")
    
    packages = [
        ("torch", "torch"),
        ("transformers", "transformers"),
        ("accelerate", "accelerate"),
    ]
    
    for package, import_name in packages:
        status, details = check_package(package, import_name)
        print_status(package, status, details)
        if not status:
            critical_failed = True
        all_passed = all_passed and status
    
    status, details = check_torch_cuda()
    print_status("PyTorch CUDA", status, details)
    if not status:
        print("    WARNING: This project requires GPU support!")
        critical_failed = True
    all_passed = all_passed and status
    
    # Memory & Knowledge Systems
    print_header("Memory & Knowledge Systems")
    
    packages = [
        ("chromadb", "chromadb"),
        ("networkx", "networkx"),
    ]
    
    for package, import_name in packages:
        status, details = check_package(package, import_name)
        print_status(package, status, details)
        if not status:
            critical_failed = True
        all_passed = all_passed and status
    
    # Development Tools
    print_header("Development & Exploration Tools")
    
    packages = [
        ("jupyter", "jupyter"),
        ("ipython", "IPython"),
        ("pytest", "pytest"),
    ]
    
    for package, import_name in packages:
        status, details = check_package(package, import_name)
        print_status(package, status, details)
        all_passed = all_passed and status
    
    # Data & Visualization
    print_header("Data & Visualization")
    
    packages = [
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("matplotlib", "matplotlib"),
        ("seaborn", "seaborn"),
    ]
    
    for package, import_name in packages:
        status, details = check_package(package, import_name)
        print_status(package, status, details)
        all_passed = all_passed and status
    
    # Monitoring (optional but recommended)
    print_header("Monitoring & Logging (Optional)")
    
    packages = [
        ("wandb", "wandb"),
        ("tensorboard", "tensorboard"),
    ]
    
    for package, import_name in packages:
        status, details = check_package(package, import_name)
        print_status(package, status, details)
        # Don't fail on optional packages
    
    # Utilities
    print_header("Utilities")
    
    packages = [
        ("tqdm", "tqdm"),
        ("rich", "rich"),
        ("pydantic", "pydantic"),
        ("psutil", "psutil"),
    ]
    
    for package, import_name in packages:
        status, details = check_package(package, import_name)
        print_status(package, status, details)
        all_passed = all_passed and status
    
    # Final Summary
    print_header("Verification Summary")
    
    if critical_failed:
        print("\n❌ CRITICAL FAILURES DETECTED")
        print("   Some essential packages or requirements are missing.")
        print("   Please review the failures above and reinstall.")
        print("\n   Rerun setup: setup.bat")
        return 1
    elif all_passed:
        print("\n✓ ALL CHECKS PASSED!")
        print("   Your environment is ready for Phase 0 implementation.")
        print("\n   Next steps:")
        print("     1. Review PHASE_0_DETAILED_PLAN.md")
        print("     2. Start with Month 1 Week 1 tasks")
        print("     3. Run: jupyter notebook")
        return 0
    else:
        print("\n⚠ SOME CHECKS FAILED")
        print("   Non-critical packages are missing but you can proceed.")
        print("   Review failures above - install missing packages if needed.")
        print("\n   Install missing packages: pip install -r requirements.txt")
        return 0

if __name__ == "__main__":
    sys.exit(main())
