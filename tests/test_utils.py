"""
Test utilities for AGI Self-Modification Research tests.

Provides:
- Proper temp directory management using D:\temp
- Database connection cleanup helpers
- Common test fixtures

Author: AGI Self-Modification Research Team
Date: November 7, 2025
"""

import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional

# Use D:\temp for all test temporary files
TEST_TEMP_BASE = Path("D:/temp/agi_tests")


def get_test_temp_dir() -> str:
    """
    Get a temporary directory for tests using D:\temp.
    
    Returns:
        Path to temporary directory
    """
    # Ensure base directory exists
    TEST_TEMP_BASE.mkdir(parents=True, exist_ok=True)
    
    # Create a temporary subdirectory
    temp_dir = tempfile.mkdtemp(dir=str(TEST_TEMP_BASE))
    return temp_dir


def cleanup_temp_dir(temp_dir: str):
    """
    Safely clean up a temporary directory.
    
    Args:
        temp_dir: Directory to clean up
    """
    if temp_dir and os.path.exists(temp_dir):
        try:
            shutil.rmtree(temp_dir)
        except PermissionError:
            # On Windows, files might still be locked
            # Try to remove what we can
            try:
                for root, dirs, files in os.walk(temp_dir, topdown=False):
                    for name in files:
                        try:
                            os.remove(os.path.join(root, name))
                        except:
                            pass
                    for name in dirs:
                        try:
                            os.rmdir(os.path.join(root, name))
                        except:
                            pass
                os.rmdir(temp_dir)
            except:
                # If we still can't remove it, just leave it
                # (will be cleaned up on next test run or manually)
                pass


def close_memory_system(memory_system):
    """
    Properly close all database connections in a memory system.
    
    Args:
        memory_system: MemorySystem instance to close
    """
    if hasattr(memory_system, 'close'):
        memory_system.close()
    else:
        # Manually close each layer if close method doesn't exist
        if hasattr(memory_system, 'observations'):
            _close_layer(memory_system.observations)
        if hasattr(memory_system, 'patterns'):
            _close_layer(memory_system.patterns)
        if hasattr(memory_system, 'theories'):
            _close_layer(memory_system.theories)
        if hasattr(memory_system, 'beliefs'):
            _close_layer(memory_system.beliefs)


def close_memory_layers(*layers):
    """
    Close database connections for multiple memory layers.
    
    Args:
        *layers: Memory layer instances to close
    """
    for layer in layers:
        _close_layer(layer)


def _close_layer(layer):
    """Close a single memory layer."""
    if layer is None:
        return
    
    if hasattr(layer, 'close'):
        layer.close()
    elif hasattr(layer, 'conn') and layer.conn:
        try:
            layer.conn.close()
        except:
            pass
