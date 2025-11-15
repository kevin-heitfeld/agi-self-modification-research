"""
Test script to verify GPU monitoring is working

This script checks that GPU snapshots are being taken and 
the summary is being printed correctly.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.gpu_monitor import GPUMonitor
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_gpu_monitor():
    """Test the GPU monitor functionality"""
    logger.info("="*80)
    logger.info("GPU MONITOR TEST")
    logger.info("="*80)
    
    # Initialize monitor
    monitor = GPUMonitor(logger=logger, gpu_total_gb=22.0)
    
    if not monitor.cuda_available:
        logger.warning("CUDA not available - test will show empty results")
        logger.warning("This is expected on CPU-only systems")
    
    # Simulate experiment lifecycle
    monitor.snapshot("session_start")
    monitor.snapshot("after_model_load")
    monitor.snapshot("after_initialization")
    
    # Simulate a few iterations
    for i in range(5):
        monitor.snapshot("generation_start", {"iteration": i+1, "conversation_turns": i*2})
        monitor.snapshot("generation_end", {"iteration": i+1, "response_length": 500})
    
    monitor.snapshot("experiment_end", {"total_iterations": 5})
    
    # Print summary
    monitor.print_summary(
        current_limits={
            "max_new_tokens": 500,
            "max_conversation_tokens": 2000,
            "keep_recent_turns": 3
        },
        include_recommendations=True
    )
    
    logger.info("\n✓ GPU monitor test complete!")
    logger.info(f"✓ Captured {len(monitor.snapshots)} snapshots")
    
    if monitor.snapshots:
        logger.info("\nSnapshot events captured:")
        for snapshot in monitor.snapshots:
            logger.info(f"  - {snapshot['event']}")

if __name__ == "__main__":
    test_gpu_monitor()
