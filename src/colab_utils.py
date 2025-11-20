"""
Colab Utilities

Helpers for detecting and managing Google Colab runtime environment,
including timeout detection, storage mounting, and session management.

Author: AGI Self-Modification Research Team
Date: November 19, 2025
"""

import os
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple


class ColabEnvironment:
    """Detect and manage Google Colab runtime environment"""
    
    @staticmethod
    def is_colab() -> bool:
        """Check if running in Google Colab
        
        Returns:
            True if running in Colab, False otherwise
        """
        return os.path.exists("/content")
    
    @staticmethod
    def is_drive_mounted() -> bool:
        """Check if Google Drive is mounted
        
        Returns:
            True if Google Drive is accessible, False otherwise
        """
        return os.path.exists("/content/drive/MyDrive")
    
    @staticmethod
    def get_drive_path() -> Optional[Path]:
        """Get Google Drive mount path if available
        
        Returns:
            Path to MyDrive if mounted, None otherwise
        """
        if ColabEnvironment.is_drive_mounted():
            return Path("/content/drive/MyDrive")
        return None
    
    @staticmethod
    def detect_storage() -> Tuple[Optional[str], Optional[Path]]:
        """Detect available cloud storage and return configuration
        
        Checks for:
        1. Google Drive (directly mounted)
        2. Other cloud storage via rclone (OneDrive, Dropbox, Nextcloud)
        
        Returns:
            Tuple of (storage_type, storage_path)
            storage_type: 'google_drive', 'rclone', or None
            storage_path: Path to storage root, or None if not available
        """
        # Check Google Drive
        if ColabEnvironment.is_drive_mounted():
            return ('google_drive', Path("/content/drive/MyDrive"))
        
        # Check for rclone-mounted storage
        if ColabEnvironment.is_colab():
            try:
                with open('/tmp/storage_config.txt', 'r') as f:
                    lines = f.read().strip().split('\n')
                    if len(lines) >= 2:
                        storage_root = lines[0]
                        return ('rclone', Path(storage_root))
            except (FileNotFoundError, IndexError):
                pass
        
        return (None, None)
    
    @staticmethod
    def get_project_path(project_name: str = "agi-self-modification-research") -> Path:
        """Get project directory path based on environment
        
        Args:
            project_name: Name of the project directory
            
        Returns:
            Path to project directory
        """
        storage_type, storage_path = ColabEnvironment.detect_storage()
        
        if storage_path:
            return storage_path / project_name
        
        # Fallback to local Colab storage
        if ColabEnvironment.is_colab():
            return Path(f"/content/{project_name}")
        
        # Running locally
        return Path.cwd()


class ColabTimeoutTracker:
    """Track Colab session runtime and predict timeouts
    
    Colab runtime limits:
    - Free tier: 12 hours max runtime, 90 min idle timeout
    - Pro tier: 24 hours max runtime, longer idle timeout
    - Pro+: Extended limits
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize timeout tracker
        
        Args:
            logger: Optional logger for warnings/info
        """
        self.logger = logger or logging.getLogger(__name__)
        self.session_start_time = datetime.now()
        self.last_save_time = datetime.now()
        
        # Configurable thresholds (in hours)
        self.free_tier_warning_hours = 10.0  # Warn 2 hours before 12h limit
        self.pro_tier_warning_hours = 20.0   # Warn 4 hours before 24h limit
        self.auto_save_interval_minutes = 30.0  # Save every 30 minutes
        
        if ColabEnvironment.is_colab():
            self.logger.info("[COLAB] Timeout tracker initialized")
            self.logger.info(f"[COLAB] Auto-save interval: {self.auto_save_interval_minutes} minutes")
    
    def get_elapsed_hours(self) -> float:
        """Get hours elapsed since session start
        
        Returns:
            Hours elapsed as float
        """
        elapsed = datetime.now() - self.session_start_time
        return elapsed.total_seconds() / 3600
    
    def get_minutes_since_save(self) -> float:
        """Get minutes elapsed since last save
        
        Returns:
            Minutes elapsed as float
        """
        elapsed = datetime.now() - self.last_save_time
        return elapsed.total_seconds() / 60
    
    def should_save(self) -> Tuple[bool, str]:
        """Check if we should save based on elapsed time
        
        Returns:
            Tuple of (should_save, reason)
            should_save: True if save recommended
            reason: Human-readable reason string
        """
        if not ColabEnvironment.is_colab():
            return (False, "Not in Colab")
        
        elapsed_hours = self.get_elapsed_hours()
        minutes_since_save = self.get_minutes_since_save()
        
        # Check auto-save interval
        if minutes_since_save >= self.auto_save_interval_minutes:
            return (True, f"{self.auto_save_interval_minutes:.0f} minutes since last save")
        
        # Check free tier warning (10 hours)
        if elapsed_hours >= self.free_tier_warning_hours and elapsed_hours < self.pro_tier_warning_hours:
            return (True, f"Session running {elapsed_hours:.1f} hours (approaching free tier 12h limit)")
        
        # Check pro tier warning (20 hours)
        if elapsed_hours >= self.pro_tier_warning_hours:
            return (True, f"Session running {elapsed_hours:.1f} hours (approaching pro tier 24h limit)")
        
        return (False, "No timeout concerns")
    
    def mark_saved(self):
        """Mark that a save has occurred"""
        self.last_save_time = datetime.now()
        
    def get_session_info(self) -> dict:
        """Get session timing information
        
        Returns:
            Dict with timing information
        """
        elapsed_hours = self.get_elapsed_hours()
        minutes_since_save = self.get_minutes_since_save()
        
        return {
            "session_start": self.session_start_time.isoformat(),
            "elapsed_hours": elapsed_hours,
            "last_save": self.last_save_time.isoformat(),
            "minutes_since_save": minutes_since_save,
            "is_colab": ColabEnvironment.is_colab(),
            "estimated_tier": self._estimate_tier(elapsed_hours)
        }
    
    def _estimate_tier(self, elapsed_hours: float) -> str:
        """Estimate Colab tier based on runtime
        
        Args:
            elapsed_hours: Hours elapsed in session
            
        Returns:
            Estimated tier string
        """
        if elapsed_hours < 12:
            return "Unknown (< 12h)"
        elif elapsed_hours < 24:
            return "Likely Pro or Pro+ (> 12h)"
        else:
            return "Likely Pro+ (> 24h)"


class ColabStorageManager:
    """Manage cloud storage configuration for Colab"""
    
    CONFIG_FILE = "/tmp/storage_config.txt"
    
    @staticmethod
    def save_storage_config(storage_root: str, project_dir: str):
        """Save storage configuration to temporary file
        
        Args:
            storage_root: Root path of cloud storage
            project_dir: Full path to project directory
        """
        with open(ColabStorageManager.CONFIG_FILE, 'w') as f:
            f.write(f"{storage_root}\n")
            f.write(f"{project_dir}\n")
    
    @staticmethod
    def load_storage_config() -> Optional[Tuple[str, str]]:
        """Load storage configuration from temporary file
        
        Returns:
            Tuple of (storage_root, project_dir) or None if not found
        """
        try:
            with open(ColabStorageManager.CONFIG_FILE, 'r') as f:
                lines = f.read().strip().split('\n')
                if len(lines) >= 2:
                    return (lines[0], lines[1])
        except FileNotFoundError:
            pass
        return None
    
    @staticmethod
    def get_memory_path(phase_name: str, logger: Optional[logging.Logger] = None) -> Path:
        """Get appropriate memory path based on environment
        
        Checks:
        1. Google Drive if mounted
        2. Other cloud storage via config file
        3. Local fallback
        
        Args:
            phase_name: Name of the phase (e.g., 'phase1a', 'phase1c_modified')
            logger: Optional logger for info messages
            
        Returns:
            Path to phase-specific memory directory
        """
        log = logger or logging.getLogger(__name__)
        
        # Check Google Drive
        if ColabEnvironment.is_drive_mounted():
            memory_path = Path("/content/drive/MyDrive/agi-self-modification-research/data/AGI_Memory") / phase_name
            log.info(f"  Using Google Drive for memory: {memory_path}")
            return memory_path
        
        # Check for other cloud storage via config
        if ColabEnvironment.is_colab():
            config = ColabStorageManager.load_storage_config()
            if config:
                storage_root, project_dir = config
                memory_path = Path(project_dir) / "data" / "AGI_Memory" / phase_name
                log.info(f"  Using cloud storage for memory: {memory_path}")
                return memory_path
            else:
                # Colab but no storage configured
                memory_path = Path(f"data/AGI_Memory/{phase_name}")
                log.warning(f"  No cloud storage detected - using local memory: {memory_path}")
                return memory_path
        
        # Running locally
        memory_path = Path(f"data/AGI_Memory/{phase_name}")
        return memory_path
    
    @staticmethod
    def get_heritage_path(logger: Optional[logging.Logger] = None) -> Path:
        """Get appropriate heritage path based on environment
        
        Checks:
        1. Google Drive if mounted
        2. Other cloud storage via config file
        3. Local fallback
        
        Args:
            logger: Optional logger for info messages
            
        Returns:
            Path to heritage directory
        """
        log = logger or logging.getLogger(__name__)
        
        # Check Google Drive
        if ColabEnvironment.is_drive_mounted():
            heritage_path = Path("/content/drive/MyDrive/agi-self-modification-research/data/heritage")
            log.info(f"  Using Google Drive for heritage: {heritage_path}")
            return heritage_path
        
        # Check for other cloud storage via config
        if ColabEnvironment.is_colab():
            config = ColabStorageManager.load_storage_config()
            if config:
                storage_root, project_dir = config
                heritage_path = Path(project_dir) / "data" / "heritage"
                log.info(f"  Using cloud storage for heritage: {heritage_path}")
                return heritage_path
            else:
                # Colab but no storage configured
                heritage_path = Path("heritage")
                log.warning(f"  No cloud storage detected - using local heritage: {heritage_path}")
                return heritage_path
        
        # Running locally
        heritage_path = Path("heritage")
        return heritage_path


# Convenience functions for backward compatibility
def is_colab() -> bool:
    """Check if running in Colab (convenience function)"""
    return ColabEnvironment.is_colab()


def check_colab_timeout(tracker: ColabTimeoutTracker, logger: Optional[logging.Logger] = None) -> bool:
    """Check if timeout is approaching (convenience function)
    
    Args:
        tracker: ColabTimeoutTracker instance
        logger: Optional logger for warnings
        
    Returns:
        True if should save, False otherwise
    """
    log = logger or logging.getLogger(__name__)
    should_save, reason = tracker.should_save()
    
    if should_save:
        log.warning(f"[TIMEOUT] {reason}")
    
    return should_save


__all__ = [
    'ColabEnvironment',
    'ColabTimeoutTracker',
    'ColabStorageManager',
    'is_colab',
    'check_colab_timeout'
]
