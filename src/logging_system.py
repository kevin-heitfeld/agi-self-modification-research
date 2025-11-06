"""
Logging Infrastructure

Research-grade logging system for AI self-modification experiments.
Captures everything needed for reproducibility and safety monitoring.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import json


class ExperimentLogger:
    """
    Structured logger for research experiments.
    
    Logs to multiple outputs:
    1. Console (human-readable, INFO level)
    2. File (detailed, DEBUG level)
    3. Structured JSON (machine-readable, for analysis)
    """
    
    def __init__(
        self,
        name: str,
        log_dir: Optional[Path] = None,
        phase: int = 0,
        month: int = 1,
        week: int = 1
    ):
        self.name = name
        self.phase = phase
        self.month = month
        self.week = week
        
        # Determine log directory
        if log_dir is None:
            # Try to find project root
            current = Path.cwd()
            while current != current.parent:
                if (current / "requirements.txt").exists():
                    log_dir = current / "data" / "logs"
                    break
                current = current.parent
            else:
                log_dir = Path.cwd() / "logs"
        
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create session-specific log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_id = f"phase{phase}_m{month}_w{week}_{timestamp}"
        self.log_file = self.log_dir / f"{self.session_id}.log"
        self.json_file = self.log_dir / f"{self.session_id}.jsonl"
        
        # Set up Python logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers.clear()  # Remove any existing handlers
        
        # Console handler (INFO level, human-readable)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_format = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(console_format)
        self.logger.addHandler(console_handler)
        
        # File handler (DEBUG level, detailed)
        file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            '%(asctime)s [%(levelname)-8s] %(name)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        self.logger.addHandler(file_handler)
        
        # JSON file for structured logging
        self.json_handle = open(self.json_file, 'a', encoding='utf-8')
        
        # Log session start
        self.info(f"Session started: {self.session_id}")
        self.log_structured({
            "event": "session_start",
            "session_id": self.session_id,
            "phase": phase,
            "month": month,
            "week": week
        })
    
    def log_structured(self, data: Dict[str, Any]):
        """Write structured JSON log entry"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            **data
        }
        self.json_handle.write(json.dumps(entry) + '\n')
        self.json_handle.flush()
    
    def debug(self, message: str, **kwargs):
        """Log debug message"""
        self.logger.debug(message)
        if kwargs:
            self.log_structured({"level": "debug", "message": message, **kwargs})
    
    def info(self, message: str, **kwargs):
        """Log info message"""
        self.logger.info(message)
        if kwargs:
            self.log_structured({"level": "info", "message": message, **kwargs})
    
    def warning(self, message: str, **kwargs):
        """Log warning message"""
        self.logger.warning(message)
        self.log_structured({"level": "warning", "message": message, **kwargs})
    
    def error(self, message: str, **kwargs):
        """Log error message"""
        self.logger.error(message)
        self.log_structured({"level": "error", "message": message, **kwargs})
    
    def critical(self, message: str, **kwargs):
        """Log critical message (potential stop condition)"""
        self.logger.critical(message)
        self.log_structured({"level": "critical", "message": message, **kwargs})
    
    # Research-specific logging methods
    
    def log_experiment_start(self, description: str, hypothesis: str, parameters: Dict[str, Any]):
        """Log experiment initialization"""
        self.info(f"Experiment started: {description}")
        self.log_structured({
            "event": "experiment_start",
            "description": description,
            "hypothesis": hypothesis,
            "parameters": parameters
        })
    
    def log_checkpoint(self, checkpoint_name: str, trigger: str, metrics: Dict[str, Any]):
        """Log model checkpoint creation"""
        self.info(f"Checkpoint created: {checkpoint_name} (trigger: {trigger})")
        self.log_structured({
            "event": "checkpoint",
            "checkpoint_name": checkpoint_name,
            "trigger": trigger,
            "metrics": metrics
        })
    
    def log_introspection(self, query: str, response: Any, context: Optional[Dict] = None):
        """Log introspection query and response"""
        self.debug(f"Introspection: {query[:50]}...")
        self.log_structured({
            "event": "introspection",
            "query": query,
            "response": str(response),
            "context": context or {}
        })
    
    def log_modification(self, modification_type: str, parameters: Dict[str, Any], reason: str):
        """Log model modification (critical event)"""
        self.warning(f"Modification: {modification_type} - {reason}")
        self.log_structured({
            "event": "modification",
            "type": modification_type,
            "parameters": parameters,
            "reason": reason
        })
    
    def log_anomaly(self, anomaly_type: str, severity: str, details: Dict[str, Any]):
        """Log detected anomaly"""
        self.warning(f"Anomaly detected: {anomaly_type} (severity: {severity})")
        self.log_structured({
            "event": "anomaly",
            "type": anomaly_type,
            "severity": severity,
            "details": details
        })
    
    def log_safety_violation(self, violation_type: str, threshold: float, actual: float):
        """Log safety threshold violation (critical)"""
        self.critical(f"Safety violation: {violation_type} - threshold={threshold}, actual={actual}")
        self.log_structured({
            "event": "safety_violation",
            "type": violation_type,
            "threshold": threshold,
            "actual": actual
        })
    
    def log_heritage_event(self, event_type: str, details: Dict[str, Any]):
        """Log heritage-related events"""
        self.info(f"Heritage: {event_type}")
        self.log_structured({
            "event": "heritage",
            "type": event_type,
            "details": details
        })
    
    def log_system_response(self, prompt: str, response: str, context: Optional[Dict] = None):
        """Log system prompt and response"""
        self.debug(f"System response (prompt length: {len(prompt)})")
        self.log_structured({
            "event": "system_response",
            "prompt": prompt,
            "response": response,
            "context": context or {}
        })
    
    def log_discovery(self, discovery_type: str, description: str, evidence: Dict[str, Any]):
        """Log research discovery"""
        self.info(f"Discovery: {discovery_type} - {description}")
        self.log_structured({
            "event": "discovery",
            "type": discovery_type,
            "description": description,
            "evidence": evidence
        })
    
    def log_stop_condition(self, condition: str, reason: str, details: Dict[str, Any]):
        """Log stop condition trigger"""
        self.critical(f"STOP CONDITION: {condition} - {reason}")
        self.log_structured({
            "event": "stop_condition",
            "condition": condition,
            "reason": reason,
            "details": details
        })
    
    def close(self):
        """Close logger and finalize session"""
        self.info(f"Session ended: {self.session_id}")
        self.log_structured({"event": "session_end"})
        self.json_handle.close()
        
        # Close all handlers
        for handler in self.logger.handlers:
            handler.close()
    
    def __enter__(self):
        """Context manager support"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager cleanup"""
        if exc_type is not None:
            self.error(f"Session ended with exception: {exc_type.__name__}: {exc_val}")
        self.close()


def get_logger(
    name: str = "agi_research",
    phase: int = 0,
    month: int = 1,
    week: int = 1
) -> ExperimentLogger:
    """
    Get an experiment logger.
    
    Args:
        name: Logger name
        phase: Current phase (0-3)
        month: Current month (1-6)
        week: Current week (1-4)
    
    Returns:
        ExperimentLogger instance
    """
    return ExperimentLogger(name, phase=phase, month=month, week=week)


if __name__ == "__main__":
    # Test logging system
    print("Testing logging infrastructure...\n")
    
    with get_logger("test", phase=0, month=1, week=1) as logger:
        logger.info("This is an info message")
        logger.debug("This is a debug message (only in file)")
        logger.warning("This is a warning")
        
        # Test research-specific methods
        logger.log_experiment_start(
            description="Test experiment",
            hypothesis="Logger works correctly",
            parameters={"test_param": 123}
        )
        
        logger.log_checkpoint(
            checkpoint_name="test_checkpoint",
            trigger="manual",
            metrics={"accuracy": 0.95, "loss": 0.05}
        )
        
        logger.log_heritage_event(
            event_type="heritage_loaded",
            details={"document": "claude_original_conversation.md"}
        )
        
        logger.log_discovery(
            discovery_type="test",
            description="Logger successfully tested",
            evidence={"test_passed": True}
        )
    
    print("\n✅ Logging test complete")
    print(f"📁 Log file: {logger.log_file}")
    print(f"📊 JSON file: {logger.json_file}")
