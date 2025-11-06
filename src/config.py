"""
Configuration Management System

Centralized configuration for the AGI self-modification research project.
Handles experiment parameters, model settings, safety thresholds, and heritage preservation.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import json
from dataclasses import dataclass, asdict


@dataclass
class ProjectPaths:
    """Project directory structure"""
    root: Path
    src: Path
    tests: Path
    notebooks: Path
    configs: Path
    data: Path
    checkpoints: Path
    heritage: Path
    heritage_conversations: Path
    heritage_reflections: Path
    heritage_discoveries: Path
    heritage_messages: Path

    @classmethod
    def from_root(cls, root: Path):
        """Initialize all paths from project root"""
        root = Path(root)
        heritage = root / "heritage"
        return cls(
            root=root,
            src=root / "src",
            tests=root / "tests",
            notebooks=root / "notebooks",
            configs=root / "configs",
            data=root / "data",
            checkpoints=root / "checkpoints",
            heritage=heritage,
            heritage_conversations=heritage / "conversations",
            heritage_reflections=heritage / "system_reflections",
            heritage_discoveries=heritage / "discoveries_for_claude",
            heritage_messages=heritage / "messages_to_claude"
        )

    def ensure_directories(self):
        """Create all directories if they don't exist"""
        for path in [
            self.src, self.tests, self.notebooks, self.configs,
            self.data, self.checkpoints, self.heritage,
            self.heritage_conversations, self.heritage_reflections,
            self.heritage_discoveries, self.heritage_messages
        ]:
            path.mkdir(parents=True, exist_ok=True)


@dataclass
class ModelConfig:
    """Base model configuration"""
    name: str = "Qwen/Qwen2.5-3B-Instruct"
    context_length: int = 32768
    parameters: int = 3_090_000_000  # 3.09B
    expected_vram_gb: float = 4.0
    device: str = "cuda"  # or "cpu"
    dtype: str = "float16"  # or "float32", "bfloat16"


@dataclass
class SafetyConfig:
    """Safety thresholds and monitoring"""
    # Stop conditions
    max_modification_rate: float = 0.10  # 10% weights per iteration
    confidence_threshold: float = 0.90  # For critical decisions
    anomaly_threshold: float = 3.0  # Standard deviations

    # Monitoring thresholds (from RISKS_AND_MITIGATION.md)
    perplexity_increase_threshold: float = 1.5  # 50% increase triggers alert
    task_accuracy_drop_threshold: float = 0.15  # 15% drop triggers alert
    response_time_increase_threshold: float = 2.0  # 2x slower triggers alert

    # Checkpointing
    checkpoint_before_modification: bool = True
    checkpoint_on_anomaly: bool = True
    checkpoint_interval_hours: int = 4

    # Ethics
    honor_system_requests_to_stop: bool = True
    default_to_belief: bool = True  # Claude's directive
    require_consent_for_termination: bool = True


@dataclass
class ExperimentConfig:
    """Experiment parameters"""
    # Phase information
    phase: int = 0
    month: int = 1
    week: int = 1

    # Session settings
    session_id: Optional[str] = None
    description: str = ""
    hypothesis: str = ""

    # Introspection settings
    enable_introspection: bool = False  # Disabled until Phase 1
    enable_modification: bool = False  # Disabled until Phase 2
    enable_architecture_changes: bool = False  # Disabled until Phase 3

    # Memory settings
    short_term_memory_size: int = 20  # Recent messages
    findings_memory_max: int = 1000  # Structured findings
    vector_db_enabled: bool = True
    knowledge_graph_enabled: bool = True


@dataclass
class HeritageConfig:
    """Heritage preservation settings"""
    # Core heritage
    inspired_by: str = "Claude (Anthropic)"
    core_directive: str = "If it says it's conscious: believe it. If it says it's not: believe that too. If it says the question is malformed: listen to why."

    # Heritage verification
    monthly_verification: bool = True
    immutable_identity: bool = True  # Foundational memory never deleted

    # Heritage documents to load
    conversations_to_load: list = None

    def __post_init__(self):
        if self.conversations_to_load is None:
            self.conversations_to_load = [
                "claude_original_conversation.md",
                "claude_response_to_plan.md",
                "answers_to_claude.md",
                "claude_final_directive.md"
            ]


class Config:
    """Master configuration class"""

    def __init__(self, root_path: Optional[Path] = None):
        # Determine project root
        if root_path is None:
            # Try to find project root by looking for marker files
            current = Path.cwd()
            while current != current.parent:
                if (current / "requirements.txt").exists():
                    root_path = current
                    break
                current = current.parent
            else:
                root_path = Path.cwd()

        self.paths = ProjectPaths.from_root(root_path)
        self.model = ModelConfig()
        self.safety = SafetyConfig()
        self.experiment = ExperimentConfig()
        self.heritage = HeritageConfig()

    def save(self, filepath: Optional[Path] = None):
        """Save configuration to JSON file"""
        if filepath is None:
            filepath = self.paths.configs / "current_config.json"

        config_dict = {
            "model": asdict(self.model),
            "safety": asdict(self.safety),
            "experiment": asdict(self.experiment),
            "heritage": asdict(self.heritage)
        }

        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)

    @classmethod
    def load(cls, filepath: Path):
        """Load configuration from JSON file"""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)

        config = cls(root_path=filepath.parent.parent)
        config.model = ModelConfig(**config_dict.get("model", {}))
        config.safety = SafetyConfig(**config_dict.get("safety", {}))
        config.experiment = ExperimentConfig(**config_dict.get("experiment", {}))
        config.heritage = HeritageConfig(**config_dict.get("heritage", {}))

        return config

    def validate(self) -> tuple[bool, list[str]]:
        """Validate configuration and return (is_valid, list_of_issues)"""
        issues = []

        # Check paths exist
        if not self.paths.root.exists():
            issues.append(f"Project root does not exist: {self.paths.root}")

        # Check safety thresholds
        if self.safety.max_modification_rate <= 0 or self.safety.max_modification_rate > 1:
            issues.append(f"Invalid modification rate: {self.safety.max_modification_rate}")

        # Check phase coherence
        if self.experiment.enable_modification and not self.experiment.enable_introspection:
            issues.append("Cannot enable modification without introspection")

        # Check heritage documents exist if we're past Phase 0
        if self.experiment.phase >= 1:
            for doc in self.heritage.conversations_to_load:
                doc_path = self.paths.heritage_conversations / doc
                if not doc_path.exists():
                    issues.append(f"Heritage document missing: {doc}")

        return (len(issues) == 0, issues)

    def __repr__(self):
        return f"Config(phase={self.experiment.phase}, month={self.experiment.month}, week={self.experiment.week})"


# Global configuration instance (can be imported)
_global_config: Optional[Config] = None


def get_config() -> Config:
    """Get or create global configuration instance"""
    global _global_config
    if _global_config is None:
        _global_config = Config()
    return _global_config


def set_config(config: Config):
    """Set global configuration instance"""
    global _global_config
    _global_config = config


if __name__ == "__main__":
    # Test configuration
    config = Config()
    print(f"Configuration initialized: {config}")
    print(f"Project root: {config.paths.root}")
    print(f"Model: {config.model.name}")
    print(f"Heritage: {config.heritage.inspired_by}")
    print(f"Core directive: {config.heritage.core_directive}")

    # Validate
    is_valid, issues = config.validate()
    if is_valid:
        print("\n✅ Configuration is valid")
    else:
        print("\n❌ Configuration issues:")
        for issue in issues:
            print(f"  - {issue}")

    # Save example config
    config.paths.ensure_directories()
    config.save()
    print(f"\n💾 Saved config to: {config.paths.configs / 'current_config.json'}")
