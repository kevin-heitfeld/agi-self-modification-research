"""
Pattern Layer (Layer 2) - Patterns and Correlations

This layer discovers and tracks patterns across observations. It automatically
detects recurring sequences, correlations, and relationships in the observation
history.

Patterns are:
- Automatically detected from observations
- Tracked with frequency and confidence
- Updated as new evidence arrives
- Used to predict future outcomes
- Foundation for theory formation (Layer 3)

Types of patterns:
- Sequential patterns (A followed by B)
- Causal patterns (A causes B)
- Correlations (A associated with B)
- Conditional patterns (If A then B)
- Negative patterns (A prevents B)

Author: AGI Self-Modification Research Team
Date: November 7, 2025
"""

import json
import time
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from enum import Enum
from collections import defaultdict, Counter
import math


class PatternType(Enum):
    """Types of patterns that can be detected."""
    SEQUENTIAL = "sequential"  # A followed by B
    CAUSAL = "causal"  # A causes B
    CORRELATION = "correlation"  # A associated with B
    CONDITIONAL = "conditional"  # If A then B
    NEGATIVE = "negative"  # A prevents B
    THRESHOLD = "threshold"  # Metric crosses threshold


@dataclass
class Pattern:
    """A detected pattern."""
    id: str
    type: PatternType
    name: str
    description: str
    components: Dict[str, Any]  # Pattern-specific structure
    support_count: int  # Number of times observed
    confidence: float  # 0.0-1.0 confidence score
    first_seen: float  # Timestamp of first detection
    last_seen: float  # Timestamp of last confirmation
    evidence: List[str]  # Observation IDs supporting this pattern
    tags: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        d = asdict(self)
        d['type'] = self.type.value
        return d

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'Pattern':
        """Create from dictionary."""
        data['type'] = PatternType(data['type'])
        return Pattern(**data)


class PatternDetector:
    """Base class for pattern detectors."""

    def __init__(self, min_support: int = 3, min_confidence: float = 0.6):
        """
        Initialize pattern detector.

        Args:
            min_support: Minimum occurrences to consider a pattern
            min_confidence: Minimum confidence score to report pattern
        """
        self.min_support = min_support
        self.min_confidence = min_confidence

    def detect(self, observations: List[Any]) -> List[Pattern]:
        """
        Detect patterns in observations.

        Args:
            observations: List of observations to analyze

        Returns:
            List of detected patterns
        """
        raise NotImplementedError


class SequentialPatternDetector(PatternDetector):
    """Detects sequential patterns (A followed by B)."""

    def detect(self, observations: List[Any]) -> List[Pattern]:
        """Detect sequential patterns."""
        patterns = []

        # Build sequences of observation types/categories
        sequences = []
        for i in range(len(observations) - 1):
            current = (observations[i].type.value, observations[i].category)
            next_obs = (observations[i+1].type.value, observations[i+1].category)
            time_gap = observations[i+1].timestamp - observations[i].timestamp

            # Only consider sequences within reasonable time windows (e.g., 1 hour)
            if time_gap < 3600:
                sequences.append((current, next_obs, time_gap))

        # Count sequence frequencies
        sequence_counts = Counter([(s[0], s[1]) for s in sequences])

        # Create patterns for frequent sequences
        for (first, second), count in sequence_counts.items():
            if count >= self.min_support:
                confidence = count / len([s for s in sequences if s[0] == first])

                if confidence >= self.min_confidence:
                    pattern_id = f"seq_{hash((first, second))}"

                    # Find supporting observations
                    evidence = []
                    for i, seq in enumerate(sequences):
                        if (seq[0], seq[1]) == (first, second):
                            if i < len(observations) - 1:
                                evidence.append(observations[i].id)

                    patterns.append(Pattern(
                        id=pattern_id,
                        type=PatternType.SEQUENTIAL,
                        name=f"{first[1]} → {second[1]}",
                        description=f"'{first[1]}' is typically followed by '{second[1]}'",
                        components={
                            'first': first,
                            'second': second,
                            'avg_time_gap': sum(s[2] for s in sequences if (s[0], s[1]) == (first, second)) / count
                        },
                        support_count=count,
                        confidence=confidence,
                        first_seen=min(obs.timestamp for obs in observations if obs.id in evidence),
                        last_seen=max(obs.timestamp for obs in observations if obs.id in evidence),
                        evidence=evidence[:10],  # Keep first 10 for space
                        tags=['sequential', first[1], second[1]]
                    ))

        return patterns


class CausalPatternDetector(PatternDetector):
    """Detects causal patterns (A causes B)."""

    def detect(self, observations: List[Any]) -> List[Pattern]:
        """Detect causal patterns."""
        patterns = []

        # Look for modification -> performance change patterns
        modifications = [o for o in observations if o.type.value == 'modification']
        performances = [o for o in observations if o.type.value == 'performance']

        # For each modification, look at performance changes shortly after
        for mod in modifications:
            # Find performance observations within 1 hour after modification
            relevant_perf = [
                p for p in performances
                if mod.timestamp < p.timestamp < mod.timestamp + 3600
            ]

            if not relevant_perf:
                continue

            # Analyze impact
            for perf in relevant_perf:
                # Extract metric values (support multiple formats)
                metric = None
                change = None
                
                # Format 1: metric_name + value (production format)
                if 'metric_name' in perf.data and 'value' in perf.data:
                    metric = perf.data['metric_name']
                    value = perf.data['value']

                    # Find baseline (performance before modification)
                    baseline_perf = [
                        p for p in performances
                        if p.timestamp < mod.timestamp and 'metric_name' in p.data and p.data['metric_name'] == metric
                    ]

                    if baseline_perf:
                        baseline = baseline_perf[-1].data['value']
                        change = ((value - baseline) / baseline) * 100
                
                # Format 2: before/after/improvement (test format)
                elif 'before' in perf.data and 'after' in perf.data:
                    metric = perf.category
                    before = perf.data['before']
                    after = perf.data['after']
                    change = ((after - before) / before) * 100
                
                # If we found a change, create pattern
                if metric and change is not None and abs(change) > 1:  # >1% change
                    pattern_id = f"causal_{mod.category}_{metric}"

                    patterns.append(Pattern(
                        id=pattern_id,
                        type=PatternType.CAUSAL,
                        name=f"{mod.category} → {metric} change",
                        description=f"Modifying '{mod.category}' affects '{metric}' by ~{change:.1f}%",
                        components={
                            'modification_type': mod.category,
                            'metric': metric,
                            'average_change_percent': change
                        },
                        support_count=1,  # Would need to track across multiple instances
                        confidence=0.7,  # Initial confidence
                        first_seen=mod.timestamp,
                        last_seen=perf.timestamp,
                        evidence=[mod.id, perf.id],
                        tags=['causal', mod.category, metric]
                    ))

        return patterns


class ThresholdPatternDetector(PatternDetector):
    """Detects threshold patterns (metric crosses threshold leading to events)."""

    def detect(self, observations: List[Any]) -> List[Pattern]:
        """Detect threshold patterns."""
        patterns = []

        # Look for performance metrics crossing thresholds before safety events
        performances = [o for o in observations if o.type.value == 'performance']
        safety_events = [o for o in observations if o.type.value == 'safety_event']

        for safety_event in safety_events:
            # Find performance observations shortly before
            preceding_perf = [
                p for p in performances
                if safety_event.timestamp - 600 < p.timestamp < safety_event.timestamp  # Within 10 min
            ]

            for perf in preceding_perf:
                # Support multiple data formats
                metric = None
                value = None
                
                # Format 1: metric_name + value (production format)
                if 'metric_name' in perf.data and 'value' in perf.data:
                    metric = perf.data['metric_name']
                    value = perf.data['value']
                # Format 2: Use category as metric, extract numeric value from data
                elif perf.category and perf.data:
                    metric = perf.category
                    # Try to find any numeric value in data
                    for key, val in perf.data.items():
                        if isinstance(val, (int, float)):
                            value = val
                            break
                
                if metric and value is not None:
                    pattern_id = f"threshold_{metric}_{safety_event.category}"

                    patterns.append(Pattern(
                        id=pattern_id,
                        type=PatternType.THRESHOLD,
                        name=f"{metric} threshold → {safety_event.category}",
                        description=f"When '{metric}' reaches {value:.2f}, '{safety_event.category}' often follows",
                        components={
                            'metric': metric,
                            'threshold_value': value,
                            'event_type': safety_event.category
                        },
                        support_count=1,
                        confidence=0.65,
                        first_seen=perf.timestamp,
                        last_seen=safety_event.timestamp,
                        evidence=[perf.id, safety_event.id],
                        tags=['threshold', metric, safety_event.category]
                    ))

        return patterns


class PatternLayer:
    """
    Layer 2: Patterns and Correlations

    Automatically detects and tracks patterns across observations. Maintains
    a database of discovered patterns with confidence scores and supporting
    evidence.

    Features:
    - Multiple pattern detection algorithms
    - Automatic pattern discovery from new observations
    - Pattern confidence tracking
    - Pattern evolution over time
    - Query interface for finding relevant patterns

    Usage:
        >>> layer = PatternLayer("data/memory/patterns", observation_layer)
        >>>
        >>> # Detect patterns (run periodically)
        >>> layer.detect_patterns()
        >>>
        >>> # Query patterns
        >>> sequential = layer.get_patterns(type=PatternType.SEQUENTIAL)
        >>> high_confidence = layer.get_patterns(min_confidence=0.8)
        >>>
        >>> # Get pattern by ID
        >>> pattern = layer.get_pattern("seq_12345")
    """

    def __init__(self, storage_dir: str, observation_layer: Any):
        """
        Initialize pattern layer.

        Args:
            storage_dir: Directory for storing patterns
            observation_layer: Reference to observation layer
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self.observation_layer = observation_layer

        # Pattern storage
        self.patterns_file = self.storage_dir / "patterns.json"
        self.patterns: Dict[str, Pattern] = {}
        self._load_patterns()

        # Pattern detectors
        self.detectors = [
            SequentialPatternDetector(min_support=3, min_confidence=0.6),
            CausalPatternDetector(min_support=2, min_confidence=0.6),
            ThresholdPatternDetector(min_support=2, min_confidence=0.6)
        ]

        # Track when patterns were last updated
        self.last_detection_time = 0.0

    def _load_patterns(self):
        """Load patterns from storage."""
        if self.patterns_file.exists():
            with open(self.patterns_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.patterns = {
                    pid: Pattern.from_dict(pdata)
                    for pid, pdata in data.items()
                }

    def _save_patterns(self):
        """Save patterns to storage."""
        with open(self.patterns_file, 'w', encoding='utf-8') as f:
            json.dump(
                {pid: pattern.to_dict() for pid, pattern in self.patterns.items()},
                f,
                indent=2
            )

    def detect_patterns(self, lookback_hours: int = 24):
        """
        Detect new patterns from recent observations.

        Args:
            lookback_hours: How many hours of observations to analyze
            
        Returns:
            Number of new patterns detected
        """
        # Get recent observations
        cutoff_time = time.time() - (lookback_hours * 3600)
        observations = self.observation_layer.query(start_time=cutoff_time)

        if len(observations) < 10:  # Need minimum data
            return 0

        # Run all detectors
        new_patterns = []
        for detector in self.detectors:
            detected = detector.detect(observations)
            new_patterns.extend(detected)

        # Track new patterns added
        new_count = 0
        
        # Merge with existing patterns
        for new_pattern in new_patterns:
            if new_pattern.id in self.patterns:
                # Update existing pattern
                existing = self.patterns[new_pattern.id]
                existing.support_count += new_pattern.support_count
                existing.last_seen = new_pattern.last_seen

                # Update confidence (weighted average)
                total_support = existing.support_count + new_pattern.support_count
                existing.confidence = (
                    (existing.confidence * existing.support_count +
                     new_pattern.confidence * new_pattern.support_count) /
                    total_support
                )

                # Add new evidence
                existing.evidence.extend(new_pattern.evidence)
                existing.evidence = list(set(existing.evidence))[:20]  # Keep unique, limit size
            else:
                # Add new pattern
                self.patterns[new_pattern.id] = new_pattern
                new_count += 1

        self.last_detection_time = time.time()
        self._save_patterns()
        
        return new_count

    def get_pattern(self, pattern_id: str) -> Optional[Pattern]:
        """
        Get a specific pattern by ID.

        Args:
            pattern_id: Pattern ID

        Returns:
            Pattern if found, None otherwise
        """
        return self.patterns.get(pattern_id)

    def get_patterns(
        self,
        pattern_type: Optional[PatternType] = None,
        tags: Optional[List[str]] = None,
        min_confidence: Optional[float] = None,
        min_support: Optional[int] = None
    ) -> List[Pattern]:
        """
        Query patterns with filters.

        Args:
            type: Filter by pattern type
            tags: Filter by tags (must have any)
            min_confidence: Minimum confidence score
            min_support: Minimum support count

        Returns:
            List of matching patterns
        """
        results = list(self.patterns.values())

        if pattern_type:
            results = [p for p in results if p.type == pattern_type]

        if tags:
            results = [p for p in results if any(tag in p.tags for tag in tags)]

        if min_confidence is not None:
            results = [p for p in results if p.confidence >= min_confidence]

        if min_support is not None:
            results = [p for p in results if p.support_count >= min_support]

        # Sort by confidence * support (most reliable patterns first)
        results.sort(key=lambda p: p.confidence * math.log(p.support_count + 1), reverse=True)

        return results

    def find_related_patterns(self, pattern_id: str) -> List[Pattern]:
        """
        Find patterns related to a given pattern.

        Args:
            pattern_id: Pattern to find relations for

        Returns:
            List of related patterns
        """
        pattern = self.patterns.get(pattern_id)
        if not pattern:
            return []

        # Find patterns with overlapping tags
        related = []
        for pid, other in self.patterns.items():
            if pid != pattern_id:
                # Check tag overlap
                overlap = set(pattern.tags) & set(other.tags)
                if len(overlap) >= 2:  # At least 2 shared tags
                    related.append(other)

        return related

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about patterns.

        Returns:
            Dictionary with statistics
        """
        if not self.patterns:
            return {
                'total_patterns': 0,
                'by_type': {},
                'average_confidence': 0.0,
                'high_confidence_count': 0
            }

        # Count by type
        by_type = defaultdict(int)
        for pattern in self.patterns.values():
            by_type[pattern.type.value] += 1

        # Average confidence
        avg_confidence = sum(p.confidence for p in self.patterns.values()) / len(self.patterns)

        # High confidence patterns (>0.8)
        high_conf = sum(1 for p in self.patterns.values() if p.confidence > 0.8)

        # Most common tags
        all_tags = []
        for pattern in self.patterns.values():
            all_tags.extend(pattern.tags)
        tag_counts = Counter(all_tags)

        return {
            'total_patterns': len(self.patterns),
            'by_type': dict(by_type),
            'average_confidence': avg_confidence,
            'high_confidence_count': high_conf,
            'most_common_tags': dict(tag_counts.most_common(10)),
            'last_detection': self.last_detection_time
        }

    def prune_patterns(self, min_confidence: float = 0.3, max_age_days: int = 90):
        """
        Remove low-confidence or outdated patterns.

        Args:
            min_confidence: Remove patterns below this confidence
            max_age_days: Remove patterns not seen in this many days
        """
        cutoff_time = time.time() - (max_age_days * 24 * 3600)

        to_remove = []
        for pid, pattern in self.patterns.items():
            if pattern.confidence < min_confidence or pattern.last_seen < cutoff_time:
                to_remove.append(pid)

        for pid in to_remove:
            del self.patterns[pid]

        self._save_patterns()

        return len(to_remove)

    def export(self, filepath: str):
        """
        Export patterns to JSON file.

        Args:
            filepath: Output file path
        """
        with open(filepath, 'w') as f:
            json.dump(
                [pattern.to_dict() for pattern in self.patterns.values()],
                f,
                indent=2
            )
