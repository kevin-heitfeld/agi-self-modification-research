"""
Theory Layer (Layer 3) - Theories and Models

This layer forms theories and causal models from detected patterns. Theories
provide explanatory power - they explain WHY patterns occur and predict WHAT
will happen in new situations.

Theories are:
- Built from multiple supporting patterns
- Tested against new observations
- Updated as evidence accumulates
- Ranked by explanatory power
- Used for prediction and planning

Types of theories:
- Causal models (X causes Y through mechanism M)
- Structural theories (Component C has role R)
- Behavioral theories (System S behaves like B)
- Optimization theories (Action A optimizes for O)

Author: AGI Self-Modification Research Team
Date: November 7, 2025
"""

import json
import time
import sqlite3
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, asdict
from pathlib import Path
from enum import Enum
from collections import defaultdict

from .base_layer import SQLiteLayerBase


class TheoryType(Enum):
    """Types of theories."""
    CAUSAL_MODEL = "causal_model"
    STRUCTURAL = "structural"
    BEHAVIORAL = "behavioral"
    OPTIMIZATION = "optimization"
    CONSTRAINT = "constraint"


@dataclass
class Theory:
    """A theory or model."""
    id: str
    type: TheoryType
    name: str
    description: str
    hypothesis: str  # The core hypothesis
    supporting_patterns: List[str]  # Pattern IDs that support this theory
    evidence_count: int  # Number of observations supporting this
    counter_evidence_count: int  # Number of observations contradicting this
    confidence: float  # 0.0-1.0 confidence based on evidence
    predictive_power: float  # 0.0-1.0 how well it predicts outcomes
    created: float
    last_updated: float
    predictions: List[Dict[str, Any]]  # Predictions made by this theory
    tags: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        d = asdict(self)
        d['type'] = self.type.value
        return d

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'Theory':
        """Create from dictionary."""
        data['type'] = TheoryType(data['type'])
        return Theory(**data)


class TheoryBuilder:
    """Base class for theory builders."""

    def build_theories(self, patterns: List[Any], observations: List[Any]) -> List[Theory]:
        """
        Build theories from patterns and observations.

        Args:
            patterns: List of detected patterns
            observations: List of observations for validation

        Returns:
            List of theories
        """
        raise NotImplementedError


class CausalModelBuilder(TheoryBuilder):
    """Builds causal models from patterns."""

    def build_theories(self, patterns: List[Any], observations: List[Any]) -> List[Theory]:
        """Build causal model theories."""
        theories = []

        # Group causal patterns by modification type
        causal_patterns = [p for p in patterns if p.type.value == 'causal']

        by_modification = defaultdict(list)
        for pattern in causal_patterns:
            mod_type = pattern.components.get('modification_type')
            if mod_type:
                by_modification[mod_type].append(pattern)

        # For each modification type, build a causal model
        for mod_type, related_patterns in by_modification.items():
            if len(related_patterns) >= 2:  # Need multiple effects for a model
                # Collect all affected metrics
                affected_metrics = {}
                for pattern in related_patterns:
                    metric = pattern.components.get('metric')
                    change = pattern.components.get('average_change_percent', 0)
                    if metric:
                        affected_metrics[metric] = change

                theory_id = f"causal_model_{mod_type}"

                # Build hypothesis
                effects = [f"{metric} by {change:+.1f}%" for metric, change in affected_metrics.items()]
                hypothesis = f"Modifying {mod_type} affects: " + ", ".join(effects)

                # Calculate confidence based on pattern strengths
                avg_confidence = sum(p.confidence for p in related_patterns) / len(related_patterns)

                theories.append(Theory(
                    id=theory_id,
                    type=TheoryType.CAUSAL_MODEL,
                    name=f"Causal Model: {mod_type}",
                    description=f"Model of how {mod_type} modifications affect system performance",
                    hypothesis=hypothesis,
                    supporting_patterns=[p.id for p in related_patterns],
                    evidence_count=sum(p.support_count for p in related_patterns),
                    counter_evidence_count=0,  # Would need to track
                    confidence=avg_confidence,
                    predictive_power=0.7,  # Initial estimate
                    created=time.time(),
                    last_updated=time.time(),
                    predictions=[],
                    tags=['causal', mod_type] + list(affected_metrics.keys())
                ))

        return theories


class StructuralTheoryBuilder(TheoryBuilder):
    """Builds theories about model structure and components."""

    def build_theories(self, patterns: List[Any], observations: List[Any]) -> List[Theory]:
        """Build structural theories."""
        theories = []

        # Look for patterns about specific layers/components
        layer_patterns = defaultdict(list)
        for pattern in patterns:
            for tag in pattern.tags:
                if 'layer' in tag.lower():
                    layer_patterns[tag].append(pattern)

        # Build theories about layer behavior
        for layer_tag, related_patterns in layer_patterns.items():
            if len(related_patterns) >= 3:  # Multiple patterns about same layer
                theory_id = f"structural_{layer_tag}"

                # Analyze what we know about this layer
                behaviors = []
                for pattern in related_patterns:
                    behaviors.append(pattern.description)

                hypothesis = f"Layer {layer_tag} exhibits consistent behavior patterns"

                theories.append(Theory(
                    id=theory_id,
                    type=TheoryType.STRUCTURAL,
                    name=f"Structure: {layer_tag}",
                    description=f"Theory about the role and behavior of {layer_tag}",
                    hypothesis=hypothesis,
                    supporting_patterns=[p.id for p in related_patterns],
                    evidence_count=sum(p.support_count for p in related_patterns),
                    counter_evidence_count=0,
                    confidence=0.65,
                    predictive_power=0.6,
                    created=time.time(),
                    last_updated=time.time(),
                    predictions=[],
                    tags=['structural', layer_tag]
                ))

        return theories


class OptimizationTheoryBuilder(TheoryBuilder):
    """Builds theories about what optimizes performance."""

    def build_theories(self, patterns: List[Any], observations: List[Any]) -> List[Theory]:
        """Build optimization theories."""
        theories = []

        # Find patterns that show consistent improvements
        positive_patterns = []
        negative_patterns = []

        for pattern in patterns:
            if pattern.type.value == 'causal':
                change = pattern.components.get('average_change_percent', 0)
                if change < -5:  # Improvement (e.g., perplexity decreased)
                    positive_patterns.append(pattern)
                elif change > 10:  # Degradation
                    negative_patterns.append(pattern)

        # Build theory about what improves performance
        if positive_patterns:
            mod_types = [p.components.get('modification_type') for p in positive_patterns]
            common_mods = set(mod_types)

            if common_mods:
                theory_id = "optimization_improvements"

                hypothesis = f"Modifications to {', '.join(common_mods)} tend to improve performance"

                theories.append(Theory(
                    id=theory_id,
                    type=TheoryType.OPTIMIZATION,
                    name="Performance Optimization Theory",
                    description="Theory about which modifications lead to improvements",
                    hypothesis=hypothesis,
                    supporting_patterns=[p.id for p in positive_patterns],
                    evidence_count=len(positive_patterns),
                    counter_evidence_count=len(negative_patterns),
                    confidence=len(positive_patterns) / (len(positive_patterns) + len(negative_patterns)),
                    predictive_power=0.75,
                    created=time.time(),
                    last_updated=time.time(),
                    predictions=[],
                    tags=['optimization', 'improvement'] + list(common_mods)
                ))

        # Build theory about what to avoid
        if negative_patterns:
            mod_types = [p.components.get('modification_type') for p in negative_patterns]
            risky_mods = set(mod_types)

            if risky_mods:
                theory_id = "optimization_risks"

                hypothesis = f"Modifications to {', '.join(risky_mods)} often degrade performance"

                theories.append(Theory(
                    id=theory_id,
                    type=TheoryType.CONSTRAINT,
                    name="Performance Risk Theory",
                    description="Theory about which modifications to avoid",
                    hypothesis=hypothesis,
                    supporting_patterns=[p.id for p in negative_patterns],
                    evidence_count=len(negative_patterns),
                    counter_evidence_count=len(positive_patterns),
                    confidence=len(negative_patterns) / (len(negative_patterns) + len(positive_patterns)),
                    predictive_power=0.7,
                    created=time.time(),
                    last_updated=time.time(),
                    predictions=[],
                    tags=['optimization', 'risk', 'constraint'] + list(risky_mods)
                ))

        return theories


class TheoryLayer(SQLiteLayerBase):
    """
    Layer 3: Theories and Models

    Forms theories and causal models from detected patterns. Theories provide
    explanatory power and enable prediction of outcomes.

    Features:
    - Multiple theory building algorithms
    - Theory validation against new evidence
    - Confidence tracking based on evidence
    - Predictive power measurement
    - Theory evolution over time

    Usage:
        >>> layer = TheoryLayer("data/memory/theories", pattern_layer)
        >>>
        >>> # Build theories from patterns
        >>> layer.build_theories()
        >>>
        >>> # Query theories
        >>> causal = layer.get_theories(type=TheoryType.CAUSAL_MODEL)
        >>> high_conf = layer.get_theories(min_confidence=0.8)
        >>>
        >>> # Test a theory's prediction
        >>> theory = layer.get_theory("causal_model_attention")
        >>> prediction = layer.predict(theory, context)
    """

    def __init__(self, storage_dir: str, pattern_layer: Any, observation_layer: Any):
        """
        Initialize theory layer.

        Args:
            storage_dir: Directory for storing theories
            pattern_layer: Reference to pattern layer
            observation_layer: Reference to observation layer
        """
        # Create directory structure if needed
        storage_path = Path(storage_dir)
        storage_path.mkdir(parents=True, exist_ok=True)
        self.storage_dir = storage_path

        self.pattern_layer = pattern_layer
        self.observation_layer = observation_layer

        # Database path
        db_path = storage_path / "theories.db"

        # Theory builders
        self.builders = [
            CausalModelBuilder(),
            StructuralTheoryBuilder(),
            OptimizationTheoryBuilder()
        ]

        self.last_build_time = 0.0
        
        # Initialize base class (establishes DB connection)
        super().__init__(str(db_path))

    def _get_table_name(self) -> str:
        """Return the main table name for this layer."""
        return "theories"
    
    def _init_schema(self):
        """Initialize the SQLite database schema."""
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS theories (
                id TEXT PRIMARY KEY,
                type TEXT NOT NULL,
                name TEXT NOT NULL,
                description TEXT NOT NULL,
                hypothesis TEXT NOT NULL,
                supporting_patterns TEXT NOT NULL,
                evidence_count INTEGER NOT NULL,
                counter_evidence_count INTEGER NOT NULL,
                confidence REAL NOT NULL,
                predictive_power REAL NOT NULL,
                created REAL NOT NULL,
                last_updated REAL NOT NULL,
                predictions TEXT NOT NULL,
                tags TEXT NOT NULL
            )
        """)

        # Create indexes for common queries
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_theory_type ON theories(type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_theory_confidence ON theories(confidence)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_theory_evidence ON theories(evidence_count)")

        self.conn.commit()

    def _save_theory(self, theory: Theory):
        """Save a single theory to the database."""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO theories
            (id, type, name, description, hypothesis, supporting_patterns, evidence_count,
             counter_evidence_count, confidence, predictive_power, created, last_updated,
             predictions, tags)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            theory.id,
            theory.type.value,
            theory.name,
            theory.description,
            theory.hypothesis,
            json.dumps(theory.supporting_patterns),
            theory.evidence_count,
            theory.counter_evidence_count,
            theory.confidence,
            theory.predictive_power,
            theory.created,
            theory.last_updated,
            json.dumps(theory.predictions),
            json.dumps(theory.tags)
        ))
        self.conn.commit()

    def _load_theory_from_row(self, row: sqlite3.Row) -> Theory:
        """Convert a database row to a Theory object."""
        return Theory(
            id=row['id'],
            type=TheoryType(row['type']),
            name=row['name'],
            description=row['description'],
            hypothesis=row['hypothesis'],
            supporting_patterns=json.loads(row['supporting_patterns']),
            evidence_count=row['evidence_count'],
            counter_evidence_count=row['counter_evidence_count'],
            confidence=row['confidence'],
            predictive_power=row['predictive_power'],
            created=row['created'],
            last_updated=row['last_updated'],
            predictions=json.loads(row['predictions']),
            tags=json.loads(row['tags'])
        )

    def build_theories(self):
        """
        Build theories from available patterns.

        Returns:
            Number of new theories built
        """
        # Get patterns from pattern layer
        patterns = self.pattern_layer.get_patterns(min_confidence=0.6)

        if len(patterns) < 3:  # Need sufficient patterns
            return 0

        # Get recent observations for validation
        observations = self.observation_layer.get_recent(limit=1000)

        # Run all theory builders
        new_theories = []
        for builder in self.builders:
            theories = builder.build_theories(patterns, observations)
            new_theories.extend(theories)

        # Track new theories added
        new_count = 0

        # Merge with existing theories
        for new_theory in new_theories:
            # Check if theory already exists
            cursor = self.conn.cursor()
            cursor.execute("SELECT * FROM theories WHERE id = ?", (new_theory.id,))
            existing_row = cursor.fetchone()

            if existing_row:
                # Update existing theory
                existing = self._load_theory_from_row(existing_row)

                # Merge supporting patterns
                all_patterns = set(existing.supporting_patterns + new_theory.supporting_patterns)
                existing.supporting_patterns = list(all_patterns)

                # Update evidence counts
                existing.evidence_count += new_theory.evidence_count
                existing.counter_evidence_count += new_theory.counter_evidence_count

                # Recalculate confidence
                total_evidence = existing.evidence_count + existing.counter_evidence_count
                if total_evidence > 0:
                    existing.confidence = existing.evidence_count / total_evidence

                existing.last_updated = time.time()
                self._save_theory(existing)
            else:
                # Add new theory
                self._save_theory(new_theory)
                new_count += 1

        self.last_build_time = time.time()

        return new_count

    def get_theory(self, theory_id: str) -> Optional[Theory]:
        """Get a specific theory by ID."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM theories WHERE id = ?", (theory_id,))
        row = cursor.fetchone()
        if row:
            return self._load_theory_from_row(row)
        return None

    def get_theories(
        self,
        theory_type: Optional[TheoryType] = None,
        tags: Optional[List[str]] = None,
        min_confidence: Optional[float] = None,
        min_evidence: Optional[int] = None
    ) -> List[Theory]:
        """
        Query theories with filters.

        Args:
            type: Filter by theory type
            tags: Filter by tags
            min_confidence: Minimum confidence score
            min_evidence: Minimum evidence count

        Returns:
            List of matching theories
        """
        # Build SQL query with filters
        query = "SELECT * FROM theories WHERE 1=1"
        params = []

        if theory_type:
            query += " AND type = ?"
            params.append(theory_type.value)

        if min_confidence is not None:
            query += " AND confidence >= ?"
            params.append(min_confidence)

        if min_evidence is not None:
            query += " AND evidence_count >= ?"
            params.append(min_evidence)

        # Order by confidence * evidence (most reliable theories first)
        query += " ORDER BY (confidence * evidence_count) DESC"

        cursor = self.conn.cursor()
        cursor.execute(query, params)
        results = [self._load_theory_from_row(row) for row in cursor.fetchall()]

        # Filter by tags in memory (JSON array filtering is complex in SQLite)
        if tags:
            results = [t for t in results if any(tag in t.tags for tag in tags)]

        return results

    def validate_theory(self, theory_id: str, observation: Any) -> bool:
        """
        Validate a theory against a new observation.

        Args:
            theory_id: Theory to validate
            observation: New observation to test against

        Returns:
            True if observation supports theory, False otherwise
        """
        theory = self.get_theory(theory_id)
        if not theory:
            return False

        # Simple validation: check if observation aligns with hypothesis
        # In practice, this would be more sophisticated

        # For now, assume any observation with matching tags supports the theory
        matching_tags = set(theory.tags) & set(observation.tags)

        if len(matching_tags) >= 2:
            theory.evidence_count += 1

            # Recalculate confidence
            total = theory.evidence_count + theory.counter_evidence_count
            theory.confidence = theory.evidence_count / total if total > 0 else 0.5

            theory.last_updated = time.time()
            self._save_theory(theory)

            return True

        return False

    def make_prediction(self, theory_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Use a theory to make a prediction.

        Args:
            theory_id: Theory to use
            context: Context for the prediction

        Returns:
            Prediction dictionary
        """
        theory = self.get_theory(theory_id)
        if not theory:
            return {'error': 'Theory not found'}

        prediction = {
            'theory_id': theory_id,
            'theory_name': theory.name,
            'confidence': theory.confidence,
            'predictive_power': theory.predictive_power,
            'prediction': None,
            'reasoning': theory.hypothesis
        }

        # Make prediction based on theory type
        if theory.type == TheoryType.CAUSAL_MODEL:
            # Predict effect of a modification
            if 'modification_type' in context:
                mod_type = context['modification_type']
                if mod_type in theory.tags:
                    prediction['prediction'] = {
                        'outcome': 'Performance will likely change',
                        'direction': 'improvement or degradation',
                        'confidence': theory.confidence
                    }

        elif theory.type == TheoryType.OPTIMIZATION:
            # Predict if an action will improve performance
            if 'proposed_action' in context:
                action = context['proposed_action']
                if any(tag in action.lower() for tag in theory.tags):
                    prediction['prediction'] = {
                        'outcome': 'Likely to improve performance',
                        'confidence': theory.confidence * theory.predictive_power
                    }

        # Record this prediction
        theory.predictions.append({
            'timestamp': time.time(),
            'context': context,
            'prediction': prediction['prediction']
        })

        self._save_theory(theory)

        return prediction

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about theories."""
        # Get base statistics from parent class
        stats = super().get_statistics()
        
        cursor = self.conn.cursor()
        
        if stats['total'] == 0:
            return {
                'total_theories': 0,
                'by_type': {},
                'average_confidence': 0.0,
                'high_confidence_count': 0,
                'total_predictions': 0,
                'last_build': self.last_build_time
            }

        # Count by type
        stats['by_type'] = self._get_grouped_counts('type')

        # Average confidence
        stats['average_confidence'] = self._get_average_value('confidence')

        # High confidence theories (>0.8)
        cursor.execute("SELECT COUNT(*) FROM theories WHERE confidence > 0.8")
        stats['high_confidence_count'] = cursor.fetchone()[0]

        # Total predictions made (need to count from JSON array)
        cursor.execute("SELECT predictions FROM theories")
        stats['total_predictions'] = sum(len(json.loads(row[0])) for row in cursor.fetchall())
        
        stats['last_build'] = self.last_build_time
        stats['total_theories'] = stats['total']

        return stats

    def prune_theories(self, min_confidence: float = 0.3, max_age_days: int = 180):
        """
        Remove low-confidence or outdated theories.

        Args:
            min_confidence: Remove theories below this confidence
            max_age_days: Remove theories not updated in this many days
        """
        cutoff_time = time.time() - (max_age_days * 24 * 3600)

        cursor = self.conn.cursor()
        cursor.execute("""
            DELETE FROM theories 
            WHERE confidence < ? OR last_updated < ?
        """, (min_confidence, cutoff_time))
        
        removed = cursor.rowcount
        self.conn.commit()

        return removed

    def export(self, filepath: str, limit: Optional[int] = None):
        """Export theories to JSON file."""
        cursor = self.conn.cursor()
        if limit:
            cursor.execute("SELECT * FROM theories LIMIT ?", (limit,))
        else:
            cursor.execute("SELECT * FROM theories")
        theories = [self._load_theory_from_row(row) for row in cursor.fetchall()]
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(
                [theory.to_dict() for theory in theories],
                f,
                indent=2
            )
