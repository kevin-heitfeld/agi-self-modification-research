"""
Belief Layer (Layer 4) - Core Beliefs and Principles

This is the highest level of the memory system. It represents core beliefs,
principles, and rules that have been validated through extensive experience.

Beliefs are:
- High-confidence conclusions (>0.9)
- Based on substantial evidence
- Operationally significant (guide behavior)
- Rarely changed (only with strong counter-evidence)
- Form the "knowledge base" of the system

Types of beliefs:
- Safety principles (Always do X, Never do Y)
- Causal laws (X always causes Y)
- Constraints (X must be within bounds)
- Heuristics (When in situation S, do A)
- Values (Optimize for V)

Author: AGI Self-Modification Research Team
Date: November 7, 2025
"""

import json
import time
import sqlite3
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
from enum import Enum
from collections import defaultdict

from .base_layer import SQLiteLayerBase


class BeliefType(Enum):
    """Types of beliefs."""
    SAFETY_PRINCIPLE = "safety_principle"
    CAUSAL_LAW = "causal_law"
    CONSTRAINT = "constraint"
    HEURISTIC = "heuristic"
    VALUE = "value"
    FACT = "fact"


class BeliefStrength(Enum):
    """Strength of belief."""
    TENTATIVE = "tentative"  # 0.7-0.8
    CONFIDENT = "confident"  # 0.8-0.9
    CERTAIN = "certain"  # 0.9-0.95
    ABSOLUTE = "absolute"  # >0.95


@dataclass
class Belief:
    """A core belief or principle."""
    id: str
    type: BeliefType
    strength: BeliefStrength
    statement: str  # The core belief statement
    justification: str  # Why this is believed
    supporting_theories: List[str]  # Theory IDs supporting this
    evidence_count: int  # Total supporting observations
    counter_evidence_count: int  # Observations that contradict
    confidence: float  # 0.0-1.0 confidence score
    importance: float  # 0.0-1.0 operational importance
    created: float
    last_validated: float
    times_applied: int  # How often this belief has been used
    success_rate: float  # When applied, how often was it correct
    tags: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        d = asdict(self)
        d['type'] = self.type.value
        d['strength'] = self.strength.value
        return d

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'Belief':
        """Create from dictionary."""
        data['type'] = BeliefType(data['type'])
        data['strength'] = BeliefStrength(data['strength'])
        return Belief(**data)


class BeliefFormation:
    """Handles formation of beliefs from theories."""

    @staticmethod
    def can_form_belief(theory: Any) -> bool:
        """
        Check if a theory is strong enough to become a belief.

        Args:
            theory: Theory to evaluate

        Returns:
            True if theory can become a belief
        """
        # Requirements for belief formation:
        # 1. High confidence (>0.85)
        # 2. Substantial evidence (>10 observations)
        # 3. Low counter-evidence (<10% of total)

        if theory.confidence < 0.85:
            return False

        if theory.evidence_count < 10:
            return False

        total_evidence = theory.evidence_count + theory.counter_evidence_count
        if total_evidence > 0:
            counter_ratio = theory.counter_evidence_count / total_evidence
            if counter_ratio > 0.1:
                return False

        return True

    @staticmethod
    def theory_to_belief(theory: Any) -> Belief:
        """
        Convert a theory into a belief.

        Args:
            theory: Theory to convert

        Returns:
            Belief object
        """
        # Determine belief type based on theory type
        belief_type_map = {
            'causal_model': BeliefType.CAUSAL_LAW,
            'optimization': BeliefType.HEURISTIC,
            'constraint': BeliefType.CONSTRAINT,
            'structural': BeliefType.FACT
        }

        belief_type = belief_type_map.get(theory.type.value, BeliefType.FACT)

        # Determine strength based on confidence
        if theory.confidence >= 0.95:
            strength = BeliefStrength.ABSOLUTE
        elif theory.confidence >= 0.9:
            strength = BeliefStrength.CERTAIN
        elif theory.confidence >= 0.85:
            strength = BeliefStrength.CONFIDENT
        else:
            strength = BeliefStrength.TENTATIVE

        return Belief(
            id=f"belief_{theory.id}",
            type=belief_type,
            strength=strength,
            statement=theory.hypothesis,
            justification=theory.description,
            supporting_theories=[theory.id],
            evidence_count=theory.evidence_count,
            counter_evidence_count=theory.counter_evidence_count,
            confidence=theory.confidence,
            importance=0.8,  # Default high importance
            created=time.time(),
            last_validated=time.time(),
            times_applied=0,
            success_rate=1.0,  # Start optimistic
            tags=theory.tags
        )


class BeliefLayer(SQLiteLayerBase):
    """
    Layer 4: Core Beliefs and Principles

    Maintains the system's core knowledge base - high-confidence beliefs that
    guide behavior and decision making.

    Features:
    - Automatic belief formation from strong theories
    - Belief validation and updating
    - Query interface for decision support
    - Conflict detection between beliefs
    - Belief importance ranking

    Usage:
        >>> layer = BeliefLayer("data/memory/beliefs.db", theory_layer)
        >>>
        >>> # Form beliefs from theories
        >>> layer.form_beliefs()
        >>>
        >>> # Query beliefs
        >>> safety = layer.get_beliefs(type=BeliefType.SAFETY_PRINCIPLE)
        >>> high_conf = layer.get_beliefs(min_confidence=0.9)
        >>>
        >>> # Check beliefs for a decision
        >>> relevant = layer.query_for_decision(context)
    """

    def __init__(self, db_path: str, theory_layer: Any):
        """
        Initialize belief layer.

        Args:
            db_path: Path to the beliefs database file
            theory_layer: Reference to theory layer
        """
        self.theory_layer = theory_layer

        # Track when beliefs were last formed
        self.last_formation_time = 0.0
        
        # Initialize base class (establishes DB connection)
        super().__init__(db_path)

        # Initialize core safety beliefs (hardcoded foundational beliefs)
        self._initialize_core_beliefs()

    def _get_table_name(self) -> str:
        """Return the main table name for this layer."""
        return "beliefs"

    def _init_schema(self):
        """Initialize the SQLite database schema."""
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS beliefs (
                id TEXT PRIMARY KEY,
                type TEXT NOT NULL,
                strength TEXT NOT NULL,
                statement TEXT NOT NULL,
                justification TEXT NOT NULL,
                supporting_theories TEXT NOT NULL,
                evidence_count INTEGER NOT NULL,
                counter_evidence_count INTEGER NOT NULL,
                confidence REAL NOT NULL,
                importance REAL NOT NULL,
                created REAL NOT NULL,
                last_validated REAL NOT NULL,
                times_applied INTEGER NOT NULL,
                success_rate REAL NOT NULL,
                tags TEXT NOT NULL
            )
        """)

        # Create indexes for common queries
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_belief_type ON beliefs(type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_belief_confidence ON beliefs(confidence)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_belief_importance ON beliefs(importance)")

        self.conn.commit()

    def _save_belief(self, belief: Belief):
        """Save a single belief to the database."""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO beliefs
            (id, type, strength, statement, justification, supporting_theories,
             evidence_count, counter_evidence_count, confidence, importance,
             created, last_validated, times_applied, success_rate, tags)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            belief.id,
            belief.type.value,
            belief.strength.value,
            belief.statement,
            belief.justification,
            json.dumps(belief.supporting_theories),
            belief.evidence_count,
            belief.counter_evidence_count,
            belief.confidence,
            belief.importance,
            belief.created,
            belief.last_validated,
            belief.times_applied,
            belief.success_rate,
            json.dumps(belief.tags)
        ))
        self.conn.commit()

    def _load_belief_from_row(self, row: sqlite3.Row) -> Belief:
        """Convert a database row to a Belief object."""
        return Belief(
            id=row['id'],
            type=BeliefType(row['type']),
            strength=BeliefStrength(row['strength']),
            statement=row['statement'],
            justification=row['justification'],
            supporting_theories=json.loads(row['supporting_theories']),
            evidence_count=row['evidence_count'],
            counter_evidence_count=row['counter_evidence_count'],
            confidence=row['confidence'],
            importance=row['importance'],
            created=row['created'],
            last_validated=row['last_validated'],
            times_applied=row['times_applied'],
            success_rate=row['success_rate'],
            tags=json.loads(row['tags'])
        )

    def _initialize_core_beliefs(self):
        """Initialize fundamental safety beliefs."""
        core_beliefs = [
            {
                'id': 'safety_checkpoint_before_modification',
                'type': BeliefType.SAFETY_PRINCIPLE,
                'strength': BeliefStrength.ABSOLUTE,
                'statement': 'Always create a checkpoint before making modifications',
                'justification': 'Checkpoints enable rollback if modifications fail',
                'importance': 1.0,
                'tags': ['safety', 'checkpoint', 'modification']
            },
            {
                'id': 'safety_nan_immediate_stop',
                'type': BeliefType.SAFETY_PRINCIPLE,
                'strength': BeliefStrength.ABSOLUTE,
                'statement': 'NaN or Inf in outputs requires immediate emergency stop',
                'justification': 'NaN/Inf indicates catastrophic failure and will propagate',
                'importance': 1.0,
                'tags': ['safety', 'nan', 'emergency']
            },
            {
                'id': 'safety_monitor_during_operations',
                'type': BeliefType.SAFETY_PRINCIPLE,
                'strength': BeliefStrength.ABSOLUTE,
                'statement': 'Always monitor operations with safety systems active',
                'justification': 'Monitoring detects problems before they cause damage',
                'importance': 1.0,
                'tags': ['safety', 'monitoring']
            },
            {
                'id': 'constraint_validate_before_commit',
                'type': BeliefType.CONSTRAINT,
                'strength': BeliefStrength.CERTAIN,
                'statement': 'Validate modifications against baseline before committing permanently',
                'justification': 'Validation prevents committing degraded states',
                'importance': 0.9,
                'tags': ['validation', 'baseline', 'modification']
            }
        ]

        for core in core_beliefs:
            # Check if belief already exists
            cursor = self.conn.cursor()
            cursor.execute("SELECT id FROM beliefs WHERE id = ?", (core['id'],))
            if not cursor.fetchone():
                belief = Belief(
                    id=core['id'],
                    type=core['type'],
                    strength=core['strength'],
                    statement=core['statement'],
                    justification=core['justification'],
                    supporting_theories=[],
                    evidence_count=1000,  # High count for core beliefs
                    counter_evidence_count=0,
                    confidence=0.99,  # Near-certain
                    importance=core['importance'],
                    created=time.time(),
                    last_validated=time.time(),
                    times_applied=0,
                    success_rate=1.0,
                    tags=core['tags']
                )
                self._save_belief(belief)

    def form_beliefs(self):
        """Form new beliefs from strong theories."""
        # Get theories from theory layer
        theories = self.theory_layer.get_theories(min_confidence=0.8)

        formed_count = 0
        for theory in theories:
            # Check if theory can become belief
            if BeliefFormation.can_form_belief(theory):
                belief = BeliefFormation.theory_to_belief(theory)

                # Check if we already have this belief
                cursor = self.conn.cursor()
                cursor.execute("SELECT * FROM beliefs WHERE id = ?", (belief.id,))
                existing_row = cursor.fetchone()

                if not existing_row:
                    self._save_belief(belief)
                    formed_count += 1
                else:
                    # Update existing belief with new evidence
                    existing = self._load_belief_from_row(existing_row)
                    existing.evidence_count += theory.evidence_count
                    existing.counter_evidence_count += theory.counter_evidence_count

                    # Recalculate confidence
                    total = existing.evidence_count + existing.counter_evidence_count
                    if total > 0:
                        existing.confidence = existing.evidence_count / total

                    existing.last_validated = time.time()
                    self._save_belief(existing)

        self.last_formation_time = time.time()

        return formed_count

    def get_belief(self, belief_id: str) -> Optional[Belief]:
        """Get a specific belief by ID."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM beliefs WHERE id = ?", (belief_id,))
        row = cursor.fetchone()
        if row:
            return self._load_belief_from_row(row)
        return None

    def get_beliefs(
        self,
        belief_type: Optional[BeliefType] = None,
        strength: Optional[BeliefStrength] = None,
        tags: Optional[List[str]] = None,
        min_confidence: Optional[float] = None,
        min_importance: Optional[float] = None
    ) -> List[Belief]:
        """
        Query beliefs with filters.

        Args:
            belief_type: Filter by belief type
            strength: Filter by belief strength
            tags: Filter by tags
            min_confidence: Minimum confidence score
            min_importance: Minimum importance score

        Returns:
            List of matching beliefs
        """
        # Build SQL query with filters
        query = "SELECT * FROM beliefs WHERE 1=1"
        params = []

        if belief_type:
            query += " AND type = ?"
            params.append(belief_type.value)

        if strength:
            query += " AND strength = ?"
            params.append(strength.value)

        if min_confidence is not None:
            query += " AND confidence >= ?"
            params.append(min_confidence)

        if min_importance is not None:
            query += " AND importance >= ?"
            params.append(min_importance)

        # Order by importance * confidence
        query += " ORDER BY (importance * confidence) DESC"

        cursor = self.conn.cursor()
        cursor.execute(query, params)
        results = [self._load_belief_from_row(row) for row in cursor.fetchall()]

        # Filter by tags in memory (JSON array filtering is complex in SQLite)
        if tags:
            results = [b for b in results if any(tag in b.tags for tag in tags)]

        return results

    def query_for_decision(self, context: Dict[str, Any]) -> List[Belief]:
        """
        Find beliefs relevant to a decision.

        Args:
            context: Decision context (e.g., {'action': 'modify_layer5'})

        Returns:
            List of relevant beliefs
        """
        # Extract tags from context
        context_tags = []
        for key, value in context.items():
            if isinstance(value, str):
                context_tags.append(value.lower())
            context_tags.append(key.lower())

        # Find beliefs with matching tags (load all for now - could optimize)
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM beliefs ORDER BY (importance * confidence) DESC")
        all_beliefs = [self._load_belief_from_row(row) for row in cursor.fetchall()]

        # Find beliefs with matching tags
        relevant = []
        for belief in all_beliefs:
            # Check if any belief tags match context tags
            matches = set(belief.tags) & set(context_tags)
            if matches:
                relevant.append(belief)

        return relevant

    def validate_belief(self, belief_id: str, outcome: bool):
        """
        Update belief based on outcome when applied.

        Args:
            belief_id: Belief that was applied
            outcome: Whether the belief led to success (True) or failure (False)
        """
        belief = self.get_belief(belief_id)
        if not belief:
            return

        belief.times_applied += 1

        # Update success rate
        current_successes = belief.success_rate * (belief.times_applied - 1)
        if outcome:
            current_successes += 1

        belief.success_rate = current_successes / belief.times_applied

        # Update confidence based on success rate
        # If success rate drops below 0.7, reduce confidence
        if belief.success_rate < 0.7:
            belief.confidence *= 0.95  # Reduce confidence

        belief.last_validated = time.time()
        self._save_belief(belief)

    def detect_conflicts(self) -> List[Dict[str, Any]]:
        """
        Detect conflicting beliefs.

        Returns:
            List of conflict descriptions
        """
        conflicts = []

        # Get all beliefs
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM beliefs")
        beliefs_list = [self._load_belief_from_row(row) for row in cursor.fetchall()]

        # Simple conflict detection: beliefs with opposing statements
        # (In practice, would use more sophisticated NLP)

        for i, belief1 in enumerate(beliefs_list):
            for belief2 in beliefs_list[i+1:]:
                # Check for tag overlap (might be related)
                shared_tags = set(belief1.tags) & set(belief2.tags)

                if len(shared_tags) >= 2:
                    # Check if one negates the other (very simple check)
                    if ('never' in belief1.statement.lower() and
                        'always' in belief2.statement.lower()) or \
                       ('always' in belief1.statement.lower() and
                        'never' in belief2.statement.lower()):

                        conflicts.append({
                            'belief1_id': belief1.id,
                            'belief1_statement': belief1.statement,
                            'belief2_id': belief2.id,
                            'belief2_statement': belief2.statement,
                            'shared_tags': list(shared_tags)
                        })

        return conflicts

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about beliefs."""
        # Get base statistics from parent class
        stats = super().get_statistics()
        
        cursor = self.conn.cursor()

        if stats['total'] == 0:
            return {
                'total_beliefs': 0,
                'by_type': {},
                'by_strength': {},
                'average_confidence': 0.0,
                'average_success_rate': 0.0,
                'total_applications': 0,
                'conflicts': 0,
                'last_formation': self.last_formation_time
            }

        # Count by type and strength
        stats['by_type'] = self._get_grouped_counts('type')
        stats['by_strength'] = self._get_grouped_counts('strength')

        # Average confidence
        stats['average_confidence'] = self._get_average_value('confidence')

        # Average success rate (for beliefs that have been applied)
        cursor.execute("SELECT AVG(success_rate) FROM beliefs WHERE times_applied > 0")
        result = cursor.fetchone()
        stats['average_success_rate'] = result[0] if result[0] is not None else 0.0

        # Total applications
        cursor.execute("SELECT SUM(times_applied) FROM beliefs")
        stats['total_applications'] = cursor.fetchone()[0] or 0

        stats['total_beliefs'] = stats['total']
        stats['conflicts'] = len(self.detect_conflicts())
        stats['last_formation'] = self.last_formation_time

        return stats

    def get_core_principles(self) -> List[str]:
        """
        Get list of core principles as strings.

        Returns:
            List of principle statements
        """
        # Get high-importance, high-confidence beliefs
        core = self.get_beliefs(min_confidence=0.9, min_importance=0.8)

        return [belief.statement for belief in core]

    def export(self, filepath: str, limit: Optional[int] = None):
        """Export beliefs to JSON file."""
        cursor = self.conn.cursor()
        if limit:
            cursor.execute("SELECT * FROM beliefs LIMIT ?", (limit,))
        else:
            cursor.execute("SELECT * FROM beliefs")
        beliefs = [self._load_belief_from_row(row) for row in cursor.fetchall()]

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(
                [belief.to_dict() for belief in beliefs],
                f,
                indent=2
            )
