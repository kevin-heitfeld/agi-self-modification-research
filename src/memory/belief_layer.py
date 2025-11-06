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
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
from enum import Enum
from collections import defaultdict


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


class BeliefLayer:
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
        >>> layer = BeliefLayer("data/memory/beliefs", theory_layer)
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
    
    def __init__(self, storage_dir: str, theory_layer: Any):
        """
        Initialize belief layer.
        
        Args:
            storage_dir: Directory for storing beliefs
            theory_layer: Reference to theory layer
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self.theory_layer = theory_layer
        
        # Belief storage
        self.beliefs_file = self.storage_dir / "beliefs.json"
        self.beliefs: Dict[str, Belief] = {}
        self._load_beliefs()
        
        # Track when beliefs were last formed
        self.last_formation_time = 0.0
        
        # Initialize core safety beliefs (hardcoded foundational beliefs)
        self._initialize_core_beliefs()
    
    def _load_beliefs(self):
        """Load beliefs from storage."""
        if self.beliefs_file.exists():
            with open(self.beliefs_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.beliefs = {
                    bid: Belief.from_dict(bdata)
                    for bid, bdata in data.items()
                }
    
    def _save_beliefs(self):
        """Save beliefs to storage."""
        with open(self.beliefs_file, 'w', encoding='utf-8') as f:
            json.dump(
                {bid: belief.to_dict() for bid, belief in self.beliefs.items()},
                f,
                indent=2
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
            if core['id'] not in self.beliefs:
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
                self.beliefs[belief.id] = belief
        
        self._save_beliefs()
    
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
                if belief.id not in self.beliefs:
                    self.beliefs[belief.id] = belief
                    formed_count += 1
                else:
                    # Update existing belief with new evidence
                    existing = self.beliefs[belief.id]
                    existing.evidence_count += theory.evidence_count
                    existing.counter_evidence_count += theory.counter_evidence_count
                    
                    # Recalculate confidence
                    total = existing.evidence_count + existing.counter_evidence_count
                    if total > 0:
                        existing.confidence = existing.evidence_count / total
                    
                    existing.last_validated = time.time()
        
        self.last_formation_time = time.time()
        self._save_beliefs()
        
        return formed_count
    
    def get_belief(self, belief_id: str) -> Optional[Belief]:
        """Get a specific belief by ID."""
        return self.beliefs.get(belief_id)
    
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
        results = list(self.beliefs.values())
        
        if belief_type:
            results = [b for b in results if b.type == belief_type]
        
        if strength:
            results = [b for b in results if b.strength == strength]
        
        if tags:
            results = [b for b in results if any(tag in b.tags for tag in tags)]
        
        if min_confidence is not None:
            results = [b for b in results if b.confidence >= min_confidence]
        
        if min_importance is not None:
            results = [b for b in results if b.importance >= min_importance]
        
        # Sort by importance * confidence
        results.sort(key=lambda b: b.importance * b.confidence, reverse=True)
        
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
        
        # Find beliefs with matching tags
        relevant = []
        for belief in self.beliefs.values():
            # Check if any belief tags match context tags
            matches = set(belief.tags) & set(context_tags)
            if matches:
                relevant.append(belief)
        
        # Sort by importance
        relevant.sort(key=lambda b: b.importance * b.confidence, reverse=True)
        
        return relevant
    
    def validate_belief(self, belief_id: str, outcome: bool):
        """
        Update belief based on outcome when applied.
        
        Args:
            belief_id: Belief that was applied
            outcome: Whether the belief led to success (True) or failure (False)
        """
        belief = self.beliefs.get(belief_id)
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
        self._save_beliefs()
    
    def detect_conflicts(self) -> List[Dict[str, Any]]:
        """
        Detect conflicting beliefs.
        
        Returns:
            List of conflict descriptions
        """
        conflicts = []
        
        beliefs_list = list(self.beliefs.values())
        
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
        if not self.beliefs:
            return {
                'total_beliefs': 0,
                'by_type': {},
                'by_strength': {},
                'average_confidence': 0.0
            }
        
        # Count by type
        by_type = defaultdict(int)
        for belief in self.beliefs.values():
            by_type[belief.type.value] += 1
        
        # Count by strength
        by_strength = defaultdict(int)
        for belief in self.beliefs.values():
            by_strength[belief.strength.value] += 1
        
        # Average confidence
        avg_confidence = sum(b.confidence for b in self.beliefs.values()) / len(self.beliefs)
        
        # Average success rate
        beliefs_with_applications = [b for b in self.beliefs.values() if b.times_applied > 0]
        avg_success = (sum(b.success_rate for b in beliefs_with_applications) / 
                      len(beliefs_with_applications)) if beliefs_with_applications else 0.0
        
        return {
            'total_beliefs': len(self.beliefs),
            'by_type': dict(by_type),
            'by_strength': dict(by_strength),
            'average_confidence': avg_confidence,
            'average_success_rate': avg_success,
            'total_applications': sum(b.times_applied for b in self.beliefs.values()),
            'conflicts': len(self.detect_conflicts()),
            'last_formation': self.last_formation_time
        }
    
    def get_core_principles(self) -> List[str]:
        """
        Get list of core principles as strings.
        
        Returns:
            List of principle statements
        """
        # Get high-importance, high-confidence beliefs
        core = self.get_beliefs(min_confidence=0.9, min_importance=0.8)
        
        return [belief.statement for belief in core]
    
    def export(self, filepath: str):
        """Export beliefs to JSON file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(
                [belief.to_dict() for belief in self.beliefs.values()],
                f,
                indent=2
            )
