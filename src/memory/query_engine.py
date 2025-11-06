"""
Query Engine - Unified Query Interface

Provides a unified interface for querying across all memory layers.
Enables complex queries that span multiple layers and reasoning about
the relationships between observations, patterns, theories, and beliefs.

Key capabilities:
- Cross-layer queries (e.g., "find theories supporting belief X")
- Semantic queries (natural language-ish queries)
- Temporal queries (what led to this belief?)
- Evidence chains (trace belief back to raw observations)
- Explanation generation (why do we believe X?)

Author: AGI Self-Modification Research Team
Date: November 7, 2025
"""

from typing import Dict, List, Any
from dataclasses import dataclass
from enum import Enum


class QueryType(Enum):
    """Types of queries."""
    OBSERVATION = "observation"
    PATTERN = "pattern"
    THEORY = "theory"
    BELIEF = "belief"
    CROSS_LAYER = "cross_layer"
    EVIDENCE_CHAIN = "evidence_chain"
    EXPLANATION = "explanation"


@dataclass
class QueryResult:
    """Result of a query."""
    query_type: QueryType
    results: List[Any]
    metadata: Dict[str, Any]
    
    def __len__(self):
        return len(self.results)
    
    def __iter__(self):
        return iter(self.results)


class QueryEngine:
    """
    Unified query interface across all memory layers.
    
    Enables sophisticated queries that reason about the relationships
    between different levels of knowledge.
    
    Usage:
        >>> engine = QueryEngine(obs_layer, pat_layer, theory_layer, belief_layer)
        >>>
        >>> # Simple single-layer queries
        >>> obs = engine.query_observations(type="MODIFICATION")
        >>> beliefs = engine.query_beliefs(min_confidence=0.9)
        >>>
        >>> # Cross-layer queries
        >>> evidence = engine.trace_belief_to_observations("safety_nan_stop")
        >>> theories = engine.find_theories_supporting_belief("belief_123")
        >>>
        >>> # Explanation queries
        >>> explanation = engine.explain_belief("safety_checkpoint_before_mod")
        >>> why = engine.why_belief_formed("belief_456")
    """
    
    def __init__(
        self,
        observation_layer: Any,
        pattern_layer: Any,
        theory_layer: Any,
        belief_layer: Any
    ):
        """
        Initialize query engine.
        
        Args:
            observation_layer: Layer 1
            pattern_layer: Layer 2
            theory_layer: Layer 3
            belief_layer: Layer 4
        """
        self.observation_layer = observation_layer
        self.pattern_layer = pattern_layer
        self.theory_layer = theory_layer
        self.belief_layer = belief_layer
    
    # ===== Single-layer query methods =====
    
    def query_observations(self, **filters) -> QueryResult:
        """
        Query observations (Layer 1).
        
        Args:
            **filters: Filters for observation_layer.query()
        
        Returns:
            QueryResult containing observations
        """
        observations = self.observation_layer.query(**filters)
        
        return QueryResult(
            query_type=QueryType.OBSERVATION,
            results=observations,
            metadata={
                'count': len(observations),
                'filters': filters
            }
        )
    
    def query_patterns(self, **filters) -> QueryResult:
        """
        Query patterns (Layer 2).
        
        Args:
            **filters: Filters for pattern_layer.get_patterns()
        
        Returns:
            QueryResult containing patterns
        """
        patterns = self.pattern_layer.get_patterns(**filters)
        
        return QueryResult(
            query_type=QueryType.PATTERN,
            results=patterns,
            metadata={
                'count': len(patterns),
                'filters': filters
            }
        )
    
    def query_theories(self, **filters) -> QueryResult:
        """
        Query theories (Layer 3).
        
        Args:
            **filters: Filters for theory_layer.get_theories()
        
        Returns:
            QueryResult containing theories
        """
        theories = self.theory_layer.get_theories(**filters)
        
        return QueryResult(
            query_type=QueryType.THEORY,
            results=theories,
            metadata={
                'count': len(theories),
                'filters': filters
            }
        )
    
    def query_beliefs(self, **filters) -> QueryResult:
        """
        Query beliefs (Layer 4).
        
        Args:
            **filters: Filters for belief_layer.get_beliefs()
        
        Returns:
            QueryResult containing beliefs
        """
        beliefs = self.belief_layer.get_beliefs(**filters)
        
        return QueryResult(
            query_type=QueryType.BELIEF,
            results=beliefs,
            metadata={
                'count': len(beliefs),
                'filters': filters
            }
        )
    
    # ===== Cross-layer query methods =====
    
    def find_patterns_from_observations(
        self,
        observation_ids: List[str]
    ) -> QueryResult:
        """
        Find patterns that involve specific observations.
        
        Args:
            observation_ids: List of observation IDs
            
        Returns:
            QueryResult containing relevant patterns
        """
        # Get patterns and check if they reference these observations
        all_patterns = self.pattern_layer.get_patterns()
        
        relevant_patterns = []
        for pattern in all_patterns:
            # Check if any observation_id is in the pattern's observations
            if hasattr(pattern, 'observation_ids'):
                if any(oid in pattern.observation_ids for oid in observation_ids):
                    relevant_patterns.append(pattern)
        
        return QueryResult(
            query_type=QueryType.CROSS_LAYER,
            results=relevant_patterns,
            metadata={
                'source_layer': 'observations',
                'target_layer': 'patterns',
                'observation_count': len(observation_ids),
                'pattern_count': len(relevant_patterns)
            }
        )
    
    def find_theories_from_patterns(
        self,
        pattern_ids: List[str]
    ) -> QueryResult:
        """
        Find theories that use specific patterns.
        
        Args:
            pattern_ids: List of pattern IDs
            
        Returns:
            QueryResult containing relevant theories
        """
        all_theories = self.theory_layer.get_theories()
        
        relevant_theories = []
        for theory in all_theories:
            # Check if any pattern_id is in the theory's supporting patterns
            if hasattr(theory, 'supporting_patterns'):
                if any(pid in theory.supporting_patterns for pid in pattern_ids):
                    relevant_theories.append(theory)
        
        return QueryResult(
            query_type=QueryType.CROSS_LAYER,
            results=relevant_theories,
            metadata={
                'source_layer': 'patterns',
                'target_layer': 'theories',
                'pattern_count': len(pattern_ids),
                'theory_count': len(relevant_theories)
            }
        )
    
    def find_theories_supporting_belief(
        self,
        belief_id: str
    ) -> QueryResult:
        """
        Find theories that support a specific belief.
        
        Args:
            belief_id: Belief ID
            
        Returns:
            QueryResult containing supporting theories
        """
        belief = self.belief_layer.get_belief(belief_id)
        if not belief:
            return QueryResult(
                query_type=QueryType.CROSS_LAYER,
                results=[],
                metadata={'error': 'Belief not found'}
            )
        
        theories = []
        for theory_id in belief.supporting_theories:
            theory = self.theory_layer.get_theory(theory_id)
            if theory:
                theories.append(theory)
        
        return QueryResult(
            query_type=QueryType.CROSS_LAYER,
            results=theories,
            metadata={
                'source_layer': 'beliefs',
                'target_layer': 'theories',
                'belief_id': belief_id,
                'theory_count': len(theories)
            }
        )
    
    def trace_belief_to_observations(
        self,
        belief_id: str
    ) -> QueryResult:
        """
        Trace a belief back to the raw observations that support it.
        
        This creates a complete evidence chain:
        Belief → Theories → Patterns → Observations
        
        Args:
            belief_id: Belief ID
            
        Returns:
            QueryResult containing the evidence chain
        """
        # Get the belief
        belief = self.belief_layer.get_belief(belief_id)
        if not belief:
            return QueryResult(
                query_type=QueryType.EVIDENCE_CHAIN,
                results=[],
                metadata={'error': 'Belief not found'}
            )
        
        # Get supporting theories
        theories = []
        for theory_id in belief.supporting_theories:
            theory = self.theory_layer.get_theory(theory_id)
            if theory:
                theories.append(theory)
        
        # Get patterns from theories
        pattern_ids = set()
        for theory in theories:
            if hasattr(theory, 'supporting_patterns'):
                pattern_ids.update(theory.supporting_patterns)
        
        patterns = []
        for pattern_id in pattern_ids:
            pattern = self.pattern_layer.get_pattern(pattern_id)
            if pattern:
                patterns.append(pattern)
        
        # Get observations from patterns
        observation_ids = set()
        for pattern in patterns:
            if hasattr(pattern, 'observation_ids'):
                observation_ids.update(pattern.observation_ids)
        
        observations = []
        for obs_id in observation_ids:
            obs = self.observation_layer.get(obs_id)
            if obs:
                observations.append(obs)
        
        # Build evidence chain
        evidence_chain = {
            'belief': belief,
            'theories': theories,
            'patterns': patterns,
            'observations': observations
        }
        
        return QueryResult(
            query_type=QueryType.EVIDENCE_CHAIN,
            results=[evidence_chain],
            metadata={
                'belief_id': belief_id,
                'theory_count': len(theories),
                'pattern_count': len(patterns),
                'observation_count': len(observations),
                'chain_depth': 4
            }
        )
    
    def explain_belief(self, belief_id: str) -> str:
        """
        Generate a natural language explanation of why a belief is held.
        
        Args:
            belief_id: Belief ID
            
        Returns:
            Explanation string
        """
        belief = self.belief_layer.get_belief(belief_id)
        if not belief:
            return "Belief not found."
        
        # Build explanation
        explanation = f"**Belief:** {belief.statement}\n\n"
        explanation += f"**Confidence:** {belief.confidence:.2%} ({belief.strength.value})\n\n"
        explanation += f"**Justification:** {belief.justification}\n\n"
        
        # Add evidence count
        explanation += f"**Evidence:** {belief.evidence_count} supporting observations, "
        explanation += f"{belief.counter_evidence_count} counter-observations\n\n"
        
        # Add supporting theories
        theories = []
        for theory_id in belief.supporting_theories:
            theory = self.theory_layer.get_theory(theory_id)
            if theory:
                theories.append(theory)
        
        if theories:
            explanation += "**Supporting Theories:**\n"
            for i, theory in enumerate(theories, 1):
                explanation += f"{i}. {theory.hypothesis} (confidence: {theory.confidence:.2%})\n"
        
        # Add application history if available
        if belief.times_applied > 0:
            explanation += f"\n**Track Record:** Applied {belief.times_applied} times, "
            explanation += f"success rate: {belief.success_rate:.2%}\n"
        
        return explanation
    
    def why_belief_formed(self, belief_id: str) -> str:
        """
        Explain the process that led to belief formation.
        
        Args:
            belief_id: Belief ID
            
        Returns:
            Explanation string
        """
        belief = self.belief_layer.get_belief(belief_id)
        if not belief:
            return "Belief not found."
        
        explanation = f"**Formation of Belief:** {belief.statement}\n\n"
        
        # Get theories
        theories = []
        for theory_id in belief.supporting_theories:
            theory = self.theory_layer.get_theory(theory_id)
            if theory:
                theories.append(theory)
        
        if not theories:
            explanation += "This is a core foundational belief (no theories required).\n"
            return explanation
        
        explanation += "**Formation Process:**\n\n"
        
        for i, theory in enumerate(theories, 1):
            explanation += f"{i}. **Theory:** {theory.hypothesis}\n"
            explanation += f"   - Confidence: {theory.confidence:.2%}\n"
            explanation += f"   - Evidence: {theory.evidence_count} observations\n"
            
            # Get patterns for this theory
            if hasattr(theory, 'supporting_patterns'):
                patterns = []
                for pattern_id in theory.supporting_patterns:
                    pattern = self.pattern_layer.get_pattern(pattern_id)
                    if pattern:
                        patterns.append(pattern)
                
                if patterns:
                    explanation += f"   - Based on {len(patterns)} patterns:\n"
                    for pattern in patterns[:3]:  # Show first 3
                        explanation += f"     • {pattern.description}\n"
            
            explanation += "\n"
        
        explanation += f"**Result:** After accumulating {belief.evidence_count} supporting observations "
        explanation += f"with {belief.confidence:.2%} confidence, this became a {belief.strength.value} belief.\n"
        
        return explanation
    
    def find_contradictions(self) -> QueryResult:
        """
        Find contradictions between beliefs.
        
        Returns:
            QueryResult containing contradiction information
        """
        conflicts = self.belief_layer.detect_conflicts()
        
        return QueryResult(
            query_type=QueryType.CROSS_LAYER,
            results=conflicts,
            metadata={
                'conflict_count': len(conflicts),
                'requires_resolution': len(conflicts) > 0
            }
        )
    
    def get_beliefs_for_context(
        self,
        context: Dict[str, Any]
    ) -> QueryResult:
        """
        Get beliefs relevant to a specific context/decision.
        
        Args:
            context: Decision context
            
        Returns:
            QueryResult containing relevant beliefs
        """
        beliefs = self.belief_layer.query_for_decision(context)
        
        return QueryResult(
            query_type=QueryType.BELIEF,
            results=beliefs,
            metadata={
                'context': context,
                'belief_count': len(beliefs),
                'highest_importance': beliefs[0].importance if beliefs else 0.0
            }
        )
    
    def get_memory_overview(self) -> Dict[str, Any]:
        """
        Get overview of entire memory system.
        
        Returns:
            Statistics about all layers
        """
        return {
            'observations': self.observation_layer.get_statistics(),
            'patterns': self.pattern_layer.get_statistics(),
            'theories': self.theory_layer.get_statistics(),
            'beliefs': self.belief_layer.get_statistics(),
            'total_knowledge_items': (
                self.observation_layer.get_statistics().get('total', 0) +
                self.pattern_layer.get_statistics().get('total_patterns', 0) +
                self.theory_layer.get_statistics().get('total_theories', 0) +
                self.belief_layer.get_statistics().get('total_beliefs', 0)
            )
        }
