"""
Memory System - Unified Memory Coordinator

The Memory System coordinates all four layers of memory and provides
a unified interface for the AGI to learn from experience.

The four-layer architecture:
1. Observation Layer: Records every event, measurement, modification
2. Pattern Layer: Detects patterns and correlations in observations
3. Theory Layer: Builds causal models and explanatory theories
4. Belief Layer: Forms core principles and reliable knowledge

This creates a knowledge hierarchy from raw data to core beliefs,
enabling the system to learn, predict, and make informed decisions.

Author: AGI Self-Modification Research Team
Date: November 7, 2025
"""

import time
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

from .observation_layer import ObservationLayer, ObservationType
from .pattern_layer import PatternLayer
from .theory_layer import TheoryLayer, Theory
from .belief_layer import BeliefLayer, Belief
from .query_engine import QueryEngine, QueryResult

# Optional import for weight sharing detection
try:
    from ..introspection import WeightInspector
except ImportError:
    WeightInspector = None

logger = logging.getLogger(__name__)


class MemorySystem:
    """
    Unified memory system coordinator.

    This class orchestrates all four memory layers and provides high-level
    methods for learning, querying, and consolidating knowledge.

    Features:
    - Automatic knowledge consolidation (observations → patterns → theories → beliefs)
    - Unified query interface via QueryEngine
    - Memory management (cleanup old data, manage storage)
    - Statistics and health monitoring
    - Export/import capabilities

    Usage:
        >>> memory = MemorySystem("data/memory")
        >>>
        >>> # Record an observation
        >>> memory.record_observation(
        ...     type=ObservationType.MODIFICATION,
        ...     description="Modified layer 5 weights",
        ...     data={'layer': 'layer5', 'percentage': 0.1}
        ... )
        >>>
        >>> # Consolidate knowledge (patterns → theories → beliefs)
        >>> memory.consolidate()
        >>>
        >>> # Query for decision support
        >>> beliefs = memory.query.get_beliefs_for_context({'action': 'modify'})
        >>>
        >>> # Get explanation
        >>> explanation = memory.query.explain_belief("safety_checkpoint_before_mod")
    """

    def __init__(self, storage_dir: str):
        """
        Initialize memory system.

        Args:
            storage_dir: Base directory for memory storage
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # Initialize layers - all databases in same directory
        self.observations = ObservationLayer(
            str(self.storage_dir)
        )

        self.patterns = PatternLayer(
            str(self.storage_dir),
            self.observations
        )

        self.theories = TheoryLayer(
            str(self.storage_dir),
            self.patterns,
            self.observations
        )

        self.beliefs = BeliefLayer(
            str(self.storage_dir),
            self.theories
        )

        # Initialize query engine
        self.query = QueryEngine(
            self.observations,
            self.patterns,
            self.theories,
            self.beliefs
        )

        # Track consolidation
        self.last_consolidation = 0.0
        self.consolidation_interval = 3600.0  # 1 hour default
        
        # Optional WeightInspector for coupled modification detection
        self._weight_inspector = None  # Optional WeightInspector instance

    # ===== High-level convenience methods =====
    
    def set_weight_inspector(self, inspector) -> None:
        """
        Set WeightInspector for coupled modification detection.
        
        This enables the memory system to detect when modifications affect
        shared weights (e.g., Qwen2.5's lm_head ↔ embed_tokens coupling).
        
        Args:
            inspector: WeightInspector instance
            
        Example:
            >>> memory = MemorySystem("data/memory")
            >>> inspector = WeightInspector(model)
            >>> memory.set_weight_inspector(inspector)
        """
        self._weight_inspector = inspector
        logger.info("WeightInspector attached to MemorySystem for coupled modification detection")

    def record_observation(
        self,
        obs_type: ObservationType,
        category: str,
        description: str,
        data: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        importance: float = 0.5
    ) -> str:
        """
        Record an observation (Layer 1).

        Args:
            obs_type: Type of observation
            category: Category (e.g., 'layer5', 'perplexity')
            description: Human-readable description
            data: Additional structured data
            tags: Tags for organization
            importance: Importance score (0.0-1.0)

        Returns:
            Observation ID
        """
        return self.observations.record(
            obs_type=obs_type,
            category=category,
            description=description,
            data=data or {},
            tags=tags or [],
            importance=importance
        )
    
    def record_modification(
        self,
        layer_name: str,
        modification_data: Dict[str, Any],
        description: Optional[str] = None,
        tags: Optional[List[str]] = None,
        importance: float = 0.8
    ) -> str:
        """
        Record a modification with weight sharing awareness.
        
        This method automatically detects if the modified layer shares weights
        with other layers and records it as a coupled modification if needed.
        This prevents the memory system from forming spurious patterns about
        architectural facts.
        
        Args:
            layer_name: Name of the layer being modified
            modification_data: Details of the modification
            description: Human-readable description (auto-generated if None)
            tags: Additional tags (auto-generated tags will be added)
            importance: Importance score (default 0.8 for modifications)
            
        Returns:
            Observation ID
            
        Example:
            >>> memory.record_modification(
            ...     layer_name="model.embed_tokens.weight",
            ...     modification_data={'change_magnitude': 0.01, 'method': 'gradient'},
            ...     tags=['experimental', 'phase1']
            ... )
        """
        # Check if WeightInspector is available and layer shares weights
        shared_layers = []
        if self._weight_inspector:
            try:
                shared_layers = self._weight_inspector.get_shared_layers(layer_name)
            except Exception as e:
                logger.warning(f"Could not check weight sharing for {layer_name}: {e}")
        
        # Prepare observation data
        obs_tags = tags or []
        obs_data = modification_data.copy()
        obs_data['layer'] = layer_name
        
        if shared_layers:
            # Record as coupled modification
            category = "coupled_modification"
            obs_data['coupled_layers'] = shared_layers
            obs_data['primary_layer'] = layer_name
            
            # Generate description
            if description is None:
                description = (
                    f"Modified {layer_name} (coupled with {', '.join(shared_layers)})"
                )
            
            # Add coupled tags
            obs_tags.extend(['modification', 'coupled', layer_name])
            obs_tags.extend(shared_layers)
            
            # Higher importance - affects multiple components
            importance = max(importance, 0.9)
            
            logger.info(
                f"Recording coupled modification: {layer_name} affects {shared_layers}"
            )
        else:
            # Record as standard modification
            category = layer_name
            
            # Generate description
            if description is None:
                description = f"Modified {layer_name}"
            
            # Add standard tags
            obs_tags.extend(['modification', layer_name])
        
        # Remove duplicates from tags
        obs_tags = list(set(obs_tags))
        
        return self.observations.record(
            obs_type=ObservationType.MODIFICATION,
            category=category,
            description=description,
            data=obs_data,
            tags=obs_tags,
            importance=importance
        )

    def auto_consolidate_if_needed(self):
        """
        Automatically consolidate if enough time has passed.

        This should be called periodically (e.g., after operations).
        """
        current_time = time.time()

        if current_time - self.last_consolidation >= self.consolidation_interval:
            self.consolidate()

    def consolidate(self, force: bool = False) -> Dict[str, int]:
        """
        Consolidate knowledge across all layers.

        This process:
        1. Detects new patterns from observations
        2. Builds new theories from patterns
        3. Forms new beliefs from theories

        Args:
            force: Force consolidation even if not enough new data

        Returns:
            Dictionary with counts of new knowledge items
        """
        stats = {
            'patterns_found': 0,
            'theories_built': 0,
            'beliefs_formed': 0
        }

        # Step 1: Detect patterns
        # Get recent observations (last 24 hours)
        cutoff_time = time.time() - (24 * 3600)
        recent_obs = self.observations.query(start_time=cutoff_time)

        if len(recent_obs) >= 10 or force:  # Minimum observations for pattern detection
            patterns_found = self.patterns.detect_patterns()
            stats['patterns_found'] = patterns_found

        # Step 2: Build theories
        # Get recent patterns (with sufficient confidence)
        recent_patterns = self.patterns.get_patterns(min_confidence=0.5)

        if len(recent_patterns) >= 5 or force:  # Minimum patterns for theory building
            theories_built = self.theories.build_theories()
            stats['theories_built'] = theories_built

        # Step 3: Form beliefs
        # Check theories that are strong enough
        strong_theories = self.theories.get_theories(min_confidence=0.85)

        if len(strong_theories) > 0 or force:
            beliefs_formed = self.beliefs.form_beliefs()
            stats['beliefs_formed'] = beliefs_formed

        self.last_consolidation = time.time()

        return stats

    def get_decision_support(
        self,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Get comprehensive decision support for a context.

        This returns:
        - Relevant beliefs
        - Supporting theories
        - Related patterns
        - Recent observations

        Args:
            context: Decision context (e.g., {'action': 'modify_layer5'})

        Returns:
            Dictionary with decision support information
        """
        # Get relevant beliefs
        beliefs_result = self.query.get_beliefs_for_context(context)
        beliefs = beliefs_result.results

        # Get theories supporting top beliefs
        theories = []
        if beliefs:
            for belief in beliefs[:3]:  # Top 3 beliefs
                theory_result = self.query.find_theories_supporting_belief(belief.id)
                theories.extend(theory_result.results)

        # Get related patterns
        patterns = []
        context_tags = list(context.values())
        if context_tags:
            pattern_result = self.query.query_patterns(tags=context_tags)
            patterns = pattern_result.results

        # Get recent related observations
        observations = []
        if context_tags:
            obs_result = self.query.query_observations(tags=context_tags)
            observations = obs_result.results[:10]  # Last 10

        return {
            'beliefs': beliefs,
            'theories': theories,
            'patterns': patterns,
            'observations': observations,
            'recommendation': self._generate_recommendation(beliefs, theories)
        }

    def _generate_recommendation(
        self,
        beliefs: List[Belief],
        theories: List[Theory]
    ) -> str:
        """Generate a recommendation based on beliefs and theories."""
        if not beliefs:
            return "No strong beliefs available for this context. Proceed with caution."

        # Get highest importance belief
        top_belief = max(beliefs, key=lambda b: b.importance * b.confidence)

        recommendation = f"**Recommendation:** {top_belief.statement}\n\n"
        recommendation += f"**Confidence:** {top_belief.confidence:.2%}\n"
        recommendation += f"**Justification:** {top_belief.justification}\n"

        if theories:
            recommendation += f"\n**Supporting theories:** {len(theories)} theories support this.\n"

        return recommendation

    def get_core_principles(self) -> List[str]:
        """
        Get list of core principles.

        Returns:
            List of principle statements
        """
        return self.beliefs.get_core_principles()

    def explain_decision(self, belief_id: str) -> str:
        """
        Explain why a particular belief/decision is held.

        Args:
            belief_id: Belief to explain

        Returns:
            Explanation string
        """
        return self.query.explain_belief(belief_id)

    def trace_to_evidence(self, belief_id: str) -> QueryResult:
        """
        Trace a belief back to raw observations.

        Args:
            belief_id: Belief to trace

        Returns:
            QueryResult with evidence chain
        """
        return self.query.trace_belief_to_observations(belief_id)

    # ===== Memory management =====

    def cleanup_old_data(
        self,
        observation_days: int = 30,
        pattern_days: int = 90,
        theory_days: int = 180
    ):
        """
        Clean up old, low-value data.

        Args:
            observation_days: Remove observations older than this
            pattern_days: Remove patterns older than this
            theory_days: Remove theories older than this
        """
        current_time = time.time()

        # Consolidate first to preserve important patterns/theories
        self.consolidate(force=True)

        # Clean observations
        cutoff = current_time - (observation_days * 86400)
        all_obs = self.observations.query()
        removed_obs = 0
        for obs in all_obs:
            if obs.timestamp < cutoff and obs.importance < 0.3:
                # Remove low-importance old observations
                # (In practice, would implement delete method)
                removed_obs += 1

        # Prune patterns
        self.patterns.prune_patterns(max_age_days=pattern_days)

        # Prune theories
        self.theories.prune_theories(max_age_days=theory_days)

        # Beliefs are rarely pruned (only manually)

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about memory system."""
        overview = self.query.get_memory_overview()

        # Add system-level stats
        overview['consolidation'] = {
            'last_consolidation': self.last_consolidation,
            'interval': self.consolidation_interval,
            'time_since_last': time.time() - self.last_consolidation
        }

        # Check for issues
        conflicts = self.query.find_contradictions()
        overview['health'] = {
            'conflicts': len(conflicts),
            'status': 'healthy' if len(conflicts) == 0 else 'conflicts_detected'
        }

        return overview

    def export_all(self, export_dir: str):
        """
        Export all memory layers.

        Args:
            export_dir: Directory to export to
        """
        export_path = Path(export_dir)
        export_path.mkdir(parents=True, exist_ok=True)

        self.observations.export(str(export_path / "observations.json"))
        self.patterns.export(str(export_path / "patterns.json"))
        self.theories.export(str(export_path / "theories.json"))
        self.beliefs.export(str(export_path / "beliefs.json"))

    def set_consolidation_interval(self, hours: float):
        """
        Set how often to auto-consolidate.

        Args:
            hours: Hours between consolidations
        """
        self.consolidation_interval = hours * 3600.0

    # ===== Introspection methods =====

    def what_do_i_know_about(self, topic: str) -> str:
        """
        Query what the system knows about a topic.

        Args:
            topic: Topic to query

        Returns:
            Summary string
        """
        summary = f"# Knowledge about: {topic}\n\n"

        # Query each layer
        beliefs = self.query.query_beliefs(tags=[topic])
        theories = self.query.query_theories(tags=[topic])
        patterns = self.query.query_patterns(tags=[topic])
        observations = self.query.query_observations(tags=[topic])

        summary += f"## Beliefs ({len(beliefs)} found)\n"
        for belief in beliefs.results[:5]:
            summary += f"- {belief.statement} (confidence: {belief.confidence:.2%})\n"

        summary += f"\n## Theories ({len(theories)} found)\n"
        for theory in theories.results[:5]:
            summary += f"- {theory.hypothesis} (confidence: {theory.confidence:.2%})\n"

        summary += f"\n## Patterns ({len(patterns)} found)\n"
        for pattern in patterns.results[:5]:
            summary += f"- {pattern.description}\n"

        summary += f"\n## Observations ({len(observations)} found)\n"
        summary += f"Total observations recorded: {len(observations)}\n"

        return summary

    def what_have_i_learned_recently(self, hours: int = 24) -> str:
        """
        Summarize recent learning.

        Args:
            hours: Time window

        Returns:
            Summary string
        """
        cutoff = time.time() - (hours * 3600)

        # Get recent items
        recent_obs = self.observations.query(start_time=cutoff)

        recent_patterns = [
            p for p in self.patterns.get_patterns()
            if p.first_seen >= cutoff
        ]

        recent_theories = [
            t for t in self.theories.get_theories()
            if t.created >= cutoff
        ]

        recent_beliefs = [
            b for b in self.beliefs.get_beliefs()
            if b.created >= cutoff
        ]

        summary = f"# Learning Summary (Last {hours} hours)\n\n"
        summary += f"- **Observations:** {len(recent_obs)} new events recorded\n"
        summary += f"- **Patterns:** {len(recent_patterns)} patterns detected\n"
        summary += f"- **Theories:** {len(recent_theories)} theories formed\n"
        summary += f"- **Beliefs:** {len(recent_beliefs)} new beliefs established\n\n"

        if recent_beliefs:
            summary += "## New Beliefs\n"
            for belief in recent_beliefs:
                summary += f"- {belief.statement}\n"

        return summary
