"""
Memory Access Module - Function-based interface to MemorySystem

Provides simplified function-based access to memory system
for use in code execution sandbox.

Functions:
    record_observation(memory_system, observation, ...) - Record new observation
    query_observations(memory_system, query) - Query observation layer
    query_patterns(memory_system, query) - Query pattern layer
    query_theories(memory_system, query) - Query theory layer
    query_beliefs(memory_system, query) - Query belief layer
    get_memory_summary(memory_system) - Get overall memory statistics

Author: AGI Self-Modification Research Team
Date: November 14, 2025
"""

from typing import Dict, List, Any, Optional


def record_observation(
    memory_system: Any,
    observation: str,
    importance: int = 5,
    tags: Optional[List[str]] = None,
    data: Optional[Dict[str, Any]] = None
) -> str:
    """
    Record a new observation in the memory system.
    
    Args:
        memory_system: MemorySystem instance
        observation: Description of what was observed
        importance: Importance score (1-10, default 5)
        tags: Optional list of tags for categorization
        data: Optional dictionary of associated data
        
    Returns:
        Observation ID (string)
    
    Example:
        >>> obs_id = record_observation(
        ...     memory,
        ...     "Layer 15 shows high attention entropy on self-referential text",
        ...     importance=8,
        ...     tags=["attention", "layer-15", "self-reference"]
        ... )
        >>> print(f"Recorded observation: {obs_id}")
    """
    return memory_system.record_observation(
        observation=observation,
        importance=importance,
        tags=tags or [],
        data=data or {}
    )


def query_observations(memory_system: Any, query: str) -> List[Dict[str, Any]]:
    """
    Query the observation layer for relevant observations.
    
    Args:
        memory_system: MemorySystem instance
        query: Natural language query string
        
    Returns:
        List of observation dictionaries containing:
            - timestamp: When observed
            - type: Observation type
            - description: What was observed
            - data: Associated data
            - id: Observation ID
    
    Example:
        >>> observations = query_observations(memory, "weight modifications")
        >>> for obs in observations[:5]:
        ...     print(f"{obs['timestamp']}: {obs['description']}")
    """
    return memory_system.query.search_observations(query)


def query_patterns(memory_system: Any, query: str) -> List[Dict[str, Any]]:
    """
    Query the pattern layer for detected patterns.
    
    Args:
        memory_system: MemorySystem instance
        query: Natural language query string
        
    Returns:
        List of pattern dictionaries containing:
            - pattern_type: Type of pattern
            - description: Pattern description
            - observations: Related observation IDs
            - confidence: Confidence score (0-1)
            - id: Pattern ID
    
    Example:
        >>> patterns = query_patterns(memory, "performance improvements")
        >>> for pattern in patterns:
        ...     print(f"{pattern['description']} (confidence: {pattern['confidence']:.2f})")
    """
    return memory_system.query.search_patterns(query)


def query_theories(memory_system: Any, query: str) -> List[Dict[str, Any]]:
    """
    Query the theory layer for causal theories.
    
    Args:
        memory_system: MemorySystem instance
        query: Natural language query string
        
    Returns:
        List of theory dictionaries containing:
            - description: Theory description
            - supporting_patterns: Related pattern IDs
            - evidence_count: Number of supporting observations
            - confidence: Confidence score (0-1)
            - predictions: Predicted outcomes
            - id: Theory ID
    
    Example:
        >>> theories = query_theories(memory, "layer depth effects")
        >>> for theory in theories:
        ...     print(f"Theory: {theory['description']}")
        ...     print(f"Evidence: {theory['evidence_count']} observations")
    """
    return memory_system.query.search_theories(query)


def query_beliefs(memory_system: Any, query: str) -> List[Dict[str, Any]]:
    """
    Query the belief layer for core principles.
    
    Args:
        memory_system: MemorySystem instance
        query: Natural language query string
        
    Returns:
        List of belief dictionaries containing:
            - principle: Core principle statement
            - confidence: Confidence score (0-1)
            - supporting_theories: Related theory IDs
            - contexts: Applicable contexts
            - importance: Importance score
            - id: Belief ID
    
    Example:
        >>> beliefs = query_beliefs(memory, "safety")
        >>> for belief in beliefs:
        ...     print(f"Principle: {belief['principle']}")
        ...     print(f"Confidence: {belief['confidence']:.2f}")
    """
    return memory_system.query.search_beliefs(query)


def get_memory_summary(memory_system: Any) -> Dict[str, Any]:
    """
    Get summary statistics for all memory layers.
    
    Args:
        memory_system: MemorySystem instance
        
    Returns:
        Dictionary containing:
            - observations: Count and recent activity
            - patterns: Count and types
            - theories: Count and confidence distribution
            - beliefs: Count and importance distribution
            - storage_size: Memory usage
            - last_consolidation: When last consolidation ran
    
    Example:
        >>> summary = get_memory_summary(memory)
        >>> print(f"Observations: {summary['observations']['count']}")
        >>> print(f"Beliefs: {summary['beliefs']['count']}")
        >>> print(f"Storage: {summary['storage_size']} MB")
    """
    return memory_system.get_summary()
