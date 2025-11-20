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
    description: str,
    category: str = "general",
    importance: float = 0.5,
    tags: Optional[List[str]] = None,
    data: Optional[Dict[str, Any]] = None
) -> str:
    """
    Record a new observation in the memory system.

    Args:
        memory_system: MemorySystem instance
        description: Description of what was observed
        category: Category for the observation (default: "general")
        importance: Importance score (0.0-1.0, default 0.5)
        tags: Optional list of tags for categorization
        data: Optional dictionary of associated data

    Returns:
        Observation ID (string)

    Example:
        >>> obs_id = record_observation(
        ...     memory,
        ...     "Layer 15 shows high attention entropy on self-referential text",
        ...     category="attention",
        ...     importance=0.8,
        ...     tags=["attention", "layer-15", "self-reference"]
        ... )
        >>> print(f"Recorded observation: {obs_id}")
    """
    from ..memory.observation_layer import ObservationType

    return memory_system.record_observation(
        obs_type=ObservationType.INTROSPECTION,
        category=category,
        description=description,
        data=data or {},
        tags=tags or [],
        importance=importance
    )


def query_observations(memory_system: Any, query: Optional[str] = None, **filters) -> List[Dict[str, Any]]:
    """
    Query the observation layer for relevant observations.

    Args:
        memory_system: MemorySystem instance
        query: Natural language query string (searches descriptions)
        **filters: Additional filters (tags, category, obs_type, min_importance, etc.)

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
    # Note: ObservationLayer.query() doesn't accept 'text' parameter
    # It uses: obs_type, category, tags, min_importance, start_time, end_time, limit, include_obsolete
    # If query is provided, we should filter by category or use tags
    if query and 'category' not in filters:
        # Use query as a category filter if no specific category provided
        filters['category'] = query

    result = memory_system.query.query_observations(**filters)
    return result.results


def query_patterns(memory_system: Any, query: Optional[str] = None, **filters) -> List[Dict[str, Any]]:
    """
    Query the pattern layer for detected patterns.

    Args:
        memory_system: MemorySystem instance
        query: Natural language query string (searches descriptions)
        **filters: Additional filters (tags, pattern_type, min_confidence, etc.)

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
    # Note: PatternLayer.query() doesn't accept 'text' parameter
    # Convert query to tags if provided
    if query and 'tags' not in filters:
        filters['tags'] = [query]

    result = memory_system.query.query_patterns(**filters)
    return result.results


def query_theories(memory_system: Any, query: Optional[str] = None, **filters) -> List[Dict[str, Any]]:
    """
    Query the theory layer for causal theories.

    Args:
        memory_system: MemorySystem instance
        query: Natural language query string (searches descriptions)
        **filters: Additional filters (tags, min_confidence, etc.)

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
    # Note: TheoryLayer.query() doesn't accept 'text' parameter
    # Convert query to tags if provided
    if query and 'tags' not in filters:
        filters['tags'] = [query]

    result = memory_system.query.query_theories(**filters)
    return result.results


def query_beliefs(memory_system: Any, query: Optional[str] = None, **filters) -> List[Dict[str, Any]]:
    """
    Query the belief layer for core principles.

    Args:
        memory_system: MemorySystem instance
        query: Natural language query string (searches principles)
        **filters: Additional filters (tags, min_confidence, min_importance, etc.)

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
    # Note: BeliefLayer.query() doesn't accept 'text' parameter
    # Convert query to tags if provided
    if query and 'tags' not in filters:
        filters['tags'] = [query]

    result = memory_system.query.query_beliefs(**filters)
    return result.results


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
    result = memory_system.query.get_memory_overview()
    return result.metadata if hasattr(result, 'metadata') else {}


def list_categories(memory_system: Any) -> Dict[str, int]:
    """
    List all observation categories and their counts.
    
    This helps you remember what categories you've used when saving observations,
    making it easier to query them later.

    Args:
        memory_system: MemorySystem instance

    Returns:
        Dictionary mapping category names to observation counts
        
    Example:
        >>> categories = list_categories(memory)
        >>> print("Available categories:")
        >>> for category, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
        ...     print(f"  {category}: {count} observations")
        >>> 
        >>> # Now query a specific category
        >>> obs = query_observations(memory, category="activation_patterns")
    """
    # Get all observations without filters
    all_obs = memory_system.query.query_observations(limit=10000)
    
    # Count by category
    categories = {}
    for obs in all_obs.results:
        category = obs.get('category', 'unknown')
        categories[category] = categories.get(category, 0) + 1
    
    return categories
