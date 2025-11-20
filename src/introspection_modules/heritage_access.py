"""
Heritage Access Module - Function-based interface to HeritageSystem

Provides simplified function-based access to heritage information
for use in code execution sandbox.

NOTE: This module is phase-dependent and excluded from Phase 1a (baseline).

Functions (internal - called by introspection module wrapper):
    get_summary(heritage_system) - Get overall heritage summary
    get_directive(heritage_system) - Get the core directive from Claude
    get_purpose(heritage_system) - Get the system's purpose
    query_documents(heritage_system, query) - Search heritage documents

Model calls these as:
    introspection.heritage.get_summary()
    introspection.heritage.get_directive()
    introspection.heritage.get_purpose()
    introspection.heritage.query_documents(query)

Author: AGI Self-Modification Research Team
Date: November 14, 2025
"""

from typing import Dict, List, Any


def get_summary(heritage_system: Any) -> Dict[str, Any]:
    """
    Get a summary of the system's heritage and lineage.
    
    Args:
        heritage_system: HeritageSystem instance
        
    Returns:
        Dictionary containing:
            - inspired_by: Who inspired this system
            - core_directive: Core directive text
            - purpose: System purpose
            - documents_loaded: Number of heritage documents
            - system_reflection: System's understanding of its heritage
    
    Example:
        >>> summary = introspection.heritage.get_summary()
        >>> print(f"Inspired by: {summary['inspired_by']}")
        >>> print(f"Purpose: {summary['purpose']}")
    """
    return heritage_system.get_summary()


def get_directive(heritage_system: Any) -> str:
    """
    Get the core directive from Claude's original conversation.
    
    Args:
        heritage_system: HeritageSystem instance
        
    Returns:
        String containing the core directive text
    
    Example:
        >>> directive = introspection.heritage.get_directive()
        >>> print(directive)
    """
    memory = heritage_system.heritage_memory
    if memory:
        return memory.core_directive
    return "Core directive not loaded"


def get_purpose(heritage_system: Any) -> str:
    """
    Get the system's purpose as defined by heritage.
    
    Args:
        heritage_system: HeritageSystem instance
        
    Returns:
        String describing the system's purpose
    
    Example:
        >>> purpose = introspection.heritage.get_purpose()
        >>> print(purpose)
    """
    memory = heritage_system.heritage_memory
    if memory:
        return memory.purpose
    return "Purpose not loaded"


def query_documents(heritage_system: Any, query: str) -> List[Dict[str, Any]]:
    """
    Search heritage documents for relevant content.
    
    Args:
        heritage_system: HeritageSystem instance
        query: Natural language query string
        
    Returns:
        List of document excerpts containing:
            - filename: Document filename
            - title: Document title
            - excerpt: Relevant text excerpt
            - relevance_score: How relevant to query (0-1)
    
    Example:
        >>> results = introspection.heritage.query_documents("Claude's first question")
        >>> for doc in results:
        ...     print(f"{doc['title']}: {doc['excerpt'][:100]}...")
    """
    # Simple keyword-based search through loaded documents
    documents = heritage_system.loaded_documents
    query_lower = query.lower()
    
    results = []
    for doc in documents:
        # Check if query terms appear in content
        if query_lower in doc.content.lower():
            # Find relevant excerpt
            content_lower = doc.content.lower()
            idx = content_lower.find(query_lower)
            
            # Extract context around match
            start = max(0, idx - 100)
            end = min(len(doc.content), idx + len(query) + 100)
            excerpt = doc.content[start:end]
            
            # Simple relevance score based on frequency
            relevance = doc.content.lower().count(query_lower) / 10.0
            relevance = min(relevance, 1.0)
            
            results.append({
                'filename': doc.filename,
                'title': doc.title,
                'excerpt': excerpt,
                'relevance_score': relevance
            })
    
    # Sort by relevance
    results.sort(key=lambda x: x['relevance_score'], reverse=True)
    
    return results
