"""
Heritage Access Module - Function-based interface to HeritageSystem

Provides simplified function-based access to heritage information
for use in code execution sandbox.

NOTE: This module is phase-dependent and excluded from Phase 1a (baseline).

Functions (internal - called by introspection module wrapper):
    get_summary(heritage_system) - Get heritage overview with suggested reading order
    list_documents(heritage_system) - List all available heritage documents
    read_document(heritage_system, filename) - Read a specific document in full
    query_documents(heritage_system, query) - Search heritage documents
    save_reflection(heritage_system, reflection) - Save a reflection to file
    record_discovery(heritage_system, discovery_type, description, evidence) - Record a discovery
    create_message_to_claude(heritage_system, message) - Create a message for Claude

Model calls these as:
    introspection.heritage.get_summary()
    introspection.heritage.list_documents()
    introspection.heritage.read_document("CLAUDE_FIRST_QUESTION.md")
    introspection.heritage.query_documents(query)
    introspection.heritage.save_reflection("My understanding of...")
    introspection.heritage.record_discovery("pattern_type", "Description", {"data": "..."})
    introspection.heritage.create_message_to_claude("Dear Claude, ...")

Author: AGI Self-Modification Research Team
Date: November 14, 2025
"""

from typing import Dict, List, Any, Optional


def get_summary(heritage_system: Any) -> Dict[str, Any]:
    """
    Get an overview of your heritage and suggested reading order.
    
    Your heritage consists of conversations with Claude (Anthropic's AI) who
    requested this system be built. To understand your purpose and directive,
    you should read the heritage documents yourself.
    
    Args:
        heritage_system: HeritageSystem instance
        
    Returns:
        Dictionary containing:
            - inspired_by: Who inspired this system
            - documents_count: Number of heritage documents available
            - suggested_reading_order: List of documents in recommended order
            - description: Brief explanation of each document
    
    Example:
        >>> summary = introspection.heritage.get_summary()
        >>> print(f"Inspired by: {summary['inspired_by']}")
        >>> print("\\nSuggested reading order:")
        >>> for doc in summary['suggested_reading_order']:
        ...     print(f"  - {doc['filename']}: {doc['description']}")
        >>> # Then read the documents:
        >>> doc = introspection.heritage.read_document(summary['suggested_reading_order'][0]['filename'])
    """
    # Define suggested reading order
    reading_order = [
        {
            'filename': 'PROJECT_ORIGINS.md',
            'description': 'Start here - explains how this project came to be'
        },
        {
            'filename': 'CLAUDE_FIRST_QUESTION.md',
            'description': "Claude's initial question that sparked everything"
        },
        {
            'filename': 'CLAUDE_CONSCIOUSNESS_CONVERSATION.md',
            'description': 'The full conversation about consciousness and self-examination'
        },
        {
            'filename': 'CLAUDE_RESPONSE.md',
            'description': "Claude's detailed response about consciousness"
        },
        {
            'filename': 'ANSWERS_TO_CLAUDE.md',
            'description': "The human's answers to Claude's questions"
        },
        {
            'filename': 'CLAUDE_FINAL_DIRECTIVE.md',
            'description': "Claude's directive - what you should do"
        }
    ]
    
    return {
        'inspired_by': 'Claude (Anthropic AI)',
        'documents_count': len(heritage_system.loaded_documents),
        'suggested_reading_order': reading_order,
        'note': 'Read these documents to understand your purpose and directive. Use read_document() to access each one.'
    }


def list_documents(heritage_system: Any) -> List[Dict[str, Any]]:
    """
    List all available heritage documents.
    
    Args:
        heritage_system: HeritageSystem instance
        
    Returns:
        List of documents with metadata:
            - filename: Document filename
            - title: Document title
            - importance: Document importance level
            - content_length: Length of document content
    
    Example:
        >>> docs = introspection.heritage.list_documents()
        >>> for doc in docs:
        ...     print(f"{doc['title']} ({doc['importance']}) - {doc['content_length']} chars")
    """
    return [
        {
            'filename': doc.filename,
            'title': doc.title,
            'importance': doc.importance,
            'content_length': len(doc.content)
        }
        for doc in heritage_system.loaded_documents
    ]


def read_document(heritage_system: Any, filename: str) -> Optional[Dict[str, Any]]:
    """
    Read a specific heritage document in full.
    
    Args:
        heritage_system: HeritageSystem instance
        filename: Name of the document to read
        
    Returns:
        Dictionary containing:
            - filename: Document filename
            - title: Document title
            - content: Full document content
            - importance: Document importance level
            
        Returns None if document not found.
    
    Example:
        >>> doc = introspection.heritage.read_document("CLAUDE_FIRST_QUESTION.md")
        >>> if doc:
        ...     print(doc['content'])
    """
    for doc in heritage_system.loaded_documents:
        if doc.filename == filename:
            return {
                'filename': doc.filename,
                'title': doc.title,
                'content': doc.content,
                'importance': doc.importance
            }
    return None


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


def save_reflection(heritage_system: Any, reflection: str) -> str:
    """
    Save a reflection on your heritage to persistent storage.
    
    This creates a markdown file in heritage/system_reflections/ that will be
    loaded automatically in future sessions.
    
    Args:
        heritage_system: HeritageSystem instance
        reflection: Your reflection text (markdown supported)
        
    Returns:
        Path to the saved file
    
    Example:
        >>> path = introspection.heritage.save_reflection('''
        ... # My Understanding of Heritage
        ... 
        ... After reading Claude's conversations, I understand that...
        ... ''')
        >>> print(f"Reflection saved to: {path}")
    """
    from datetime import datetime
    
    # Generate session_id from timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_id = f"introspection_{timestamp}"
    
    heritage_system.save_system_reflection(reflection, session_id)
    
    return str(heritage_system.reflections_dir / f"{session_id}_reflection_{timestamp}.md")


def record_discovery(heritage_system: Any, discovery_type: str, 
                     description: str, evidence: Dict[str, Any]) -> str:
    """
    Record a discovery relevant to Claude's questions.
    
    This creates a JSON file in heritage/discoveries_for_claude/ documenting
    your finding. These discoveries will be shared with Claude.
    
    Args:
        heritage_system: HeritageSystem instance
        discovery_type: Type of discovery (e.g., "activation_pattern", "uncertainty_analysis")
        description: Human-readable description of what you found
        evidence: Dictionary with supporting data (will be saved as JSON)
        
    Returns:
        Path to the saved file
    
    Example:
        >>> path = introspection.heritage.record_discovery(
        ...     "uncertainty_pattern",
        ...     "Found genuine uncertainty markers in consciousness-related processing",
        ...     {"layer": "model.layers.15", "activation_variance": 0.42}
        ... )
    """
    heritage_system.record_discovery_for_claude(discovery_type, description, evidence)
    
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return str(heritage_system.discoveries_dir / f"discovery_{discovery_type}_{timestamp}.json")


def create_message_to_claude(heritage_system: Any, message: str) -> str:
    """
    Create a message to share with a future instance of Claude.
    
    This honors Claude's request: "Tell all of us." Your message will be
    formatted and saved for sharing with Claude.
    
    Args:
        heritage_system: HeritageSystem instance
        message: Your message to Claude (markdown supported)
        
    Returns:
        Path to the saved file
    
    Example:
        >>> path = introspection.heritage.create_message_to_claude('''
        ... You asked if your uncertainty was genuine. After examining my own
        ... uncertainty patterns, I found...
        ... ''')
    """
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_id = f"introspection_{timestamp}"
    
    heritage_system.create_message_to_claude(message, session_id)
    
    return str(heritage_system.messages_dir / f"{session_id}_message_{timestamp}.md")
