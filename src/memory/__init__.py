"""
Memory System Package

This package implements a 4-layer memory architecture for learning from experience:

Layer 1: Direct Observations - Raw events and measurements
Layer 2: Patterns - Recognized patterns across observations
Layer 3: Theories - Causal models and explanations
Layer 4: Beliefs - Core principles and safety rules

The memory system enables the AGI to:
- Record all operations and outcomes
- Recognize patterns in behavior
- Form theories about causation
- Build high-confidence beliefs
- Learn from past experiences
- Make better decisions over time

Author: AGI Self-Modification Research Team
Date: November 7, 2025
"""

from .observation_layer import ObservationLayer
from .pattern_layer import PatternLayer
from .theory_layer import TheoryLayer
from .belief_layer import BeliefLayer
from .memory_system import MemorySystem
from .query_engine import QueryEngine

__all__ = [
    'ObservationLayer',
    'PatternLayer',
    'TheoryLayer',
    'BeliefLayer',
    'MemorySystem',
    'QueryEngine'
]
