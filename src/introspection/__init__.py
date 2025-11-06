"""
Introspection APIs for Self-Examining AGI
Provides tools for the system to examine its own architecture, weights, and activations
"""

from .weight_inspector import WeightInspector
from .activation_monitor import ActivationMonitor
from .architecture_navigator import ArchitectureNavigator

__all__ = ['WeightInspector', 'ActivationMonitor', 'ArchitectureNavigator']
