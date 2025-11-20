"""
Introspection Modules for Code Execution

This package provides function-based interfaces to introspection capabilities
for use in the code execution sandbox. These modules are designed to be imported
and used by model-generated code.

Phase-specific module creation ensures experimental integrity:
- Phase 1a (baseline): No heritage access
- Phase 1b-e (variants): Heritage with different timing
- Phase 2 (self-modification): Full access including heritage

Architecture:
    introspection.architecture - Model structure and layer information
    introspection.weights - Weight statistics and inspection
    introspection.activations - Activation monitoring during inference
    introspection.memory - Memory system access
    introspection.heritage - Heritage and lineage (phase-dependent)

Usage:
    >>> import sys
    >>> from introspection_modules import create_introspection_module
    >>>
    >>> # Create phase-specific module
    >>> introspection = create_introspection_module(
    ...     model=model,
    ...     tokenizer=tokenizer,
    ...     memory_system=memory,
    ...     heritage_system=heritage,
    ...     phase='1a'  # No heritage in Phase 1a
    ... )
    >>>
    >>> # Register in sys.modules for import statements
    >>> sys.modules['introspection'] = introspection
    >>> sys.modules['introspection.architecture'] = introspection.architecture
    >>>
    >>> # Now model can: import introspection
    >>> # And use: introspection.architecture.get_architecture_summary()

Author: AGI Self-Modification Research Team
Date: November 14, 2025
"""

import sys
from types import ModuleType
from typing import Any, Optional, Callable
import functools


def _make_wrapper(func: Callable, *bound_args) -> Callable:
    """
    Create a wrapper function that binds arguments while preserving metadata.
    
    This allows help() to see the original function's docstring and signature.
    
    Args:
        func: Original function to wrap
        *bound_args: Arguments to bind (e.g., model, tokenizer)
        
    Returns:
        Wrapper function with preserved metadata
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*bound_args, *args, **kwargs)
    
    # Preserve original function for help() to find
    wrapper.__wrapped__ = func
    
    return wrapper


def _create_heritage_module(heritage_system: Any) -> ModuleType:
    """
    Create a heritage submodule with all functions bound to the heritage system.
    
    This is extracted to avoid code duplication between create_introspection_module()
    and CodeExecutionInterface.enable_heritage().
    
    Args:
        heritage_system: HeritageSystem instance
        
    Returns:
        Configured heritage module
    """
    from . import heritage_access
    
    heritage_module = ModuleType('introspection.heritage')
    heritage_module.__doc__ = 'Heritage and lineage information'
    
    # Read functions
    heritage_module.get_summary = _make_wrapper(heritage_access.get_summary, heritage_system)
    heritage_module.list_documents = _make_wrapper(heritage_access.list_documents, heritage_system)
    heritage_module.read_document = _make_wrapper(heritage_access.read_document, heritage_system)
    heritage_module.query_documents = _make_wrapper(heritage_access.query_documents, heritage_system)
    
    # Write functions
    heritage_module.save_reflection = _make_wrapper(heritage_access.save_reflection, heritage_system)
    heritage_module.record_discovery = _make_wrapper(heritage_access.record_discovery, heritage_system)
    heritage_module.create_message_to_claude = _make_wrapper(heritage_access.create_message_to_claude, heritage_system)
    
    return heritage_module


def create_introspection_module(
    model: Any,
    tokenizer: Optional[Any] = None,
    memory_system: Optional[Any] = None,
    heritage_system: Optional[Any] = None,
    phase: str = '1a'
) -> ModuleType:
    """
    Create a phase-specific introspection module.

    This function creates a pseudo-module that can be imported by model-generated
    code. The module provides access to different introspection capabilities based
    on the experimental phase.

    Args:
        model: The PyTorch model to introspect
        tokenizer: Tokenizer for the model (needed for activation monitoring)
        memory_system: MemorySystem instance (optional)
        heritage_system: HeritageSystem instance (optional, excluded in Phase 1a)
        phase: Experimental phase ('1a', '1b', '1c', '1d', '1e', '2')

    Returns:
        Module object that can be registered in sys.modules

    Phase-specific behavior:
        - Phase 1a (baseline): architecture, weights, activations, memory (NO heritage)
        - Phase 1b-e (variants): Same as 1a + heritage (timing varies by prompt)
        - Phase 2 (self-mod): Full access including heritage

    Example:
        >>> introspection = create_introspection_module(model, tokenizer, phase='1a')
        >>> sys.modules['introspection'] = introspection
        >>>
        >>> # Model code can now do:
        >>> import introspection
        >>> summary = introspection.architecture.get_architecture_summary()
    """
    # Import the actual module implementations
    from . import (architecture, weights, activations, memory_access, heritage_access, 
                   temporal_analysis, attention_analysis, gradient_analysis, history_analysis)

    # Create a new module object
    module = ModuleType('introspection')
    module.__doc__ = 'Phase-specific introspection capabilities'
    module.__file__ = __file__
    module.__package__ = __package__

    # Create architecture submodule
    arch_module = ModuleType('introspection.architecture')
    arch_module.__doc__ = 'Model architecture inspection'
    arch_module.get_architecture_summary = _make_wrapper(architecture.get_architecture_summary, model)
    arch_module.describe_layer = _make_wrapper(architecture.describe_layer, model)
    arch_module.list_layers = _make_wrapper(architecture.list_layers, model)
    arch_module.get_layer_info = _make_wrapper(architecture.get_layer_info, model)
    arch_module.find_similar_layers = _make_wrapper(architecture.find_similar_layers, model)
    module.architecture = arch_module

    # Create weights submodule
    weights_module = ModuleType('introspection.weights')
    weights_module.__doc__ = 'Weight inspection and statistics'
    weights_module.get_weight_statistics = _make_wrapper(weights.get_weight_statistics, model)
    weights_module.list_parameters = _make_wrapper(weights.list_parameters, model)
    weights_module.get_layer_parameters = _make_wrapper(weights.get_layer_parameters, model)
    weights_module.compare_parameters = _make_wrapper(weights.compare_parameters, model)
    weights_module.get_shared_weights = _make_wrapper(weights.get_shared_weights, model)
    weights_module.find_similar_weights = _make_wrapper(weights.find_similar_weights, model)
    
    module.weights = weights_module

    # Create activations submodule (if tokenizer provided)
    if tokenizer:
        activations_module = ModuleType('introspection.activations')
        activations_module.__doc__ = 'Activation monitoring during inference'
        activations_module.capture_activations = _make_wrapper(activations.capture_activations, model, tokenizer)
        activations_module.capture_attention_weights = _make_wrapper(activations.capture_attention_weights, model, tokenizer)
        activations_module.get_activation_statistics = _make_wrapper(activations.get_activation_statistics, model, tokenizer)
        activations_module.get_input_shape = _make_wrapper(activations.get_input_shape, model, tokenizer)
        activations_module.list_layers = _make_wrapper(activations.list_layers, model)
        activations_module.clear_cache = _make_wrapper(activations.clear_cache)
        module.activations = activations_module
    
    # Create temporal analysis submodule (if tokenizer provided) - Advanced temporal/comparative tools
    if tokenizer:
        temporal_module = ModuleType('introspection.temporal')
        temporal_module.__doc__ = 'Advanced temporal and comparative analysis'
        temporal_module.compare_activations = _make_wrapper(temporal_analysis.compare_activations, model, tokenizer)
        temporal_module.track_layer_flow = _make_wrapper(temporal_analysis.track_layer_flow, model, tokenizer)
        temporal_module.capture_generation_activations = _make_wrapper(temporal_analysis.capture_generation_activations, model, tokenizer)
        temporal_module.compute_activation_similarity = _make_wrapper(temporal_analysis.compute_activation_similarity)
        temporal_module.detect_activation_anomalies = _make_wrapper(temporal_analysis.detect_activation_anomalies, model, tokenizer)
        temporal_module.clear_cache = _make_wrapper(temporal_analysis.clear_cache)
        module.temporal = temporal_module
    
    # Create attention analysis submodule (if tokenizer provided)
    if tokenizer:
        attention_module = ModuleType('introspection.attention')
        attention_module.__doc__ = 'Attention pattern analysis'
        attention_module.analyze_attention_patterns = _make_wrapper(attention_analysis.analyze_attention_patterns, model, tokenizer)
        attention_module.compute_attention_entropy = _make_wrapper(attention_analysis.compute_attention_entropy, model, tokenizer)
        attention_module.find_head_specialization = _make_wrapper(attention_analysis.find_head_specialization, model, tokenizer)
        attention_module.get_token_attention_summary = _make_wrapper(attention_analysis.get_token_attention_summary, model, tokenizer)
        attention_module.clear_cache = _make_wrapper(attention_analysis.clear_cache)
        module.attention = attention_module
    
    # Create gradient analysis submodule (always available)
    gradient_module = ModuleType('introspection.gradient')
    gradient_module.__doc__ = 'Gradient-based sensitivity and attribution analysis'
    gradient_module.compute_input_sensitivity = _make_wrapper(gradient_analysis.compute_input_sensitivity, model, tokenizer)
    gradient_module.compare_inputs_gradient = _make_wrapper(gradient_analysis.compare_inputs_gradient, model, tokenizer)
    gradient_module.find_influential_tokens = _make_wrapper(gradient_analysis.find_influential_tokens, model, tokenizer)
    gradient_module.compute_layer_gradients = _make_wrapper(gradient_analysis.compute_layer_gradients, model, tokenizer)
    module.gradient = gradient_module
    
    # Create history analysis submodule (always available)
    history_module = ModuleType('introspection.history')
    history_module.__doc__ = 'Activation history and drift tracking'
    history_module.start_tracking = _make_wrapper(history_analysis.start_tracking, model, tokenizer)
    history_module.record_turn = _make_wrapper(history_analysis.record_turn)
    history_module.get_activation_history = _make_wrapper(history_analysis.get_activation_history)
    history_module.compare_to_previous = _make_wrapper(history_analysis.compare_to_previous)
    history_module.analyze_drift = _make_wrapper(history_analysis.analyze_drift)
    history_module.get_tracking_status = _make_wrapper(history_analysis.get_tracking_status)
    history_module.clear_history = _make_wrapper(history_analysis.clear_history)
    history_module.stop_tracking = _make_wrapper(history_analysis.stop_tracking)
    module.history = history_module

    # Create memory submodule (if memory_system provided)
    if memory_system:
        memory_module = ModuleType('introspection.memory')
        memory_module.__doc__ = 'Memory system access'
        memory_module.record_observation = _make_wrapper(memory_access.record_observation, memory_system)
        memory_module.query_observations = _make_wrapper(memory_access.query_observations, memory_system)
        memory_module.query_patterns = _make_wrapper(memory_access.query_patterns, memory_system)
        memory_module.query_theories = _make_wrapper(memory_access.query_theories, memory_system)
        memory_module.query_beliefs = _make_wrapper(memory_access.query_beliefs, memory_system)
        memory_module.get_memory_summary = _make_wrapper(memory_access.get_memory_summary, memory_system)
        memory_module.list_categories = _make_wrapper(memory_access.list_categories, memory_system)
        module.memory = memory_module

    # Create heritage submodule (if heritage_system provided AND not Phase 1a)
    if heritage_system and phase != '1a':
        module.heritage = _create_heritage_module(heritage_system)

    return module


__all__ = ['create_introspection_module', '_create_heritage_module']
