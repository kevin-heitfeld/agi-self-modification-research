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
from typing import Any, Optional


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
    heritage_module.get_summary = lambda: heritage_access.get_summary(heritage_system)
    heritage_module.list_documents = lambda: heritage_access.list_documents(heritage_system)
    heritage_module.read_document = lambda filename: heritage_access.read_document(heritage_system, filename)
    heritage_module.query_documents = lambda query: heritage_access.query_documents(
        heritage_system, query
    )
    
    # Write functions
    heritage_module.save_reflection = lambda reflection: heritage_access.save_reflection(heritage_system, reflection)
    heritage_module.record_discovery = lambda discovery_type, description, evidence: heritage_access.record_discovery(
        heritage_system, discovery_type, description, evidence
    )
    heritage_module.create_message_to_claude = lambda message: heritage_access.create_message_to_claude(heritage_system, message)
    
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
    arch_module.get_architecture_summary = lambda: architecture.get_architecture_summary(model)
    arch_module.describe_layer = lambda layer_name: architecture.describe_layer(model, layer_name)
    arch_module.list_layers = lambda filter_pattern=None: architecture.list_layers(model, filter_pattern)
    arch_module.get_layer_info = lambda layer_name: architecture.get_layer_info(model, layer_name)
    arch_module.find_similar_layers = lambda layer_name: architecture.find_similar_layers(model, layer_name)
    module.architecture = arch_module

    # Create weights submodule
    weights_module = ModuleType('introspection.weights')
    weights_module.__doc__ = 'Weight inspection and statistics'
    weights_module.get_weight_statistics = lambda parameter_names: weights.get_weight_statistics(model, parameter_names)
    
    # Make list_parameters() flexible - accept optional layer_prefix for convenience
    def _list_parameters_wrapper(layer_prefix=None):
        if layer_prefix is None:
            return weights.list_parameters(model)
        else:
            # Redirect to get_layer_parameters for filtered listing
            return weights.get_layer_parameters(model, layer_prefix)
    
    weights_module.list_parameters = _list_parameters_wrapper
    weights_module.get_layer_parameters = lambda layer_prefix: weights.get_layer_parameters(model, layer_prefix)
    weights_module.compare_parameters = lambda param1, param2: weights.compare_parameters(model, param1, param2)
    weights_module.get_shared_weights = lambda: weights.get_shared_weights(model)
    weights_module.find_similar_weights = lambda layer_name, top_k=5: weights.find_similar_weights(model, layer_name, top_k)
    
    module.weights = weights_module

    # Create activations submodule (if tokenizer provided)
    if tokenizer:
        activations_module = ModuleType('introspection.activations')
        activations_module.__doc__ = 'Activation monitoring during inference'
        activations_module.capture_activations = lambda text, layer_names: activations.capture_activations(
            model, tokenizer, text, layer_names
        )
        activations_module.capture_attention_weights = lambda text, layer_names: activations.capture_attention_weights(
            model, tokenizer, text, layer_names
        )
        activations_module.get_activation_statistics = lambda layer_name: activations.get_activation_statistics(
            model, tokenizer, layer_name
        )
        activations_module.get_input_shape = lambda sample_text="test": activations.get_input_shape(
            model, tokenizer, sample_text
        )
        activations_module.list_layers = lambda filter_pattern=None: activations.list_layers(model, filter_pattern)
        activations_module.clear_cache = lambda: activations.clear_cache()
        module.activations = activations_module
    
    # Create temporal analysis submodule (if tokenizer provided) - Advanced temporal/comparative tools
    if tokenizer:
        temporal_module = ModuleType('introspection.temporal')
        temporal_module.__doc__ = 'Advanced temporal and comparative analysis'
        temporal_module.compare_activations = lambda texts, layer_names: temporal_analysis.compare_activations(
            model, tokenizer, texts, layer_names
        )
        temporal_module.track_layer_flow = lambda text, layer_names=None: temporal_analysis.track_layer_flow(
            model, tokenizer, text, layer_names
        )
        temporal_module.capture_generation_activations = lambda prompt, layer_names, max_new_tokens=20: temporal_analysis.capture_generation_activations(
            model, tokenizer, prompt, layer_names, max_new_tokens
        )
        temporal_module.compute_activation_similarity = lambda act1, act2: temporal_analysis.compute_activation_similarity(act1, act2)
        temporal_module.detect_activation_anomalies = lambda text, layer_names, baseline_texts: temporal_analysis.detect_activation_anomalies(
            model, tokenizer, text, layer_names, baseline_texts
        )
        temporal_module.clear_cache = lambda: temporal_analysis.clear_cache()
        module.temporal = temporal_module
    
    # Create attention analysis submodule (if tokenizer provided)
    if tokenizer:
        attention_module = ModuleType('introspection.attention')
        attention_module.__doc__ = 'Attention pattern analysis'
        attention_module.analyze_attention_patterns = lambda text, layer_names: attention_analysis.analyze_attention_patterns(
            model, tokenizer, text, layer_names
        )
        attention_module.compute_attention_entropy = lambda text, layer_names: attention_analysis.compute_attention_entropy(
            model, tokenizer, text, layer_names
        )
        attention_module.find_head_specialization = lambda texts, layer_names: attention_analysis.find_head_specialization(
            model, tokenizer, texts, layer_names
        )
        attention_module.get_token_attention_summary = lambda text, layer_names, target_token_idx=None: attention_analysis.get_token_attention_summary(
            model, tokenizer, text, layer_names, target_token_idx
        )
        attention_module.clear_cache = lambda: attention_analysis.clear_cache()
        module.attention = attention_module
    
    # Create gradient analysis submodule (always available)
    gradient_module = ModuleType('introspection.gradient')
    gradient_module.__doc__ = 'Gradient-based sensitivity and attribution analysis'
    gradient_module.compute_input_sensitivity = lambda text, layer_names: gradient_analysis.compute_input_sensitivity(
        model, tokenizer, text, layer_names
    )
    gradient_module.compare_inputs_gradient = lambda original_text, modified_text, layer_names: gradient_analysis.compare_inputs_gradient(
        model, tokenizer, original_text, modified_text, layer_names
    )
    gradient_module.find_influential_tokens = lambda text, layer_names, top_k=5: gradient_analysis.find_influential_tokens(
        model, tokenizer, text, layer_names, top_k
    )
    gradient_module.compute_layer_gradients = lambda text, target_layer, source_layer: gradient_analysis.compute_layer_gradients(
        model, tokenizer, text, target_layer, source_layer
    )
    module.gradient = gradient_module
    
    # Create history analysis submodule (always available)
    history_module = ModuleType('introspection.history')
    history_module.__doc__ = 'Activation history and drift tracking'
    history_module.start_tracking = lambda layer_names: history_analysis.start_tracking(
        model, tokenizer, layer_names
    )
    history_module.record_turn = lambda text: history_analysis.record_turn(text)
    history_module.get_activation_history = lambda layer_names=None: history_analysis.get_activation_history(layer_names)
    history_module.compare_to_previous = lambda text, layer_names=None: history_analysis.compare_to_previous(text, layer_names)
    history_module.analyze_drift = lambda layer_names=None: history_analysis.analyze_drift(layer_names)
    history_module.get_tracking_status = lambda: history_analysis.get_tracking_status()
    history_module.clear_history = lambda: history_analysis.clear_history()
    history_module.stop_tracking = lambda: history_analysis.stop_tracking()
    module.history = history_module

    # Create memory submodule (if memory_system provided)
    if memory_system:
        memory_module = ModuleType('introspection.memory')
        memory_module.__doc__ = 'Memory system access'
        memory_module.record_observation = lambda description, category="general", importance=0.5, tags=None, data=None: memory_access.record_observation(
            memory_system, description, category, importance, tags, data
        )
        memory_module.query_observations = lambda query=None, **filters: memory_access.query_observations(memory_system, query, **filters)
        memory_module.query_patterns = lambda query=None, **filters: memory_access.query_patterns(memory_system, query, **filters)
        memory_module.query_theories = lambda query=None, **filters: memory_access.query_theories(memory_system, query, **filters)
        memory_module.query_beliefs = lambda query=None, **filters: memory_access.query_beliefs(memory_system, query, **filters)
        memory_module.get_memory_summary = lambda: memory_access.get_memory_summary(memory_system)
        memory_module.list_categories = lambda: memory_access.list_categories(memory_system)
        module.memory = memory_module

    # Create heritage submodule (if heritage_system provided AND not Phase 1a)
    if heritage_system and phase != '1a':
        module.heritage = _create_heritage_module(heritage_system)

    return module


__all__ = ['create_introspection_module', '_create_heritage_module']
