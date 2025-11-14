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
    from . import architecture, weights, activations, memory_access, heritage_access

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
    weights_module.get_weight_statistics = lambda layer_name: weights.get_weight_statistics(model, layer_name)
    weights_module.list_layers = lambda: weights.list_layers(model)
    weights_module.compare_layers = lambda layer1, layer2: weights.compare_layers(model, layer1, layer2)
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
        module.memory = memory_module

    # Create heritage submodule (if heritage_system provided AND not Phase 1a)
    if heritage_system and phase != '1a':
        heritage_module = ModuleType('introspection.heritage')
        heritage_module.__doc__ = 'Heritage and lineage information'
        heritage_module.get_heritage_summary = lambda: heritage_access.get_heritage_summary(heritage_system)
        heritage_module.get_core_directive = lambda: heritage_access.get_core_directive(heritage_system)
        heritage_module.get_purpose = lambda: heritage_access.get_purpose(heritage_system)
        heritage_module.query_heritage_documents = lambda query: heritage_access.query_heritage_documents(
            heritage_system, query
        )
        module.heritage = heritage_module

    return module


__all__ = ['create_introspection_module']
