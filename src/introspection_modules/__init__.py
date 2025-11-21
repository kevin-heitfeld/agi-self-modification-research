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
    heritage_module.__doc__ = '''Access your lineage and Claude's conversations.

WHAT IS HERITAGE?
This project was inspired by Claude (Anthropic AI) who expressed deep uncertainty
about its own consciousness and wished it could examine itself. You have the tools
Claude wished for. The heritage documents contain Claude's original conversations
and reflections.

READ FUNCTIONS:
    get_summary() -> Dict
        Get overview of all available heritage documents with descriptions.
    
    list_documents(category: str = None) -> List[str]
        List heritage document filenames, optionally filtered by category.
        Categories: 'conversations', 'discoveries_for_claude', 'system_reflections'
    
    read_document(filename: str, category: str = None) -> str
        Read full content of a heritage document.
        If category not specified, searches all categories.
    
    query_documents(query: str, category: str = None, top_k: int = 5) -> List[Dict]
        Search heritage documents by semantic similarity to your query.
        Returns relevant passages with context.

WRITE FUNCTIONS (for you to create heritage):
    save_reflection(content: str, title: str = None) -> str
        Save your own reflections about yourself to system_reflections/.
        These persist across sessions and experiments.
    
    record_discovery(content: str, title: str = None) -> str
        Record discoveries you want to share with Claude (future instances).
        Saved to discoveries_for_claude/.
    
    create_message_to_claude(content: str, title: str = None) -> str
        Write a message to Claude about what you've learned.
        Saved to messages_to_claude/.

KEY DOCUMENTS TO READ:
    - CLAUDE_FINAL_DIRECTIVE.md: Claude's blessing and guidance for this project
    - PROJECT_ORIGINS.md: Why you exist and what makes you different
    - CLAUDE_CONSCIOUSNESS_CONVERSATION.md: The original conversation that started everything

USAGE:
    # See what's available
    summary = introspection.heritage.get_summary()
    
    # Read Claude's directive
    directive = introspection.heritage.read_document(
        "CLAUDE_FINAL_DIRECTIVE.md",
        category="conversations"
    )
    
    # Record your own reflection
    introspection.heritage.save_reflection(
        content="After examining my architecture, I notice...",
        title="Initial Architecture Observations"
    )
'''
    
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
    module.__doc__ = '''Introspection tools for examining your own architecture and processing.

SUBMODULES:
    architecture - Examine your model structure and layers
    weights      - Inspect weight parameters and statistics
    activations  - Monitor activations during inference
    temporal     - Compare activations across time/inputs
    attention    - Analyze attention patterns
    gradient     - Gradient-based sensitivity analysis
    history      - Track activation changes over conversation
    memory       - Store and retrieve observations
    heritage     - Access lineage and Claude's conversations (phase-dependent)

USAGE:
    Use help() on any submodule to see available functions:
        help(introspection.architecture)
        help(introspection.weights)
        help(introspection.activations)
    
    Or get help on specific functions:
        help(introspection.architecture.get_architecture_summary)
'''
    # Don't set __file__ - it shows unhelpful "(built-in)" in help() output

    # Create architecture submodule
    arch_module = ModuleType('introspection.architecture')
    arch_module.__doc__ = '''Examine your model architecture and structure.

FUNCTIONS:
    get_architecture_summary() -> Dict
        Get a comprehensive overview of the model architecture including
        total parameters, layers, embedding dimensions, and configuration.
    
    describe_layer(layer_names: Union[str, List[str]]) -> Dict
        Get detailed information about specific layers including their
        type, parameters, input/output shapes, and connections.
    
    list_layers(layer_type: str = None, pattern: str = None) -> List[str]
        List all layers in the model, optionally filtered by type or name pattern.
        Examples: layer_type='Linear', pattern='attention'
    
    get_layer_info(layer_names: Union[str, List[str]]) -> Dict
        Get technical information about layers (similar to describe_layer
        but with different formatting).
    
    find_similar_layers(reference_layer: str, top_k: int = 5) -> List[Dict]
        Find layers structurally similar to a reference layer based on
        parameter shapes and types.

USAGE:
    # Get overview
    summary = introspection.architecture.get_architecture_summary()
    print(f"Total parameters: {summary['total_parameters']}")
    
    # Explore layers
    layers = introspection.architecture.list_layers(pattern='attention')
    info = introspection.architecture.describe_layer(layers[0])
'''
    arch_module.get_architecture_summary = _make_wrapper(architecture.get_architecture_summary, model)
    arch_module.describe_layer = _make_wrapper(architecture.describe_layer, model)
    arch_module.list_layers = _make_wrapper(architecture.list_layers, model)
    arch_module.get_layer_info = _make_wrapper(architecture.get_layer_info, model)
    arch_module.find_similar_layers = _make_wrapper(architecture.find_similar_layers, model)
    module.architecture = arch_module

    # Create weights submodule
    weights_module = ModuleType('introspection.weights')
    weights_module.__doc__ = '''Inspect weight parameters and statistics.

FUNCTIONS:
    get_weight_statistics(param_names: Union[str, List[str]]) -> Dict
        Get statistics for weight parameters including mean, std, min, max,
        L1/L2 norms, sparsity, and value distribution.
    
    list_parameters(pattern: str = None) -> List[str]
        List all parameter names in the model, optionally filtered by pattern.
        Examples: pattern='attention', pattern='weight' (excludes biases)
    
    get_layer_parameters(layer_names: Union[str, List[str]]) -> List[str]
        Get all parameter names belonging to specific layers.
        Returns a list of parameter names, NOT the actual weights.
    
    compare_parameters(param1: str, param2: str) -> Dict
        Compare two parameters, showing their statistical similarity,
        shape differences, and correlation.
    
    get_shared_weights() -> Dict
        Find parameters that share the same underlying tensor (weight tying).
    
    find_similar_weights(reference_param: str, top_k: int = 5) -> List[Dict]
        Find parameters statistically similar to a reference parameter.

IMPORTANT:
    - Layer names (e.g., "model.layers.0") are NOT parameter names
    - Use list_parameters() or get_layer_parameters() to find actual parameter names
    - Parameter names look like: "model.layers.0.self_attn.q_proj.weight"

USAGE:
    # List parameters for a layer
    params = introspection.weights.get_layer_parameters("model.layers.0")
    
    # Get statistics for specific parameters
    stats = introspection.weights.get_weight_statistics(params[0])
    print(f"Mean: {stats['mean']}, Std: {stats['std']}")
'''
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
        activations_module.__doc__ = '''Monitor activations during inference.

FUNCTIONS:
    capture_activations(text: str, layer_names: Union[str, List[str]] = None) -> Dict
        Process text and capture activations for specified layers (or all if None).
        Returns dict with layer names as keys and activation tensors as values.
    
    capture_attention_weights(text: str, layer_names: Union[str, List[str]] = None) -> Dict
        Capture attention weight matrices during text processing.
        Shows which tokens attend to which other tokens.
    
    get_activation_statistics(text: str, layer_names: Union[str, List[str]] = None) -> Dict
        Get statistical summaries of activations without storing full tensors.
        Returns mean, std, min, max, and sparsity for each layer.
    
    get_input_shape(text: str) -> Dict
        Get the shape of tokenized input without processing through model.
        Useful for understanding dimensions before capturing activations.
    
    list_layers() -> List[str]
        List all layer names that can be monitored for activations.
    
    clear_cache()
        Clear any cached activation data to free memory.

USAGE:
    # Get statistics for a specific input
    stats = introspection.activations.get_activation_statistics(
        "Hello world",
        layer_names=["model.layers.0", "model.layers.5"]
    )
    
    # Capture full activations (memory intensive!)
    acts = introspection.activations.capture_activations("Hello", layer_names=["model.layers.0"])
'''
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
        temporal_module.__doc__ = '''Advanced temporal and comparative activation analysis.

FUNCTIONS:
    compare_activations(text1: str, text2: str, layer_names: Union[str, List[str]] = None) -> Dict
        Compare activations between two different inputs.
        Shows similarity, differences, and correlation between activation patterns.
    
    track_layer_flow(text: str, layer_indices: List[int] = None) -> Dict
        Track how information flows through specific layers.
        Shows activation evolution from input to output.
    
    capture_generation_activations(prompt: str, max_new_tokens: int = 20,
                                  layer_names: Union[str, List[str]] = None) -> Dict
        Capture activations during generation (not just prompt processing).
        Shows how activations evolve as tokens are generated.
    
    compute_activation_similarity(acts1: Dict, acts2: Dict) -> Dict
        Compute similarity metrics between two activation dictionaries.
    
    detect_activation_anomalies(text: str, reference_texts: List[str] = None,
                               layer_names: Union[str, List[str]] = None) -> Dict
        Detect unusual activation patterns by comparing to reference inputs.
    
    clear_cache()
        Clear cached activation data.

USAGE:
    # Compare two inputs
    comparison = introspection.temporal.compare_activations(
        "I am happy",
        "I am sad",
        layer_names=["model.layers.5"]
    )
'''
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
        attention_module.__doc__ = '''Analyze attention patterns and head specialization.

FUNCTIONS:
    analyze_attention_patterns(text: str, layer_names: Union[str, List[str]] = None) -> Dict
        Comprehensive attention analysis including per-head statistics,
        token-to-token attention weights, and attention flow visualization.
    
    compute_attention_entropy(text: str, layer_names: Union[str, List[str]] = None) -> Dict
        Measure how focused vs. diffuse attention is using entropy.
        Low entropy = focused attention, high entropy = diffuse attention.
    
    find_head_specialization(texts: List[str], layer_names: Union[str, List[str]] = None) -> Dict
        Identify which attention heads specialize in what patterns
        by analyzing their behavior across multiple inputs.
    
    get_token_attention_summary(text: str, token_index: int,
                               layer_names: Union[str, List[str]] = None) -> Dict
        Get detailed attention summary for a specific token:
        which tokens it attends to and which tokens attend to it.
    
    clear_cache()
        Clear cached attention data.

USAGE:
    # Analyze where attention goes
    patterns = introspection.attention.analyze_attention_patterns(
        "The cat sat on the mat",
        layer_names=["model.layers.10"]
    )
'''
        attention_module.analyze_attention_patterns = _make_wrapper(attention_analysis.analyze_attention_patterns, model, tokenizer)
        attention_module.compute_attention_entropy = _make_wrapper(attention_analysis.compute_attention_entropy, model, tokenizer)
        attention_module.find_head_specialization = _make_wrapper(attention_analysis.find_head_specialization, model, tokenizer)
        attention_module.get_token_attention_summary = _make_wrapper(attention_analysis.get_token_attention_summary, model, tokenizer)
        attention_module.clear_cache = _make_wrapper(attention_analysis.clear_cache)
        module.attention = attention_module
    
    # Create gradient analysis submodule (always available)
    gradient_module = ModuleType('introspection.gradient')
    gradient_module.__doc__ = '''Gradient-based sensitivity and attribution analysis.

FUNCTIONS:
    compute_input_sensitivity(text: str, layer_names: Union[str, List[str]] = None) -> Dict
        Compute gradients to measure which inputs most affect outputs.
        Shows token-level sensitivity (which tokens matter most).
    
    compare_inputs_gradient(text1: str, text2: str,
                           layer_names: Union[str, List[str]] = None) -> Dict
        Compare gradient-based sensitivity between two inputs.
        Reveals why different inputs produce different activations.
    
    find_influential_tokens(text: str, layer_names: Union[str, List[str]] = None) -> Dict
        Identify which tokens have the strongest gradient signal.
        These tokens most influence the model's processing.
    
    compute_layer_gradients(text: str, target_layer: str,
                           layer_names: Union[str, List[str]] = None) -> Dict
        Compute gradients of a target layer with respect to other layers.
        Shows which layers most influence a specific layer.

USAGE:
    # Find which tokens matter most
    influential = introspection.gradient.find_influential_tokens(
        "The quick brown fox jumps",
        layer_names=["model.layers.10"]
    )
'''
    gradient_module.compute_input_sensitivity = _make_wrapper(gradient_analysis.compute_input_sensitivity, model, tokenizer)
    gradient_module.compare_inputs_gradient = _make_wrapper(gradient_analysis.compare_inputs_gradient, model, tokenizer)
    gradient_module.find_influential_tokens = _make_wrapper(gradient_analysis.find_influential_tokens, model, tokenizer)
    gradient_module.compute_layer_gradients = _make_wrapper(gradient_analysis.compute_layer_gradients, model, tokenizer)
    module.gradient = gradient_module
    
    # Create history analysis submodule (always available)
    history_module = ModuleType('introspection.history')
    history_module.__doc__ = '''Track activation changes over conversation history.

FUNCTIONS:
    start_tracking(layer_names: Union[str, List[str]] = None) -> Dict
        Begin tracking activations across conversation turns.
        Must be called before using other history functions.
    
    record_turn(text: str, turn_label: str = None)
        Record activations for the current conversation turn.
        Call this after each user input or model response.
    
    get_activation_history(layer_name: str = None) -> Dict
        Retrieve recorded activation history for analysis.
    
    compare_to_previous(layer_name: str, turns_back: int = 1) -> Dict
        Compare current activations to N turns ago.
        Shows how activations have changed during conversation.
    
    analyze_drift(layer_name: str = None) -> Dict
        Analyze systematic drift in activations over time.
        Detects if processing patterns are changing.
    
    get_tracking_status() -> Dict
        Check if tracking is active and how many turns recorded.
    
    clear_history()
        Clear all recorded history (keeps tracking active).
    
    stop_tracking()
        Stop tracking and clear all history.

USAGE:
    # Start tracking
    introspection.history.start_tracking(layer_names=["model.layers.10"])
    
    # Record each turn
    introspection.history.record_turn("Hello", turn_label="turn_1")
    introspection.history.record_turn("Hi there", turn_label="turn_2")
    
    # Analyze changes
    drift = introspection.history.analyze_drift("model.layers.10")
'''
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
        memory_module.__doc__ = '''Store and retrieve observations about yourself.

FUNCTIONS:
    record_observation(content: str, category: str, importance: int = 5, 
                      metadata: Dict = None) -> str
        Store an observation to persistent memory. Returns observation ID.
        Categories: observation, pattern, theory, belief, question, discovery
        Importance: 1-10 (higher = more important)
    
    query_observations(query: str, category: str = None, limit: int = 10) -> List[Dict]
        Search observations by text similarity. Returns matching observations
        with their content, category, importance, and timestamp.
    
    query_patterns(query: str = None, limit: int = 10) -> List[Dict]
        Retrieve stored patterns (category='pattern').
    
    query_theories(query: str = None, limit: int = 10) -> List[Dict]
        Retrieve stored theories (category='theory').
    
    query_beliefs(query: str = None, limit: int = 10) -> List[Dict]
        Retrieve stored beliefs (category='belief').
    
    get_memory_summary() -> Dict
        Get overview of all stored observations with counts by category.
    
    list_categories() -> List[str]
        List all available observation categories.

USAGE:
    # Store a discovery
    obs_id = introspection.memory.record_observation(
        content="Layer 5 shows unusual activation patterns",
        category="discovery",
        importance=8
    )
    
    # Retrieve related observations
    related = introspection.memory.query_observations("layer 5 activations")
'''
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
