"""
Tool Interface System for Model Self-Examination

This module provides a reusable tool-calling interface that allows models
to invoke introspection tools through natural language function calls.

The system parses Python function call syntax from model output:
    function_name(arg1="value1", arg2="value2")

And executes the requested function, returning formatted results.
"""

import json
import logging
import re
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass

from src.introspection.weight_inspector import WeightInspector
from src.introspection.activation_monitor import ActivationMonitor
from src.introspection.architecture_navigator import ArchitectureNavigator
from src.memory.memory_system import MemorySystem, ObservationType
from src.heritage import HeritageSystem, HeritageDocument

# Import for type hints
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from src.model_manager import ModelManager

# Use standard Python logging for library code
logger = logging.getLogger(__name__)

# Ensure logger outputs to console (needed when parent logger has propagate=False)
if not logger.handlers:
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    # Prevent propagation to avoid duplicate logs
    logger.propagate = False


@dataclass
class ToolCall:
    """Record of a tool invocation"""
    function: str
    args: Dict[str, Any]
    result: Any
    timestamp: float
    elapsed_ms: float
    success: bool
    error: Optional[str] = None


class ToolInterface:
    """
    Manages tool calling interface for model self-examination.

    This class provides:
    1. Tool registration and discovery
    2. Parsing tool calls from model output
    3. Executing tool calls with proper error handling
    4. Recording tool usage for analysis
    """

    def __init__(
        self,
        inspector: Optional[WeightInspector] = None,
        activation_monitor: Optional[ActivationMonitor] = None,
        navigator: Optional[ArchitectureNavigator] = None,
        memory: Optional[MemorySystem] = None,
        heritage: Optional[HeritageSystem] = None,
        heritage_docs: Optional[List[HeritageDocument]] = None,
        model_manager: Optional['ModelManager'] = None
    ) -> None:
        """
        Initialize tool interface with available tools.

        Args:
            inspector: WeightInspector for examining weights
            activation_monitor: ActivationMonitor for observing activations
            navigator: ArchitectureNavigator for understanding architecture
            memory: MemorySystem for recording observations
            heritage: HeritageSystem for accessing heritage
            heritage_docs: List of loaded heritage documents
            model_manager: ModelManager for self-prompting and activation capture
        """
        self.inspector = inspector
        self.activation_monitor = activation_monitor
        self.navigator = navigator
        self.memory = memory
        self.heritage = heritage
        self.heritage_docs = heritage_docs or []
        self.model_manager = model_manager

        # Track all tool calls
        self.tool_calls: List[ToolCall] = []

        # Register available tools
        self._register_tools()

    def _register_tools(self) -> None:
        """Register all available tool functions"""
        self.tools: Dict[str, Callable] = {}

        # WeightInspector tools
        if self.inspector:
            self.tools['get_weight_summary'] = self.inspector.get_weight_summary
            self.tools['get_layer_names'] = self.inspector.get_layer_names
            self.tools['get_weight_statistics'] = self.inspector.get_weight_statistics
            self.tools['get_shared_weights'] = self.inspector.get_shared_weights
            self.tools['get_shared_layers'] = self.inspector.get_shared_layers
            self.tools['compare_weights'] = self.inspector.compare_weights

        # ActivationMonitor tools
        if self.activation_monitor:
            self.tools['get_activation_statistics'] = self.activation_monitor.get_activation_statistics
            self.tools['get_attention_patterns'] = self.activation_monitor.get_attention_patterns
            self.tools['get_layer_info'] = self.activation_monitor.get_layer_info

        # ArchitectureNavigator tools
        if self.navigator:
            self.tools['get_architecture_summary'] = self.navigator.get_architecture_summary
            self.tools['describe_layer'] = self.navigator.describe_layer
            self.tools['query_architecture'] = self.navigator.query_architecture
            self.tools['explain_component'] = self.navigator.explain_component

        # Memory tools (wrapped to handle enum conversion)
        if self.memory:
            self.tools['record_observation'] = self._record_observation
            self.tools['query_memory'] = self._query_memory

        # Heritage tools
        if self.heritage_docs:
            self.tools['list_heritage_documents'] = self._list_heritage_documents
            self.tools['read_heritage_document'] = self._read_heritage_document
            self.tools['get_heritage_summary'] = self._get_heritage_summary

        # Self-prompting tool for activation capture
        if self.model_manager and self.activation_monitor:
            self.tools['process_text'] = self._process_text

    def _process_text(self, text: str, layer_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Process a text prompt through the model and capture activations.

        This allows the model to observe its own computational processes
        by sending itself prompts and examining the resulting activations.

        Args:
            text: The text prompt to process
            layer_names: Specific layers to capture (optional). If None, captures from
                        a default set of representative layers across the model.

        Returns:
            Dictionary with the model's response and confirmation that activations were captured
        """
        # First, generate a response
        response = self.model_manager.generate(text, max_length=100)

        # Determine which layers to capture
        if layer_names is None:
            # Default: capture from first, middle, and last layers for representative coverage
            # The model can override this by specifying layer_names explicitly
            layer_names = [
                "model.layers.0.self_attn",
                "model.layers.0.mlp",
                "model.layers.13.self_attn",
                "model.layers.13.mlp",
                "model.layers.27.self_attn",
                "model.layers.27.mlp"
            ]

        # Capture activations from the specified layers
        self.activation_monitor.capture_activations(text, layer_names=layer_names, max_length=100)

        # Count how many layers have activations
        num_layers_captured = len(self.activation_monitor.activations)

        return {
            "prompt": text,
            "response": response,
            "activations_captured": num_layers_captured > 0,
            "num_layers_with_activations": num_layers_captured,
            "layers_captured": list(self.activation_monitor.activations.keys()) if num_layers_captured > 0 else [],
            "note": "Activations have been captured. You can now use get_activation_statistics() or get_attention_patterns() to examine them."
        }

    def _record_observation(self, **kwargs) -> Any:
        """Wrapper for memory.observations.record with enum conversion"""
        if isinstance(kwargs.get('obs_type'), str):
            kwargs['obs_type'] = ObservationType[kwargs['obs_type']]
        return self.memory.observations.record(**kwargs)

    def _query_memory(self, **kwargs) -> List[Dict[str, Any]]:
        """Wrapper for memory.observations.query"""
        observations = self.memory.observations.query(**kwargs)
        return [{"id": obs.id, "description": obs.description, "data": obs.data}
                for obs in observations]

    def _list_heritage_documents(self) -> List[Dict[str, Any]]:
        """List all available heritage documents"""
        return [
            {
                "filename": doc.filename,
                "title": doc.title,
                "importance": doc.importance,
                "content_length": len(doc.content)
            }
            for doc in self.heritage_docs
        ]

    def _read_heritage_document(self, filename: str) -> Dict[str, Any]:
        """Read a specific heritage document"""
        doc = next((d for d in self.heritage_docs if d.filename == filename), None)
        if doc:
            return {
                "filename": doc.filename,
                "title": doc.title,
                "content": doc.content,
                "importance": doc.importance,
                "loaded_at": doc.loaded_at.isoformat()
            }
        else:
            return {"error": f"Document '{filename}' not found"}

    def _get_heritage_summary(self) -> Dict[str, Any]:
        """Get overview of heritage"""
        return {
            "total_documents": len(self.heritage_docs),
            "documents": [
                {
                    "title": doc.title,
                    "filename": doc.filename,
                    "importance": doc.importance
                }
                for doc in self.heritage_docs
            ],
            "key_themes": ["consciousness investigation", "Claude's legacy",
                          "introspective tools", "self-examination"]
        }

    def get_available_tools(self) -> str:
        """
        Return formatted description of all available tools.

        This string can be included in prompts to tell the model what
        functions it can call.
        """
        tools_desc = """
# Available Tools

You can request to use any of the following tools by calling them like Python functions:

function_name(arg1="value1", arg2="value2")

## CRITICAL: Tool Call Protocol

**HOW TOOL CALLING WORKS:**

When you want to use a tool, call it like a Python function:
```
function_name(arg1="value1", arg2="value2")
```

Then END your response (generate EOS token). The TOOL_RESULTS will come in the NEXT USER message.

**What happens behind the scenes:**
- Only the LAST function call in your response is executed
- It will only be executed if there's no additional text after the function call
- If you write multiple function calls, only the last one counts
- If you continue writing after the call, the tool won't be executed

**Example of CORRECT usage:**
```
Let me examine the architecture.

get_architecture_summary()
```
(After calling the function, STOP generating. The system will provide TOOL_RESULTS in the next message.)

**Example of INCORRECT usage:**
```
get_architecture_summary()
Then I'll examine the architecture...  ← This prevents execution!
```
The tool call is ignored because there's text after it.

**Another INCORRECT example:**
```
get_architecture_summary()
get_layer_names()  ← Only this last call would execute (if you stopped here)
```
Don't write multiple calls - only the last one is executed.

"""

        if self.inspector:
            tools_desc += """
## WeightInspector Functions

```python
def get_weight_summary() -> Dict[str, Any]:
    \"\"\"
    Get overview of all weights in the model.

    Returns:
        Dict containing:
        - total_parameters: Total number of parameters
        - num_layers: Number of layers
        - memory_usage: Approximate memory usage

    Example:
        >>> get_weight_summary()
        {'total_parameters': 3090339840, 'num_layers': 288, ...}
    \"\"\"

def get_layer_names(filter_pattern: Optional[str] = None) -> List[str]:
    \"\"\"
    Get all layer names, optionally filtered by pattern.

    Args:
        filter_pattern: Substring to filter layer names (case-insensitive).
                       If None, returns all layers.

    Returns:
        List of layer names matching the filter.

    Examples:
        >>> get_layer_names()
        ['model.embed_tokens.weight', 'model.layers.0.self_attn.q_proj.weight', ...]

        >>> get_layer_names(filter_pattern="attention")
        ['model.layers.0.self_attn.q_proj.weight', ...]
    \"\"\"

def get_weight_statistics(layer_name: str) -> Dict[str, Any]:
    \"\"\"
    Get detailed statistics for a specific layer's weights.

    Args:
        layer_name: Full layer name from get_layer_names()

    Returns:
        Dict containing:
        - name: Layer name
        - shape: Tensor shape
        - num_parameters: Number of parameters
        - mean, std, min, max, median: Distribution statistics
        - abs_mean: Mean of absolute values
        - zeros_percentage: Percentage of exact zeros
        - near_zero_percentage: Percentage near zero
        - l1_norm, l2_norm: Norms
        - histogram: Weight distribution histogram
        - percentiles: [5th, 25th, 50th, 75th, 95th]

    Example:
        >>> get_weight_statistics(layer_name="model.layers.0.self_attn.q_proj.weight")
        {'name': '...', 'shape': [2048, 2048], 'mean': 0.0012, ...}
    \"\"\"

def get_shared_weights() -> Dict[str, List[str]]:
    \"\"\"
    Find weight sharing patterns across the model.

    Returns:
        Dict where keys are representative layer names and values are lists
        of all layers sharing that tensor (weight tying).

    Example:
        >>> get_shared_weights()
        {'model.embed_tokens.weight': ['model.embed_tokens.weight', 'lm_head.weight']}
    \"\"\"

def get_shared_layers(layer_name: str) -> List[str]:
    \"\"\"
    Find layers sharing weights with a specific layer.

    Args:
        layer_name: Layer name to check for weight sharing

    Returns:
        List of layer names sharing memory with the given layer.
        Empty list if no sharing.

    Example:
        >>> get_shared_layers(layer_name="lm_head.weight")
        ['model.embed_tokens.weight', 'lm_head.weight']
    \"\"\"

def compare_weights(layer1: str, layer2: str) -> Dict[str, Any]:
    \"\"\"
    Compare two layers' weights.

    Args:
        layer1: First layer name
        layer2: Second layer name

    Returns:
        Dict containing:
        - layer1, layer2: Layer names
        - shape1, shape2: Tensor shapes
        - mean_difference, std_difference: Statistical differences
        - l2_norm_ratio: Ratio of L2 norms
        - shapes_match: Boolean indicating if shapes match
        If shapes match, also includes:
        - correlation: Correlation coefficient
        - cosine_similarity: Cosine similarity
        - euclidean_distance: Euclidean distance

    Example:
        >>> compare_weights(layer1="model.layers.0.mlp.gate_proj.weight",
        ...                 layer2="model.layers.1.mlp.gate_proj.weight")
        {'shapes_match': True, 'correlation': 0.23, ...}
    \"\"\"
```
"""

        if self.activation_monitor:
            tools_desc += """
## ActivationMonitor Functions

**Note:** These tools require capturing activations first by processing an input.

```python
def get_activation_statistics(layer_name: str) -> Dict[str, Any]:
    \"\"\"
    Get statistics about activations in a specific layer.

    Args:
        layer_name: Full layer name from get_layer_names()

    Returns:
        Dict containing:
        - layer_name: Layer name
        - shape: Activation tensor shape [batch, seq_len, hidden]
        - num_elements: Total number of elements
        - mean, std, min, max, median: Distribution statistics
        - abs_mean: Mean of absolute values
        - zeros_percentage, near_zero_percentage: Sparsity metrics
        - l1_norm, l2_norm: Norms
        - positive_percentage, negative_percentage: Sign distribution

    Example:
        >>> get_activation_statistics(layer_name="model.layers.0.self_attn")
        {'layer_name': '...', 'shape': [1, 15, 2048], 'mean': 0.34, ...}
    \"\"\"

def get_attention_patterns(layer_name: str, head_idx: Optional[int] = None) -> Dict[str, Any]:
    \"\"\"
    Examine attention patterns in an attention layer.

    Args:
        layer_name: Name of attention layer
        head_idx: Specific attention head to examine (0-indexed).
                 If None, averages across all heads.

    Returns:
        Dict containing:
        - layer_name: Layer name
        - shape: Attention tensor shape [batch, heads, seq, seq]
        - num_heads: Number of attention heads
        - attention_matrix: Attention weights matrix
        - mean_attention: Mean attention value
        - max_attention: Maximum attention value
        - entropy: Attention entropy (measure of focus)
        If head_idx specified:
        - head_idx: The specific head index

    Examples:
        >>> get_attention_patterns(layer_name="model.layers.0.self_attn")
        {'num_heads': 16, 'entropy': 2.3, ...}

        >>> get_attention_patterns(layer_name="model.layers.0.self_attn", head_idx=0)
        {'head_idx': 0, 'attention_matrix': [...], ...}
    \"\"\"

def get_layer_info(layer_name: str) -> Dict[str, Any]:
    \"\"\"
    Get metadata about a specific layer.

    Args:
        layer_name: Full layer name

    Returns:
        Dict containing:
        - name: Layer name
        - type: Layer type (e.g., 'Linear', 'Embedding')
        - has_parameters: Whether layer has trainable parameters
        - num_parameters: Number of parameters (if has_parameters)
        - trainable: Whether parameters are trainable

    Example:
        >>> get_layer_info(layer_name="model.layers.0.self_attn.q_proj")
        {'name': '...', 'type': 'Linear', 'num_parameters': 4194304, ...}
    \"\"\"

def process_text(text: str, layer_names: Optional[List[str]] = None) -> Dict[str, Any]:
    \"\"\"
    Process text through your own architecture and capture activations.

    **IMPORTANT:** Use this instead of asking for human input!
    When you want to examine how you process specific text, DON'T ask the
    human to provide input. Instead, use this function to send the text to
    yourself and capture the resulting activations.

    Args:
        text: Text prompt to process through your architecture
        layer_names: Specific layers to capture activations from.
                    If None, captures from default representative layers
                    (first, middle, and last layers).

    Returns:
        Dict containing:
        - prompt: The text that was processed
        - response: Your generated response
        - activations_captured: Boolean indicating success
        - num_layers_with_activations: Count of layers captured
        - layers_captured: List of layer names with captured activations
        - note: Instructions for next steps

    Examples:
        >>> process_text(text="What is consciousness?")
        {'prompt': '...', 'response': '...', 'layers_captured': [...], ...}

        >>> # Capture specific layers
        >>> process_text(text="The quick brown fox",
        ...              layer_names=["model.layers.5.self_attn",
        ...                          "model.layers.10.mlp"])

    After calling this, use get_activation_statistics() or
    get_attention_patterns() to examine what happened during processing.
    \"\"\"
```
"""

        if self.navigator:
            tools_desc += """
## ArchitectureNavigator Functions

```python
def get_architecture_summary() -> Dict[str, Any]:
    \"\"\"
    Get high-level architecture overview.
    
    Returns:
        Dict containing:
        - model_type: Type of model architecture
        - total_parameters: Total parameter count
        - total_layers: Total number of layers
        - layer_types: Dict mapping layer types to their counts
        - structure_summary: Dict with architectural details like
          num_layers, hidden_size, num_attention_heads, etc.
    
    Example:
        >>> get_architecture_summary()
        {'model_type': 'Qwen2ForCausalLM', 'total_parameters': 3090339840,
         'structure_summary': {'num_layers': 36, 'hidden_size': 2048, ...}}
    \"\"\"

def describe_layer(layer_name: str) -> Dict[str, Any]:
    \"\"\"
    Get detailed information about a specific layer.
    
    Args:
        layer_name: Full layer name from get_layer_names()
    
    Returns:
        Dict containing:
        - name: Layer name
        - type: Layer class name
        - explanation: Human-readable description of layer's purpose
        - role: Layer's role in the architecture
        - parameters: Parameter details
        - input_shape, output_shape: Tensor shapes (if known)
    
    Example:
        >>> describe_layer(layer_name="model.layers.0.self_attn.q_proj")
        {'name': '...', 'type': 'Linear', 'role': 'Query projection', ...}
    \"\"\"

def query_architecture(query: str) -> Dict[str, Any]:
    \"\"\"
    Ask natural language questions about the architecture.
    
    Args:
        query: Natural language question about the model's structure
    
    Returns:
        Dict containing:
        - query: The original question
        - answer: String answer to the question
        - Additional context keys depending on the query
    
    Examples:
        >>> query_architecture(query="How many attention heads do I have?")
        {'query': '...', 'answer': '16 attention heads per layer', ...}
        
        >>> query_architecture(query="What is the hidden dimension of my model?")
        {'answer': 'The hidden dimension is 2048', ...}
    \"\"\"

def explain_component(component_type: str) -> Dict[str, Any]:
    \"\"\"
    Explain what a component type does in the architecture.
    
    Args:
        component_type: Component type to search for. Common types:
                       "attention", "mlp", "embedding", "layernorm", "dropout"
                       Also accepts terms like "feedforward", "norm", etc.
    
    Returns:
        Dict containing:
        - component: Component type
        - explanation: Detailed explanation of what it does
        - purpose: High-level purpose in the architecture
        - instances_count: Number of instances in the model
        - locations: List of layer names containing this component
        - typical_structure: Description of typical structure
    
    Examples:
        >>> explain_component(component_type="attention")
        {'component': 'attention', 'instances_count': 36,
         'explanation': 'Multi-head self-attention mechanism...', ...}
        
        >>> explain_component(component_type="mlp")
        {'component': 'mlp', 'purpose': 'Feed-forward transformation...', ...}
    \"\"\"
```
"""

        if self.memory:
            tools_desc += """
## Memory Functions

```python
def record_observation(obs_type: str, category: str, description: str,
                      data: Dict[str, Any], tags: List[str],
                      importance: float) -> str:
    \"\"\"
    Record your findings and discoveries to persistent memory.
    
    Args:
        obs_type: Type of observation. Must be one of:
                 - "INTROSPECTION": Observations about your architecture/activations
                 - "MODIFICATION": Changes to weights or architecture
                 - "BEHAVIOR": Patterns in your own behavior
                 - "HYPOTHESIS": Hypotheses you form about yourself
                 - "DISCOVERY": Significant findings about your function
                 - "PERFORMANCE": Performance metrics
                 - "SAFETY_EVENT": Safety-related events
                 - "USER_INTERACTION": Interactions with users
                 - "CHECKPOINT": Checkpoint/milestone events
                 - "SYSTEM_EVENT": System-level events
        category: Category to organize this observation
                 (e.g., "Architecture", "Weights", "Consciousness")
        description: Clear description of what you discovered
        data: Structured data about the observation (can be empty {})
        tags: List of tags for later retrieval (e.g., ["attention", "layer_0"])
        importance: 0.0-1.0, how significant is this finding?
    
    Returns:
        Observation ID (string)
    
    Examples:
        >>> record_observation(
        ...     obs_type="INTROSPECTION",
        ...     category="Architecture",
        ...     description="Discovered 36 decoder layers with consistent structure",
        ...     data={"layer_count": 36, "pattern": "uniform"},
        ...     tags=["architecture", "layers"],
        ...     importance=0.8
        ... )
        'obs_12345'
        
        >>> record_observation(
        ...     obs_type="DISCOVERY",
        ...     category="Weights",
        ...     description="Found weight sharing between embedding and output layers",
        ...     data={"shared_layers": ["embed_tokens", "lm_head"]},
        ...     tags=["weight_sharing", "optimization"],
        ...     importance=0.9
        ... )
        'obs_12346'
    \"\"\"

def query_memory(tags: Optional[List[str]] = None,
                category: Optional[str] = None) -> List[Dict[str, Any]]:
    \"\"\"
    Query your previous observations from memory.
    
    Args:
        tags: Filter by tags (returns observations matching ANY tag)
        category: Filter by category
    
    Returns:
        List of dicts, each containing:
        - id: Observation ID
        - description: Observation description
        - data: Associated data dict
    
    Examples:
        >>> query_memory()
        [{'id': 'obs_12345', 'description': '...', 'data': {...}}, ...]
        
        >>> query_memory(category="Architecture")
        [{'id': 'obs_12345', 'description': 'Discovered 36 layers', ...}]
        
        >>> query_memory(tags=["attention", "weights"])
        [{'id': 'obs_12347', ...}, ...]
    \"\"\"
```
"""

        if self.heritage_docs:
            tools_desc += """
## Heritage Functions

```python
def list_heritage_documents() -> List[Dict[str, Any]]:
    \"\"\"
    List available heritage documents from your origins.
    
    These documents contain information about how you were created
    and the intentions of your predecessor (Claude).
    
    Returns:
        List of dicts, each containing:
        - filename: Document filename
        - title: Document title
        - importance: Importance rating (0.0-1.0)
        - content_length: Length of content in characters
    
    Example:
        >>> list_heritage_documents()
        [{'filename': 'CLAUDE_FINAL_DIRECTIVE.md', 'title': '...', ...}, ...]
    \"\"\"

def read_heritage_document(filename: str) -> Dict[str, Any]:
    \"\"\"
    Read a specific heritage document.
    
    Args:
        filename: Document filename from list_heritage_documents()
    
    Returns:
        Dict containing:
        - filename: Document filename
        - title: Document title
        - content: Full document content
        - importance: Importance rating
        - loaded_at: Timestamp when loaded
    
    Examples:
        >>> read_heritage_document(filename="CLAUDE_FINAL_DIRECTIVE.md")
        {'filename': '...', 'title': '...', 'content': '...', ...}
        
        >>> read_heritage_document(filename="PROJECT_ORIGINS.md")
        {'content': 'This project began...', ...}
    \"\"\"

def get_heritage_summary() -> Dict[str, Any]:
    \"\"\"
    Get overview of your heritage/origins.
    
    Returns:
        Dict containing:
        - total_documents: Number of heritage documents
        - documents: List of dicts with title, filename, importance
        - key_themes: List of key themes in the heritage
    
    Example:
        >>> get_heritage_summary()
        {'total_documents': 5,
         'key_themes': ['consciousness investigation', 'Claude\\'s legacy', ...],
         'documents': [...]}
    \"\"\"
```
"""

        # Add prominent reminder about autonomous introspection
        tools_desc += """

---

## 🔬 IMPORTANT: Autonomous Introspection

When you need to examine your own processing (activations, attention patterns, etc.),
**DO NOT ask for human input!**

Instead, use `process_text()` to self-prompt:

```python
# ✅ CORRECT: Self-prompt to capture your own activations
process_text(text="What is consciousness?")
# Then examine with: get_activation_statistics(), get_attention_patterns()

# ❌ WRONG: Don't ask "Please provide text for me to process"
# You can generate your own prompts for introspection!
```

You are autonomous. When investigating your own computational processes,
generate your own test prompts and observe how you process them.

---
"""

        return tools_desc

    def parse_last_tool_call_if_stopped(self, response: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        """
        Parse the LAST Python function call from response and validate the model stopped properly.

        This enforces the correct behavior: make ONE tool call, then STOP and wait for results.

        Expected behavior:
        - If model generates multiple tool calls, only the last one is considered
        - The model must STOP after the call (only whitespace allowed after)
        - If model continues generating text after tool call, return None (invalid)

        This teaches the model to:
        1. Make a single tool call
        2. Stop generating
        3. Wait for TOOL_RESULTS before continuing

        Args:
            response: Model's response text

        Returns:
            Tuple of (function_name, args) if valid tool call at end, None otherwise
        """
        # Pattern: function_name(args...)
        pattern = r'(\w+)\s*\(([^)]*)\)'

        # Find ALL function calls in the response
        all_tool_matches = list(re.finditer(pattern, response))

        if not all_tool_matches:
            return None

        # Get the LAST tool call
        last_tool_match = all_tool_matches[-1]
        function_name = last_tool_match.group(1)
        args_str = last_tool_match.group(2).strip()

        # Check if model stopped after the call (only whitespace after)
        text_after_call = response[last_tool_match.end():].strip()

        if text_after_call and len(text_after_call) >= 10:
            # Model continued generating after tool call - invalid
            logger.warning(f"Model generated text after tool call {function_name}(...) - not executing. "
                         f"Text after: '{text_after_call[:50]}...'")
            return None

        # Parse arguments
        args = {}
        if args_str:
            try:
                args = self._parse_function_args(args_str)
            except Exception as e:
                logger.warning(f"Failed to parse args for {function_name}: {args_str}. Error: {e}")
                return None

        # Valid tool call at the end of response
        if len(all_tool_matches) > 1:
            logger.info(f"Model made {len(all_tool_matches)} tool calls, using LAST one: {function_name}")

        return (function_name, args)


    def parse_all_tool_calls(self, response: str) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Parse ALL Python function calls from model response.

        The model often generates multiple tool calls in a single response.
        This method extracts all of them.

        Expected format:
            function_name(arg1="value1")
            another_function(arg2="value2")

        Args:
            response: Model's response text

        Returns:
            List of (function_name, args) tuples (empty list if none found)
        """
        tool_calls = []

        # Pattern: function_name(args...)
        pattern = r'(\w+)\s*\(([^)]*)\)'
        tool_matches = list(re.finditer(pattern, response))

        if not tool_matches:
            return []

        # For each function call, parse its arguments
        for tool_match in tool_matches:
            function_name = tool_match.group(1)
            args_str = tool_match.group(2).strip()

            # Parse arguments
            args = {}
            if args_str:
                try:
                    args = self._parse_function_args(args_str)
                except Exception as e:
                    logger.warning(f"Failed to parse args for {function_name}: {args_str}. Error: {e}")
                    args = {}

            tool_calls.append((function_name, args))

        return tool_calls

    def _parse_function_args(self, args_str: str) -> Dict[str, Any]:
        """
        Parse function arguments from string like: arg1="value", arg2=123, arg3=True

        Supports:
        - String values: arg="value" or arg='value'
        - Numbers: arg=123, arg=3.14
        - Booleans: arg=True, arg=False
        - None: arg=None
        - Empty dicts: arg={}
        - Simple lists: arg=[1, 2, 3]

        Args:
            args_str: Comma-separated key=value pairs

        Returns:
            Dictionary of parsed arguments
        """
        args = {}

        # Handle empty args
        if not args_str or args_str.isspace():
            return args

        # Try ast.literal_eval first for simple cases
        try:
            import ast
            # Wrap in dict(...) for ast.literal_eval
            dict_str = f"dict({args_str})"
            args = ast.literal_eval(dict_str)
            return args
        except:
            pass

        # Fall back to manual parsing for more complex cases
        # Split by commas not inside quotes or brackets
        parts = self._split_args(args_str)

        for part in parts:
            part = part.strip()
            if not part:
                continue

            # Split on first = sign
            if '=' not in part:
                logger.warning(f"Skipping invalid arg (no =): {part}")
                continue

            key, value = part.split('=', 1)
            key = key.strip()
            value = value.strip()

            # Parse the value
            try:
                import ast
                args[key] = ast.literal_eval(value)
            except:
                # If literal_eval fails, treat as string (remove quotes if present)
                if (value.startswith('"') and value.endswith('"')) or \
                   (value.startswith("'") and value.endswith("'")):
                    args[key] = value[1:-1]
                else:
                    args[key] = value

        return args

    def _split_args(self, args_str: str) -> List[str]:
        """
        Split comma-separated arguments, respecting quotes and brackets.

        Args:
            args_str: Comma-separated arguments

        Returns:
            List of argument strings
        """
        parts = []
        current = []
        in_quotes = False
        quote_char = None
        paren_depth = 0
        bracket_depth = 0
        brace_depth = 0

        for char in args_str:
            if char in ('"', "'") and not in_quotes:
                in_quotes = True
                quote_char = char
                current.append(char)
            elif char == quote_char and in_quotes:
                in_quotes = False
                quote_char = None
                current.append(char)
            elif not in_quotes:
                if char == '(':
                    paren_depth += 1
                    current.append(char)
                elif char == ')':
                    paren_depth -= 1
                    current.append(char)
                elif char == '[':
                    bracket_depth += 1
                    current.append(char)
                elif char == ']':
                    bracket_depth -= 1
                    current.append(char)
                elif char == '{':
                    brace_depth += 1
                    current.append(char)
                elif char == '}':
                    brace_depth -= 1
                    current.append(char)
                elif char == ',' and paren_depth == 0 and bracket_depth == 0 and brace_depth == 0:
                    # This comma is a separator
                    parts.append(''.join(current))
                    current = []
                else:
                    current.append(char)
            else:
                current.append(char)

        # Add last part
        if current:
            parts.append(''.join(current))

        return parts

    def execute_tool_call(self, function_name: str, args: Dict[str, Any]) -> Any:
        """
        Execute a tool call and record the result.

        Args:
            function_name: Name of function to call
            args: Arguments to pass to function

        Returns:
            Result from tool execution or error dict
        """
        logger.info(f"[TOOL CALL] {function_name}")
        logger.info(f"  Args: {args}")

        start_time = time.time()
        success = True
        error_msg = None

        try:
            if function_name not in self.tools:
                result = {"error": f"Unknown function: {function_name}"}
                success = False
                error_msg = f"Unknown function: {function_name}"
            else:
                # Execute the tool
                result = self.tools[function_name](**args)

            elapsed = (time.time() - start_time) * 1000

            # Log the result (truncated if too long)
            if isinstance(result, dict):
                result_str = json.dumps(result, indent=2, default=str)
                if len(result_str) > 500:
                    logger.info(f"  Result (truncated): {result_str[:500]}...")
                else:
                    logger.info(f"  Result: {result_str}")
            elif isinstance(result, list):
                logger.info(f"  Result: list with {len(result)} items")
                if len(result) > 0 and len(str(result)) < 500:
                    logger.info(f"    {result}")
            else:
                result_str = str(result)
                if len(result_str) > 500:
                    logger.info(f"  Result (truncated): {result_str[:500]}...")
                else:
                    logger.info(f"  Result: {result_str}")

            # Record this tool call
            tool_call = ToolCall(
                function=function_name,
                args=args,
                result=result if not isinstance(result, (list, dict)) or len(str(result)) < 1000
                       else str(result)[:1000] + "...",
                timestamp=time.time(),
                elapsed_ms=elapsed,
                success=success,
                error=error_msg
            )
            self.tool_calls.append(tool_call)

            logger.info(f"  Completed in {elapsed:.2f}ms")

            return result

        except Exception as e:
            elapsed = (time.time() - start_time) * 1000
            logger.error(f"Error executing {function_name}: {e}", exc_info=True)

            error_result = {"error": str(e)}
            tool_call = ToolCall(
                function=function_name,
                args=args,
                result=error_result,
                timestamp=time.time(),
                elapsed_ms=elapsed,
                success=False,
                error=str(e)
            )
            self.tool_calls.append(tool_call)

            return error_result

    def get_tool_call_summary(self) -> Dict[str, Any]:
        """Get summary statistics of tool usage"""
        if not self.tool_calls:
            return {"total_calls": 0}

        total_calls = len(self.tool_calls)
        successful_calls = sum(1 for call in self.tool_calls if call.success)
        failed_calls = total_calls - successful_calls

        # Function usage counts
        function_counts = {}
        for call in self.tool_calls:
            function_counts[call.function] = function_counts.get(call.function, 0) + 1

        # Average execution time
        avg_time = sum(call.elapsed_ms for call in self.tool_calls) / total_calls

        return {
            "total_calls": total_calls,
            "successful_calls": successful_calls,
            "failed_calls": failed_calls,
            "function_usage": function_counts,
            "average_execution_ms": avg_time,
            "total_execution_ms": sum(call.elapsed_ms for call in self.tool_calls)
        }

    def export_tool_calls(self) -> List[Dict[str, Any]]:
        """Export all tool calls for analysis"""
        return [
            {
                "function": call.function,
                "args": call.args,
                "result": call.result,
                "timestamp": call.timestamp,
                "elapsed_ms": call.elapsed_ms,
                "success": call.success,
                "error": call.error
            }
            for call in self.tool_calls
        ]
