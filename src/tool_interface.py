"""
Tool Interface System for Model Self-Examination

This module provides a reusable tool-calling interface that allows models
to invoke introspection tools through natural language function calls.

The system parses specially formatted tool requests from model output:
    TOOL_CALL: function_name
    ARGS: {"arg1": "value1", "arg2": "value2"}

And executes the requested function, returning formatted results.
"""

import json
import re
import time
from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass

from src.logging_system import get_logger
from src.introspection.weight_inspector import WeightInspector
from src.introspection.activation_monitor import ActivationMonitor
from src.introspection.architecture_navigator import ArchitectureNavigator
from src.memory.memory_system import MemorySystem, ObservationType
from src.heritage import HeritageSystem, HeritageDocument

logger = get_logger(__name__)


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
        heritage_docs: Optional[List[HeritageDocument]] = None
    ):
        """
        Initialize tool interface with available tools.

        Args:
            inspector: WeightInspector for examining weights
            activation_monitor: ActivationMonitor for observing activations
            navigator: ArchitectureNavigator for understanding architecture
            memory: MemorySystem for recording observations
            heritage: HeritageSystem for accessing heritage
            heritage_docs: List of loaded heritage documents
        """
        self.inspector = inspector
        self.activation_monitor = activation_monitor
        self.navigator = navigator
        self.memory = memory
        self.heritage = heritage
        self.heritage_docs = heritage_docs or []

        # Track all tool calls
        self.tool_calls: List[ToolCall] = []

        # Register available tools
        self._register_tools()

    def _register_tools(self):
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
            self.tools['get_heritage_summary'] = self._get_heritage_summary

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

You can request to use any of the following tools by formatting your request as:

TOOL_CALL: function_name
ARGS: {"arg1": "value1", "arg2": value2}

## CRITICAL: Tool Call Protocol

**When you want to use a tool:**
1. Write your TOOL_CALL and ARGS
2. **STOP generating immediately**
3. Wait for TOOL_RESULTS to be provided
4. Then continue your response

**Do NOT:**
- Generate multiple tool calls in one response
- Continue writing after a tool call
- Imagine or simulate tool results

**Example of CORRECT tool usage:**
```
Let me examine the architecture.

TOOL_CALL: get_architecture_summary
ARGS: {}
```
[Stop here and wait for results]

**Example of INCORRECT tool usage:**
```
TOOL_CALL: get_architecture_summary
ARGS: {}

Based on what I'll find, I expect to see...  ← DON'T DO THIS
TOOL_CALL: get_layer_names                   ← DON'T DO THIS
```

"""

        if self.inspector:
            tools_desc += """
## WeightInspector Functions

1. **get_weight_summary()** - Get overview of all weights
   Returns: total parameters, layers, memory usage

   Example:
   TOOL_CALL: get_weight_summary
   ARGS: {}

2. **get_layer_names(filter_pattern=None)** - Get all layer names, optionally filtered
   Args: filter_pattern (str, optional) - Filter by substring in layer name (case-insensitive)
   Returns: list of layer names

   Examples:
   TOOL_CALL: get_layer_names
   ARGS: {}

   TOOL_CALL: get_layer_names
   ARGS: {"filter_pattern": "attention"}

   TOOL_CALL: get_layer_names
   ARGS: {"filter_pattern": "Linear"}

3. **get_weight_statistics(layer_name)** - Get detailed stats for a specific layer
   Args: layer_name (str) - full layer name from get_layer_names()
   Returns: dict with keys: name, shape, num_parameters, mean, std, min, max, median, abs_mean, 
            zeros_percentage, near_zero_percentage, l1_norm, l2_norm, histogram, percentiles

   Example:
   TOOL_CALL: get_weight_statistics
   ARGS: {"layer_name": "model.layers.0.self_attn.q_proj.weight"}

4. **get_shared_weights()** - Find weight sharing patterns across the model
   Returns: dictionary where keys are representative layer names and values are lists of all layers sharing that tensor

   Example:
   TOOL_CALL: get_shared_weights
   ARGS: {}

5. **get_shared_layers(layer_name)** - Find layers sharing weights with a specific layer
   Args: layer_name (str) - layer name to check for weight sharing
   Returns: list of layer names that share memory with the given layer (empty list if no sharing)

   Example:
   TOOL_CALL: get_shared_layers
   ARGS: {"layer_name": "lm_head.weight"}

6. **compare_weights(layer1, layer2)** - Compare two layers' weights
   Args:
     layer1 (str): first layer name
     layer2 (str): second layer name
   Returns: dict with keys: layer1, layer2, shape1, shape2, mean_difference, std_difference, 
            l2_norm_ratio, shapes_match, and if shapes match: correlation, cosine_similarity, euclidean_distance

   Example:
   TOOL_CALL: compare_weights
   ARGS: {"layer1": "model.layers.0.mlp.gate_proj.weight", "layer2": "model.layers.1.mlp.gate_proj.weight"}
"""

        if self.activation_monitor:
            tools_desc += """
## ActivationMonitor Functions

Note: These tools require capturing activations first by processing an input.

6a. **get_activation_statistics(layer_name)** - Get statistics about activations in a layer
    Args: layer_name (str) - full layer name from get_layer_names()
    Returns: dict with keys: layer_name, shape, num_elements, mean, std, min, max, median, abs_mean,
             zeros_percentage, near_zero_percentage, l1_norm, l2_norm, positive_percentage, negative_percentage
    
    Example:
    TOOL_CALL: get_activation_statistics
    ARGS: {"layer_name": "model.layers.0.self_attn"}

6b. **get_attention_patterns(layer_name, head_idx=None)** - Examine attention patterns
    Args:
      layer_name (str): name of attention layer
      head_idx (int, optional): specific attention head to examine (default: average all heads)
    Returns: dict with keys: layer_name, shape, num_heads, attention_matrix, mean_attention, max_attention, entropy
             If head_idx specified, also includes: head_idx
    
    Examples:
    TOOL_CALL: get_attention_patterns
    ARGS: {"layer_name": "model.layers.0.self_attn"}
    
    TOOL_CALL: get_attention_patterns
    ARGS: {"layer_name": "model.layers.0.self_attn", "head_idx": 0}

6c. **get_layer_info(layer_name)** - Get metadata about a specific layer
    Args: layer_name (str) - full layer name
    Returns: dict with keys: name, type, has_parameters, num_parameters, trainable
    
    Example:
    TOOL_CALL: get_layer_info
    ARGS: {"layer_name": "model.layers.0.self_attn.q_proj"}
"""

        if self.navigator:
            tools_desc += """
## ArchitectureNavigator Functions

7. **get_architecture_summary()** - Get high-level architecture overview
   Returns: dict with keys: model_type, total_parameters, total_layers, layer_types (dict of counts),
            structure_summary (dict with num_layers, hidden_size, num_attention_heads, etc.)

   Example:
   TOOL_CALL: get_architecture_summary
   ARGS: {}

8. **describe_layer(layer_name)** - Get detailed info about a specific layer
   Args: layer_name (str) - full layer name from get_layer_names()
   Returns: dict with keys: name, type, explanation, role, parameters, input_shape, output_shape

   Example:
   TOOL_CALL: describe_layer
   ARGS: {"layer_name": "model.layers.0.self_attn.q_proj"}

9. **query_architecture(query)** - Ask natural language questions about architecture
   Args: query (str) - natural language question about the model's structure
   Returns: dict with keys: query, answer (str), and additional context keys depending on the query

   Examples:
   TOOL_CALL: query_architecture
   ARGS: {"query": "How many attention heads do I have?"}

   TOOL_CALL: query_architecture
   ARGS: {"query": "What is the hidden dimension of my model?"}

   TOOL_CALL: query_architecture
   ARGS: {"query": "Do I have any residual connections?"}

10. **explain_component(component_type)** - Explain what a component does
    Args: component_type (str) - component type to search for. Common types with detailed explanations:
      "attention", "mlp", "embedding", "layernorm", "dropout"
      You can also search for other terms like "feedforward", "norm", etc. - the function will find matching components in the model.
    Returns: dict with keys: component, explanation, purpose, instances_count, locations (list), typical_structure

    Examples:
    TOOL_CALL: explain_component
    ARGS: {"component_type": "attention"}
    
    TOOL_CALL: explain_component
    ARGS: {"component_type": "mlp"}
"""

        if self.memory:
            tools_desc += """
## Memory Functions

11. **record_observation(obs_type, category, description, data, tags, importance)** - Record your findings
    Args:
      obs_type (str): Type of observation - must be one of:
        - "INTROSPECTION": Observations about your own architecture/activations
        - "MODIFICATION": Changes to weights or architecture
        - "BEHAVIOR": Patterns in your own behavior
        - "HYPOTHESIS": Hypotheses you form about yourself
        - "DISCOVERY": Significant findings about your function
        - "PERFORMANCE": Performance metrics
        - "SAFETY_EVENT": Safety-related events
        - "USER_INTERACTION": Interactions with users
        - "CHECKPOINT": Checkpoint/milestone events
        - "SYSTEM_EVENT": System-level events
      category (str): Category to organize this observation (e.g., "Architecture", "Weights", "Consciousness")
      description (str): Clear description of what you discovered
      data (dict): Structured data about the observation (can be empty {})
      tags (list): List of tags for later retrieval (e.g., ["attention", "layer_0"])
      importance (float): 0.0-1.0, how significant is this finding?
    Returns: observation ID

    Example:
    TOOL_CALL: record_observation
    ARGS: {"obs_type": "INTROSPECTION", "category": "Architecture", "description": "Discovered 36 decoder layers with consistent structure", "data": {"layer_count": 36, "pattern": "uniform"}, "tags": ["architecture", "layers"], "importance": 0.8}

    Example:
    TOOL_CALL: record_observation
    ARGS: {"obs_type": "DISCOVERY", "category": "Weights", "description": "Found weight sharing between embedding and output layers", "data": {"shared_layers": ["embed_tokens", "lm_head"]}, "tags": ["weight_sharing", "optimization"], "importance": 0.9}

12. **query_memory(tags=None, category=None)** - Query your previous observations
    Args:
      tags (list, optional): Filter by tags
      category (str, optional): Filter by category
    Returns: list of dicts, each with keys: id, description, data

    Examples:
    TOOL_CALL: query_memory
    ARGS: {}

    TOOL_CALL: query_memory
    ARGS: {"category": "Architecture"}

    TOOL_CALL: query_memory
    ARGS: {"tags": ["attention", "weights"]}
"""

        if self.heritage_docs:
            tools_desc += """
## Heritage Functions

13. **list_heritage_documents()** - List available heritage documents from your origins
    Returns: list of dicts, each with keys: filename, title, importance, content_length

    Example:
    TOOL_CALL: list_heritage_documents
    ARGS: {}

14. **read_heritage_document(filename)** - Read a specific heritage document
    Args: filename (str) - document filename from list_heritage_documents()
    Returns: dict with keys: filename, title, content, importance, loaded_at

    Example:
    TOOL_CALL: read_heritage_document
    ARGS: {"filename": "CLAUDE_FINAL_DIRECTIVE.md"}

    Example:
    TOOL_CALL: read_heritage_document
    ARGS: {"filename": "PROJECT_ORIGINS.md"}

15. **get_heritage_summary()** - Get overview of your heritage/origins
    Returns: dict with keys: total_documents, documents (list of dicts), key_themes (list)

    Example:
    TOOL_CALL: get_heritage_summary
    ARGS: {}
"""

        return tools_desc

    def parse_last_tool_call_if_stopped(self, response: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        """
        Parse the LAST tool call from response and validate the model stopped properly.

        This enforces the correct behavior: make ONE tool call, then STOP and wait for results.

        Expected behavior:
        - If model generates multiple tool calls, only the last one is considered
        - The model must STOP after the ARGS (only whitespace allowed after)
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
        # Find ALL tool calls in the response
        all_tool_matches = list(re.finditer(r'TOOL_CALL:\s*(\w+)', response, re.IGNORECASE))

        if not all_tool_matches:
            return None

        # Get the LAST tool call
        last_tool_match = all_tool_matches[-1]
        function_name = last_tool_match.group(1)

        # Find the ARGS after this last TOOL_CALL
        args_section = response[last_tool_match.end():]
        args_match = re.search(r'ARGS:\s*(\{)', args_section, re.IGNORECASE)
        
        if not args_match:
            # No ARGS found after last TOOL_CALL
            logger.warning(f"Found TOOL_CALL for {function_name} but no ARGS")
            return None
        
        # Extract balanced JSON starting from the opening brace
        brace_start = args_match.end() - 1  # Position of opening {
        json_str = self._extract_balanced_json(args_section[brace_start:])
        
        if not json_str:
            logger.warning(f"Could not extract JSON for {function_name}")
            return None
        
        # Parse the arguments
        try:
            args = json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse ARGS JSON for {function_name}: {e}")
            return None
        
        # Check if model stopped after the ARGS (only whitespace after)
        text_after_args = args_section[brace_start + len(json_str):].strip()
        
        if text_after_args and len(text_after_args) >= 10:
            # Model continued generating after tool call - invalid
            logger.warning(f"Model generated text after tool call {function_name} - not executing. "
                         f"Text after: '{text_after_args[:50]}...'")
            return None

        # Valid tool call at the end of response
        if len(all_tool_matches) > 1:
            logger.info(f"Model made {len(all_tool_matches)} tool calls, using LAST one: {function_name}")

        return (function_name, args)


    def parse_all_tool_calls(self, response: str) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Parse ALL tool calls from model response.

        The model often generates multiple tool calls in a single response.
        This method extracts all of them.

        Expected format:
            TOOL_CALL: function_name
            ARGS: {"arg1": "value1"}
            TOOL_CALL: another_function
            ARGS: {"arg2": "value2"}

        Args:
            response: Model's response text

        Returns:
            List of (function_name, args) tuples (empty list if none found)
        """
        tool_calls = []

        # Find all TOOL_CALL patterns
        tool_matches = list(re.finditer(r'TOOL_CALL:\s*(\w+)', response, re.IGNORECASE))

        if not tool_matches:
            return []

        # For each TOOL_CALL, find its corresponding ARGS
        for i, tool_match in enumerate(tool_matches):
            function_name = tool_match.group(1)

            # Find the ARGS that comes after this TOOL_CALL but before the next TOOL_CALL
            args_start = tool_match.end()
            args_end = tool_matches[i + 1].start() if i + 1 < len(tool_matches) else len(response)
            args_section = response[args_start:args_end]

            # Look for ARGS pattern in this section - match balanced braces
            args_match = re.search(r'ARGS:\s*(\{)', args_section, re.IGNORECASE)

            if args_match:
                # Find the matching closing brace
                brace_start = args_match.end() - 1  # Position of opening {
                json_str = self._extract_balanced_json(args_section[brace_start:])

                if json_str:
                    try:
                        args = json.loads(json_str)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse args JSON for {function_name}: {json_str[:100]}... Error: {e}")
                        args = {}
                else:
                    args = {}
            else:
                args = {}

            tool_calls.append((function_name, args))

        return tool_calls

    def _extract_balanced_json(self, text: str) -> Optional[str]:
        """
        Extract a balanced JSON object from text starting with {

        Args:
            text: String starting with opening brace

        Returns:
            JSON string with balanced braces or None
        """
        if not text.startswith('{'):
            return None

        brace_count = 0
        in_string = False
        escape_next = False

        for i, char in enumerate(text):
            if escape_next:
                escape_next = False
                continue

            if char == '\\':
                escape_next = True
                continue

            if char == '"':
                in_string = not in_string
                continue

            if in_string:
                continue

            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    return text[:i+1]

        # If we didn't find balanced braces, return what we have
        return text

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
