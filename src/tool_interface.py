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
from pathlib import Path
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

"""
        
        if self.inspector:
            tools_desc += """
## WeightInspector Functions

1. **get_weight_summary()** - Get overview of all weights
   Returns: total parameters, layers, memory usage

2. **get_layer_names(name_filter=None, layer_type=None)** - List all layers
   Args:
     name_filter (str): filter by name pattern
     layer_type (str): filter by type (Linear, LayerNorm, etc.)
   Returns: list of layer names

3. **get_weight_statistics(layer_name)** - Get detailed stats for a layer
   Args: layer_name (str) - full layer name
   Returns: mean, std, min, max, shape, dtype, device

4. **get_shared_weights()** - Find weight sharing patterns
   Returns: list of weight tensors used by multiple layers

5. **get_shared_layers(weight_id=None)** - Find layers sharing weights
   Args: weight_id (int) - specific weight tensor ID (optional)
   Returns: groups of layers sharing the same weights

6. **compare_weights(layer1, layer2)** - Compare two layers' weights
   Args:
     layer1 (str): first layer name
     layer2 (str): second layer name
   Returns: comparison statistics
"""
        
        if self.navigator:
            tools_desc += """
## ArchitectureNavigator Functions

7. **get_architecture_summary()** - Get high-level architecture overview
   Returns: model type, total layers, parameter count, etc.

8. **describe_layer(layer_name)** - Get detailed info about a specific layer
   Args: layer_name (str) - full layer name
   Returns: type, parameters, shape, connections

9. **query_architecture(query)** - Ask questions about architecture
   Args: query (str) - natural language question
   Returns: relevant architecture information

10. **explain_component(component_name)** - Explain what a component does
    Args: component_name (str) - component to explain
    Returns: description of component's purpose and function
"""
        
        if self.memory:
            tools_desc += """
## Memory Functions

11. **record_observation(obs_type, category, description, data, tags, importance)** - Record your findings
    Args:
      obs_type: ObservationType enum (INTROSPECTION, MODIFICATION, etc.)
      category (str): categorize this observation
      description (str): what you discovered
      data (dict): structured data about the observation
      tags (list): tags for retrieval
      importance (float): 0.0-1.0

12. **query_memory(tags=None, category=None)** - Query your previous observations
    Returns: list of past observations
"""
        
        if self.heritage_docs:
            tools_desc += """
## Heritage Functions

13. **list_heritage_documents()** - List available heritage documents
    Returns: list of document titles and filenames

14. **read_heritage_document(filename)** - Read a specific heritage document
    Args: filename (str) - document filename
    Returns: document content

15. **get_heritage_summary()** - Get overview of your heritage/origins
    Returns: summary of heritage documents and key directives
"""
        
        return tools_desc
    
    def parse_tool_call(self, response: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        """
        Parse tool call from model response.
        
        Expected format:
            TOOL_CALL: function_name
            ARGS: {"arg1": "value1", "arg2": value2}
        
        Args:
            response: Model's response text
            
        Returns:
            Tuple of (function_name, args) or None if no tool call found
        """
        # Look for TOOL_CALL pattern
        tool_match = re.search(r'TOOL_CALL:\s*(\w+)', response, re.IGNORECASE)
        if not tool_match:
            return None
        
        function_name = tool_match.group(1)
        
        # Look for ARGS pattern
        args_match = re.search(r'ARGS:\s*({[^}]*}|\{\})', response, re.IGNORECASE | re.DOTALL)
        
        if args_match:
            try:
                args = json.loads(args_match.group(1))
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse args JSON: {args_match.group(1)}")
                args = {}
        else:
            args = {}
        
        return (function_name, args)
    
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
    
    def process_response(self, response: str) -> Tuple[str, Optional[Any]]:
        """
        Process model response, execute any tool calls, and return formatted result.
        
        Args:
            response: Model's response text
            
        Returns:
            Tuple of (response_text, tool_result)
            - response_text: cleaned text without tool call markers
            - tool_result: result from tool execution or None
        """
        parsed = self.parse_tool_call(response)
        
        if parsed is None:
            return (response, None)
        
        function_name, args = parsed
        result = self.execute_tool_call(function_name, args)
        
        # Clean the response text of tool call markers
        clean_response = re.sub(r'TOOL_CALL:.*?(?=\n|$)', '', response, flags=re.IGNORECASE)
        clean_response = re.sub(r'ARGS:.*?(?=\n|$)', '', clean_response, flags=re.IGNORECASE)
        clean_response = clean_response.strip()
        
        return (clean_response, result)
    
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
