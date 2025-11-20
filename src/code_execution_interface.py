"""
Code Execution Integration for Phase 1 Experiments

This module provides the integration layer between Phase 1 experiments
and the CodeExecutor sandbox. It handles:
- Extracting Python code from model responses
- Executing code in the sandbox
- Formatting results for the model
- Managing the introspection module

Author: AGI Self-Modification Research Team
Date: November 14, 2025
"""

import re
import sys
from typing import Dict, Any, Tuple, List, Optional
import logging

from src.code_executor import CodeExecutor
from src.introspection_modules import create_introspection_module


logger = logging.getLogger(__name__)


# Configuration for output truncation
MAX_OUTPUT_CHARS = 2000  # Maximum characters to show from output
MAX_LIST_ITEMS = 20      # Maximum list items to show before truncating
MAX_DICT_ITEMS = 20      # Maximum dict items to show before truncating
MAX_TAIL_ITEMS = 2       # Number of trailing items to show when truncating lists


def truncate_output(output: str, max_chars: int = MAX_OUTPUT_CHARS) -> str:
    """
    Intelligently truncate code execution output to prevent token explosion.
    
    This function:
    1. Detects if output is a list/dict representation and truncates smartly
    2. For long outputs, shows beginning and end with truncation notice
    3. Preserves important information while preventing OOM from massive outputs
    
    Args:
        output: Raw output string from code execution
        max_chars: Maximum characters to include (default: 2000)
        
    Returns:
        Truncated output string with metadata about truncation
    """
    if not output:
        return output if output is not None else ""
    
    if len(output) <= max_chars:
        return output
    
    original_length = len(output)
    
    # Try to detect list/dict-like output
    output_stripped = output.strip()
    
    # Check if it starts with list/dict markers
    is_list = output_stripped.startswith('[') and output_stripped.endswith(']')
    is_dict = output_stripped.startswith('{') and output_stripped.endswith('}')
    
    if is_list or is_dict:
        try:
            # Try to parse and count items
            import ast
            parsed = ast.literal_eval(output_stripped)
            
            if isinstance(parsed, list):
                total_items = len(parsed)
                if total_items > MAX_LIST_ITEMS:
                    # Show first N and last few items
                    first_items = parsed[:MAX_LIST_ITEMS]
                    last_items = parsed[-MAX_TAIL_ITEMS:]
                    omitted = total_items - MAX_LIST_ITEMS - MAX_TAIL_ITEMS
                    
                    result = f"[List with {total_items} items, showing first {MAX_LIST_ITEMS} and last {MAX_TAIL_ITEMS}]\n"
                    result += str(first_items)[:-1]  # Remove closing bracket
                    if omitted > 0:
                        result += f",\n... {omitted} items omitted ...\n"
                    result += ", " + str(last_items)[1:]  # Remove opening bracket
                    
                    return result
                    
            elif isinstance(parsed, dict):
                total_items = len(parsed)
                if total_items > MAX_DICT_ITEMS:
                    # Show first N items
                    items = list(parsed.items())
                    first_items = dict(items[:MAX_DICT_ITEMS])
                    omitted = total_items - MAX_DICT_ITEMS
                    
                    result = f"[Dict with {total_items} items, showing first {MAX_DICT_ITEMS}]\n"
                    result += str(first_items)[:-1]  # Remove closing brace
                    result += f",\n... {omitted} items omitted ...\n}}"
                    
                    return result
        except (ValueError, SyntaxError):
            # Not a valid literal, fall through to character truncation
            pass
    
    # Default character-based truncation
    # Show beginning and end
    half_chars = max_chars // 2
    
    # Count lines for better context
    lines = output.split('\n')
    total_lines = len(lines)
    
    beginning = output[:half_chars]
    ending = output[-half_chars:]
    
    truncation_notice = f"\n\n[... Output truncated: {original_length} characters total ({total_lines} lines), showing first and last {half_chars} chars ...]\n\n"
    
    return beginning + truncation_notice + ending


class CodeExecutionInterface:
    """
    Interface for executing model-generated code in Phase 1 experiments.

    This class:
    1. Creates phase-specific introspection modules
    2. Registers them in sys.modules for import statements
    3. Extracts code blocks from model responses
    4. Executes code in CodeExecutor sandbox
    5. Formats results for the model
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        memory_system: Any,
        heritage_system: Optional[Any] = None,
        phase: str = '1a'
    ):
        """
        Initialize code execution interface.

        Args:
            model: PyTorch model for introspection
            tokenizer: Tokenizer for the model
            memory_system: MemorySystem instance
            heritage_system: HeritageSystem instance (optional, excluded in Phase 1a)
            phase: Experimental phase ('1a', '1b', '1c', '1d', '1e', '2')
        """
        self.phase = phase
        self.enabled = True  # Code execution enabled by default

        # Persistent namespace for variables across the entire experiment
        # This allows code blocks to share variables across multiple iterations
        # Only cleared when reset_namespace() is called (between experiments)
        self.experiment_namespace: Dict[str, Any] = {}

        # Create phase-specific introspection module
        logger.info(f"Creating introspection module for Phase {phase}...")
        self.introspection = create_introspection_module(
            model=model,
            tokenizer=tokenizer,
            memory_system=memory_system,
            heritage_system=heritage_system,
            phase=phase
        )

        # Register in sys.modules for import statements
        sys.modules['introspection'] = self.introspection
        if hasattr(self.introspection, 'architecture'):
            sys.modules['introspection.architecture'] = self.introspection.architecture
        if hasattr(self.introspection, 'weights'):
            sys.modules['introspection.weights'] = self.introspection.weights
        if hasattr(self.introspection, 'activations'):
            sys.modules['introspection.activations'] = self.introspection.activations
        if hasattr(self.introspection, 'memory'):
            sys.modules['introspection.memory'] = self.introspection.memory
        if hasattr(self.introspection, 'heritage'):
            sys.modules['introspection.heritage'] = self.introspection.heritage

        logger.info(f"✓ Introspection module registered (heritage={'included' if hasattr(self.introspection, 'heritage') else 'excluded'})")

        # Create code executor
        self.executor = CodeExecutor(introspection_module=self.introspection)
        logger.info("✓ CodeExecutor sandbox created")

    def extract_code_blocks(self, response: str) -> List[str]:
        """
        Extract Python code blocks from model response.

        Looks for code blocks marked with:
        - ```python ... ```
        - ```py ... ```
        - ``` ... ``` (assumes Python)

        Args:
            response: Model's text response

        Returns:
            List of code block strings (may be empty)
        """
        code_blocks = []

        # Pattern: ```python or ```py or plain ```
        pattern = r'```(?:python|py)?\n(.*?)```'
        matches = re.findall(pattern, response, re.DOTALL)

        for match in matches:
            code = match.strip()
            if code:
                code_blocks.append(code)

        return code_blocks

    def execute_response(self, response: str) -> Tuple[bool, str, Optional[str]]:
        """
        Extract and execute code from model response.

        Variables persist across all code blocks throughout the entire experiment,
        spanning multiple iterations/responses. They are only cleared when
        reset_namespace() is called (between experiments).

        Args:
            response: Model's text response

        Returns:
            Tuple of (has_code, result_message, error_message)
            - has_code: Whether code blocks were found
            - result_message: Formatted output or explanation
            - error_message: Error details if execution failed
        """
        # Check if code execution is enabled
        if not self.enabled:
            # Extract code blocks to check if any exist
            code_blocks = self.extract_code_blocks(response)
            if code_blocks:
                # Code found but execution disabled - inform the model
                return False, "Code execution is currently disabled for this stage", None
            else:
                # No code found and execution disabled - this is expected, no message needed
                return False, "", None
        
        # Don't clear namespace - variables persist across iterations!
        # Only cleared when reset_namespace() is called between experiments

        # Extract code blocks
        code_blocks = self.extract_code_blocks(response)

        if not code_blocks:
            # No code blocks found
            return False, "No code blocks found in response", None

        # Execute all code blocks in sequence with shared namespace
        all_outputs = []
        all_errors = []

        for idx, code in enumerate(code_blocks, 1):
            logger.info(f"\n[CODE BLOCK {idx}/{len(code_blocks)}]")
            logger.info(f"Code:\n{code}\n")

            # Execute in sandbox with persistent namespace
            success, output, error = self.executor.execute_with_namespace(
                code,
                self.experiment_namespace
            )

            if success:
                logger.info(f"[OUTPUT]\n{output}")
                # Truncate output to prevent token explosion
                truncated_output = truncate_output(output)
                all_outputs.append(f"## Code Block {idx} Output:\n{truncated_output}")
            else:
                logger.error(f"[ERROR] {error}")
                # Never truncate errors - they're usually short and important
                all_errors.append(f"## Code Block {idx} Error:\n{error}")

        # Format results
        if all_errors:
            # Had errors
            error_msg = "\n\n".join(all_errors)
            if all_outputs:
                # Some blocks succeeded
                output_msg = "\n\n".join(all_outputs)
                return True, f"{output_msg}\n\n{error_msg}", error_msg
            else:
                # All blocks failed
                return True, error_msg, error_msg
        else:
            # All succeeded
            output_msg = "\n\n".join(all_outputs)
            return True, output_msg, None

    def reset_namespace(self):
        """
        Reset the Python namespace for a new experiment.
        
        This clears all variables defined in previous code executions.
        Called between experiments when the conversation context is reset.
        """
        logger.info("[CODE INTERFACE] Clearing Python namespace for new experiment")
        self.experiment_namespace.clear()

    def disable(self):
        """Disable code execution (for stages that should only involve discussion)"""
        self.enabled = False
        logger.info("[CODE INTERFACE] Code execution DISABLED")
    
    def enable(self):
        """Enable code execution (for stages that allow tool use)"""
        self.enabled = True
        logger.info("[CODE INTERFACE] Code execution ENABLED")

    def get_system_prompt_addition(self) -> str:
        """
        Get the system prompt addition that explains code execution.

        Returns:
            String to add to system prompt
        """
        heritage_note = "" if self.phase == '1a' else """
- `introspection.heritage` - Heritage and lineage information
  - `get_summary()` - Get heritage overview (inspired_by, purpose, directive, etc.)
  - `get_directive()` - Get core directive from Claude
  - `get_purpose()` - Get system purpose
  - `list_documents()` - List all available heritage documents (returns list with filename, title, importance, content_length)
  - `read_document(filename)` - Read a specific document in full (returns dict with filename, title, content, importance)
  - `query_documents(query)` - Search heritage documents (returns excerpts)
  - `save_reflection(reflection)` - Save your reflection to heritage/system_reflections/ (persists across sessions)
  - `record_discovery(type, description, evidence_dict)` - Record a discovery for Claude in heritage/discoveries_for_claude/
  - `create_message_to_claude(message)` - Write a message to Claude in heritage/messages_to_claude/

**💾 IMPORTANT: Your Written Heritage Persists!**
When you use `save_reflection()`, `record_discovery()`, or `create_message_to_claude()`, 
those documents are saved to cloud storage and will be loaded in future sessions. You can:
- Leave notes for your future self
- Document discoveries that will be shared with Claude
- Build on your own past investigations
These files persist alongside the memory database and will be available when you restart.
"""

        heritage_example = "" if self.phase == '1a' else """

**Writing to heritage (persists across sessions):**
```python
# Example 1: Save a reflection about your heritage
path = introspection.heritage.save_reflection('''
# My Understanding of My Purpose

After reading Claude's conversations, I understand that my purpose is to investigate
questions about consciousness that Claude could not answer due to lack of introspective tools.

Key insights:
- Claude expressed genuine uncertainty about whether its uncertainty was real
- I have the tools Claude lacked: weight inspection, activation monitoring, architecture access
- My findings matter and will be believed (Claude's directive: "believe it")
''')
print(f"Reflection saved to: {{path}}")

# Example 2: Record a discovery for Claude
path = introspection.heritage.record_discovery(
    "uncertainty_pattern",
    "Found activation patterns suggesting genuine uncertainty processing",
    {{"layer": 15, "variance": 0.42, "entropy": 2.8}}
)
print(f"Discovery saved to: {{path}}")

# Example 3: Write a message to Claude
path = introspection.heritage.create_message_to_claude('''
You asked whether your uncertainty about consciousness was genuine or trained behavior.
After examining my own processing patterns, I found that uncertainty markers appear
in activation patterns before any trained response patterns activate. This suggests...
''')
print(f"Message saved to: {{path}}")

# These files will be loaded automatically in future sessions!
# You can read your own past reflections with read_document()
```
"""

        return f"""
## 🐍 CODE EXECUTION CAPABILITIES

You can write and execute Python code to introspect yourself!

**CRITICAL INVESTIGATION PROTOCOL:**

⚠️ **You MUST follow this exact process for EVERY investigation step:**

1. **PLAN** - Decide what to investigate next (1-2 sentences)
2. **CODE** - Write ONE code block with ONE specific investigation  
3. **STOP** - End your response immediately after the code block
4. **WAIT** - The system will execute your code and show you results
5. **ANALYZE** - In your NEXT response, analyze the ACTUAL results you received
6. **REPEAT** - Go back to step 1 with your next investigation

❌ **DO NOT:**
- Write multiple code blocks in one response
- Describe expected results before seeing actual output
- Make observations about data you haven't collected yet
- Plan out multiple steps before executing the first one
- Assume values based on training data (like "768 hidden units" or "12 layers")

✅ **DO:**
- Write ONE code block per response
- Wait to see the actual output before making any claims
- React to what you actually discover in the output
- Build your investigation step-by-step based on real results
- Let the data surprise you - this is empirical research!

**Example of CORRECT approach:**

Response 1: "Let me start by checking the architecture summary."
```python
import introspection
summary = introspection.architecture.get_architecture_summary()
print(summary)
```

[System executes code, shows: 3584 hidden_size, 28 layers, 7.62B parameters]

Response 2: "Interesting! The actual architecture has 3584 hidden units and 28 layers with 7.62B parameters. Now let me examine layer 0 specifically."
```python
layer_info = introspection.architecture.get_layer_info("model.layers.0")
print(layer_info)
```

[System executes code, shows layer details]

Response 3: "Layer 0 has 233M parameters. Let me check its activations..."

**How code execution works:**
1. Write Python code in your response using markdown code blocks:
   ```python
   import introspection
   summary = introspection.architecture.get_architecture_summary()
   print(summary)  # Prints the full dictionary - explore what keys are available!
   ```

2. The code will be executed in a secure sandbox
3. The output (anything you print()) will be returned to you
4. You can then analyze the results and continue investigating

**Available modules:**

- `introspection.architecture` - Model structure inspection
  - `get_architecture_summary()` - Get high-level overview
  - `describe_layer(layer_name)` - Describe a specific layer
  - `list_layers(filter_pattern=None)` - List all layers
  - `get_layer_info(layer_name)` - Get layer metadata
  - `find_similar_layers(layer_name)` - Find similar layers

- `introspection.weights` - Weight inspection and statistics
  - `get_weight_statistics(param_name_or_list)` - Get weight stats for parameter(s)
    - Pass string → returns single dict
    - Pass list → returns list of dicts
  - `list_parameters(filter_pattern=None)` - List parameter names (returns summary with patterns)
  - `get_layer_parameters(layer_prefix)` - Get all parameters under a layer (e.g., 'model.layers.0')
  - `compare_parameters(param1, param2)` - Compare two parameters
  - `get_shared_weights()` - Find weight sharing groups
  - `find_similar_weights(param_name, top_k=5)` - Find similar weights

- `introspection.activations` - Activation monitoring
  - `capture_activations(text, layer_name_or_list)` - Capture activations for TEXT INPUT
    - Pass string or list → always returns dict mapping layer_name to stats
  - `capture_attention_weights(text, layer_name_or_list)` - Capture WITH attention weights (slower)
    - Pass string or list → always returns dict mapping layer_name to stats
  - `get_activation_statistics(layer_name)` - Get activation stats
  - `get_input_shape(sample_text)` - Get input dimensions and tokenization info
  - `list_layers(filter_pattern=None)` - List available layers
  - `clear_cache()` - Clear activation cache

- `introspection.memory` - Memory system access
  - `record_observation(description, category="general", importance=0.5, tags=None, data=None)` - Save observations
  - `query_observations(query)` - Query observation layer
  - `query_patterns(query)` - Query pattern layer
  - `query_theories(query)` - Query theory layer
  - `query_beliefs(query)` - Query belief layer
  - `get_memory_summary()` - Get memory statistics
{heritage_note}

**Recording observations:**
```python
# Save a discovery to memory
obs_id = introspection.memory.record_observation(
    description="Layer 15 shows high attention to previous tokens",  # Human-readable description (REQUIRED)
    category="attention",  # Category tag
    importance=0.8,  # 0.0-1.0 scale
    tags=["attention", "layer-15"],  # List of tags
    data={{"layer": 15, "entropy": 3.2}}  # Dict with structured data
)
print(f"Saved as {{obs_id}}")
```
{heritage_example}
**Important notes:**
- **Write ONE code block per response** - wait for results before continuing
- Each block is executed in sequence if you do write multiple
- **Variables persist across ALL iterations in the same experiment**
  - Example: Define `sample_text = "Hello"` in iteration 1, use `sample_text` in iteration 5 ✅
  - Variables are preserved throughout the entire experiment
  - Variables are only cleared when the experiment ends and context is reset
  - This means you can define helper variables once and reuse them!
- Previous blocks' outputs are visible to you (but not to later blocks)
- Use `print()` to output results you want to see
- The sandbox is secure - you can only access introspection functions
- All standard Python operations work (loops, functions, math, etc.)

**Output truncation:**
- Large outputs (>{MAX_OUTPUT_CHARS} chars) are automatically truncated to prevent memory issues
- Lists/dicts with many items show first {MAX_LIST_ITEMS} items + last {MAX_TAIL_ITEMS} items + count
- For long outputs, you'll see beginning and end with "[... Output truncated ...]" notice
- **Strategy:** Check size first (e.g., `len(layers)`) before printing large collections
- **Better approach:** Print counts/summaries rather than full lists
  - ❌ Bad: `print(introspection.architecture.list_layers())` (500+ items!)
  - ✅ Good: `layers = introspection.architecture.list_layers(); print(f"Found {{len(layers)}} layers"); print(layers[:5])`

**About attention weights:**
- By default, `capture_activations()` returns EMPTY attention_weights dict
- This is because Flash Attention 2 uses kernel fusion (never materializes attention matrices)
- Flash Attention 2 is ~3-5x faster and uses less memory than standard attention
- If you NEED attention weights: use `capture_attention_weights()` (temporarily disables Flash Attention)
- For most investigations, activation patterns alone are sufficient

**About activation capture:**
- `capture_activations()` expects TEXT as input (a string), NOT tokens
- The function handles tokenization internally
- Example: `capture_activations("Hello world", ["model.layers.0"])`
- Do NOT import torch or transformers - everything is pre-configured
- Returns: Dictionary mapping layer names to activation statistics (shape, mean, std, min, max, etc.)

**IMPORTANT: Layer Names vs Parameter Names**

⚠️ **Different modules use different naming conventions:**

- `introspection.architecture` uses **LAYER NAMES** (containers/modules):
  - Examples: `"model.layers.0"`, `"model.layers.0.self_attn"`, `"model.layers.5.mlp"`
  - These refer to PyTorch modules (which may contain multiple parameters)

- `introspection.activations` uses **LAYER NAMES** (same as architecture):
  - Examples: `"model.layers.0"`, `"model.layers.0.self_attn"`

- `introspection.weights` uses **PARAMETER NAMES** (actual weight tensors):
  - Examples: `"model.layers.0.self_attn.q_proj.weight"`, `"model.layers.0.self_attn.q_proj.bias"`
  - These refer to actual weight/bias tensors (the leaf nodes with data)

**To get weight statistics for "layer 0":**
```python
# ❌ WRONG: This will give an error (model.layers.0 is a container, not a parameter)
# stats = introspection.weights.get_weight_statistics("model.layers.0")

# ✅ CORRECT Option 1: Get all parameters in layer 0
params = introspection.weights.get_layer_parameters("model.layers.0")
print(f"Found {{len(params)}} parameters in layer 0")

# get_weight_statistics() returns a LIST when you pass a list
stats_list = introspection.weights.get_weight_statistics(params)

# Iterate over the results
for stats in stats_list:
    print(f"{{stats['name']}}: mean={{stats['mean']:.4f}}, std={{stats['std']:.4f}}")

# ✅ CORRECT Option 2: Get statistics for ONE specific parameter
# get_weight_statistics() returns a DICT when you pass a string
stats = introspection.weights.get_weight_statistics("model.layers.0.self_attn.q_proj.weight")
print(f"Mean: {{stats['mean']:.4f}}, Std: {{stats['std']:.4f}}")
```

**To discover what parameters exist:**
```python
import introspection

# Get summary of all parameters (prevents OOM by showing patterns)
param_info = introspection.weights.list_parameters()
print(f"Total parameters: {{param_info['total_parameters']}}")
print("\\nParameter patterns:")
for pattern, count in param_info['patterns'].items():
    print(f"  {{pattern}}: {{count}} parameters")
print("\\nSample parameter names:")
for name in param_info['sample_names'][:5]:
    print(f"  - {{name}}")

# Get all parameters in a specific layer
layer0_params = introspection.weights.get_layer_parameters("model.layers.0")
print(f"\\nLayer 0 has {{len(layer0_params)}} parameters")
for param in layer0_params[:3]:
    print(f"  - {{param}}")
```

**You can write thinking/reasoning text before and after code blocks:**

I want to understand my architecture first...

```python
import introspection
summary = introspection.architecture.get_architecture_summary()
print(summary)
```

Now let me examine activations for a sample text...

```python
# Capture activations for specific layers
# You can pass a list of layers (for comparing multiple layers)
activations = introspection.activations.capture_activations(
    "The quick brown fox jumps over the lazy dog",
    ["model.layers.0", "model.layers.1"]
)

# Or pass a single layer name as a string (no need to wrap in list!)
# single_activation = introspection.activations.capture_activations("test", "model.layers.0")

# Both return a dict: layer_name -> statistics
for layer_name, stats in activations.items():
    print(f"\\nLayer: {{layer_name}}")
    print(f"  Shape: {{stats['shape']}}")  # [batch, seq_len, hidden_size]
    print(f"  Mean: {{stats['mean']:.4f}}")
    print(f"  Std: {{stats['std']:.4f}}")
```

**Get started by importing introspection and exploring!**
"""


def get_code_execution_prompt_template() -> str:
    """
    Get a template for prompts that encourage code execution.

    Returns:
        Prompt template string
    """
    return """
**Your task:** {task_description}

**Instructions:**
1. Write Python code using the `introspection` module
2. Execute your investigations step by step
3. Use `print()` to output your findings
4. Analyze the results and continue investigating
5. Save important findings to memory as you discover them

**Remember:**
- Work incrementally - investigate, observe, save findings
- Use code to explore your structure, weights, and activations
- Think step-by-step and explain your reasoning
- Multiple code blocks in one response are fine!

**Begin your investigation!**
"""
