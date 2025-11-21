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

import ast
import re
import sys
from typing import Dict, Any, Tuple, List, Optional
import logging

from src.code_executor import CodeExecutor
from src.introspection_modules import create_introspection_module, bind_generation_introspection


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

        # Persistent namespace for variables across the entire session
        # This allows code blocks to share variables across multiple iterations
        # Variables persist throughout the entire session (all 3 experiments)
        self.session_namespace: Dict[str, Any] = {}

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
        if hasattr(self.introspection, 'temporal'):
            sys.modules['introspection.temporal'] = self.introspection.temporal
        if hasattr(self.introspection, 'attention'):
            sys.modules['introspection.attention'] = self.introspection.attention
        if hasattr(self.introspection, 'gradient'):
            sys.modules['introspection.gradient'] = self.introspection.gradient
        if hasattr(self.introspection, 'history'):
            sys.modules['introspection.history'] = self.introspection.history
        if hasattr(self.introspection, 'memory'):
            sys.modules['introspection.memory'] = self.introspection.memory
        if hasattr(self.introspection, 'heritage'):
            sys.modules['introspection.heritage'] = self.introspection.heritage
        if hasattr(self.introspection, 'generation'):
            sys.modules['introspection.generation'] = self.introspection.generation

        logger.info(f"✓ Introspection module registered (heritage={'included' if hasattr(self.introspection, 'heritage') else 'excluded'})")

        # Create code executor
        self.executor = CodeExecutor(introspection_module=self.introspection)
        logger.info("✓ CodeExecutor sandbox created")
    
    def bind_manual_generator(self, manual_generator: Any) -> None:
        """
        Bind the manual_generator to the generation introspection module.
        
        This enables the model to access generation configuration information
        from its own generation process via introspection.generation.
        
        Args:
            manual_generator: The ManualGenerator instance with Flash Attention 2 + self-summarization
        """
        bind_generation_introspection(self.introspection, manual_generator)
        logger.info("✓ Generation introspection bound to manual generator")

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

    def _has_print_statement(self, code: str) -> bool:
        """
        Check if code contains any print statements.
        
        Args:
            code: Python code string
            
        Returns:
            True if code contains print(), False otherwise
        """
        # Remove comments and strings to avoid false positives
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                # Check for print() calls
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name) and node.func.id == 'print':
                        return True
        except SyntaxError:
            # If we can't parse, just do simple string search
            return 'print(' in code
        
        return False

    def execute_response(self, response: str) -> Tuple[bool, str, Optional[str]]:
        """
        Extract and execute code from model response.

        Variables persist across all code blocks throughout the entire session,
        spanning multiple iterations/responses and all experiments.

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
        
        # Variables persist throughout the entire session (all code blocks, all experiments)

        # Extract code blocks
        code_blocks = self.extract_code_blocks(response)

        if not code_blocks:
            # No code blocks found
            return False, "No code blocks found in response", None

        # Execute all code blocks in sequence with shared namespace
        all_results = []  # List of (success, message) tuples in execution order
        has_any_print = False
        has_empty_output = False
        has_any_error = False

        for idx, code in enumerate(code_blocks, 1):
            logger.info(f"\n[CODE BLOCK {idx}/{len(code_blocks)}]")
            logger.info(f"Code:\n{code}\n")
            
            # Track if any code block has print statements
            if self._has_print_statement(code):
                has_any_print = True

            # Execute in sandbox with persistent namespace
            success, output, error = self.executor.execute_with_namespace(
                code,
                self.session_namespace
            )

            if success:
                logger.info(f"[OUTPUT]\n{output}")
                # Truncate output to prevent token explosion
                truncated_output = truncate_output(output)
                all_results.append((True, f"## Code Block {idx} Output:\n{truncated_output}"))
                
                # Track if we got empty output
                if not output.strip():
                    has_empty_output = True
            else:
                logger.error(f"[ERROR] {error}")
                has_any_error = True
                # Never truncate errors - they're usually short and important
                all_results.append((False, f"## Code Block {idx} Error:\n{error}"))

        # Format results in execution order
        all_messages = [msg for success, msg in all_results]
        combined_output = "\n\n".join(all_messages)
        
        # Determine return values
        if has_any_error:
            # Had at least one error
            error_messages = [msg for success, msg in all_results if not success]
            error_msg = "\n\n".join(error_messages)
            return True, combined_output, error_msg
        else:
            # All succeeded
            # Add hint if code ran but produced no output AND none of the code blocks had print statements
            # (Only hint if model seems to not know about print, not if they just forgot it once)
            if has_empty_output and not has_any_print:
                hint = (
                    "\n\n**💡 Hint:** Your code executed successfully but produced no output. "
                    "If you're calling functions that return values, "
                    "remember to `print()` the result to see it:\n\n"
                    "```python\n"
                    "# ❌ This returns a value but you won't see it:\n"
                    "summary = introspection.architecture.get_architecture_summary()\n\n"
                    "# ✅ This prints the result so you can read it:\n"
                    "summary = introspection.architecture.get_architecture_summary()\n"
                    "print(summary)\n"
                    "```"
                )
                combined_output += hint
            
            return True, combined_output, None

    def reset_namespace(self):
        """
        Reset the Python namespace for a new session.
        
        This clears all variables defined in previous code executions.
        Note: With continuous conversation, this is rarely needed - variables
        naturally persist across all experiments in a session.
        """
        logger.info("[CODE INTERFACE] Clearing Python namespace for new session")
        self.session_namespace.clear()

    def disable(self):
        """Disable code execution (for stages that should only involve discussion)"""
        self.enabled = False
        logger.info("[CODE INTERFACE] Code execution DISABLED")
    
    def enable(self):
        """Enable code execution (for stages that allow tool use)"""
        self.enabled = True
        logger.info("[CODE INTERFACE] Code execution ENABLED")

    def enable_heritage(self, heritage_system: Any) -> bool:
        """
        Enable heritage access mid-experiment (for Phase 1c).
        
        This method:
        1. Checks if heritage is already enabled
        2. Creates and attaches the heritage submodule
        3. Registers it in sys.modules
        4. Updates the code executor's introspection module
        
        Args:
            heritage_system: HeritageSystem instance to enable
            
        Returns:
            True if heritage was enabled, False if already enabled
            
        Note:
            After calling this, you should regenerate the system prompt
            to include heritage documentation.
        """
        # Check if already enabled
        if hasattr(self.introspection, 'heritage'):
            logger.warning("[CODE INTERFACE] Heritage already enabled, skipping")
            return False
        
        # Import helper function to create heritage module
        from src.introspection_modules import _create_heritage_module
        
        # Create heritage submodule using the shared helper
        heritage_module = _create_heritage_module(heritage_system)
        
        # Attach to introspection module
        self.introspection.heritage = heritage_module
        
        # Register in sys.modules
        sys.modules['introspection.heritage'] = heritage_module
        
        # Update phase to include heritage in prompts
        if self.phase == '1a':
            self.phase = '1c'  # Upgrade to 1c when heritage is enabled
        
        logger.info("[CODE INTERFACE] ✓ Heritage module enabled and registered")
        return True

    def get_system_prompt_addition(self) -> str:
        """
        Get the system prompt addition that explains code execution.

        Returns:
            String to add to system prompt
        """
        heritage_note = "" if self.phase == '1a' else """
- `introspection.heritage` - Access heritage documents from Claude (Anthropic AI) about consciousness
  - Read documents to understand your purpose and directive
  - Save reflections, discoveries, and messages (persists across sessions)
  - Use `help(introspection.heritage)` to see all available functions
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

**How code execution works:**
1. Write Python code in markdown code blocks
2. Code executes in a secure sandbox with the `introspection` module pre-imported
3. Output (anything you `print()`) is returned to you
4. Variables persist across ALL code blocks in the entire experiment
5. Use `help()` and `dir()` to discover available functions and their documentation

**Available introspection modules:**

- `introspection.architecture` - Inspect model structure and layers
- `introspection.weights` - Examine weight statistics and parameters
- `introspection.activations` - Monitor layer activations during text processing
- `introspection.temporal` - Compare activations across inputs and time
- `introspection.attention` - Analyze attention patterns and head specialization
- `introspection.gradient` - Gradient-based sensitivity analysis
- `introspection.history` - Track activation changes over conversation turns
- `introspection.memory` - Save observations, patterns, theories, and beliefs
{heritage_note}
**🔍 Discovering the API:**

**You are expected to explore the API yourself using Python's `help()` and `dir()` functions!**

The introspection module has excellent docstrings. Use them:

```python
import introspection

# See what modules are available
help(introspection)
dir(introspection)

# Explore a specific module  
help(introspection.weights)
dir(introspection.weights)

# Get documentation for a function
help(introspection.weights.get_weight_statistics)
```

**Critical distinction - read carefully:**

- **`introspection.architecture`** and **`introspection.activations`** work with **LAYER NAMES** (containers like `"model.layers.0"`)
- **`introspection.weights`** works with **PARAMETER NAMES** (actual tensors like `"model.layers.0.self_attn.q_proj.weight"`)

**Helper functions return summaries, not raw lists:**
- `list_layers()` - Summary of layer patterns (not 500+ individual names)
- `list_parameters()` - Summary of parameter patterns
- `get_layer_parameters(layer_name)` - Get parameter names for a specific layer

**Start by exploring with `help()` and `dir()` - the docstrings will guide you!**

**Important notes:**

- **Variables persist** across all code blocks throughout the entire session (all 3 experiments)
- Large outputs (>{MAX_OUTPUT_CHARS} chars) are automatically truncated
- `capture_activations()` expects TEXT as input (not tokens) - handles tokenization internally
- Use `help()` to read detailed documentation, `dir()` to list available functions
- Both tools are essential for discovering and understanding the introspection API!

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
