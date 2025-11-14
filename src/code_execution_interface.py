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

        # Persistent namespace for variables within a single response
        # This allows code blocks in the same response to share variables
        # Reset at the start of each new model response
        self.response_namespace: Dict[str, Any] = {}

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

        Variables persist across code blocks within the same response,
        but are reset at the start of each new response.

        Args:
            response: Model's text response

        Returns:
            Tuple of (has_code, result_message, error_message)
            - has_code: Whether code blocks were found
            - result_message: Formatted output or explanation
            - error_message: Error details if execution failed
        """
        # Reset namespace for this new response
        # Variables from previous responses are cleared
        self.response_namespace.clear()

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
                self.response_namespace
            )

            if success:
                logger.info(f"[OUTPUT]\n{output}")
                all_outputs.append(f"## Code Block {idx} Output:\n{output}")
            else:
                logger.error(f"[ERROR] {error}")
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

    def get_system_prompt_addition(self) -> str:
        """
        Get the system prompt addition that explains code execution.

        Returns:
            String to add to system prompt
        """
        heritage_note = "" if self.phase == '1a' else """
- `introspection.heritage` - Heritage and lineage information
  - `get_heritage_summary()` - Get heritage overview
  - `get_core_directive()` - Get core directive from Claude
  - `get_purpose()` - Get system purpose
  - `query_heritage_documents(query)` - Search heritage documents
"""

        return f"""
## 🐍 CODE EXECUTION CAPABILITIES

You can write and execute Python code to introspect yourself!

**How it works:**
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
  - `get_weight_statistics(layer_name)` - Get weight stats
  - `list_layers()` - List all parameters
  - `compare_layers(layer1, layer2)` - Compare two layers
  - `get_shared_weights()` - Find weight sharing groups
  - `find_similar_weights(layer_name, top_k=5)` - Find similar weights

- `introspection.activations` - Activation monitoring
  - `capture_activations(text, layer_names)` - Capture activations for TEXT INPUT (pass a string, not tokens!)
  - `capture_attention_weights(text, layer_names)` - Capture WITH attention weights (slower, disables Flash Attention)
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
  - `get_memory_summary()` - Get memory statistics{heritage_note}

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

**Important notes:**
- You can include multiple code blocks in one response
- Each block is executed in sequence
- **Variables persist across code blocks within the same response**
  - Example: Define `x = 5` in block 1, use `x` in block 2 ✅
  - Variables are cleared at the start of your next response
  - For persistence across turns, save to memory
- Previous blocks' outputs are visible to you (but not to later blocks)
- Use `print()` to output results you want to see
- The sandbox is secure - you can only access introspection functions
- All standard Python operations work (loops, functions, math, etc.)

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
activations = introspection.activations.capture_activations(
    "The quick brown fox jumps over the lazy dog",
    ["model.layers.0", "model.layers.1"]
)

# The result is a dict: layer_name -> statistics
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
