"""
Test script for code block highlighting in colored output

Demonstrates how code blocks appear in model responses.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import logging
from scripts.experiments.phase1_base import ColoredFormatter

# Setup logging with colored formatter
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setFormatter(ColoredFormatter())
logger.addHandler(console_handler)

# Test data
model_responses = [
    # Response with single code block
    """[MODEL] Let me examine my architecture using code:

```python
import introspection
summary = introspection.architecture.get_architecture_summary()
print(f"Total layers: {summary['total_layers']}")
print(f"Model type: {summary['model_type']}")
```

This will help us understand the model structure.""",
    
    # Response with multiple code blocks
    """[MODEL] I'll investigate this in two steps:

First, let's check the layer count:

```python
layers = introspection.architecture.list_layers()
print(f"Found {len(layers)} layers")
```

Then examine a specific layer:

```py
layer_info = introspection.architecture.get_layer_info("model.layers.0")
print(layer_info)
```

Now we can analyze the results.""",

    # Response with inline code (should not be affected)
    "[MODEL] I think `introspection.memory.record_observation()` would be useful here. Let me try it.",
    
    # Response without code
    "[MODEL] Based on the previous observations, I believe the attention patterns show strong locality in the first few layers.",
]

def test_code_highlighting():
    """Test code block highlighting in various scenarios"""
    print("\n" + "=" * 80)
    print("CODE BLOCK HIGHLIGHTING TEST")
    print("=" * 80 + "\n")
    
    for i, response in enumerate(model_responses, 1):
        print(f"\n{'─' * 80}")
        print(f"Test Case {i}:")
        print('─' * 80)
        logger.info(response)
        print()
    
    print("\n" + "=" * 80)
    print("LEGEND:")
    print("=" * 80)
    print("- Code blocks have light gray background (black text)")
    print("- Standard markdown backticks (```) used for code blocks")
    print("- Regular text appears normally")
    print("- [MODEL] tag is in bold blue")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    test_code_highlighting()
