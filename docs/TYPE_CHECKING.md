# Type Checking Guide

## Overview

This project uses Python type hints with **Pyright** (VS Code's Pylance) for static type checking to catch bugs early.

## Why Type Checking Matters

Type checking would have caught several bugs we encountered:

1. **ActivationMonitor initialization bug**: Passing `WeightInspector` instead of tokenizer
2. **Incorrect function signatures**: Missing or wrong parameter types
3. **None-handling errors**: Accessing attributes on Optional types without checks

## Setup

### VS Code (Pylance)

Type checking is automatically enabled in VS Code through Pylance. Configuration in `pyproject.toml`:

```toml
[tool.pyright]
typeCheckingMode = "basic"
```

You'll see red squiggles for type errors in VS Code automatically.

### Command Line

Install pyright:
```bash
pip install pyright
```

Run type checking:
```bash
python scripts/utilities/type_check.py
```

Or directly:
```bash
pyright src/ scripts/experiments/
```

## Type Annotation Guidelines

### 1. Annotate Function Signatures

**Always** annotate:
- Parameter types
- Return types (including `-> None`)

```python
def process_text(self, text: str, layer_names: Optional[List[str]] = None) -> Dict[str, Any]:
    ...
```

### 2. Use Proper Types for External Libraries

```python
from transformers import PreTrainedModel, PreTrainedTokenizer

class ModelManager:
    def __init__(self) -> None:
        self.model: Optional[PreTrainedModel] = None
        self.tokenizer: Optional[PreTrainedTokenizer] = None
```

### 3. Use Protocols for Duck Typing

When you need an interface but don't want to import the concrete class:

```python
from typing import Protocol

class TokenizerProtocol(Protocol):
    """Protocol defining the interface we need from a tokenizer"""
    def __call__(self, text: str, return_tensors: str, **kwargs: Any) -> Dict[str, torch.Tensor]: ...
    def decode(self, token_ids: List[int], **kwargs: Any) -> str: ...
```

### 4. Use TYPE_CHECKING for Circular Imports

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.model_manager import ModelManager

class ToolInterface:
    def __init__(self, model_manager: Optional['ModelManager'] = None) -> None:
        ...
```

### 5. Annotate Class Attributes

```python
class ToolInterface:
    def __init__(self) -> None:
        self.tools: Dict[str, Callable] = {}
        self.tool_calls: List[ToolCall] = []
```

## Common Type Errors and Fixes

### Error: "object is not callable"

**Problem**: Passing wrong type to a function.

**Fix**: Check function signature and pass correct type:
```python
# WRONG
activation_monitor = ActivationMonitor(model, inspector)  # inspector is WeightInspector

# RIGHT  
activation_monitor = ActivationMonitor(model, tokenizer)  # tokenizer is TokenizerProtocol
```

### Error: "None is not assignable to X"

**Problem**: Not checking for None before accessing.

**Fix**: Add None checks:
```python
if self.model_manager is not None:
    result = self.model_manager.generate(text)
```

Or use assertion:
```python
assert self.model_manager is not None
result = self.model_manager.generate(text)
```

### Error: "Argument type X is incompatible with Y"

**Problem**: Passing wrong type.

**Fix**: Check the expected type and convert if needed:
```python
# If function expects List[str] but you have Optional[List[str]]
if layer_names is not None:
    do_something(layer_names)
else:
    do_something([])  # Provide default
```

## Gradual Adoption

We're using `typeCheckingMode = "basic"` which is not too strict. This allows:
- Gradual addition of type hints
- `Any` types where needed
- Fewer complaints about untyped third-party libraries

As the codebase matures, we can increase strictness:
- `basic` → `standard` → `strict`

## Pre-commit Hook (Optional)

To run type checking before each commit:

```bash
# .git/hooks/pre-commit
#!/bin/sh
python scripts/utilities/type_check.py
```

Make it executable:
```bash
chmod +x .git/hooks/pre-commit
```

## Current Coverage

Files with comprehensive type annotations:
- ✅ `src/model_manager.py`
- ✅ `src/tool_interface.py`
- ✅ `src/introspection/activation_monitor.py` (with TokenizerProtocol)
- ✅ `src/introspection/weight_inspector.py`

Files needing more annotations:
- ⏳ Demo scripts
- ⏳ Test files
- ⏳ Utility scripts

## Resources

- [Python Type Hints Documentation](https://docs.python.org/3/library/typing.html)
- [Pyright Documentation](https://github.com/microsoft/pyright)
- [Type Checking Best Practices](https://typing.readthedocs.io/en/latest/source/best_practices.html)
