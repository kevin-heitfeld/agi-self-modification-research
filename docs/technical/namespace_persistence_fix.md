# Python Namespace Persistence Fix

**Date:** November 15, 2025  
**Issue:** Model repeatedly getting `NameError` for variables like `sample_text` that it defined in previous iterations

## Problem

The model was defining variables in one iteration (e.g., iteration 8) and trying to use them in later iterations (e.g., iteration 10), but getting `NameError: name 'sample_text' is not defined`. This happened repeatedly throughout experiments.

**Root cause:** The Python namespace was being cleared at the start of **each response/iteration**, not just between experiments.

## Solution

Changed the namespace scope from **per-response** to **per-experiment**:

### Before
- Variables persisted only within a single response (across multiple code blocks in that response)
- Namespace cleared at the start of each new model response
- Model had to redefine variables in every iteration

### After
- Variables persist throughout the **entire experiment** (across all iterations)
- Namespace only cleared when `reset_namespace()` is called (between experiments)
- Model can define variables once and reuse them across iterations

## Changes Made

### 1. `src/code_execution_interface.py`

**Renamed namespace variable:**
```python
# Before
self.response_namespace: Dict[str, Any] = {}

# After
self.experiment_namespace: Dict[str, Any] = {}
```

**Removed namespace clearing in `execute_response()`:**
```python
# Before
def execute_response(self, response: str):
    # Reset namespace for this new response
    self.response_namespace.clear()  # ❌ Cleared every response
    ...

# After
def execute_response(self, response: str):
    # Don't clear namespace - variables persist across iterations!
    # Only cleared when reset_namespace() is called between experiments
    ...
```

**Added `reset_namespace()` method:**
```python
def reset_namespace(self):
    """
    Reset the Python namespace for a new experiment.
    
    This clears all variables defined in previous code executions.
    Called between experiments when the conversation context is reset.
    """
    logger.info("[CODE INTERFACE] Clearing Python namespace for new experiment")
    self.experiment_namespace.clear()
```

**Updated system prompt:**
```python
**Variables persist across ALL iterations in the same experiment**
  - Example: Define `sample_text = "Hello"` in iteration 1, use `sample_text` in iteration 5 ✅
  - Variables are preserved throughout the entire experiment
  - Variables are only cleared when the experiment ends and context is reset
  - This means you can define helper variables once and reuse them!
```

### 2. `scripts/experiments/phase1_base.py`

**Call `reset_namespace()` when resetting conversation:**
```python
def reset_conversation(self):
    """Reset conversation history for next experiment"""
    self.logger.info("[SYSTEM] Resetting conversation history")
    
    # Take snapshot before reset
    self.gpu_monitor.snapshot("before_reset")
    
    self.conversation_history = []
    self.conversation_kv_cache = None
    
    # Reset Python namespace for code execution  # ← NEW
    self.code_interface.reset_namespace()        # ← NEW

    # Aggressive memory cleanup
    gc.collect()
    ...
```

### 3. `src/code_executor.py`

**Added `__build_class__` to safe builtins:**
```python
safe_builtins = {
    ...
    # Class creation (required for defining classes)
    '__build_class__': __builtins__['__build_class__'],  # ← NEW
}
```

This allows the model to define classes in its code, which is useful for organizing investigation data.

### 4. `tests/test_namespace_persistence.py`

Created comprehensive tests covering:
- ✅ Variables persist across multiple executions
- ✅ Namespace cleared when `.clear()` is called (simulating experiment reset)
- ✅ Functions persist across executions
- ✅ Introspection imports persist
- ✅ Classes persist across executions

All 5 tests pass.

## Benefits

1. **Fewer errors:** Model doesn't get NameError for variables it defined earlier
2. **More efficient:** Model can define helper variables/functions once and reuse them
3. **Better code organization:** Model can define classes and helper functions
4. **Matches intuition:** Variables work like they do in a normal Python session

## Backwards Compatibility

This change is fully backwards compatible:
- Old experiments that redefined variables each iteration will continue to work
- New experiments can take advantage of persistent variables
- Memory is still cleaned between experiments (when conversation resets)

## Example Usage

**Before (had to redefine each time):**
```python
# Iteration 8
sample_text = "The quick brown fox"
print(len(sample_text))

# Iteration 10
sample_text = "The quick brown fox"  # Had to redefine! ❌
print(sample_text.upper())
```

**After (define once, use everywhere):**
```python
# Iteration 8
sample_text = "The quick brown fox"
print(len(sample_text))

# Iteration 10
print(sample_text.upper())  # Still available! ✅
```
