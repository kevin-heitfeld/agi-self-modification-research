# Output Truncation System

## Problem

During Phase 1a experiments, the model would sometimes print massive outputs from code execution, causing:
- Token count explosion (e.g., 500+ layer names = ~15K characters)
- Out of Memory (OOM) errors on iteration 5
- Context window overflow in long experiments

Example from `phase1a_research.log`:
```python
layers = introspection.architecture.list_layers()
print(f"List of Layers:\n{layers}")
```

This returned 500+ layer names, consuming massive context space in the next prompt.

## Solution: Smart Truncation with Type Awareness

Implemented Option 1+3: Intelligent truncation that:
1. **Detects output type** (list, dict, or plain text)
2. **Truncates smartly** based on type
3. **Preserves useful information** while preventing OOM

### Configuration

```python
MAX_OUTPUT_CHARS = 2000  # Maximum characters to show
MAX_LIST_ITEMS = 10      # First N items to show from lists
MAX_DICT_ITEMS = 10      # First N items to show from dicts
```

### Truncation Strategies

#### 1. Lists (>10 items)
Shows: first 10 items + omission count + last 2 items

```
[List with 500 items, showing first 10 and last 2]
['lm_head', 'model', 'model.embed_tokens', ...]
... 485 items omitted ...
[..., 'model.norm', 'model.rotary_emb']
```

#### 2. Dictionaries (>10 items)
Shows: first 10 items + omission count

```
[Dict with 100 items, showing first 10]
{'layer_0': {...}, 'layer_1': {...}, ...}
... 90 items omitted ...
}
```

#### 3. Long Text (>2000 chars)
Shows: first 1000 chars + truncation notice + last 1000 chars

```
Beginning of output...

[... Output truncated: 15234 characters total (500 lines), showing first and last 1000 chars ...]

...end of output
```

#### 4. Short Output (≤2000 chars)
**No truncation** - passes through unchanged

### Implementation

**File:** `src/code_execution_interface.py`

**Function:** `truncate_output(output: str, max_chars: int = 2000) -> str`

**Integration:**
```python
# In execute_response()
if success:
    logger.info(f"[OUTPUT]\n{output}")
    # Truncate output to prevent token explosion
    truncated_output = truncate_output(output)
    all_outputs.append(f"## Code Block {idx} Output:\n{truncated_output}")
```

**System Prompt Addition:**
```
**Output truncation:**
- Large outputs (>2000 chars) are automatically truncated to prevent memory issues
- Lists/dicts with many items show first 10 items + last 2 items + count
- For long outputs, you'll see beginning and end with "[... Output truncated ...]" notice
- **Strategy:** Check size first (e.g., `len(layers)`) before printing large collections
- **Better approach:** Print counts/summaries rather than full lists
  - ❌ Bad: `print(introspection.architecture.list_layers())` (500+ items!)
  - ✅ Good: `layers = introspection.architecture.list_layers(); print(f"Found {len(layers)} layers"); print(layers[:5])`
```

## Testing

**File:** `tests/test_output_truncation.py`

12 comprehensive tests covering:
- Short output unchanged ✓
- Empty output handling ✓
- Exact limit handling ✓
- Long text truncation ✓
- List truncation ✓
- Dict truncation ✓
- Small collections unchanged ✓
- Invalid syntax fallback ✓
- Multiline output ✓
- Structure preservation ✓
- **Real scenario: 500+ layer list** ✓
- Activation stats scenario ✓

**All tests passing:** 12/12 ✓

## Demonstration

**File:** `scripts/demos/demo_output_truncation.py`

Run: `python scripts/demos/demo_output_truncation.py`

Shows:
- Original OOM scenario (82.2% reduction)
- Large dictionary truncation (79.7% reduction)
- Long text truncation (96.9% reduction)
- Small output unchanged
- Before/after comparison (98.4% reduction in worst case)

## Results

### Performance Improvements

| Scenario | Before | After | Reduction |
|----------|--------|-------|-----------|
| Layer list (500 items) | 11,766 chars | 2,099 chars | 82.2% |
| Dict stats (100 items) | 10,463 chars | 2,129 chars | 79.7% |
| Long text (1000 lines) | 67,889 chars | 2,102 chars | 96.9% |
| Small output (<2000 chars) | 51 chars | 51 chars | 0% (unchanged) |

### Token Savings

- **Per execution**: ~9,000 tokens saved (worst case)
- **Per experiment (20 iterations)**: ~180,000 tokens saved
- **Context safety**: Prevents overflow in long experiments
- **OOM prevention**: Eliminates token explosion errors

### Model Behavior

The model receives:
- ✓ Enough information to understand what happened
- ✓ Counts and structure metadata
- ✓ Sample items from beginning and end
- ✓ Clear truncation notices

This allows the model to:
- Understand collection sizes without seeing all items
- Adjust strategy (e.g., query specific items instead)
- Continue experiments without hitting OOM

## Best Practices for Model

The system prompt now teaches the model to:

**❌ Avoid:**
```python
layers = introspection.architecture.list_layers()
print(layers)  # 500+ items!
```

**✅ Prefer:**
```python
layers = introspection.architecture.list_layers()
print(f"Found {len(layers)} layers")
print(f"First 5: {layers[:5]}")
print(f"Last 5: {layers[-5:]}")
```

## Files Changed

1. **`src/code_execution_interface.py`**
   - Added `truncate_output()` function (100 lines)
   - Modified `execute_response()` to apply truncation
   - Updated system prompt with truncation guidance

2. **`tests/test_output_truncation.py`** (new)
   - 12 comprehensive tests
   - Integration tests with real scenarios

3. **`scripts/demos/demo_output_truncation.py`** (new)
   - Interactive demonstration
   - Shows all truncation strategies
   - Performance metrics

4. **`.gitignore`**
   - Added `rclone_access_tokens.txt` to ignore list

## Commit

```
ab9cb89 - Add intelligent output truncation to prevent OOM from code execution
```

## Future Enhancements

Potential improvements:
- [ ] Configurable limits per experiment
- [ ] Progressive truncation (reduce limit as conversation grows)
- [ ] Smart detection of important vs. redundant data
- [ ] Compression for repeated patterns
- [ ] Model feedback loop (learn what to truncate based on what model queries)

## References

- Issue: OOM error in `data/logs/phase1a_research.log` at iteration 5
- Root cause: Model printed 500+ layer names from `list_layers()`
- Solution: Smart truncation preserves info while preventing OOM
- Status: ✅ Implemented, tested, and deployed
