# VS Code Configuration

This document explains how to configure VS Code for optimal development experience with this project.

## Quick Setup

1. **Copy the example configuration files:**
   ```bash
   # Windows PowerShell
   Copy-Item .vscode\settings.json.example .vscode\settings.json
   Copy-Item .env.example .env
   
   # Windows CMD
   copy .vscode\settings.json.example .vscode\settings.json
   copy .env.example .env
   
   # Linux/Mac
   cp .vscode/settings.json.example .vscode/settings.json
   cp .env.example .env
   ```

2. **Reload VS Code** to apply the settings:
   - Press `Ctrl+Shift+P` (Windows/Linux) or `Cmd+Shift+P` (Mac)
   - Type "Reload Window" and press Enter

## What These Settings Fix

### 1. Import Resolution Errors ❌ → ✅

**Before:** Scripts couldn't import from `src` folder
```python
from model_manager import ModelManager  # ❌ Unable to import
from introspection import WeightInspector  # ❌ Unable to import
```

**After:** All imports work correctly
```python
from model_manager import ModelManager  # ✅ Works!
from introspection import WeightInspector  # ✅ Works!
```

**How:** Added `src` to Python analysis extra paths

### 2. Excessive Pylint Warnings 🔴 → 🟢

**Disabled overly strict rules:**
- `C0103` - Naming conventions (allows short variable names like `i`, `x`, `id`, `db`)
- `C0114-C0116` - Missing docstrings (we have docstrings where needed)
- `W0703` - Catching broad `Exception` (needed for safety catches)
- `W1203` - Using f-strings in logging (more readable than lazy %)
- `W0612-W0613` - Unused variables/arguments (common in hooks/callbacks)
- `R0913-R0915` - Too many arguments/locals/statements (complex functions are OK)
- `R0801` - Duplicate code (some patterns repeat intentionally)

### 3. Test Discovery ✅

**Configured pytest integration:**
- Tests appear in VS Code Test Explorer
- Can run individual tests with play button
- Debugging works for tests

### 4. Python Environment 🐍

**Automatically activates the venv:**
- Terminal opens with virtual environment active
- Correct Python interpreter selected
- All dependencies available

## Key Settings Explained

### Python Analysis
```json
"python.analysis.extraPaths": ["${workspaceFolder}/src"]
```
Tells Pylance where to find the `src` modules for import resolution.

### Python Environment
```json
"python.defaultInterpreterPath": "${workspaceFolder}/venv/Scripts/python.exe"
```
Uses the project's virtual environment automatically.

### Testing
```json
"python.testing.pytestEnabled": true
```
Enables pytest test discovery and running in VS Code.

### Pylint Configuration
```json
"python.linting.pylintArgs": [
    "--disable=W1203",  // Allow f-strings in logging
    "--max-line-length=120",  // Allow longer lines
    ...
]
```
Disables overly strict rules while keeping important ones.

## File Exclusions

The configuration hides clutter from search and file explorer:
- `__pycache__/` - Python bytecode cache
- `.pytest_cache/` - Pytest cache
- `htmlcov/` - Coverage reports
- `venv/` - Virtual environment (large, not needed in search)
- `models/` - Downloaded models (very large)

## Troubleshooting

### Still seeing import errors?

1. **Reload VS Code window:**
   - `Ctrl+Shift+P` → "Reload Window"

2. **Check Python interpreter:**
   - Bottom-left corner should show "Python 3.11.x ('venv')"
   - If not, click it and select `./venv/Scripts/python.exe`

3. **Verify .env file:**
   - Should contain: `PYTHONPATH=${workspaceFolder}/src`

### Tests not appearing?

1. **Check pytest is installed:**
   ```bash
   pytest --version
   ```

2. **Reload tests:**
   - Click the testing beaker icon (left sidebar)
   - Click the refresh button

3. **Check output:**
   - View → Output → Select "Python Test Log" from dropdown

### Linting too strict or too loose?

Edit `.vscode/settings.json` and adjust the `pylintArgs`:
- To disable more rules: Add to `--disable=` list
- To enable more rules: Remove from `--disable=` list
- See: https://pylint.pycqa.org/en/latest/user_guide/messages/messages_overview.html

## Recommended VS Code Extensions

Install these for the best experience:

1. **Python** (`ms-python.python`) - Essential
2. **Pylance** (`ms-python.vscode-pylance`) - Language server
3. **Python Test Explorer** - Included with Python extension
4. **autoDocstring** (`njpwerner.autodocstring`) - Generate docstrings
5. **Better Comments** (`aaron-bond.better-comments`) - Highlight TODOs
6. **GitLens** (`eamodio.gitlens`) - Git supercharged

## Alternative: Minimal Configuration

If you prefer minimal linting, create a simpler `.vscode/settings.json`:

```json
{
    "python.analysis.extraPaths": ["${workspaceFolder}/src"],
    "python.defaultInterpreterPath": "${workspaceFolder}/venv/Scripts/python.exe",
    "python.linting.enabled": false  // Disable all linting
}
```

## Notes

- `.vscode/settings.json` and `.env` are in `.gitignore` (not committed)
- `.vscode/settings.json.example` and `.env.example` are templates (committed)
- You can customize your local settings without affecting others
- Settings apply workspace-wide (all open folders)

## Getting Help

If you're still seeing errors:

1. Check the **Output** panel: View → Output → Select "Python"
2. Check the **Problems** panel: View → Problems (`Ctrl+Shift+M`)
3. Check Pylance status: Bottom status bar shows "Pylance: Ready"
4. Restart VS Code: Sometimes needed for settings to fully apply

---

**After setup, you should see:**
- ✅ No import errors
- ✅ Minimal lint warnings (only important ones)
- ✅ Tests discoverable in Test Explorer
- ✅ Virtual environment active in terminal
- ✅ Fast code navigation and auto-completion
