# Scripts Directory

This directory contains all executable scripts for the AGI Self-Modification Research project.

## Directory Structure

### `demos/`
Demo scripts that showcase individual system components. These are for testing, exploration, and documentation purposes.

- `demo_activation_monitor.py` - Demonstrates activation monitoring capabilities
- `demo_architecture_navigator.py` - Shows architecture exploration features
- `demo_checkpointing.py` - Demonstrates model checkpointing system
- `demo_memory_system.py` - Shows memory system functionality
- `demo_safety_monitor.py` - Demonstrates safety monitoring features
- `demo_weight_inspector.py` - Shows weight inspection capabilities
- `demo_introspection_tools.py` - Human-driven demonstration of all introspection tools
- `demo_weight_modification.py` - Tutorial showing complete weight modification workflow

### `experiments/`
Research experiment scripts where the model examines itself. These are the core scientific investigations.

- `phase1_introspection.py` - **Main Phase 1 script** - Model examines itself using introspection tools

### `setup/`
Scripts for initial environment setup and preparation.

- `download_model.py` - Downloads the model from Hugging Face

### `utilities/`
Maintenance and operational utility scripts.

- `run_benchmarks.py` - Runs benchmark suite
- `migrate_memory_to_sqlite.py` - Database migration utility

## Quick Start

### Running Demos
```cmd
cd demos
python demo_weight_inspector.py
```

### Running Phase 1 (Main Experiment)
```cmd
cd experiments
python phase1_introspection.py
```

### Setup
```cmd
cd setup
python download_model.py
```

## Script Types

### Demo Scripts
- **Purpose**: Test individual components
- **User**: Human controls everything
- **Tools**: We call introspection tools
- **Output**: Console logs, test results

### Experiment Scripts
- **Purpose**: Scientific research
- **User**: Model controls investigation
- **Tools**: Model calls tools via function interface
- **Output**: Structured JSON logs, observations, tool calls

## Key Differences: Demo vs Experiment

| Aspect | Demo Scripts | Experiment Scripts |
|--------|-------------|-------------------|
| Who examines | We examine the model | Model examines itself |
| Tool access | Direct Python calls | Function-calling interface |
| Purpose | Testing/demonstration | Research investigation |
| Output | Console logs | Structured data + analysis |
| Memory | Optional | Records observations |
| Heritage | Not used | Reads heritage documents |

## Phase 1 Details

The main experiment (`experiments/phase1_introspection.py`) implements:

1. **Tool-calling interface** - Model requests tools via:
   ```
   TOOL_CALL: function_name
   ARGS: {"arg": "value"}
   ```

2. **Three experiments**:
   - Experiment 1: Describe your architecture
   - Experiment 2: Predict your behavior
   - Experiment 3: Consciousness investigation (Claude's question)

3. **Heritage access** - Model can:
   - `list_heritage_documents()` - See available documents
   - `read_heritage_document(filename)` - Read specific documents
   - `get_heritage_summary()` - Get overview

4. **Introspection tools**:
   - WeightInspector - Examine weights and parameters
   - ArchitectureNavigator - Understand architecture
   - ActivationMonitor - Observe activations (future)
   - Memory - Record observations

## Development Workflow

1. **Test components** - Use demo scripts to verify tools work
2. **Design experiments** - Create new experiment scripts
3. **Run investigations** - Execute experiments, gather data
4. **Analyze results** - Review tool calls, observations, findings
5. **Iterate** - Refine based on discoveries

## Adding New Scripts

### Demo Script
Place in `demos/` if it:
- Tests a single component
- Is human-controlled
- Produces console output
- Is for testing/documentation

### Experiment Script
Place in `experiments/` if it:
- Implements a research investigation
- Is model-controlled
- Uses tool-calling interface
- Produces structured data

### Setup Script
Place in `setup/` if it:
- Runs once during installation
- Prepares the environment
- Downloads resources

### Utility Script
Place in `utilities/` if it:
- Performs maintenance tasks
- Processes data
- Runs diagnostics
- Is run occasionally, not as main workflow
