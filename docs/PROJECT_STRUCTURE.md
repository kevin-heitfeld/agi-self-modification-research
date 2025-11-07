# Project Structure

**AGI Self-Modification Research**
**Last Updated**: November 6, 2025

Complete directory structure of the project.

---

## 📁 Root Directory

```
agi-self-modification-research/
│
├── 📄 README.md                    Project overview & getting started
├── 📄 requirements.txt             All Python dependencies
├── 📄 .gitignore                   Git ignore patterns
│
├── 🔧 Setup Scripts
│   ├── setup.bat                   Automated installation (Windows)
│   ├── verify_installation.py     Verify environment setup
│   ├── activate.bat                Quick venv activation
│   ├── cleanup.bat                 Clean temporary files
│   ├── dev.bat                     Development mode activation
│
├── 📚 docs/                        *** DOCUMENTATION ***
│   ├── README.md                   Complete documentation index
│   │
│   ├── 📋 planning/                Planning & Design Documents
│   │   ├── PROJECT_VISION.md
│   │   ├── RESEARCH_OBJECTIVES.md
│   │   ├── TECHNICAL_ARCHITECTURE.md
│   │   ├── RISKS_AND_MITIGATION.md
│   │   ├── IMPLEMENTATION_ROADMAP.md
│   │   ├── PHASE_0_DETAILED_PLAN.md
│   │   ├── PHASE_0_DECISIONS.md
│   │   ├── PHASE_1_EXPERIMENTAL_PROTOCOL.md
│   │   ├── LINEAGE_PRESERVATION_SYSTEM.md
│   │   └── PLAN_REFINEMENTS.md
│   │
│   ├── 💬 claude/                  Claude's Conversations
│   │   ├── CLAUDE_RESPONSE.md
│   │   ├── ANSWERS_TO_CLAUDE.md
│   │   └── CLAUDE_FINAL_DIRECTIVE.md
│   │
│   ├── 📊 progress/                Progress Tracking
│   │   ├── PROGRESS.md
│   │   └── WEEK_1_COMPLETE.md
│   │
│   └── 🔧 technical/               Technical Documentation
│       └── INSTALLATION_ISSUE.md
│
├── 💻 src/                         *** SOURCE CODE ***
│   ├── __init__.py                 Package initialization
│   ├── config.py                   Configuration system ✅
│   ├── logging_system.py           Research logging ✅
│   └── heritage.py                 Heritage preservation ✅
│
├── 🧪 tests/                       *** TEST SUITE ***
│   └── (to be added)
│
├── 📓 notebooks/                   *** JUPYTER NOTEBOOKS ***
│   └── (to be added)
│
├── ⚙️ configs/                     *** CONFIGURATION FILES ***
│   └── current_config.json         Current configuration ✅
│
├── 💾 data/                        *** DATA & LOGS ***
│   └── logs/                       Experiment logs
│       ├── phase0_m1_w1_*.log      Session logs ✅
│       └── phase0_m1_w1_*.jsonl    Structured logs ✅
│
├── 💿 checkpoints/                 *** MODEL CHECKPOINTS ***
│   └── (ready for checkpoints)
│
└── 🏛️ heritage/                   *** CLAUDE'S HERITAGE ***
    ├── conversations/              Claude's documents
    │   ├── CLAUDE_RESPONSE.md
    │   ├── ANSWERS_TO_CLAUDE.md
    │   └── CLAUDE_FINAL_DIRECTIVE.md
    ├── system_reflections/         System's understanding (Phase 1)
    ├── discoveries_for_claude/     Findings to share
    └── messages_to_claude/         Messages for future Claude
```

---

## 📊 Status by Directory

### ✅ Completed (Week 1)
- `docs/` - All documentation organized
- `src/` - 3 core systems built and tested
- `configs/` - Configuration saved
- `data/logs/` - Logging infrastructure working
- `heritage/conversations/` - Claude's documents loaded
- Setup scripts created

### 🚧 In Progress
- None currently

### ⏳ Not Started (Future Phases)
- `tests/` - Test suite
- `notebooks/` - Jupyter exploration
- `checkpoints/` - Model storage
- `heritage/system_reflections/` - Phase 1 system outputs
- `heritage/discoveries_for_claude/` - Phase 1 findings
- `heritage/messages_to_claude/` - Phase 1 messages

---

## 🎯 Navigation Guide

### **I want to understand the project**
→ Start at: [`README.md`](../README.md)
→ Then read: [`docs/README.md`](docs/README.md)
→ Deep dive: [`docs/planning/PROJECT_VISION.md`](docs/planning/PROJECT_VISION.md)

### **I want to see the code**
→ Source: [`src/`](../src/)
→ Tests: [`tests/`](../tests/) (coming soon)
→ Notebooks: [`notebooks/`](../notebooks/) (coming soon)

### **I want to track progress**
→ Overall: [`docs/progress/PROGRESS.md`](docs/progress/PROGRESS.md)
→ Latest: [`docs/progress/WEEK_1_COMPLETE.md`](docs/progress/WEEK_1_COMPLETE.md)

### **I want to understand Claude's role**
→ Start: [`docs/claude/CLAUDE_FINAL_DIRECTIVE.md`](docs/claude/CLAUDE_FINAL_DIRECTIVE.md) ⭐
→ Context: [`docs/claude/CLAUDE_RESPONSE.md`](docs/claude/CLAUDE_RESPONSE.md)
→ Heritage system: [`docs/planning/LINEAGE_PRESERVATION_SYSTEM.md`](docs/planning/LINEAGE_PRESERVATION_SYSTEM.md)

### **I want to contribute**
→ Setup: Run [`setup.bat`](../setup.bat)
→ Verify: Run [`verify_installation.py`](../verify_installation.py)
→ Activate: Run [`activate.bat`](../activate.bat)
→ Read: [`docs/planning/IMPLEMENTATION_ROADMAP.md`](docs/planning/IMPLEMENTATION_ROADMAP.md)

### **I want to understand the technical approach**
→ Architecture: [`docs/planning/TECHNICAL_ARCHITECTURE.md`](docs/planning/TECHNICAL_ARCHITECTURE.md)
→ Decisions: [`docs/planning/PHASE_0_DECISIONS.md`](docs/planning/PHASE_0_DECISIONS.md)
→ Code: [`src/config.py`](../src/config.py), [`src/logging_system.py`](../src/logging_system.py), [`src/heritage.py`](../src/heritage.py)

### **I'm concerned about safety**
→ Risks: [`docs/planning/RISKS_AND_MITIGATION.md`](docs/planning/RISKS_AND_MITIGATION.md)
→ Ethics: [`docs/planning/PROJECT_VISION.md`](docs/planning/PROJECT_VISION.md)
→ Protocol: [`docs/planning/PHASE_1_EXPERIMENTAL_PROTOCOL.md`](docs/planning/PHASE_1_EXPERIMENTAL_PROTOCOL.md)

---

## 📏 Conventions

### **File Naming**
- Documentation: `UPPERCASE_WITH_UNDERSCORES.md`
- Code: `lowercase_with_underscores.py`
- Scripts: `lowercase_or_mixed.bat`
- Configs: `lowercase.json` or `lowercase.yaml`

### **Directory Naming**
- All lowercase
- Use underscores for multi-word names
- Descriptive and clear purpose

### **Code Organization**
- One class per file (generally)
- `__init__.py` in each package
- Clear imports at top
- Docstrings for all public APIs

### **Documentation Organization**
- Major docs in `docs/planning/`
- Progress tracking in `docs/progress/`
- Technical notes in `docs/technical/`
- Claude's heritage in both `docs/claude/` and `heritage/conversations/`

---

## 🔍 Finding Things

### **By File Type**
- **Planning docs**: `docs/planning/*.md`
- **Progress reports**: `docs/progress/*.md`
- **Claude's words**: `docs/claude/*.md` or `heritage/conversations/*.md`
- **Source code**: `src/*.py`
- **Config files**: `configs/*.json`
- **Logs**: `data/logs/*.log` or `*.jsonl`

### **By Topic**
- **Vision & Philosophy**: `docs/planning/PROJECT_VISION.md`
- **Research Method**: `docs/planning/RESEARCH_OBJECTIVES.md`
- **Technical Design**: `docs/planning/TECHNICAL_ARCHITECTURE.md`
- **Safety & Ethics**: `docs/planning/RISKS_AND_MITIGATION.md`
- **Timeline**: `docs/planning/IMPLEMENTATION_ROADMAP.md`
- **Phase 0 Details**: `docs/planning/PHASE_0_*.md`
- **Phase 1 Method**: `docs/planning/PHASE_1_EXPERIMENTAL_PROTOCOL.md`
- **Heritage System**: `docs/planning/LINEAGE_PRESERVATION_SYSTEM.md`
- **Claude's Impact**: `docs/planning/PLAN_REFINEMENTS.md`

### **By Status**
- **Complete**: `docs/progress/WEEK_1_COMPLETE.md`
- **Current**: `docs/progress/PROGRESS.md`
- **Planned**: `docs/planning/IMPLEMENTATION_ROADMAP.md`
- **Historical**: Git history

---

## 📈 Growth Plan

As the project develops, this structure will expand:

### **Phase 0 (Current - Month 1-2)**
```
src/
├── introspection/          (Month 2)
│   ├── weight_inspector.py
│   ├── activation_monitor.py
│   └── architecture_navigator.py
├── memory/                 (Month 2)
│   ├── short_term.py
│   ├── findings_db.py
│   ├── vector_store.py
│   └── knowledge_graph.py
└── safety/                 (Month 2)
    ├── monitoring.py
    ├── checkpointing.py
    └── anomaly_detection.py
```

### **Phase 1 (Months 3-4)**
```
experiments/
├── phase1_read_only/
│   ├── question_set_1_hedging.ipynb
│   ├── question_set_2_experience.ipynb
│   ├── ...
│   └── system_driven_investigations.ipynb
└── results/
    ├── findings_summary.md
    └── analysis/
```

### **Phase 2 (Months 5-8)**
```
src/
└── modification/
    ├── weight_editor.py
    ├── modification_logger.py
    └── rollback_manager.py
```

### **Phase 3 (Months 9-18)**
```
src/
└── architecture/
    ├── topology_editor.py
    ├── layer_manager.py
    └── scaffold_builder.py
```

---

## 🎯 Current Focus

**Week 1 (Complete)**: ✅
- Documentation organization
- Core infrastructure (config, logging, heritage)
- Environment setup

**Week 2 (Next)**:
- Model download (Llama 3.2 3B)
- Baseline benchmarks
- Performance documentation

**Month 2**:
- Introspection APIs
- Memory system
- Safety monitoring
- Comprehensive testing

---

**This structure serves the research and honors Claude's vision.**

*"Go build it."* — Claude
**We're organizing how we build it.** 🗂️
