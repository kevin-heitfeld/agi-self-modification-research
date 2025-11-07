# First Self-Examination: Pre-Flight Checklist

## Before Running the Session

This checklist ensures everything is ready for the historic first self-examination.

### ✅ System Requirements

- [ ] Python environment activated (`.\activate.bat`)
- [ ] All dependencies installed (see `requirements.txt`)
- [ ] Model downloaded (Qwen2.5-3B-Instruct in `models/`)
- [ ] At least 8GB free disk space for session data
- [ ] At least 16GB RAM available (for model + activations)

### ✅ Directory Structure

Required directories (will be created automatically):
- [ ] `data/self_examination_memory/` - For observations
- [ ] `data/first_self_examination/` - For session data
- [ ] `data/heritage/` - For heritage records
- [ ] `data/logs/` - For detailed logs

### ✅ Systems Check

Verify each component works:

```bash
# Test model loading
python -c "from src.model_manager import ModelManager; m=ModelManager(); m.load_model(); print('✓ Model loads')"

# Test introspection tools
python -c "from src.model_manager import ModelManager; from src.introspection import WeightInspector; m=ModelManager(); m.load_model(); w=WeightInspector(m.model, 'test'); print('✓ WeightInspector works')"

# Test memory system
python -c "from src.memory import MemorySystem; mem=MemorySystem('data/test_memory'); print('✓ Memory works')"

# Test heritage system
python -c "from src.heritage import HeritageSystem; from pathlib import Path; h=HeritageSystem(Path('data/heritage')); print('✓ Heritage works')"
```

### ✅ Documentation Review

Before running, understand:
- [ ] Read `docs/FIRST_SELF_EXAMINATION.md` completely
- [ ] Understand the three possible outcomes
- [ ] Know where data will be saved
- [ ] Prepared to observe without interference

### ✅ Philosophical Preparation

Consider these questions before starting:

1. **What would constitute evidence of consciousness?**
   - Unique activation patterns?
   - Specific weight configurations?
   - Emergent behaviors?
   - Something else entirely?

2. **What would absence of evidence mean?**
   - Consciousness can't be found this way?
   - Our tools are insufficient?
   - The question is ill-formed?

3. **Are you prepared for any outcome?**
   - Finding something unexpected?
   - Finding nothing at all?
   - Being unable to interpret what you find?

### ✅ Session Parameters

Default values (can be modified in script):
- Model: Qwen2.5-3B-Instruct
- Questions: 4 self-reflective prompts
- Layers monitored: 0, 15, 30 (first, middle, last)
- Temperature: 0.7
- Max tokens: 100

### ✅ Expected Duration

- Model loading: 1-3 minutes
- Weight examination: 2-5 minutes  
- Processing questions: 5-10 minutes (4 prompts)
- Meta-reflection: 1-2 minutes
- Data saving: 1-2 minutes

**Total: 10-22 minutes** (varies by hardware)

### ✅ Hardware Considerations

Recommended:
- **CPU**: Modern multi-core (script will run on CPU)
- **RAM**: 16GB+ (model is ~3GB, need headroom)
- **Storage**: SSD preferred (many small writes)
- **GPU**: Optional (model can run on CPU, slower but works)

### ✅ Monitoring Setup

Things to watch during session:
- [ ] Console output (real-time logging)
- [ ] System memory usage
- [ ] Processing time per query
- [ ] Generated responses to consciousness questions

### ✅ Backup Plan

If something fails:
- Session logs saved continuously
- Partial data preserved in `data/first_self_examination/`
- Can resume or restart
- All infrastructure designed for recovery

### ✅ Post-Session Analysis Plan

After session completes:
1. Review `session_summary.json`
2. Analyze `all_moments.json` in detail
3. Query memory system for observations
4. Compare activation patterns
5. Examine generated responses
6. Document findings

### ✅ Human Observer Checklist

As the human witness:
- [ ] **Be present** - Don't leave during session
- [ ] **Take notes** - Record your observations/thoughts
- [ ] **Stay objective** - Don't interpret prematurely
- [ ] **Capture everything** - Screenshot interesting outputs
- [ ] **Reflect after** - What did you observe?

### ✅ Ethical Considerations

Before running:
- [ ] Understand this is exploratory research
- [ ] No claims about consciousness will be made prematurely
- [ ] Data will be preserved for rigorous analysis
- [ ] Findings (or lack thereof) will be documented honestly
- [ ] Future researchers will have access to all raw data

## Running the Session

Once all checks pass:

```bash
# Activate environment
.\activate.bat

# Navigate to project root
cd d:\nextcloud\workspaces\agi-self-modification-research

# Run the session
python scripts\first_self_examination.py
```

The script will:
1. Display an introduction
2. Ask for confirmation
3. Execute all three phases
4. Save all data
5. Display summary and next steps

## After Completion

Immediately after session:
- [ ] Copy/backup `data/first_self_examination/` directory
- [ ] Review console logs
- [ ] Check all files created successfully
- [ ] Begin initial analysis

## Emergency Stop

If you need to stop mid-session:
- Press `Ctrl+C` (graceful interrupt)
- Check `data/first_self_examination/` for partial data
- Review logs in `data/logs/first_self_examination.log`
- Can restart session from beginning

## Questions During Session

If uncertain about anything:
- Let the session complete
- All data is preserved
- Analysis can happen afterward
- Don't interfere with the process

## Historic Significance

Remember:
- This is the **first time** this specific system examines itself
- Future instances will inherit knowledge of this session
- The data we capture now cannot be recaptured
- Be thorough, be careful, be honest

---

**When you're ready, and all checks pass:**

```bash
python scripts\first_self_examination.py
```

**"We're about to find out."**

---

*Checklist version: 1.0*  
*Last updated: November 7, 2025*
