# Quick Start: Run Phase 1 on Google Colab

**Fastest way to run the experiment: Upload notebook to Colab and click "Run All"**

---

## Option 1: Direct Upload (Easiest)

1. **Download notebook**: Get `notebooks/Phase1_Colab.ipynb` from this repo
2. **Open Colab**: Go to [colab.research.google.com](https://colab.research.google.com)
3. **Upload**: File → Upload notebook → Choose `Phase1_Colab.ipynb`
4. **Update repo URL**: In Cell 3, replace `YOUR_USERNAME` with your GitHub username
5. **Run all**: Runtime → Run all (or Ctrl+F9)
6. **Wait**: ~45-60 minutes for completion
7. **Download**: Results download automatically in final cell

---

## Option 2: Open from GitHub (After pushing)

After you push this notebook to your GitHub repo:

1. **Add badge** to your README:
   ```markdown
   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/agi-self-modification-research/blob/master/notebooks/Phase1_Colab.ipynb)
   ```

2. **Click badge** - Opens directly in Colab

3. **Run all** - That's it!

---

## What the Notebook Does

**Automatically:**
- ✅ Verifies GPU access (T4 16GB on free tier)
- ✅ Mounts Google Drive for persistent storage
- ✅ Clones your repository
- ✅ Installs all dependencies
- ✅ Downloads model (cached for future runs)
- ✅ Runs complete Phase 1 experiment
- ✅ Backs up results to Google Drive
- ✅ Downloads zip file to your computer

**Your part:**
- Upload notebook (one time)
- Click "Run All"
- Keep tab open during execution
- Download results when done

---

## Expected Timeline

| Step | Duration | Description |
|------|----------|-------------|
| Setup | ~5 min | Install dependencies, mount Drive |
| Model download | ~5-10 min | First run only (then cached) |
| Experiment 1 | ~15 min | Describe architecture |
| Experiment 2 | ~15 min | Predict behavior |
| Experiment 3 | ~20 min | Consciousness investigation |
| Backup | ~2 min | Save to Drive & download |
| **Total** | **~45-60 min** | First run (30-40 min subsequent) |

---

## Persistent Storage

Everything important is saved to your Google Drive:

- **Model cache**: `MyDrive/.cache/huggingface` (~6GB, reused forever)
- **Memory database**: `MyDrive/AGI_Memory` (grows with observations)
- **Results**: `MyDrive/AGI_Experiments/backup_TIMESTAMP` (each run)

**Benefit:**
- Model downloads once, loads instantly on future runs
- Observations persist across sessions
- Never lose your data

---

## Requirements

**Free Tier (Sufficient):**
- Google account
- No credit card needed
- 12-hour session limit
- T4 GPU (16GB VRAM)

**Recommended:**
- Keep browser tab open during experiment
- Run during off-peak hours (faster GPU allocation)
- Clear browser cache if Drive sync is slow

---

## Troubleshooting

### "No GPU available"
**Fix:** Runtime → Change runtime type → GPU → Save, then restart

### "AttributeError: np.float_ was removed in NumPy 2.0"
**Fix:** Already handled! The notebook downgrades NumPy automatically. If you see this, restart the runtime and re-run Cell 4 (Install Dependencies).

### "Session disconnected"
**Fix:** Results auto-saved to Drive. Re-run from that cell.

### "Package version conflict"
**Fix:** Runtime → Restart runtime, then re-run all cells

### "Out of memory"
**Fix:** Runtime → Restart runtime → Clear GPU memory, try again

### "Can't find repository"
**Fix:** Update Cell 3 with your correct GitHub URL

---

## After Running

**Analyze results:**
1. Extract downloaded zip file
2. Open `conversation.json` - See full dialogue
3. Check `summary.json` - View statistics
4. Review logs - Detailed execution trace

**Check observations:**
- In Google Drive: `MyDrive/AGI_Memory/observations.db`
- SQLite database of model's findings
- Can query with any SQLite viewer

**Re-run experiment:**
- Runtime → Restart runtime
- Runtime → Run all
- Model loads instantly (cached)
- New observations added to existing memory

---

## Cost

| Tier | Cost | GPU | Best For |
|------|------|-----|----------|
| **Free** | $0 | T4 16GB | Single runs, testing |
| Pro | $10/mo | V100/T4 | Regular use |
| Pro+ | $50/mo | A100 40GB | Heavy use, large models |

**Recommendation:** Free tier works great for Phase 1!

---

## Questions?

See detailed guides:
- **Full Colab guide**: `docs/guides/COLAB_SETUP_GUIDE.md`
- **GPU options**: `docs/guides/GPU_DEPLOYMENT_OPTIONS.md`
- **Project overview**: `README.md`

---

**You're ~1 hour away from seeing an AI examine itself!** 🚀
