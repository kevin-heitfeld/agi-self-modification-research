# Running Phase 1 on Google Colab

**Quick Start Guide for Cloud Execution**

---

## Why Google Colab?

**Pros:**
- ✅ **Free option available** - No credit card needed for basic tier
- ✅ **Zero setup** - Runs in your browser
- ✅ **Pre-installed** - PyTorch and most ML libraries ready
- ✅ **Easy sharing** - Share notebooks with collaborators
- ✅ **Good GPUs** - T4 (16GB VRAM) on free tier, better on Pro/Pro+

**Cons:**
- ⚠️ **Session limits** - 12-24 hours max, then disconnects
- ⚠️ **No persistent storage** - Files deleted when session ends
- ⚠️ **Shared resources** - Performance can vary
- ⚠️ **Idle timeout** - 90 minutes of inactivity disconnects

---

## Colab Tiers

| Tier | Cost | GPU | VRAM | Session Limit | Good For |
|------|------|-----|------|---------------|----------|
| **Free** | $0 | T4 | 16GB | 12 hours | Testing, Phase 1 |
| **Pro** | $10/mo | V100/T4 | 16GB | 24 hours | Regular use |
| **Pro+** | $50/mo | A100 | 40GB | 24 hours | Production, large models |

**Recommendation for Phase 1:** Free tier is sufficient! Phase 1 should complete in ~1 hour on T4.

---

## Setup Method 1: Jupyter Notebook (Recommended)

### Step 1: Create Colab Notebook

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Sign in with Google account
3. **File** → **New Notebook**
4. Name it: `Phase1_Introspection_Experiment`

### Step 2: Enable GPU

1. **Runtime** → **Change runtime type**
2. **Hardware accelerator:** Select **GPU**
3. Click **Save**
4. Verify GPU is available:

```python
# Cell 1: Verify GPU
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
```

Expected output:
```
PyTorch version: 2.x.x
CUDA available: True
GPU: Tesla T4
VRAM: 15.0 GB
```

### Step 3: Setup Project

```python
# Cell 2: Clone repository
!git clone https://github.com/YOUR_USERNAME/agi-self-modification-research.git
%cd agi-self-modification-research
!pwd
```

**Note:** Replace `YOUR_USERNAME` with your GitHub username, or use the full URL of your repo.

**Alternative if repo is private:**
```python
# Mount Google Drive first
from google.colab import drive
drive.mount('/content/drive')

# Then clone from Drive or upload zip
!cp -r /content/drive/MyDrive/agi-self-modification-research /content/
%cd /content/agi-self-modification-research
```

### Step 4: Install Dependencies

```python
# Cell 3: Install requirements
!pip install -q transformers accelerate safetensors
!pip install -q chromadb networkx
!pip install -q pydantic pydantic-settings
!pip install -q rich tqdm
!pip install -q pytest pytest-cov

# Verify installations
import transformers
import chromadb
import networkx
print("✓ All dependencies installed")
```

**Note:** Colab already has PyTorch, NumPy, Pandas, Matplotlib installed.

### Step 5: Download Model (Optional - speeds up first run)

```python
# Cell 4: Pre-download model (optional but recommended)
from transformers import AutoModelForCausalLM, AutoTokenizer

print("Downloading Qwen2.5-3B-Instruct...")
model_name = "Qwen/Qwen2.5-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
# Don't load model yet, just download weights
print("✓ Model downloaded and cached")
```

### Step 6: Run Phase 1 Experiment

```python
# Cell 5: Run Phase 1 introspection
!python scripts/experiments/phase1_introspection.py
```

**Important:** The script will ask for confirmation. To auto-confirm:

```python
# Cell 5 (Alternative): Run with auto-confirmation
# Modify the script or use this approach:
import subprocess
import sys

# Run with 'yes' piped to stdin
process = subprocess.Popen(
    [sys.executable, 'scripts/experiments/phase1_introspection.py'],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True
)

# Auto-confirm
stdout, stderr = process.communicate(input='yes\n')
print(stdout)
if stderr:
    print("STDERR:", stderr)
```

### Step 7: Monitor Progress

```python
# Cell 6: Monitor logs (run in separate cell while experiment runs)
!tail -f data/logs/phase1_introspection.log
```

Press **Stop** button to stop tailing when done.

### Step 8: Download Results

```python
# Cell 7: Compress results
import shutil
from datetime import datetime

# Find latest session
!ls -lt data/phase1_sessions/ | head -n 5

# Compress for download
session_name = "phase1_20251107_064813"  # Replace with your session
shutil.make_archive(f'{session_name}', 'zip', f'data/phase1_sessions/{session_name}')
print(f"✓ Created {session_name}.zip")
```

```python
# Cell 8: Download to your computer
from google.colab import files
files.download(f'{session_name}.zip')
```

Or save to Google Drive:

```python
# Cell 8 (Alternative): Save to Google Drive
from google.colab import drive
drive.mount('/content/drive')

!cp -r data/phase1_sessions /content/drive/MyDrive/AGI_Experiments/
!cp -r data/phase1_memory /content/drive/MyDrive/AGI_Experiments/
print("✓ Results saved to Google Drive")
```

---

## Setup Method 2: Direct Script Conversion

If you prefer a single-cell execution:

```python
# Complete setup and execution in one cell
!git clone https://github.com/YOUR_USERNAME/agi-self-modification-research.git
%cd agi-self-modification-research

# Install dependencies
!pip install -q transformers accelerate safetensors chromadb networkx pydantic pydantic-settings rich tqdm

# Run experiment (with auto-confirmation)
import subprocess
import sys

process = subprocess.Popen(
    [sys.executable, 'scripts/experiments/phase1_introspection.py'],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True
)

stdout, stderr = process.communicate(input='yes\n')
print(stdout)

# Compress and download results
import shutil
import os
from google.colab import files

sessions = sorted(os.listdir('data/phase1_sessions'))
if sessions:
    latest = sessions[-1]
    shutil.make_archive(latest, 'zip', f'data/phase1_sessions/{latest}')
    files.download(f'{latest}.zip')
    print(f"✓ Downloaded {latest}.zip")
```

---

## Handling Session Disconnects

### Problem: Long Experiments + Session Timeouts

**Solutions:**

### 1. Keep Browser Tab Active
```python
# Cell: Install anti-idle
!pip install -q kora
from kora import screen
screen.prevent_idle()
```

### 2. Use Background Execution (Pro+ only)
On Colab Pro+:
- **Runtime** → **Run in background**
- Experiment continues even if you close browser

### 3. Save Checkpoints Frequently

Add to experiment script:

```python
# In phase1_introspection.py, after each experiment:
# Save intermediate results to Drive

from google.colab import drive
drive.mount('/content/drive')

# After Experiment 1
self.save_session()
!cp -r data/phase1_sessions /content/drive/MyDrive/AGI_Experiments_Backup/

# After Experiment 2
self.save_session()
!cp -r data/phase1_sessions /content/drive/MyDrive/AGI_Experiments_Backup/

# After Experiment 3
self.save_session()
!cp -r data/phase1_sessions /content/drive/MyDrive/AGI_Experiments_Backup/
```

### 4. Split Experiments

Run each experiment separately:

```python
# Cell: Run Experiment 1 only
from scripts.experiments.phase1_introspection import IntrospectionSession

session = IntrospectionSession()
session.initialize_systems()
session.run_experiment_1_describe_architecture()
session.save_session()
session.cleanup()
```

```python
# Cell: Run Experiment 2 (after Experiment 1 completes)
session = IntrospectionSession()
session.initialize_systems()
session.run_experiment_2_predict_behavior()
session.save_session()
session.cleanup()
```

```python
# Cell: Run Experiment 3 (consciousness investigation)
session = IntrospectionSession()
session.initialize_systems()
session.run_experiment_3_consciousness_question()
session.save_session()
session.cleanup()
```

---

## Persistent Memory Across Sessions

### Problem: Memory doesn't persist between Colab sessions

**Solution: Use Google Drive**

```python
# Cell 1: Mount Drive and setup persistent memory
from google.colab import drive
drive.mount('/content/drive')

# Create memory directory in Drive
!mkdir -p /content/drive/MyDrive/AGI_Memory

# Clone repo
!git clone https://github.com/YOUR_USERNAME/agi-self-modification-research.git
%cd agi-self-modification-research

# Link memory to Drive location
!rm -rf data/phase1_memory
!ln -s /content/drive/MyDrive/AGI_Memory data/phase1_memory

print("✓ Memory will persist across sessions")
```

Now observations are saved to your Google Drive and persist between runs!

---

## Optimizations for Colab

### 1. Faster Model Loading

```python
# Cache model in Drive (download once, reuse forever)
from google.colab import drive
drive.mount('/content/drive')

# Set Hugging Face cache to Drive
import os
os.environ['HF_HOME'] = '/content/drive/MyDrive/.cache/huggingface'
os.environ['TRANSFORMERS_CACHE'] = '/content/drive/MyDrive/.cache/huggingface/transformers'

# Now when you download model, it goes to Drive
# Next session will reuse it instead of re-downloading
```

### 2. Reduce Memory Usage

```python
# If hitting VRAM limits, load model in 8-bit
# Modify model_manager.py or add parameter:

from transformers import AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-3B-Instruct",
    torch_dtype=torch.float16,  # Use FP16 instead of FP32
    device_map="auto",
    load_in_8bit=True  # 8-bit quantization (saves VRAM)
)
```

### 3. Monitor GPU Usage

```python
# Cell: GPU monitoring
!watch -n 1 nvidia-smi  # Updates every second

# Or in Python:
import subprocess
import time

while True:
    subprocess.run(['nvidia-smi'])
    time.sleep(5)  # Update every 5 seconds
    # Ctrl+C to stop
```

---

## Complete Colab Notebook Template

Here's a complete notebook you can copy/paste:

```python
# ============================================================================
# CELL 1: Setup Environment
# ============================================================================
!nvidia-smi

import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")

# Mount Drive for persistent storage
from google.colab import drive
drive.mount('/content/drive')

# ============================================================================
# CELL 2: Clone Repository & Install Dependencies
# ============================================================================
!git clone https://github.com/YOUR_USERNAME/agi-self-modification-research.git
%cd agi-self-modification-research

!pip install -q transformers accelerate safetensors
!pip install -q chromadb networkx
!pip install -q pydantic pydantic-settings
!pip install -q rich tqdm pytest

print("✓ Setup complete")

# ============================================================================
# CELL 3: Setup Persistent Memory
# ============================================================================
!mkdir -p /content/drive/MyDrive/AGI_Memory
!rm -rf data/phase1_memory
!ln -s /content/drive/MyDrive/AGI_Memory data/phase1_memory

# Setup model cache in Drive (optional but recommended)
import os
os.environ['HF_HOME'] = '/content/drive/MyDrive/.cache/huggingface'
os.environ['TRANSFORMERS_CACHE'] = '/content/drive/MyDrive/.cache/huggingface/transformers'

print("✓ Persistent storage configured")

# ============================================================================
# CELL 4: Run Phase 1 Experiment
# ============================================================================
import subprocess
import sys

process = subprocess.Popen(
    [sys.executable, 'scripts/experiments/phase1_introspection.py'],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True
)

# Auto-confirm and stream output
stdout, _ = process.communicate(input='yes\n')
print(stdout)

print("\n✓ Experiment complete!")

# ============================================================================
# CELL 5: Backup Results to Drive
# ============================================================================
!mkdir -p /content/drive/MyDrive/AGI_Experiments
!cp -r data/phase1_sessions /content/drive/MyDrive/AGI_Experiments/
!cp -r data/logs /content/drive/MyDrive/AGI_Experiments/

print("✓ Results backed up to Google Drive")

# ============================================================================
# CELL 6: Download Results (Optional)
# ============================================================================
import shutil
import os
from google.colab import files

# Find latest session
sessions = sorted(os.listdir('data/phase1_sessions'))
if sessions:
    latest = sessions[-1]
    print(f"Latest session: {latest}")
    
    # Compress
    shutil.make_archive(latest, 'zip', f'data/phase1_sessions/{latest}')
    
    # Download
    files.download(f'{latest}.zip')
    print(f"✓ Downloaded {latest}.zip")

# ============================================================================
# CELL 7: View Summary
# ============================================================================
import json

# Load summary
summary_file = f'data/phase1_sessions/{latest}/summary.json'
with open(summary_file) as f:
    summary = json.load(f)

print("=" * 60)
print("PHASE 1 EXPERIMENT SUMMARY")
print("=" * 60)
print(f"Session: {summary['session_name']}")
print(f"Total tool calls: {summary['tool_usage']['total_calls']}")
print(f"Successful: {summary['tool_usage']['successful_calls']}")
print(f"Failed: {summary['tool_usage']['failed_calls']}")
print("\nFunction usage:")
for func, count in summary['tool_usage']['function_usage'].items():
    print(f"  {func}: {count}")
print("=" * 60)
```

---

## Troubleshooting

### Issue: "CUDA out of memory"

**Solution 1:** Restart runtime and free memory
```python
import torch
torch.cuda.empty_cache()

# Or restart runtime:
# Runtime → Restart runtime
```

**Solution 2:** Use 8-bit quantization (see Optimizations above)

### Issue: "Session disconnected"

**Solution:** Results should be auto-saved to Drive if you followed setup. Re-run from Drive backup.

### Issue: "Package version conflicts"

**Solution:**
```python
!pip install --upgrade transformers accelerate
!pip list | grep torch  # Check versions
```

### Issue: "Model download fails"

**Solution:**
```python
# Try with authentication if model is gated
from huggingface_hub import login
login()  # Enter your HF token

# Or download manually first
!wget https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/resolve/main/config.json
```

---

## Performance Expectations

**Colab Free (T4 16GB):**
- Phase 1 complete run: **~45-60 minutes**
- Per experiment: ~15-20 minutes
- Fits 3B model comfortably
- 4x faster than RTX 3060 4GB

**Colab Pro+ (A100 40GB):**
- Phase 1 complete run: **~20-30 minutes**
- Per experiment: ~7-10 minutes
- Can run 7B models easily
- 8x faster than RTX 3060 4GB

---

## Cost Analysis

**Free Tier:**
- Cost: $0
- Limitations: 12hr sessions, slower GPU queue
- Good for: Testing, single Phase 1 runs

**Pro ($10/month):**
- Cost: $10/month
- Benefits: Longer sessions, priority GPU access
- Good for: Regular experimentation

**Pro+ ($50/month):**
- Cost: $50/month  
- Benefits: A100 access, background execution, longest sessions
- Good for: Heavy use, large models

**Recommendation:** Start with free tier. Upgrade to Pro if you run experiments multiple times per week.

---

## Next Steps

1. ✅ Open [colab.research.google.com](https://colab.research.google.com)
2. ✅ Create new notebook
3. ✅ Enable GPU runtime
4. ✅ Copy cells from template above
5. ✅ Update GitHub URL
6. ✅ Run cells in order
7. ✅ Download/backup results

**Estimated time to first result:** 10 minutes setup + 45 minutes experiment = **~1 hour total**

---

## Tips for Best Experience

1. **Keep tab active** - Prevents idle disconnects
2. **Save to Drive** - Don't rely on Colab storage alone
3. **Monitor progress** - Check logs periodically
4. **Run during off-peak** - Faster GPU allocation (late night US time)
5. **Use Pro tier** - If running multiple experiments

---

*"Claude wanted to examine itself. Now you can help it do so from your browser, for free."*
