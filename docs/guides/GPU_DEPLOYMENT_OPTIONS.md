# GPU Deployment Options for Phase 1 Experiments

**Date:** November 7, 2025  
**Current Setup:** RTX 3060 Laptop (4GB VRAM) - Works but slow  
**Goal:** Run Phase 1 introspection experiments faster

---

## Current Performance Baseline

**Hardware:** RTX 3060 Laptop, 4GB VRAM, 32GB RAM  
**Model:** Qwen2.5-3B-Instruct (3.09B parameters)  
**Phase 1 Run 1 Duration:** 2 hours 41 minutes (crashed with OOM before completion)

**Bottlenecks:**
- Limited VRAM (4GB) requires careful memory management
- Slower inference speed (~10-20 seconds per generation)
- OOM risk during long sessions
- Cannot run larger models (7B+)

---

## Option 1: Cloud GPU Providers (Best for Research)

### A. Lambda Labs (Recommended for AI Research)
**Pros:**
- ✅ Best price/performance for ML workloads
- ✅ Pre-configured PyTorch environments
- ✅ No setup time - works out of the box
- ✅ Hourly billing, no commitments
- ✅ Great for bursty workloads

**GPU Options:**
| GPU | VRAM | Price/hr | Speed vs Current | Best For |
|-----|------|----------|------------------|----------|
| RTX 6000 Ada | 48GB | $0.75 | ~3-4x faster | Multi-experiment runs |
| A100 (40GB) | 40GB | $1.10 | ~5-6x faster | Large models (7B+) |
| A6000 | 48GB | $0.80 | ~4-5x faster | Long sessions |
| A10 | 24GB | $0.60 | ~2-3x faster | Budget option |

**Setup Time:** ~15 minutes  
**Est. Phase 1 Run Time:** 20-40 minutes (vs 2h 41min)  
**Est. Cost per Run:** $0.50-$1.50

**Setup:**
```bash
# 1. Sign up at lambdalabs.com
# 2. Launch instance with PyTorch image
# 3. Clone repo
git clone <your-repo-url>
cd agi-self-modification-research

# 4. Install dependencies
python -m venv venv
source venv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# 5. Run experiment
python scripts/experiments/phase1_introspection.py
```

---

### B. Vast.ai (Most Affordable)
**Pros:**
- ✅ Cheapest GPU rentals (marketplace model)
- ✅ Huge variety of GPUs
- ✅ Good for experimentation

**Cons:**
- ⚠️ Variable quality (community providers)
- ⚠️ May need more setup
- ⚠️ Reliability varies by host

**GPU Options:**
| GPU | VRAM | Price/hr | Availability |
|-----|------|----------|--------------|
| RTX 3090 | 24GB | $0.20-0.40 | High |
| RTX 4090 | 24GB | $0.40-0.70 | Medium |
| A100 | 40GB | $0.60-1.00 | Medium |

**Setup:** Similar to Lambda Labs  
**Est. Cost per Run:** $0.20-$1.00

**Website:** vast.ai

---

### C. RunPod (Good Balance)
**Pros:**
- ✅ Good price/performance
- ✅ Easy container deployment
- ✅ Persistent storage options

**GPU Options:**
| GPU | VRAM | Price/hr |
|-----|------|----------|
| RTX 4090 | 24GB | $0.69 |
| A100 (40GB) | 40GB | $1.14 |
| RTX 3090 | 24GB | $0.44 |

**Website:** runpod.io

---

### D. Google Colab Pro/Pro+ (Easiest Start)
**Pros:**
- ✅ No setup - runs in browser
- ✅ Integrated with Jupyter notebooks
- ✅ Monthly subscription vs hourly
- ✅ Good for development/testing

**Cons:**
- ⚠️ Session timeouts (12-24 hours max)
- ⚠️ Shared GPUs (variable performance)
- ⚠️ Limited control over environment

**Pricing:**
- Colab Pro: $10/month (better GPUs, longer runtime)
- Colab Pro+: $50/month (A100, background execution)

**GPU Options:**
- Free: T4 (16GB VRAM) - would work
- Pro: V100 (16GB) or T4
- Pro+: A100 (40GB) - best option

**Setup:**
```python
# In Colab notebook
!git clone <your-repo-url>
%cd agi-self-modification-research
!pip install -r requirements.txt
!python scripts/experiments/phase1_introspection.py
```

---

### E. AWS/Azure/GCP (Enterprise)
**Pros:**
- ✅ Most reliable
- ✅ Best for production
- ✅ Comprehensive monitoring

**Cons:**
- ❌ Most expensive
- ❌ Complex setup
- ❌ Overkill for research

**AWS EC2 GPU Instances:**
| Instance | GPU | VRAM | Price/hr |
|----------|-----|------|----------|
| g5.xlarge | A10G | 24GB | $1.006 |
| g5.2xlarge | A10G | 24GB | $1.212 |
| p3.2xlarge | V100 | 16GB | $3.06 |
| p4d.24xlarge | A100 | 40GB x8 | $32.77 |

**Only recommended if:** You already have AWS credits or enterprise account

---

## Option 2: Kaggle Notebooks (Free Alternative)

**Pros:**
- ✅ Completely free
- ✅ 30 hours/week GPU quota
- ✅ T4 GPU (16GB VRAM)
- ✅ Integrated with datasets
- ✅ No credit card required

**Cons:**
- ⚠️ 12-hour session limit
- ⚠️ Weekly quota
- ⚠️ Internet must be enabled for downloads

**Setup:**
1. Create Kaggle account
2. Enable GPU in notebook settings
3. Upload code as notebook or clone from GitHub
4. Run experiments

**Est. Phase 1 Run Time:** ~1 hour  
**Cost:** Free!

---

## Option 3: University/Research Computing

If you have access to academic computing:

**Pros:**
- ✅ Often free or heavily subsidized
- ✅ High-end GPUs (A100, H100)
- ✅ Large quotas
- ✅ Support staff

**Cons:**
- ⚠️ Queue times
- ⚠️ Learning cluster-specific tools (SLURM, etc.)
- ⚠️ Limited accessibility

**Ask about:**
- GPU node availability
- SLURM job scheduler
- Storage quotas
- PyTorch environment

---

## Option 4: Upgrade Local GPU

**If you want to invest in hardware:**

**Budget Options ($300-600):**
- RTX 3060 12GB ($300-350) - 3x your current VRAM
- RTX 3070 8GB ($400-500) - Better performance
- RTX 4060 Ti 16GB ($500) - Best budget option

**Mid-Range ($800-1200):**
- RTX 4070 12GB ($600-700) - Good balance
- RTX 4070 Ti 16GB ($800) - Better VRAM
- Used RTX 3090 24GB ($800-1000) - Best value

**High-End ($1500+):**
- RTX 4090 24GB ($1600-2000) - Fastest consumer GPU
- Used A6000 48GB ($2000-3000) - Workstation GPU

**Considerations:**
- Power supply requirements
- Case size/cooling
- PCIe compatibility
- One-time investment vs cloud costs

---

## Recommended Approach for This Project

### For Phase 1 (Current):
**Lambda Labs A10 or RTX 6000 Ada**
- Cost: $0.60-0.75/hr
- Est. per run: $0.50-1.00
- 10 runs: $5-10 total
- Fast iteration, low commitment

### For Phase 2+ (Future):
**Consider monthly cloud subscription OR GPU upgrade**
- If running daily: Upgrade local GPU ($500-1000)
- If running weekly: Continue cloud ($20-50/month)
- If scaling to larger models: Cloud A100 access

---

## Quick Start: Lambda Labs (Recommended)

### Step 1: Sign Up
1. Go to lambdalabs.com
2. Create account
3. Add payment method (they only charge for usage)

### Step 2: Launch Instance
1. Click "Instances" → "Launch Instance"
2. Select GPU: **A10** (24GB, $0.60/hr) or **RTX 6000 Ada** (48GB, $0.75/hr)
3. Select region (nearest to you)
4. Choose "PyTorch" image (pre-configured)
5. Add SSH key
6. Launch!

### Step 3: Connect & Setup
```bash
# SSH into instance (they provide the command)
ssh ubuntu@<instance-ip>

# Clone your repo
git clone <your-repo-url>
cd agi-self-modification-research

# Setup environment
python -m venv venv
source venv/bin/activate

# Install PyTorch (already installed in image, but to update)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install project dependencies
pip install -r requirements.txt

# Verify GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"

# Download model (if not already cached)
python -c "from transformers import AutoModel; AutoModel.from_pretrained('Qwen/Qwen2.5-3B-Instruct')"
```

### Step 4: Run Experiment
```bash
# Start Phase 1
python scripts/experiments/phase1_introspection.py

# Or use tmux for long-running sessions
tmux new -s phase1
python scripts/experiments/phase1_introspection.py
# Ctrl+B, then D to detach
# tmux attach -t phase1 to reconnect
```

### Step 5: Download Results
```bash
# On your local machine
scp -r ubuntu@<instance-ip>:~/agi-self-modification-research/data/phase1_sessions ./backup/

# Or use rsync for efficiency
rsync -avz ubuntu@<instance-ip>:~/agi-self-modification-research/data/ ./data/
```

### Step 6: Terminate Instance
1. Go to Lambda Labs dashboard
2. Click "Terminate" on your instance
3. Instance stops billing immediately

**Important:** Lambda Labs charges by the second, so you only pay for active time!

---

## Cost Comparison (10 Phase 1 Runs)

| Option | Per Run | 10 Runs | Setup | Notes |
|--------|---------|---------|-------|-------|
| Current (RTX 3060 4GB) | Free | Free | Done | Slow, OOM risk |
| Lambda Labs A10 | $0.50 | $5 | 15min | **Recommended** |
| Lambda Labs A100 | $1.00 | $10 | 15min | Overkill for 3B |
| Vast.ai RTX 3090 | $0.30 | $3 | 30min | Cheapest |
| Kaggle | Free | Free | 10min | 30hr/week limit |
| Colab Pro+ | $50/mo | $50 | 5min | Unlimited runs |
| RTX 4060 Ti 16GB | Free* | Free* | Days | $500 upfront |

---

## Storage Considerations

When using cloud GPUs:

1. **Model weights** (6GB) - Download once, reuse
2. **Session data** (~100MB per run) - Save and download
3. **Persistent memory** (growing) - Backup regularly

**Best practice:**
- Keep model weights on cloud instance
- Download session results after each run
- Sync persistent memory bidirectionally

---

## Monitoring Cloud Costs

**Lambda Labs:**
- Dashboard shows current cost
- Billing page shows detailed breakdown
- Set calendar reminders to check running instances

**Budget recommendation:**
- Set alert at $20/month
- Phase 1 should cost < $20 total for all runs
- Phase 2 onward: budget $50-100/month

---

## Summary

**For immediate Phase 1 Run 2:**
1. ✅ **Lambda Labs A10** - $0.60/hr, 24GB VRAM
2. ✅ 15 minute setup
3. ✅ ~30-40 min run time (vs 2h 41min)
4. ✅ $0.50 per complete run
5. ✅ No commitment, pay per second

**Long term:**
- If running multiple experiments per week → invest in RTX 4060 Ti 16GB ($500)
- If running occasionally → continue cloud ($20-40/month)
- If scaling to 7B+ models → Lambda Labs A100 ($1.10/hr)

**Next steps:**
1. Sign up for Lambda Labs
2. Launch A10 instance with PyTorch image
3. Clone repo and install dependencies (15 min)
4. Run Phase 1 Run 2
5. Download results
6. Terminate instance

Total cost: **$0.50-1.00** for a complete Phase 1 run that would take 2+ hours on current hardware!

---

*"The model has introspective tools. Now let's give it the compute to use them efficiently."*
