# Llama Model Usage Guide for Iridis5

This guide provides step-by-step instructions for using Llama models with OPRO on Iridis5.

## Prerequisites

1. Access to Iridis5 cluster
2. Active conda environment with required packages
3. GPU allocation permissions

## Workflow Overview

### Phase 1: Model Download (On Login Node)
### Phase 2: GPU Job Submission (On Compute Node)
### Phase 3: Results Analysis

---

## Phase 1: Download Models (Login Node)

### Option 1A: Interactive Download
```bash
# SSH to Iridis5 login node
ssh username@iridis5.soton.ac.uk

# Navigate to your project directory
cd path/to/your/OPRO_Streamlined

# Activate your environment
conda activate final

# Run interactive download
python download_models_login.py
```

### Option 1B: Batch Download Job
```bash
# Submit download job (if login nodes don't have direct internet)
sbatch download_models.slurm

# Check job status
squeue -u $USER

# Check results
cat download_llama_JOBID.out
```

### Verify Download Success
```bash
# Check downloaded models
ls -la ~/.cache/huggingface/hub/
du -sh ~/.cache/huggingface/
```

---

## Phase 2: Submit GPU Job

### Basic GPU Job Submission
```bash
# Navigate to project directory
cd path/to/your/OPRO_Streamlined

# Submit GPU job
sbatch submit_llama_job.slurm

# Check job status
squeue -u $USER

# Monitor job output (replace JOBID with actual job ID)
tail -f llama_opro_JOBID.out
```

### Interactive GPU Session (Alternative)
```bash
# Request interactive GPU session
sinteractive -p lyceum --gres=gpu:1 --mem=32G --time=2:00:00

# Once on GPU node, run directly
cd path/to/your/OPRO_Streamlined
conda activate final
python run_opro_with_llama.py
```

---

## Phase 3: Check Results

### After Job Completion
```bash
# Check job output
cat llama_opro_JOBID.out
cat llama_opro_JOBID.err

# Check for Llama results
if [ -f "llama_opro_results.json" ]; then
    echo "Llama model was used successfully"
    cat llama_opro_results.json
else
    echo "Fell back to offline mode"
    cat ICD11_OPRO/prompts/optimization_history.json
fi
```

---

## Troubleshooting

### Common Issues and Solutions

#### Issue 1: No Internet Access
```
ERROR: No internet access detected
```
**Solution**: Run download on login node or use `download_models.slurm`

#### Issue 2: Insufficient GPU Memory
```
ERROR: CUDA out of memory
```
**Solution**: The scripts automatically try smaller models:
1. Llama-3.2-1B-Instruct (2GB)
2. Llama-3.2-3B-Instruct (4GB)  
3. Llama-3-8B-Instruct (8GB with quantization)

#### Issue 3: Missing bitsandbytes
```
ERROR: bitsandbytes not available
```
**Solution**: Install quantization support:
```bash
python install_quantization.py
# or manually:
pip install bitsandbytes accelerate
```

#### Issue 4: Model Download Failed
```
ERROR: Failed to download model
```
**Solution**: Check internet connectivity and try again:
```bash
curl -I https://huggingface.co
python download_models_login.py
```

---

## Job Script Customization

### Adjust Resource Requirements

Edit `submit_llama_job.slurm`:

```bash
# For longer optimization
#SBATCH --time=04:00:00

# For more memory
#SBATCH --mem=64G

# For specific GPU type (if available)
#SBATCH --gres=gpu:rtx3090:1
```

### Adjust Model Selection

Edit `run_opro_with_llama.py` line 53-57:
```python
models_to_try = [
    "meta-llama/Llama-3.2-1B-Instruct",  # Fastest
    # "meta-llama/Llama-3.2-3B-Instruct",  # Comment out unwanted models
    # "meta-llama/Meta-Llama-3-8B-Instruct"
]
```

---

## Performance Expectations

### Model Performance (on GTX 1080 Ti 11GB):

| Model | Memory Usage | Speed | Quality |
|-------|-------------|--------|---------|
| Llama-3.2-1B | ~2GB | Fast | Good |
| Llama-3.2-3B | ~4GB | Medium | Better |
| Llama-3-8B | ~8GB | Slower | Best |

### Fallback Behavior:
- If no models can be loaded: Falls back to offline OPRO
- Offline OPRO has proven effective in previous tests
- Both modes produce valid optimization results

---

## File Outputs

### Successful Llama Execution:
- `llama_opro_results.json` - Llama optimization results
- `llama_opro_config.json` - Configuration used
- `llama_opro_JOBID.out` - Job output log

### Offline Mode Fallback:
- `ICD11_OPRO/prompts/optimization_history.json` - Offline results
- `ICD11_OPRO/prompts/optimized_prompt.txt` - Final optimized prompt

---

## Quick Start Commands

```bash
# Complete workflow
cd OPRO_Streamlined

# Step 1: Download models (login node)
python download_models_login.py

# Step 2: Submit GPU job
sbatch submit_llama_job.slurm

# Step 3: Check results
squeue -u $USER
cat llama_opro_*.out
```

---

## Support

If you encounter issues:
1. Check job output files (.out and .err)
2. Verify conda environment is activated
3. Check GPU allocation with `nvidia-smi`
4. Use offline mode as reliable fallback

The system is designed to gracefully handle failures and provide useful results in either Llama or offline mode. 