# Iridis5 NumPy Compatibility Quick Fix

## 🚨 EMERGENCY FIX (For NumPy 2.2.6 Error)

**Your error shows NumPy 2.2.6 is installed, which is incompatible with SciPy. Fix immediately:**

```bash
# CRITICAL: Run these commands in your current environment
pip uninstall -y numpy scipy transformers torch torchvision torchaudio
pip install numpy==1.24.3 scipy==1.10.1
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install transformers==4.30.2
pip install -r requirements_iridis5.txt

# Verify the fix worked:
python -c "import numpy, scipy, torch, transformers; print('SUCCESS: All imports work')"
```

**After running these commands, try OPRO optimization again.**

## 🌐 NETWORK CONNECTIVITY ISSUE (HuggingFace Download)

**If you see: "We couldn't connect to 'https://huggingface.co' to load this file"**

This happens because Iridis5 compute nodes often have restricted internet access. **The system will automatically fallback to text-based optimization** (no LLM required).

### Expected Behavior:
```
Error loading language model: We couldn't connect to 'https://huggingface.co'...
This is likely due to network connectivity issues on Iridis5
Will use fallback text modification methods instead
>>> System will use fallback method (not true LLM optimization)
```

### Options:

**Option 1: Accept Fallback Mode (Recommended)**
- The system will still work using enhanced text modification
- No internet required
- Faster execution
- Still provides meaningful prompt improvements

**Option 2: Pre-download Models (Advanced)**
```bash
# On a machine with internet access, download the model:
python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; AutoModelForCausalLM.from_pretrained('gpt2'); AutoTokenizer.from_pretrained('gpt2')"

# Then transfer the ~/.cache/huggingface directory to Iridis5
```

**Option 3: Use Llama Model Only**
The system will only attempt to load:
- meta-llama/Meta-Llama-3-8B-Instruct

If this fails due to network issues, the system will use enhanced text modification methods instead.

---

## Problem Description
You're encountering a numpy binary compatibility error on Iridis5:
```
ValueError: numpy.dtype size changed, may indicate binary incompatibility. Expected 96 from C header, got 88 from PyObject
```

This is a common issue on supercomputing environments due to version conflicts between numpy, scipy, and transformers.

## Quick Solution

### Option 1: Automated Setup (Recommended)

Run the automated setup script:

```bash
# Make the script executable
chmod +x setup_iridis5.sh

# Run the setup script
./setup_iridis5.sh
```

This will create a clean conda environment with compatible versions.

### Option 2: Manual Fix

If you prefer manual setup:

```bash
# Load required modules
module load conda
module load cuda/11.8

# Remove existing environment
conda env remove -n opro_streamlined -y

# Create fresh environment
conda create -n opro_streamlined python=3.10 -y
conda activate opro_streamlined

# Install numpy and scipy first with specific versions
conda install numpy=1.24.3 scipy=1.10.1 -c conda-forge -y

# Install PyTorch with CUDA 11.8
pip install torch==2.0.1+cu118 torchaudio==2.0.2+cu118 torchvision==0.15.2+cu118 -f https://download.pytorch.org/whl/torch_stable.html

# Install transformers with compatible version
pip install transformers==4.30.2 tokenizers==0.13.3 accelerate==0.20.3

# Install remaining dependencies
pip install tqdm==4.65.0 python-dotenv==1.0.0
```

### Option 3: Use Iridis5-Specific Version

Use the specialized Iridis5 version of the optimizer:

```bash
# Activate your environment
conda activate opro_streamlined

# Copy the Iridis5-specific optimizer
cp core/opro_optimizer_iridis5.py core/opro_optimizer.py

# Run with enhanced error handling
python run_opro.py --mode info
```

## Verification

After setup, verify your installation:

```bash
python -c "import numpy; print(f'NumPy: {numpy.__version__}')"
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
```

Expected output:
```
NumPy: 1.24.3
PyTorch: 2.0.1+cu118
Transformers: 4.30.2
```

## Run OPRO

Once the environment is fixed, run OPRO:

```bash
conda activate opro_streamlined
python run_opro.py --mode optimize
```

## Alternative: CPU-Only Mode

If CUDA issues persist, you can run in CPU-only mode:

```bash
# Install CPU-only PyTorch
pip uninstall torch torchaudio torchvision -y
pip install torch==2.0.1+cpu torchaudio==2.0.2+cpu torchvision==0.15.2+cpu -f https://download.pytorch.org/whl/torch_stable.html
```

## Troubleshooting

### If setup script fails:
1. Check module availability: `module avail conda`
2. Ensure you have write permissions in your home directory
3. Check disk space: `df -h $HOME`

### If imports still fail:
1. Completely remove conda environment: `conda env remove -n opro_streamlined -y`
2. Clear conda cache: `conda clean --all -y`
3. Restart your session and try again

### If transformers loading fails:
The Iridis5 version includes fallback modes that work without transformers for basic optimization.

## Contact Support

If issues persist, the system includes comprehensive error reporting. Check the logs in `logs/` directory for detailed error information. 