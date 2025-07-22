#!/bin/bash
#SBATCH --job-name=download_llama
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=download_llama_%j.out
#SBATCH --error=download_llama_%j.err

# This script should be run on a node with internet access
# Usually login nodes or specific download nodes

# Set environment variables
export TRANSFORMERS_CACHE=$HOME/.cache/huggingface
export HF_HOME=$HOME/.cache/huggingface

# Navigate to working directory
cd $SLURM_SUBMIT_DIR/OPRO_Streamlined

# Activate conda environment if using one
# conda activate final

echo "Job ID: $SLURM_JOB_ID"
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"

# Test internet connectivity
echo "Testing internet connectivity..."
if curl -s --head https://huggingface.co | head -n 1 | grep "HTTP/1.[01] [23].." > /dev/null; then
    echo "SUCCESS: Internet access confirmed"
else
    echo "ERROR: No internet access detected"
    echo "This script must be run on a node with internet connectivity"
    exit 1
fi

# Run the download script
echo "Starting model download..."
python download_models_login.py << EOF
1
EOF

# Verify downloads
echo "Checking downloaded models..."
if [ -d "$HOME/.cache/huggingface/hub" ]; then
    echo "Downloaded models:"
    ls -la $HOME/.cache/huggingface/hub/ | grep llama
    echo "Cache size:"
    du -sh $HOME/.cache/huggingface/
else
    echo "WARNING: No models found in cache"
fi

echo "Download job completed at: $(date)" 