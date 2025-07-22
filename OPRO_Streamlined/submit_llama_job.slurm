#!/bin/bash
#SBATCH --job-name=llama_opro
#SBATCH --partition=lyceum
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=llama_opro_%j.out
#SBATCH --error=llama_opro_%j.err

# Load required modules (adjust based on available modules)
# module load python/3.10
# module load cuda/11.8
# module load pytorch/2.0.1

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export TRANSFORMERS_CACHE=$HOME/.cache/huggingface
export HF_HOME=$HOME/.cache/huggingface

# Navigate to working directory
cd $SLURM_SUBMIT_DIR/OPRO_Streamlined

# Activate conda environment if using one
# conda activate final

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "GPU allocated: $CUDA_VISIBLE_DEVICES"

# Check GPU availability
nvidia-smi

# Check Python environment
python --version
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# Check if quantization libraries are available
python -c "try:
    import bitsandbytes as bnb
    print('bitsandbytes available')
except ImportError:
    print('bitsandbytes not available - installing...')
    import subprocess
    subprocess.run(['pip', 'install', 'bitsandbytes'])
"

# Run the main script
echo "Starting Llama OPRO optimization..."
python run_opro.py

# Check results
if [ -f "llama_opro_results.json" ]; then
    echo "SUCCESS: Llama OPRO completed successfully"
    echo "Results:"
    cat llama_opro_results.json
else
    echo "INFO: No Llama results found, checking offline results..."
    if [ -f "ICD11_OPRO/prompts/optimization_history.json" ]; then
        echo "Offline OPRO results found"
        tail -20 ICD11_OPRO/prompts/optimization_history.json
    fi
fi

echo "Job completed at: $(date)" 