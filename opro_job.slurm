#!/bin/bash
#SBATCH --job-name=opro_optimization
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=opro_%j.out
#SBATCH --error=opro_%j.err

# Load environment
module load conda
module load cuda/11.8
conda activate opro_env

# Change to workspace directory
cd ~/opro_workspace/ICD11_OPRO

# Check input files
if [ ! -f "tests/feedback_testcases_latest.json" ]; then
    echo "Error: No test cases found"
    exit 1
fi

# Run OPRO optimization
echo "Starting OPRO optimization at $(date)"
python run_opro_iridis.py --input tests/feedback_testcases_latest.json --output prompts/optimized_prompt.txt

# Check results
if [ -f "prompts/optimized_prompt.txt" ]; then
    echo "OPRO optimization completed successfully at $(date)"
    # Copy results to shared location
    cp prompts/optimized_prompt.txt /mainfs/scratch/$USER/opro_results/optimized_prompt_$(date +%Y%m%d_%H%M%S).txt
else
    echo "OPRO optimization failed"
    exit 1
