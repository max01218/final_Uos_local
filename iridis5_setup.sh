#!/bin/bash
# Iridis5 OPRO Environment Setup Script

# Create workspace directory
mkdir -p ~/opro_workspace
cd ~/opro_workspace

# Load modules
module load conda
module load cuda/11.8
module load gcc/9.3.0

# Create conda environment
conda create -n opro_env python=3.9 -y
conda activate opro_env

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers accelerate bitsandbytes
pip install openai anthropic
pip install numpy pandas tqdm
pip install huggingface_hub

# Test GPU
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')"

echo "Environment setup complete!" 