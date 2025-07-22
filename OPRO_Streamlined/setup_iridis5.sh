#!/bin/bash
# Iridis5 Environment Setup Script for OPRO Streamlined
# This script sets up a clean conda environment to avoid numpy compatibility issues

set -e  # Exit on any error

echo "Setting up OPRO Streamlined environment on Iridis5..."

# Check if we're running in an existing environment with the numpy issue
if python -c "import numpy; print(numpy.__version__)" 2>/dev/null | grep -q "^2\."; then
    echo "DETECTED: NumPy 2.x in current environment - this causes compatibility issues!"
    echo "Do you want to:"
    echo "1) Fix current environment (fast)"
    echo "2) Create new clean environment (recommended)"
    read -p "Choice (1 or 2): " choice
    
    if [ "$choice" = "1" ]; then
        echo "Applying emergency fix to current environment..."
        pip uninstall -y numpy scipy transformers torch torchvision torchaudio
        pip install numpy==1.24.3 scipy==1.10.1
        pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 --index-url https://download.pytorch.org/whl/cu118
        pip install transformers==4.30.2
        pip install -r requirements_iridis5.txt
        echo "✓ Emergency fix applied. You can now run OPRO optimization."
        exit 0
    fi
fi

# Load required modules
module load conda
module load cuda/11.8

# Environment name
ENV_NAME="opro_streamlined"

# Remove existing environment if it exists
echo "Removing existing environment (if any)..."
conda env remove -n $ENV_NAME -y 2>/dev/null || true

# Create new conda environment with specific Python version
echo "Creating new conda environment: $ENV_NAME"
conda create -n $ENV_NAME python=3.10 -y

# Activate environment
echo "Activating environment..."
source activate $ENV_NAME

# Install numpy and scipy first to establish correct versions
echo "Installing core numerical libraries..."
conda install -n $ENV_NAME numpy=1.24.3 scipy=1.10.1 -c conda-forge -y

# Install PyTorch with CUDA support
echo "Installing PyTorch with CUDA 11.8..."
pip install torch==2.0.1+cu118 torchaudio==2.0.2+cu118 torchvision==0.15.2+cu118 -f https://download.pytorch.org/whl/torch_stable.html

# Install transformers and related packages
echo "Installing transformers..."
pip install transformers==4.30.2 tokenizers==0.13.3 accelerate==0.20.3 safetensors==0.3.1

# Install remaining requirements
echo "Installing remaining dependencies..."
pip install tqdm==4.65.0 python-dotenv==1.0.0

# Verify installation
echo "Verifying installation..."
python -c "import numpy; print(f'NumPy version: {numpy.__version__}')"
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"

echo "Environment setup completed successfully!"
echo "To activate the environment, run: conda activate $ENV_NAME" 