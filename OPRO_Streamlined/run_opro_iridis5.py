#!/usr/bin/env python3
"""
OPRO Runner for Iridis5 Supercomputer
Specialized version with enhanced error handling and environment detection
"""

import os
import sys
import argparse
import logging
from datetime import datetime
from pathlib import Path

# Add core directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'core'))

# Import with fallback handling
try:
    from opro_optimizer_iridis5 import OPROOptimizer, OptimizationResult
    print("Using Iridis5-optimized OPRO module")
except ImportError:
    try:
        from opro_optimizer import OPROOptimizer, OptimizationResult
        print("Using standard OPRO module")
    except ImportError as e:
        print(f"Error importing OPRO module: {e}")
        print("Please check your environment setup")
        sys.exit(1)

# Configure logging for Iridis5
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler('logs/opro_iridis5.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def check_iridis5_environment():
    """Check if we're running on Iridis5 and verify environment"""
    print("Checking Iridis5 environment...")
    
    # Check if we're on Iridis5
    hostname = os.environ.get('HOSTNAME', '')
    if 'iridis' not in hostname.lower():
        print(f"Warning: Not detected as Iridis5 environment (hostname: {hostname})")
    else:
        print(f"Detected Iridis5 environment: {hostname}")
    
    # Check SLURM environment
    if 'SLURM_JOB_ID' in os.environ:
        print(f"Running in SLURM job: {os.environ['SLURM_JOB_ID']}")
        print(f"Job name: {os.environ.get('SLURM_JOB_NAME', 'unknown')}")
        print(f"Partition: {os.environ.get('SLURM_JOB_PARTITION', 'unknown')}")
    
    # Check CUDA environment
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        print(f"CUDA devices: {os.environ['CUDA_VISIBLE_DEVICES']}")
    
    # Check conda environment
    conda_env = os.environ.get('CONDA_DEFAULT_ENV', 'unknown')
    print(f"Conda environment: {conda_env}")
    
    # Check Python version
    print(f"Python version: {sys.version}")
    
    print("Environment check completed.\n")

def check_llama_availability():
    """Check if Llama model is actually available and usable"""
    print("\n=== Llama Model Availability Check ===")
    
    # Check transformers
    try:
        from transformers import AutoTokenizer
        print("Transformers library available")
    except ImportError:
        print("Transformers library not available")
        return False
    except Exception as e:
        print(f"Transformers import error: {e}")
        return False
    
    # Check torch
    try:
        import torch
        print("PyTorch library available")
    except ImportError:
        print("PyTorch library not available")
        return False
    
    # Check GPU memory
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        total_memory = props.total_memory / (1024**3)  # GB
        print(f"GPU memory: {total_memory:.1f} GB ({props.name})")
        if total_memory < 16:
            print("GPU memory may be insufficient for Llama 3-8B (requires ~16GB)")
    
    # Check if model files exist in cache
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct")
    if os.path.exists(cache_dir):
        print("Llama 3-8B model files cached")
    else:
        print("Llama 3-8B model files not cached, download required (~16GB)")
    
    # Check HF token
    token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')
    if token:
        print("HuggingFace token configured")
    else:
        print("No HuggingFace token set, may be unable to download some models")
    
    return True

def run_optimization():
    """Run OPRO optimization with Iridis5-specific settings"""
    try:
        print("\nStarting OPRO optimization on Iridis5...")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Check Llama availability
        check_llama_availability()
        
        # Initialize optimizer with environment checking
        print(f"\n{'='*50}")
        optimizer = OPROOptimizer()
        
        # Test actual Llama model usage
        print("\n=== Testing Llama Model Actual Call ===")
        try:
            from OPRO_Streamlined.core.opro_optimizer_iridis5 import call_local_llm, _transformers_available
            
            print(f"Transformers available: {_transformers_available}")
            
            if _transformers_available:
                print("Attempting to call Llama 3-8B model...")
                try:
                    test_response = call_local_llm("Hello, this is a test.", max_new_tokens=20, temperature=0.7)
                    print(f"Llama 3-8B model call successful: {test_response[:100]}...")
                    print("System will use Llama 3-8B for prompt variant generation")
                except Exception as llm_error:
                    if "huggingface.co" in str(llm_error):
                        print("Network connectivity issue - cannot download Llama 3-8B from HuggingFace")
                        print("This is normal on Iridis5 compute nodes with restricted internet")
                        print(">>> System will use enhanced text modification methods (still effective)")
                    else:
                        print(f"Llama 3-8B model call failed: {llm_error}")
                        print("System will fallback to enhanced text modifications")
            else:
                print("Transformers not available, using enhanced text modifications")
                print("System will use intelligent fallback method")
        except Exception as e:
            print(f"Error testing Llama 3-8B model: {e}")
            print("System will fallback to enhanced text modifications")
        
        # Run optimization
        print("\nRunning ENHANCED OPRO optimization, please wait...")
        print("This may take several minutes on Iridis5...")
        print("Enhanced features:")
        print("- Improved evaluation metrics")
        print("- Extended iteration count")  
        print("- Better logging and file saving")
        
        result = optimizer.optimize_prompts()
        
        # Display detailed results
        print(f"\n{'='*50}")
        print(f"OPTIMIZATION COMPLETED SUCCESSFULLY!")
        print(f"{'='*50}")
        print(f"Final score: {result.final_score:.3f}")
        print(f"Total improvement: {result.improvement_achieved:.3f}")
        print(f"Total iterations: {result.total_iterations}")
        print(f"Time elapsed: {result.time_elapsed:.1f} seconds")
        
        print(f"\nResults saved to:")
        print(f"   Optimized prompt: prompts/optimized_prompt.txt")
        print(f"   Optimization history: prompts/optimization_history.json")
        print(f"   Summary report: prompts/optimization_summary.txt")
        print(f"   Logs: logs/opro_iridis5.log")
        
        # Check if files were actually created
        files_to_check = [
            "prompts/optimized_prompt.txt",
            "prompts/optimization_history.json", 
            "prompts/optimization_summary.txt"
        ]
        
        print(f"\nFile verification:")
        for file_path in files_to_check:
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                print(f"   {file_path} ({file_size} bytes)")
            else:
                print(f"   {file_path} (missing)")
        
        print(f"\n{'='*50}")
        print("Enhanced optimization completed!")
        
        return result
        
    except Exception as e:
        print(f"Optimization failed: {e}")
        logger.error(f"Optimization error: {e}")
        
        # Provide helpful error information
        print("\nTroubleshooting tips:")
        print("1. Check if your conda environment is properly set up")
        print("2. Verify numpy/scipy compatibility with: python -c 'import numpy, scipy; print(\"OK\")'")
        print("3. Check available GPU memory with: nvidia-smi")
        print("4. Review logs in logs/opro_iridis5.log for detailed error information")
        
        return None

def evaluate_prompt(prompt_file: str):
    """Evaluate a specific prompt file"""
    try:
        if not os.path.exists(prompt_file):
            print(f"Error: Prompt file {prompt_file} not found")
            return None
        
        print(f"\nEvaluating prompt file: {prompt_file}")
        
        # Load prompt
        with open(prompt_file, 'r', encoding='utf-8') as f:
            prompt_content = f.read().strip()
        
        # Initialize optimizer for evaluation
        optimizer = OPROOptimizer()
        
        # Evaluate prompt
        score = optimizer._evaluate_prompt(prompt_content)
        
        print(f"Evaluation completed!")
        print(f"Prompt score: {score:.3f}")
        print(f"Prompt length: {len(prompt_content)} characters")
        print(f"Word count: {len(prompt_content.split())} words")
        
        return score
        
    except Exception as e:
        print(f"Evaluation failed: {e}")
        logger.error(f"Evaluation error: {e}")
        return None

def display_system_info():
    """Display system information for Iridis5"""
    print("\nOPRO Streamlined System Information (Iridis5)")
    print("=" * 60)
    print(f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check system files
    required_files = [
        "config/config.json",
        "core/opro_optimizer.py",
        "core/opro_optimizer_iridis5.py",
        "core/scheduler.py"
    ]
    
    print("\nSystem files:")
    for file_path in required_files:
        status = "OK" if os.path.exists(file_path) else "MISSING"
        print(f"   {file_path}: {status}")
    
    # Check directories
    required_dirs = ["prompts", "prompts/seeds", "tests", "logs", "config"]
    print("\nDirectories:")
    for dir_path in required_dirs:
        status = "OK" if os.path.exists(dir_path) else "MISSING"
        if os.path.exists(dir_path):
            file_count = len([f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))])
            print(f"   {dir_path}/: {status} ({file_count} files)")
        else:
            print(f"   {dir_path}/: {status}")
    
    # Check Iridis5 specific environment
    check_iridis5_environment()

def create_slurm_job_script():
    """Create a SLURM job script for Iridis5"""
    slurm_script = """#!/bin/bash
#SBATCH --job-name=opro_optimization
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/opro_%j.out
#SBATCH --error=logs/opro_%j.err

# Load environment
module load conda
module load cuda/11.8

# Activate conda environment
conda activate opro_streamlined

# Change to working directory
cd $SLURM_SUBMIT_DIR

# Run OPRO optimization
echo "Starting OPRO optimization at $(date)"
python run_opro_iridis5.py --mode optimize

echo "OPRO optimization completed at $(date)"
"""
    
    with open("submit_opro_job.sh", "w") as f:
        f.write(slurm_script)
    
    print("Created SLURM job script: submit_opro_job.sh")
    print("To submit the job, run: sbatch submit_opro_job.sh")

def main():
    """Main function"""
    # Ensure logs directory exists
    os.makedirs("logs", exist_ok=True)
    
    parser = argparse.ArgumentParser(description='OPRO Streamlined System for Iridis5')
    parser.add_argument('--mode', choices=['optimize', 'evaluate', 'info', 'create-job'], 
                       default='info', help='Operation mode')
    parser.add_argument('--prompt-file', help='Path to prompt file for evaluation')
    
    args = parser.parse_args()
    
    # Log the command
    logger.info(f"Running OPRO with mode: {args.mode}")
    
    if args.mode == 'optimize':
        result = run_optimization()
        if result:
            print("\nOptimization completed successfully!")
            logger.info("Optimization completed successfully")
        else:
            print("\nOptimization failed!")
            logger.error("Optimization failed")
            sys.exit(1)
    
    elif args.mode == 'evaluate':
        if not args.prompt_file:
            print("Error: --prompt-file required for evaluation mode")
            sys.exit(1)
        
        score = evaluate_prompt(args.prompt_file)
        if score is not None:
            print(f"\nEvaluation completed! Score: {score:.3f}")
            logger.info(f"Evaluation completed with score: {score:.3f}")
        else:
            print("\nEvaluation failed!")
            logger.error("Evaluation failed")
            sys.exit(1)
    
    elif args.mode == 'info':
        display_system_info()
    
    elif args.mode == 'create-job':
        create_slurm_job_script()
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 