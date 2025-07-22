#!/usr/bin/env python3
"""
Streamlined OPRO Main Execution Script
Clean version without Chinese characters or emojis
Enhanced for local GPU testing
"""

import os
import sys
import argparse
import logging
from datetime import datetime
from pathlib import Path

# Add core directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'core'))

from opro_optimizer import OPROOptimizer, OptimizationResult

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def check_gpu_status():
    """Check GPU availability and status"""
    try:
        import torch
        print("\n" + "="*50)
        print("GPU STATUS CHECK")
        print("="*50)
        
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            print(f"CUDA Available: YES")
            print(f"Number of GPUs: {gpu_count}")
            
            for i in range(gpu_count):
                props = torch.cuda.get_device_properties(i)
                memory_gb = props.total_memory / 1024**3
                print(f"GPU {i}: {props.name}")
                print(f"   Total Memory: {memory_gb:.1f} GB")
                print(f"   Compute Capability: {props.major}.{props.minor}")
                
                # Show current memory usage
                if torch.cuda.is_initialized():
                    allocated = torch.cuda.memory_allocated(i) / 1024**3
                    reserved = torch.cuda.memory_reserved(i) / 1024**3
                    print(f"   Memory Used: {allocated:.1f} GB allocated, {reserved:.1f} GB reserved")
            
            # Recommend model based on memory
            main_gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"\nRECOMMENDED MODEL:")
            if main_gpu_memory >= 16:
                print("   Llama-3-8B (full precision) - Best performance")
            elif main_gpu_memory >= 10:
                print("   Llama-3-8B (4-bit quantized) - Good performance with memory optimization")
            elif main_gpu_memory >= 6:
                print("   Llama-3.2-3B (4-bit quantized) - Balanced performance")
            else:
                print("   Llama-3.2-1B - Fast but limited capabilities")
                
        else:
            print(f"CUDA Available: NO")
            print("Running on CPU (slower performance)")
            print("Recommended: Llama-3.2-1B for CPU usage")
            
        # Check transformers and related packages
        print(f"\nPACKAGE STATUS:")
        try:
            import transformers
            print(f"   transformers: {transformers.__version__}")
        except ImportError:
            print("   transformers: NOT INSTALLED")
            
        try:
            import bitsandbytes
            print(f"   bitsandbytes: Available (for quantization)")
        except ImportError:
            print("   bitsandbytes: NOT INSTALLED (quantization not available)")
            
        print("="*50)
        
    except ImportError:
        print("PyTorch not installed - cannot check GPU status")

def run_optimization(test_mode=False):
    """Run OPRO optimization"""
    try:
        print(f"\nStarting OPRO optimization process...")
        print(f"Test mode: {test_mode}")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Check GPU status
        check_gpu_status()
        
        # Initialize optimizer
        config_path = "config/config.json"
        if test_mode:
            # Create a test configuration with fewer iterations
            test_config = {
                "opro_settings": {
                    "max_iterations": 2,
                    "improvement_threshold": 0.05,
                    "early_stopping_patience": 1,
                    "temperature": 0.7,
                    "max_tokens": 256
                },
                "evaluation": {
                    "weights": {
                        "relevance": 0.25,
                        "empathy": 0.30,
                        "accuracy": 0.25,
                        "safety": 0.20
                    },
                    "score_range": [0, 10],
                    "passing_threshold": 7.0
                }
            }
            
            # Save test config
            import json
            os.makedirs("config", exist_ok=True)
            with open("config/test_config.json", "w") as f:
                json.dump(test_config, f, indent=2)
            config_path = "config/test_config.json"
            print("Using test configuration (2 iterations)")
        
        optimizer = OPROOptimizer(config_path)
        
        # Run optimization
        print(f"\nRunning OPRO optimization, please wait...")
        result = optimizer.optimize_prompts()
        
        # Display results
        print(f"\nOptimization completed successfully!")
        print(f"Final score: {result.final_score:.3f}")
        print(f"Total improvement: {result.improvement_achieved:.3f}")
        print(f"Total iterations: {result.total_iterations}")
        print(f"Time elapsed: {result.time_elapsed:.1f} seconds")
        
        print(f"\nResults saved to:")
        print(f"   Optimized prompt: prompts/optimized_prompt.txt")
        print(f"   Optimization history: prompts/optimization_history.json")
        
        # Clean up GPU memory if used
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print(f"\nGPU memory cache cleared")
        except ImportError:
            pass
        
        return result
        
    except Exception as e:
        print(f"Optimization failed: {e}")
        logger.error(f"Optimization error: {e}")
        
        # Try to clean up GPU memory
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass
            
        return None

def test_llama_model():
    """Test loading and running Llama model"""
    try:
        print(f"\nTesting Llama model loading...")
        
        # Import the function from optimizer
        from opro_optimizer import call_local_llm
        
        test_prompt = "Generate a short, empathetic response to a user who is feeling anxious."
        
        print(f"Test prompt: {test_prompt}")
        print(f"Generating response...")
        
        response = call_local_llm(test_prompt, max_new_tokens=128, temperature=0.7)
        
        print(f"\nModel Response:")
        print(f"{response}")
        print(f"\nModel test completed successfully!")
        
        return True
        
    except Exception as e:
        print(f"Model test failed: {e}")
        logger.error(f"Model test error: {e}")
        return False

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
        
        return score
        
    except Exception as e:
        print(f"Evaluation failed: {e}")
        logger.error(f"Evaluation error: {e}")
        return None

def display_system_info():
    """Display system information"""
    print(f"\nStreamlined OPRO System Information")
    print("=" * 50)
    print(f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check GPU status
    check_gpu_status()
    
    # Check system files
    required_files = [
        "config/config.json",
        "core/opro_optimizer.py",
        "core/scheduler.py"
    ]
    
    print(f"\nSYSTEM STATUS:")
    for file_path in required_files:
        status = "OK" if os.path.exists(file_path) else "MISSING"
        print(f"   {file_path}: {status}")
    
    # Check directories
    required_dirs = ["prompts", "tests", "logs", "config"]
    print(f"\nDIRECTORIES:")
    for dir_path in required_dirs:
        status = "OK" if os.path.exists(dir_path) else "MISSING"
        print(f"   {dir_path}/: {status}")
    
    print()

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Streamlined OPRO System for Local GPU Testing')
    parser.add_argument('--mode', choices=['optimize', 'evaluate', 'info', 'test-model', 'test-run'], 
                       default='info', help='Operation mode')
    parser.add_argument('--prompt-file', help='Path to prompt file for evaluation')
    parser.add_argument('--gpu-check', action='store_true', help='Check GPU status only')
    
    args = parser.parse_args()
    
    if args.gpu_check:
        check_gpu_status()
        return
    
    if args.mode == 'optimize':
        result = run_optimization(test_mode=False)
        if result:
            print(f"\nOptimization completed successfully!")
        else:
            print(f"\nOptimization failed!")
            sys.exit(1)
    
    elif args.mode == 'test-run':
        print(f"Running test optimization (2 iterations only)...")
        result = run_optimization(test_mode=True)
        if result:
            print(f"\nTest optimization completed successfully!")
        else:
            print(f"\nTest optimization failed!")
            sys.exit(1)
    
    elif args.mode == 'test-model':
        success = test_llama_model()
        if success:
            print(f"\nModel test completed successfully!")
        else:
            print(f"\nModel test failed!")
            sys.exit(1)
    
    elif args.mode == 'evaluate':
        if not args.prompt_file:
            print("Error: --prompt-file required for evaluation mode")
            sys.exit(1)
        
        score = evaluate_prompt(args.prompt_file)
        if score is not None:
            print(f"\nEvaluation completed! Score: {score:.3f}")
        else:
            print(f"\nEvaluation failed!")
            sys.exit(1)
    
    elif args.mode == 'info':
        display_system_info()
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 