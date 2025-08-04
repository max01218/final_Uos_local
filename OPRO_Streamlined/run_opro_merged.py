#!/usr/bin/env python3
"""
OPRO (Optimization by PROmpting) - Merged Version

This script combines the complete functionality of ICD11_OPRO with the optimized features of OPRO_Streamlined.
Features:
- Complete OPRO optimization logic
- Robust LLaMA integration with quantization
- Enhanced error handling and memory management
- Comprehensive evaluation system
- Automated tools and scheduling
"""

import os
import sys
import json
import logging
import torch
from datetime import datetime
from pathlib import Path

# Add core directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'core'))

from opro_optimizer_merged import OPROOptimizer, OptimizationResult

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
    """Run OPRO optimization with enhanced functionality"""
    try:
        print(f"\nStarting OPRO optimization process...")
        print(f"Test mode: {test_mode}")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Check GPU status
        check_gpu_status()
        
        # Initialize optimizer
        config_path = "config/config.json"
        optimizer = OPROOptimizer(config_path)
        
        if test_mode:
            print("Running in test mode with limited iterations...")
            optimizer.config["optimization"]["max_iterations"] = 2
        
        # Run optimization
        print("\nStarting optimization...")
        result = optimizer.optimize_prompts()
        
        # Display results
        print(f"\n" + "="*50)
        print("OPTIMIZATION RESULTS")
        print("="*50)
        print(f"Final Score: {result.final_score:.3f}")
        print(f"Improvement: +{result.improvement_achieved:.3f}")
        print(f"Total Iterations: {result.total_iterations}")
        print(f"Time Elapsed: {result.time_elapsed:.1f} seconds")
        print(f"Best Prompt: {result.best_prompt.content[:100]}...")
        
        # Save detailed results
        save_detailed_results(result)
        
        return result
        
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        print(f"ERROR: Optimization failed - {e}")
        return None

def save_detailed_results(result: OptimizationResult):
    """Save detailed optimization results"""
    try:
        # Create results directory
        results_dir = "results"
        os.makedirs(results_dir, exist_ok=True)
        
        # Save detailed report
        report_file = os.path.join(results_dir, f"optimization_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'final_score': result.final_score,
                'improvement_achieved': result.improvement_achieved,
                'total_iterations': result.total_iterations,
                'time_elapsed': result.time_elapsed
            },
            'best_prompt': {
                'content': result.best_prompt.content,
                'score': result.best_prompt.score,
                'iteration': result.best_prompt.iteration,
                'generation_method': result.best_prompt.generation_method
            },
            'optimization_history': [
                {
                    'content': candidate.content,
                    'score': candidate.score,
                    'iteration': candidate.iteration,
                    'generation_method': candidate.generation_method,
                    'timestamp': candidate.timestamp
                }
                for candidate in result.optimization_history
            ]
        }
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        print(f"Detailed results saved to: {report_file}")
        
    except Exception as e:
        logger.error(f"Failed to save detailed results: {e}")

def test_llama_model():
    """Test LLaMA model functionality"""
    try:
        print("\n" + "="*50)
        print("LLAMA MODEL TEST")
        print("="*50)
        
        from core.opro_optimizer_merged import call_local_llm
        
        test_prompt = "Hello, how are you today?"
        print(f"Testing with prompt: {test_prompt}")
        
        response = call_local_llm(test_prompt, max_new_tokens=50)
        print(f"Response: {response}")
        
        print("LLaMA model test successful!")
        return True
        
    except Exception as e:
        print(f"LLaMA model test failed: {e}")
        return False

def evaluate_prompt(prompt_file: str):
    """Evaluate a specific prompt file"""
    try:
        if not os.path.exists(prompt_file):
            print(f"Prompt file not found: {prompt_file}")
            return
        
        with open(prompt_file, 'r', encoding='utf-8') as f:
            prompt_content = f.read().strip()
        
        print(f"\nEvaluating prompt from: {prompt_file}")
        print(f"Prompt content: {prompt_content[:100]}...")
        
        # Initialize optimizer for evaluation
        optimizer = OPROOptimizer("config/config.json")
        score = optimizer._evaluate_prompt(prompt_content)
        
        print(f"Evaluation score: {score:.3f}")
        
    except Exception as e:
        print(f"Prompt evaluation failed: {e}")

def display_system_info():
    """Display system information"""
    print("\n" + "="*50)
    print("SYSTEM INFORMATION")
    print("="*50)
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    print(f"Configuration file: config/config.json")
    print(f"Logs directory: logs/")
    print(f"Results directory: results/")
    print("="*50)

def main():
    """Main function with argument parsing"""
    import argparse
    
    parser = argparse.ArgumentParser(description="OPRO Optimization - Merged Version")
    parser.add_argument("--test", action="store_true", help="Run in test mode with limited iterations")
    parser.add_argument("--test-llama", action="store_true", help="Test LLaMA model functionality")
    parser.add_argument("--evaluate", type=str, help="Evaluate a specific prompt file")
    parser.add_argument("--info", action="store_true", help="Display system information")
    
    args = parser.parse_args()
    
    print("OPRO Optimization - Merged Version")
    print("="*50)
    
    if args.info:
        display_system_info()
        return
    
    if args.test_llama:
        test_llama_model()
        return
    
    if args.evaluate:
        evaluate_prompt(args.evaluate)
        return
    
    # Run main optimization
    result = run_optimization(test_mode=args.test)
    
    if result:
        print("\n✅ Optimization completed successfully!")
    else:
        print("\n❌ Optimization failed!")

if __name__ == "__main__":
    main() 