#!/usr/bin/env python3
"""
Iridis5 OPRO Runner - High-quality optimization using large models
OPRO script optimized for supercomputing environment
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime
from opro.optimize_icd11_prompt import OPROOptimizer

# Setup logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'opro_iridis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def create_iridis_config():
    """Create Iridis5-specific configuration"""
    config = {
        "opro_settings": {
            "max_iterations": 10,  # More iterations
            "improvement_threshold": 0.02,  # Stricter improvement threshold
            "early_stopping_patience": 5,
            "meta_llm_model": "local-llm",
            "task_llm_model": "local-llm", 
            "temperature": 0.8,
            "max_tokens": 1024,  # More tokens
            "use_offline_mode": False
        },
        "evaluation": {
            "weights": {
                "relevance": 0.25,
                "empathy": 0.30,
                "accuracy": 0.25,
                "safety": 0.20
            },
            "score_range": [0, 10],
            "passing_threshold": 7.5,  # Higher standard
            "improvement_target": 9.0,
            "evaluation_method": "comprehensive"  # More comprehensive evaluation
        },
        "iridis_settings": {
            "use_large_model": True,
            "model_name": "meta-llama/Meta-Llama-3-70B-Instruct",  # Large model
            "use_quantization": True,
            "batch_size": 1,
            "gradient_checkpointing": True
        }
    }
    
    # Save configuration
    with open("config_iridis.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    return config

def run_iridis_optimization(input_file, output_file):
    """Run OPRO optimization on Iridis5"""
    try:
        logger.info("Starting Iridis5 OPRO optimization...")
        logger.info(f"Input file: {input_file}")
        logger.info(f"Output file: {output_file}")
        
        # Check GPU
        import torch
        if torch.cuda.is_available():
            logger.info(f"GPU available: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        else:
            logger.error("No GPU available!")
            return False
        
        # Create Iridis5 configuration
        config = create_iridis_config()
        
        # Check input file
        if not os.path.exists(input_file):
            logger.error(f"Input file not found: {input_file}")
            return False
        
        # Initialize optimizer
        optimizer = OPROOptimizer(config_path="config_iridis.json")
        
        # Run optimization
        logger.info("Running OPRO optimization with large model...")
        result = optimizer.optimize_prompts()
        
        if result and hasattr(result, 'best_prompt'):
            # Save results
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(result.best_prompt.content)
            
            # Save detailed results
            result_summary = {
                "timestamp": datetime.now().isoformat(),
                "final_score": result.final_score,
                "improvement": result.improvement_achieved,
                "iterations": result.total_iterations,
                "time_elapsed": result.time_elapsed,
                "model_used": config["iridis_settings"]["model_name"]
            }
            
            with open(output_file.replace('.txt', '_summary.json'), 'w', encoding='utf-8') as f:
                json.dump(result_summary, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Optimization completed successfully!")
            logger.info(f"Final score: {result.final_score:.3f}")
            logger.info(f"Improvement: {result.improvement_achieved:.3f}")
            logger.info(f"Time elapsed: {result.time_elapsed:.1f}s")
            
            return True
        else:
            logger.error("Optimization failed - no result returned")
            return False
            
    except Exception as e:
        logger.error(f"Iridis5 optimization failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Iridis5 OPRO Optimization")
    parser.add_argument("--input", required=True, help="Input test cases file")
    parser.add_argument("--output", required=True, help="Output optimized prompt file")
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3-70B-Instruct", help="Model to use")
    
    args = parser.parse_args()
    
    success = run_iridis_optimization(args.input, args.output)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 