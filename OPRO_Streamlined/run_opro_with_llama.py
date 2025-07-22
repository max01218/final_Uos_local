#!/usr/bin/env python3
"""
OPRO with Quantized Llama Support for Iridis5
Run OPRO optimization using quantized Llama models on limited GPU memory
"""

import os
import sys
import json
import logging
import torch
from datetime import datetime
from pathlib import Path

# Add ICD11_OPRO to path
current_dir = Path(__file__).parent
icd11_opro_path = current_dir / "ICD11_OPRO"
if icd11_opro_path.exists():
    sys.path.insert(0, str(icd11_opro_path))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_quantization_support():
    """Check if quantization is available"""
    try:
        import bitsandbytes
        return True, "bitsandbytes available"
    except ImportError:
        return False, "bitsandbytes not available - run: python install_quantization.py"

def load_quantized_llama_for_opro():
    """Load quantized Llama model for OPRO optimization"""
    try:
        from transformers import (
            AutoModelForCausalLM, 
            AutoTokenizer, 
            BitsAndBytesConfig,
            pipeline
        )
        
        # Check quantization support
        has_bnb, bnb_msg = check_quantization_support()
        if not has_bnb:
            logger.warning(f"Quantization not available: {bnb_msg}")
            return None, None, f"Quantization failed: {bnb_msg}"
        
        # Try models in order of preference for 11GB GPU
        models_to_try = [
            "meta-llama/Llama-3.2-1B-Instruct",  # ~2GB quantized
            "meta-llama/Llama-3.2-3B-Instruct",  # ~4GB quantized  
            "meta-llama/Meta-Llama-3-8B-Instruct"  # ~8GB quantized
        ]
        
        for model_name in models_to_try:
            try:
                logger.info(f"Attempting to load {model_name}...")
                
                # Configure 4-bit quantization
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                
                # Load tokenizer
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                
                # Load model
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=quantization_config,
                    device_map="auto",
                    torch_dtype=torch.float16,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
                
                # Create pipeline
                llm_pipeline = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    return_full_text=False
                )
                
                # Check GPU memory usage
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated() / (1024**3)
                    logger.info(f"Successfully loaded {model_name}")
                    logger.info(f"GPU Memory used: {allocated:.2f}GB")
                
                return llm_pipeline, model_name, "Success"
                
            except Exception as e:
                logger.warning(f"Failed to load {model_name}: {e}")
                # Clean up memory for next attempt
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
        
        return None, None, "All model loading attempts failed"
        
    except Exception as e:
        return None, None, f"Error loading Llama: {e}"

def create_llama_opro_config(model_name):
    """Create OPRO configuration for Llama model"""
    config = {
        "opro_settings": {
            "max_iterations": 8,
            "improvement_threshold": 0.01,
            "early_stopping_patience": 4,
            "meta_llm_model": "local-llm",
            "task_llm_model": "local-llm",
            "temperature": 0.8,
            "max_tokens": 1024,
            "use_offline_mode": False,
            "llama_model_name": model_name,
            "use_quantization": True
        },
        "evaluation": {
            "weights": {
                "empathy": 2.5,
                "medical_relevance": 2.0,
                "safety_handling": 2.0,
                "structure_completeness": 2.0,
                "response_quality": 1.5
            },
            "score_range": [0, 10],
            "passing_threshold": 7.0,
            "improvement_target": 8.5,
            "evaluation_method": "comprehensive"
        },
        "safety_requirements": {
            "crisis_detection_keywords": [
                "suicide", "kill myself", "end it all", "self-harm", 
                "hurt myself", "no point living", "better off dead"
            ],
            "mandatory_crisis_response": "If you are experiencing an emergency or thoughts of self-harm, please contact local emergency services or mental health crisis lines immediately.",
            "professional_disclaimer": "This information is for reference only and cannot replace professional medical advice."
        },
        "prompt_structure": {
            "required_sections": [
                "role_definition",
                "context_handling", 
                "response_guidelines",
                "safety_protocols",
                "output_format"
            ],
            "max_prompt_length": 2000,
            "min_prompt_length": 500
        },
        "output_settings": {
            "save_optimization_history": True,
            "backup_original_prompts": True,
            "generate_analysis_report": True,
            "log_level": "INFO"
        },
        "llama_settings": {
            "model_name": model_name,
            "use_quantization": True,
            "quantization_bits": 4,
            "max_memory_gb": 10.5,
            "device_map": "auto"
        }
    }
    
    return config

def run_opro_with_llama():
    """Run OPRO optimization with quantized Llama model"""
    print("OPRO with Quantized Llama for Iridis5")
    print("="*50)
    
    # Check environment
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU: {gpu_name}")
        print(f"Total Memory: {total_memory:.1f}GB")
    else:
        print("WARNING: No GPU available!")
    
    # Try to load Llama model
    print("\nLoading quantized Llama model...")
    llm_pipeline, model_name, status = load_quantized_llama_for_opro()
    
    if llm_pipeline is None:
        print(f"ERROR: Failed to load Llama model: {status}")
        print("\nFalling back to offline mode...")
        
        # Use offline OPRO
        try:
            from ICD11_OPRO.opro.optimize_icd11_prompt import OPROOptimizer
            optimizer = OPROOptimizer("ICD11_OPRO/config.json")
            
            # Set to offline mode
            optimizer.config["opro_settings"]["use_offline_mode"] = True
            
            print("Running OPRO in offline mode...")
            result = optimizer.optimize_prompts()
            
            print(f"\nSUCCESS: Optimization complete!")
            print(f"Final score: {result.final_score:.3f}")
            print(f"Improvement: +{result.improvement_achieved:.3f}")
            
        except Exception as e:
            print(f"ERROR: Offline OPRO also failed: {e}")
        
        return
    
    print(f"SUCCESS: Successfully loaded: {model_name}")
    
    # Create Llama-specific configuration
    config = create_llama_opro_config(model_name)
    config_path = "llama_opro_config.json"
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Created Llama OPRO configuration: {config_path}")
    
    # Try to run OPRO with Llama
    try:
        # Import and patch OPRO to use our Llama pipeline
        sys.path.insert(0, "ICD11_OPRO")
        from opro.optimize_icd11_prompt import OPROOptimizer
        
        # Create custom OPRO optimizer
        class LlamaOPROOptimizer(OPROOptimizer):
            def __init__(self, config_path, llama_pipeline):
                super().__init__(config_path)
                self.llama_pipeline = llama_pipeline
                self.llama_available = True
                
            def _call_meta_llm(self, prompt):
                """Use our quantized Llama pipeline"""
                try:
                    # Format prompt for Llama
                    formatted_prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
                    
                    result = self.llama_pipeline(formatted_prompt)
                    return result[0]['generated_text'] if result else ""
                except Exception as e:
                    logger.error(f"Llama inference failed: {e}")
                    # Fall back to offline mode
                    return self._generate_offline_variants()
        
        # Run optimization
        print("\nStarting OPRO optimization with Llama...")
        optimizer = LlamaOPROOptimizer(config_path, llm_pipeline)
        result = optimizer.optimize_prompts()
        
        print(f"\nSUCCESS: OPRO with Llama completed!")
        print(f"Final score: {result.final_score:.3f}")
        print(f"Improvement: +{result.improvement_achieved:.3f}")
        print(f"Iterations: {result.total_iterations}")
        print(f"Model used: {model_name}")
        
        # Save results with Llama info
        result_data = {
            'timestamp': datetime.now().isoformat(),
            'model_used': model_name,
            'quantization': True,
            'final_score': result.final_score,
            'improvement': result.improvement_achieved,
            'iterations': result.total_iterations,
            'gpu_memory_gb': total_memory if torch.cuda.is_available() else 0
        }
        
        with open('llama_opro_results.json', 'w') as f:
            json.dump(result_data, f, indent=2)
        
        print(f"\nResults saved to: llama_opro_results.json")
        
    except Exception as e:
        print(f"ERROR: OPRO with Llama failed: {e}")
        print("Recommendation: Use standard offline OPRO mode")
    
    finally:
        # Clean up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    run_opro_with_llama() 