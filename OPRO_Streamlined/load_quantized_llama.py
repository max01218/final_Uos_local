#!/usr/bin/env python3
"""
Quantized Llama Model Loader for Iridis5
Load Llama models with 4-bit quantization to fit in 11GB GPU memory
"""

import torch
import logging
from typing import Optional, Tuple
import os
import sys

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_quantization_support():
    """Check if quantization libraries are available"""
    try:
        import bitsandbytes as bnb
        return True, "bitsandbytes available"
    except ImportError:
        return False, "bitsandbytes not available - install with: pip install bitsandbytes"

def load_quantized_llama(model_name: str = "meta-llama/Llama-3.2-3B-Instruct", 
                        use_4bit: bool = True) -> Tuple[Optional[object], Optional[object], str]:
    """
    Load a quantized Llama model
    
    Args:
        model_name: Model to load (smaller models for limited memory)
        use_4bit: Use 4-bit quantization
    
    Returns:
        (model, tokenizer, status_message)
    """
    
    # Check quantization support
    has_bnb, bnb_msg = check_quantization_support()
    if not has_bnb and use_4bit:
        logger.warning(f"Quantization not available: {bnb_msg}")
        return None, None, f"Failed: {bnb_msg}"
    
    try:
        from transformers import (
            AutoModelForCausalLM, 
            AutoTokenizer, 
            BitsAndBytesConfig
        )
        
        # Configure quantization
        if use_4bit and has_bnb:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            logger.info("Using 4-bit quantization configuration")
        else:
            quantization_config = None
            logger.info("Loading without quantization")
        
        # Check if model exists locally first
        if os.path.exists(model_name):
            logger.info(f"Loading local model: {model_name}")
        else:
            logger.info(f"Attempting to load model: {model_name}")
        
        # Load tokenizer
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True,
            trust_remote_code=True
        )
        
        # Set pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info("Set pad_token to eos_token")
        
        # Load model with quantization
        logger.info("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        # Print memory usage
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024**3)
            reserved = torch.cuda.memory_reserved() / (1024**3)
            logger.info(f"GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")
        
        return model, tokenizer, "Successfully loaded quantized model"
        
    except Exception as e:
        error_msg = f"Failed to load model: {str(e)}"
        logger.error(error_msg)
        return None, None, error_msg

def test_model_inference(model, tokenizer, test_prompt: str = "Hello, how are you?") -> str:
    """Test model inference"""
    try:
        # Prepare input
        inputs = tokenizer(test_prompt, return_tensors="pt", padding=True)
        
        # Move to GPU if available
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
        
    except Exception as e:
        return f"Inference failed: {str(e)}"

def main():
    """Main function to test quantized model loading"""
    print("Quantized Llama Model Loader for Iridis5")
    print("="*50)
    
    # Check GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU: {gpu_name}")
        print(f"Total Memory: {total_memory:.1f}GB")
    else:
        print("No GPU available - will use CPU")
    
    # Try different models in order of preference
    models_to_try = [
        ("meta-llama/Llama-3.2-1B-Instruct", "Smallest Llama 3.2 model"),
        ("meta-llama/Llama-3.2-3B-Instruct", "Medium Llama 3.2 model"), 
        ("meta-llama/Meta-Llama-3-8B-Instruct", "Full Llama 3-8B with quantization")
    ]
    
    for model_name, description in models_to_try:
        print(f"\nTrying {description}...")
        print(f"Model: {model_name}")
        
        model, tokenizer, status = load_quantized_llama(model_name, use_4bit=True)
        
        if model is not None and tokenizer is not None:
            print(f"Success: {status}")
            
            # Test inference
            print("\nTesting inference...")
            test_response = test_model_inference(model, tokenizer, "What is mental health?")
            print(f"Test response: {test_response[:200]}...")
            
            # Clean up
            del model, tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            print(f"\nSUCCESS: Successfully loaded and tested {description}")
            break
        else:
            print(f"Failed: {status}")
    
    else:
        print("\nAll model loading attempts failed")
        print("Recommendation: Use offline mode (already implemented)")

if __name__ == "__main__":
    main() 