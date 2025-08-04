#!/usr/bin/env python3
"""
Test LLaMA model access and functionality
"""

import json
import os
from huggingface_hub import login, HfApi
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoConfig
import torch

# Global config
config = None

def load_config():
    """Load configuration from file"""
    global config
    try:
        with open("config/llama_access_config.json", "r", encoding="utf-8") as f:
            config = json.load(f)
        print("✓ Config loaded successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to load config: {e}")
        return False

def login_to_hf():
    """Login to Hugging Face"""
    try:
        print("Logging in to Hugging Face...")
        login(token=config["huggingface_token"])
        print("✓ Hugging Face login successful")
        return True
    except Exception as e:
        print(f"❌ Hugging Face login failed: {e}")
        return False

def test_model_access():
    """Test access to different LLaMA models"""
    print("\nTesting API access...")
    
    accessible_models = []
    for model_name in config["accessible_llama_models"]:
        try:
            # Test API access
            api = HfApi()
            model_info = api.model_info(model_name)
            print(f"✅ {model_name} - API access successful")
            accessible_models.append(model_name)
        except Exception as e:
            print(f"❌ {model_name} - API access failed: {e}")
    
    if not accessible_models:
        print("❌ No models are accessible")
        return False
    
    # Use the first accessible model for testing
    test_model = accessible_models[0]
    print(f"\nUsing accessible model for testing: {test_model}")
    
    return test_model

def simple_test(model_name):
    """Simple test with accessible model"""
    print("\n============================================================")
    print("Simple LLaMA Test")
    print("============================================================")
    print(f"Attempting to load: {model_name}")
    
    try:
        # Test tokenizer loading
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=config["huggingface_token"],
            trust_remote_code=True
        )
        print("✅ Tokenizer loaded successfully")
        
        # Test model loading (just config to save time)
        model_config = AutoConfig.from_pretrained(
            model_name,
            token=config["huggingface_token"],
            trust_remote_code=True
        )
        print("✅ Model config loaded successfully")
        
        print(f"✅ Simple test successful with {model_name}")
        return True
        
    except Exception as e:
        print(f"❌ Simple test failed: {e}")
        return False

def main():
    """Main test function"""
    print("Starting LLaMA access test...")
    print("=" * 60)
    print("LLaMA Access Test")
    print("=" * 60)
    
    # Load config
    if not load_config():
        return False
    
    # Login to Hugging Face
    if not login_to_hf():
        return False
    
    # Test model access
    test_model = test_model_access()
    if not test_model:
        print("❌ No accessible models found")
        return False
    
    # Test simple LLaMA with accessible model
    if not simple_test(test_model):
        print("❌ All tests failed")
        return False
    
    print("✅ All tests passed!")
    return True

if __name__ == "__main__":
    main() 