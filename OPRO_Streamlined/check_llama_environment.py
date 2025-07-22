#!/usr/bin/env python3
"""
Llama Environment Checker for Iridis5
Check system capabilities and provide solutions for using Llama models
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def check_gpu_memory():
    """Check GPU memory availability"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            print(f"GPU Count: {gpu_count}")
            
            for i in range(gpu_count):
                gpu_props = torch.cuda.get_device_properties(i)
                total_memory_gb = gpu_props.total_memory / (1024**3)
                gpu_name = torch.cuda.get_device_name(i)
                print(f"GPU {i}: {gpu_name}")
                print(f"  Total Memory: {total_memory_gb:.1f}GB")
                
                # Check available memory
                torch.cuda.empty_cache()
                allocated = torch.cuda.memory_allocated(i) / (1024**3)
                cached = torch.cuda.memory_reserved(i) / (1024**3)
                free = total_memory_gb - allocated - cached
                print(f"  Available: {free:.1f}GB")
                
                return total_memory_gb, free
        else:
            print("No CUDA GPUs available")
            return 0, 0
    except ImportError:
        print("PyTorch not available")
        return 0, 0

def check_network_connectivity():
    """Check if we can access HuggingFace"""
    try:
        import requests
        response = requests.get("https://huggingface.co", timeout=5)
        return response.status_code == 200
    except:
        return False

def check_huggingface_cache():
    """Check existing HuggingFace cache"""
    cache_dir = Path.home() / ".cache" / "huggingface"
    if cache_dir.exists():
        print(f"HuggingFace cache directory: {cache_dir}")
        
        # Look for existing models
        hub_dir = cache_dir / "hub"
        if hub_dir.exists():
            models = [d.name for d in hub_dir.iterdir() if d.is_dir()]
            print(f"Cached models: {len(models)}")
            
            # Look for Llama models specifically
            llama_models = [m for m in models if "llama" in m.lower()]
            if llama_models:
                print(f"Llama models found: {llama_models}")
                return llama_models
            else:
                print("No Llama models in cache")
        else:
            print("No hub cache directory")
    else:
        print("No HuggingFace cache directory")
    
    return []

def check_iridis_modules():
    """Check available modules on Iridis5"""
    try:
        result = subprocess.run(['module', 'avail'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            output = result.stdout + result.stderr  # module output goes to stderr
            
            # Look for relevant modules
            relevant = []
            for line in output.split('\n'):
                if any(keyword in line.lower() for keyword in 
                      ['pytorch', 'cuda', 'python', 'anaconda', 'miniconda']):
                    relevant.append(line.strip())
            
            if relevant:
                print("Relevant modules available:")
                for module in relevant[:10]:  # Show first 10
                    print(f"  {module}")
            else:
                print("No relevant modules found")
                
        return result.returncode == 0
    except FileNotFoundError:
        print("Module system not available")
        return False

def suggest_solutions(gpu_memory, has_network, cached_models):
    """Suggest solutions based on environment"""
    print("\n" + "="*60)
    print("RECOMMENDED SOLUTIONS:")
    print("="*60)
    
    if gpu_memory < 12:
        print("\nMEMORY LIMITATION (Need >12GB for Llama 3-8B)")
        print("Solutions:")
        print("1. Use quantized models (4-bit, 8-bit)")
        print("2. Use smaller models (Llama 3.2-1B, 3B)")
        print("3. Use CPU inference (slower)")
        
    if not has_network:
        print("\n NETWORK LIMITATION (No internet on compute nodes)")
        print("Solutions:")
        print("1. Pre-download models on login nodes")
        print("2. Use shared/scratch storage")
        print("3. Use offline mode (current system fallback)")
        
    if not cached_models:
        print("\n NO CACHED MODELS")
        print("Solutions:")
        print("1. Download on login node with internet")
        print("2. Copy from shared storage")
        print("3. Request system admin assistance")
    
    print("\n IMMEDIATE ACTIONS:")
    if gpu_memory >= 8:
        print(" Try quantized Llama 3-8B (4-bit)")
        print("   Command: Load with load_in_4bit=True")
    
    if gpu_memory >= 4:
        print(" Try Llama 3.2-3B model")
        print("   Model: meta-llama/Llama-3.2-3B-Instruct")
    
    print(" Use current system offline mode")
    print("   Already implemented and working!")

def main():
    print("Llama Environment Checker for Iridis5")
    print("="*50)
    
    # System info
    print(f"Python version: {sys.version}")
    print(f"Platform: {sys.platform}")
    print(f"Current directory: {os.getcwd()}")
    
    # Check GPU
    print("\n Checking GPU...")
    gpu_memory, free_memory = check_gpu_memory()
    
    # Check network
    print("\n Checking network...")
    has_network = check_network_connectivity()
    print(f"Internet access: {has_network}")
    
    # Check cache
    print("\n Checking HuggingFace cache...")
    cached_models = check_huggingface_cache()
    
    # Check modules
    print("\n Checking available modules...")
    has_modules = check_iridis_modules()
    
    # Suggest solutions
    suggest_solutions(gpu_memory, has_network, cached_models)
    
    # Generate environment report
    report = {
        "timestamp": str(subprocess.run(['date'], capture_output=True, text=True).stdout.strip()),
        "gpu_memory_gb": gpu_memory,
        "free_memory_gb": free_memory,
        "has_network": has_network,
        "cached_models": cached_models,
        "has_module_system": has_modules
    }
    
    with open("environment_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\n Environment report saved to: environment_report.json")

if __name__ == "__main__":
    main() 