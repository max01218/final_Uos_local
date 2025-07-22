#!/usr/bin/env python3
"""
Iridis5 Setup Checker
Determine current capabilities and recommend next steps
"""

import os
import sys
import subprocess
from pathlib import Path

def check_environment():
    """Check current environment and capabilities"""
    print("Iridis5 Llama Setup Checker")
    print("="*40)
    
    # Basic environment
    print(f"Python version: {sys.version}")
    print(f"Current directory: {os.getcwd()}")
    
    # Check if on login vs compute node
    try:
        hostname = subprocess.run(['hostname'], capture_output=True, text=True).stdout.strip()
        print(f"Hostname: {hostname}")
        
        if any(pattern in hostname.lower() for pattern in ['login', 'head']):
            node_type = "login"
        elif any(pattern in hostname.lower() for pattern in ['compute', 'node', 'cn']):
            node_type = "compute"
        else:
            node_type = "unknown"
        print(f"Node type: {node_type}")
    except:
        node_type = "unknown"
        print("Node type: unknown")
    
    # Check GPU availability
    gpu_available = False
    try:
        import torch
        if torch.cuda.is_available():
            gpu_available = True
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"GPU: {gpu_name}")
            print(f"GPU Memory: {total_memory:.1f}GB")
        else:
            print("GPU: Not available")
    except ImportError:
        print("PyTorch: Not available")
    
    # Check internet connectivity
    internet_available = False
    try:
        import requests
        response = requests.get("https://huggingface.co", timeout=5)
        internet_available = response.status_code == 200
        print(f"Internet access: {'Yes' if internet_available else 'No'}")
    except:
        print("Internet access: No")
    
    # Check for existing models
    cache_dir = Path.home() / ".cache" / "huggingface"
    models_available = False
    if cache_dir.exists():
        hub_dir = cache_dir / "hub"
        if hub_dir.exists():
            models = [d.name for d in hub_dir.iterdir() if d.is_dir() and "llama" in d.name.lower()]
            if models:
                models_available = True
                print(f"Cached models: {len(models)} Llama models found")
            else:
                print("Cached models: None found")
        else:
            print("Cached models: No cache directory")
    else:
        print("Cached models: No cache directory")
    
    # Check quantization support
    quantization_available = False
    try:
        import bitsandbytes
        quantization_available = True
        print("Quantization: bitsandbytes available")
    except ImportError:
        print("Quantization: bitsandbytes not available")
    
    # Check OPRO system
    opro_available = os.path.exists("ICD11_OPRO/run_opro.py")
    print(f"OPRO system: {'Available' if opro_available else 'Not found'}")
    
    print("\n" + "="*40)
    print("RECOMMENDATIONS:")
    print("="*40)
    
    # Provide recommendations based on current state
    if node_type == "login" and internet_available:
        print("1. DOWNLOAD MODELS (Recommended next step)")
        print("   python download_models_login.py")
        print("   This will download Llama models to your cache")
        
    elif node_type == "compute" or (node_type == "unknown" and gpu_available):
        if models_available and quantization_available:
            print("1. RUN LLAMA OPRO (Ready to go!)")
            print("   python run_opro_with_llama.py")
            
        elif models_available and not quantization_available:
            print("1. INSTALL QUANTIZATION SUPPORT")
            print("   python install_quantization.py")
            print("2. THEN RUN LLAMA OPRO")
            print("   python run_opro_with_llama.py")
            
        elif not models_available:
            print("1. DOWNLOAD MODELS (on login node first)")
            print("   Exit to login node and run: python download_models_login.py")
            print("2. OR USE OFFLINE MODE (works now)")
            print("   cd ICD11_OPRO && python run_opro.py")
            
        else:
            print("1. USE OFFLINE MODE (recommended)")
            print("   cd ICD11_OPRO && python run_opro.py")
    
    elif not internet_available and not models_available:
        if opro_available:
            print("1. USE OFFLINE MODE (best current option)")
            print("   cd ICD11_OPRO && python run_opro.py")
        else:
            print("1. CHECK OPRO SYSTEM AVAILABILITY")
            print("   Ensure ICD11_OPRO directory exists")
    
    # Job submission recommendations
    if node_type == "login":
        print("\n2. SUBMIT GPU JOB (after downloading models)")
        print("   sbatch submit_llama_job.slurm")
    
    # Always mention offline fallback
    if opro_available:
        print(f"\nFALLBACK OPTION:")
        print("  Offline OPRO is always available and proven effective")
        print("  cd ICD11_OPRO && python run_opro.py")
    
    return {
        'node_type': node_type,
        'gpu_available': gpu_available,
        'internet_available': internet_available,
        'models_available': models_available,
        'quantization_available': quantization_available,
        'opro_available': opro_available
    }

def main():
    try:
        status = check_environment()
        
        # Quick start suggestion
        print(f"\nQUICK START:")
        if status['opro_available']:
            if status['node_type'] == 'login' and status['internet_available']:
                print("  python download_models_login.py")
            elif status['gpu_available'] and status['models_available']:
                print("  python run_opro_with_llama.py")
            else:
                print("  cd ICD11_OPRO && python run_opro.py")
        else:
            print("  Check that OPRO_Streamlined/ICD11_OPRO exists")
            
    except Exception as e:
        print(f"Error during environment check: {e}")
        print("Try running: cd ICD11_OPRO && python run_opro.py")

if __name__ == "__main__":
    main() 