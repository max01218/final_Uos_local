#!/usr/bin/env python3
"""
Model Download Script for Iridis5 Login Nodes
Download Llama models on login nodes for use on compute nodes
"""

import os
import sys
from pathlib import Path
import json
import subprocess
from datetime import datetime

def check_network():
    """Check if we have internet access"""
    try:
        import requests
        response = requests.get("https://huggingface.co", timeout=10)
        return response.status_code == 200
    except:
        return False

def check_node_type():
    """Try to determine if we're on a login node or compute node"""
    hostname = subprocess.run(['hostname'], capture_output=True, text=True).stdout.strip()
    
    # Iridis5 login nodes typically have specific naming patterns
    if any(pattern in hostname.lower() for pattern in ['login', 'head', 'master']):
        return "login"
    elif any(pattern in hostname.lower() for pattern in ['compute', 'node', 'cn']):
        return "compute"
    else:
        return "unknown"

def download_model(model_name, cache_dir=None):
    """Download a model using transformers"""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        print(f"Downloading model: {model_name}")
        
        # Set cache directory if specified
        if cache_dir:
            os.environ['TRANSFORMERS_CACHE'] = str(cache_dir)
            os.environ['HF_HOME'] = str(cache_dir)
        
        # Download tokenizer first (smaller)
        print("  Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Download model
        print("  Downloading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            low_cpu_mem_usage=True
        )
        
        print(f"SUCCESS: Successfully downloaded {model_name}")
        return True
        
    except Exception as e:
        print(f"ERROR: Failed to download {model_name}: {e}")
        return False

def get_model_info(model_name):
    """Get information about a model"""
    try:
        import requests
        url = f"https://huggingface.co/api/models/{model_name}"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            size_gb = data.get('modelSize', 0) / (1024**3) if data.get('modelSize') else 0
            return {
                'size_gb': size_gb,
                'tags': data.get('tags', []),
                'downloads': data.get('downloads', 0)
            }
    except:
        pass
    return {'size_gb': 0, 'tags': [], 'downloads': 0}

def main():
    print("Llama Model Download Script for Iridis5")
    print("="*50)
    
    # Check environment
    node_type = check_node_type()
    has_network = check_network()
    
    print(f"Node type: {node_type}")
    print(f"Network access: {'Yes' if has_network else 'No'}")
    
    if node_type == "compute":
        print("\nWARNING: You appear to be on a compute node!")
        print("This script should be run on a LOGIN NODE with internet access.")
        print("Please run this on a login node first, then use the models on compute nodes.")
        return
    
    if not has_network:
        print("\nERROR: No internet access detected!")
        print("Please ensure you're on a login node with internet connectivity.")
        return
    
    # Setup cache directory
    cache_dir = Path.home() / ".cache" / "huggingface"
    print(f"\nCache directory: {cache_dir}")
    
    # Models to download (ordered by size/preference)
    models_to_download = [
        {
            'name': 'meta-llama/Llama-3.2-1B-Instruct',
            'description': 'Llama 3.2 1B (smallest, ~2GB)',
            'priority': 1
        },
        {
            'name': 'meta-llama/Llama-3.2-3B-Instruct', 
            'description': 'Llama 3.2 3B (medium, ~6GB)',
            'priority': 2
        },
        {
            'name': 'meta-llama/Meta-Llama-3-8B-Instruct',
            'description': 'Llama 3 8B (large, ~16GB)',
            'priority': 3
        }
    ]
    
    print("\nAvailable models to download:")
    for i, model in enumerate(models_to_download, 1):
        info = get_model_info(model['name'])
        print(f"  {i}. {model['description']}")
        print(f"     Size: ~{info['size_gb']:.1f}GB, Downloads: {info['downloads']:,}")
    
    # Get user choice
    print("\nWhich models would you like to download?")
    print("  1. Download smallest model only (recommended)")
    print("  2. Download small + medium models")
    print("  3. Download all models")
    print("  4. Custom selection")
    
    choice = input("Enter choice (1-4): ").strip()
    
    models_selected = []
    if choice == '1':
        models_selected = [models_to_download[0]]
    elif choice == '2':
        models_selected = models_to_download[:2]
    elif choice == '3':
        models_selected = models_to_download
    elif choice == '4':
        print("\nEnter model numbers to download (e.g., 1,3):")
        indices = input("Models: ").strip().split(',')
        try:
            models_selected = [models_to_download[int(i.strip())-1] for i in indices]
        except (ValueError, IndexError):
            print("Invalid selection")
            return
    else:
        print("Invalid choice")
        return
    
    # Download selected models
    print(f"\nDownloading {len(models_selected)} models...")
    success_count = 0
    
    for model in models_selected:
        print(f"\n{'='*60}")
        if download_model(model['name'], cache_dir):
            success_count += 1
    
    print(f"\n{'='*60}")
    print(f"Download complete: {success_count}/{len(models_selected)} models downloaded")
    
    # Create download report
    report = {
        'timestamp': datetime.now().isoformat(),
        'node_type': node_type,
        'cache_directory': str(cache_dir),
        'models_attempted': [m['name'] for m in models_selected],
        'successful_downloads': success_count,
        'total_attempts': len(models_selected)
    }
    
    with open('download_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    if success_count > 0:
        print(f"\nSUCCESS: Models downloaded to: {cache_dir}")
        print("\nNext steps:")
        print("1. Copy your OPRO_Streamlined directory to the compute node")
        print("2. Run the quantized model loader: python load_quantized_llama.py")
        print("3. Run OPRO optimization with Llama support")
        
        print(f"\nDownload report saved to: download_report.json")
    else:
        print("\nERROR: No models were downloaded successfully")
        print("Please check your internet connection and try again")

if __name__ == "__main__":
    main() 