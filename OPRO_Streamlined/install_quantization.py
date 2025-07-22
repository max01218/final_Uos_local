#!/usr/bin/env python3
"""
Install Quantization Support for Iridis5
Install bitsandbytes and other required libraries for 4-bit quantization
"""

import subprocess
import sys
import os

def install_package(package_name, package_version=None):
    """Install a package using pip"""
    try:
        if package_version:
            cmd = [sys.executable, "-m", "pip", "install", f"{package_name}=={package_version}"]
        else:
            cmd = [sys.executable, "-m", "pip", "install", package_name]
        
        print(f"Installing {package_name}...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"SUCCESS: Successfully installed {package_name}")
            return True
        else:
            print(f"ERROR: Failed to install {package_name}")
            print(f"Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"ERROR: Exception installing {package_name}: {e}")
        return False

def check_installation(package_name):
    """Check if a package is installed"""
    try:
        __import__(package_name)
        print(f"OK: {package_name} is already installed")
        return True
    except ImportError:
        print(f"MISSING: {package_name} is not installed")
        return False

def main():
    print("Installing Quantization Support for Iridis5")
    print("="*50)
    
    # Check current environment
    print(f"Python version: {sys.version}")
    print(f"Platform: {sys.platform}")
    
    # List of packages to install
    packages = [
        ("bitsandbytes", None),  # For 4-bit quantization
        ("accelerate", "0.20.3"),  # For distributed loading
    ]
    
    # Check which packages are already installed
    print("\nChecking current installations...")
    for package_name, version in packages:
        check_installation(package_name.replace("-", "_"))
    
    # Install missing packages
    print("\nInstalling missing packages...")
    success_count = 0
    
    for package_name, version in packages:
        if install_package(package_name, version):
            success_count += 1
    
    print(f"\nInstallation complete: {success_count}/{len(packages)} packages installed")
    
    # Test quantization support
    print("\nTesting quantization support...")
    try:
        import bitsandbytes as bnb
        import torch
        
        print("SUCCESS: bitsandbytes imported successfully")
        
        if torch.cuda.is_available():
            print("SUCCESS: CUDA is available")
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("WARNING: CUDA not available")
        
        print("\nQuantization support ready!")
        
    except ImportError as e:
        print(f"ERROR: Quantization test failed: {e}")
        print("You may need to install manually:")
        print("  pip install bitsandbytes accelerate")

if __name__ == "__main__":
    main() 