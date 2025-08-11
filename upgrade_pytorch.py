#!/usr/bin/env python3
"""
Script to upgrade PyTorch to resolve security vulnerability
"""

import subprocess
import sys
import os

def check_current_torch_version():
    """Check current PyTorch version"""
    try:
        import torch
        print(f"Current PyTorch version: {torch.__version__}")
        return torch.__version__
    except ImportError:
        print("PyTorch not installed")
        return None

def upgrade_pytorch():
    """Upgrade PyTorch to version 2.6+ to resolve security vulnerability"""
    print("Upgrading PyTorch to resolve security vulnerability...")
    print("=" * 60)
    
    # Check current version
    current_version = check_current_torch_version()
    
    # Upgrade commands for different scenarios
    upgrade_commands = [
        # Try pip upgrade first
        [sys.executable, "-m", "pip", "install", "--upgrade", "torch>=2.6.0"],
        
        # If that fails, try with --force-reinstall
        [sys.executable, "-m", "pip", "install", "--force-reinstall", "torch>=2.6.0"],
        
        # Alternative: install specific version
        [sys.executable, "-m", "pip", "install", "torch==2.6.0"],
    ]
    
    for i, cmd in enumerate(upgrade_commands):
        print(f"\nAttempt {i+1}: {' '.join(cmd)}")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print("Success!")
            print("Output:", result.stdout)
            break
        except subprocess.CalledProcessError as e:
            print(f"Failed: {e}")
            print("Error output:", e.stderr)
            if i == len(upgrade_commands) - 1:
                print("\nAll upgrade attempts failed. Please try manual installation:")
                print("pip install torch>=2.6.0 --force-reinstall")
                return False
    
    # Verify upgrade
    new_version = check_current_torch_version()
    if new_version and new_version != current_version:
        print(f"\nPyTorch successfully upgraded from {current_version} to {new_version}")
        return True
    else:
        print("\nPyTorch upgrade verification failed")
        return False

def install_safetensors():
    """Install safetensors for safer model loading"""
    print("\nInstalling safetensors for safer model loading...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "safetensors"], check=True)
        print("Safetensors installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to install safetensors: {e}")
        return False

def main():
    """Main upgrade process"""
    print("PyTorch Security Vulnerability Fix")
    print("=" * 60)
    print("This script will upgrade PyTorch to resolve CVE-2025-32434")
    print("See: https://nvd.nist.gov/vuln/detail/CVE-2025-32434")
    print()
    
    # Upgrade PyTorch
    if upgrade_pytorch():
        print("\nPyTorch upgrade completed successfully!")
    else:
        print("\nPyTorch upgrade failed. Please try manual installation.")
        return False
    
    # Install safetensors
    install_safetensors()
    
    print("\n" + "=" * 60)
    print("Upgrade process completed!")
    print("You can now restart your FastAPI server.")
    return True

if __name__ == "__main__":
    main() 