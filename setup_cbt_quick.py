#!/usr/bin/env python3
"""
Quick CBT Setup Script
Helps set up CBT data for the system
"""

import os
import sys
import subprocess

def check_cbt_directory():
    """Check if CBT_System directory exists"""
    if not os.path.exists("CBT_System"):
        print("❌ CBT_System directory not found!")
        return False
    print("✅ CBT_System directory found")
    return True

def check_cbt_data():
    """Check if CBT data exists"""
    cbt_data_dir = "CBT_System/cbt_data"
    if not os.path.exists(cbt_data_dir):
        print("❌ CBT data directory not found")
        return False
    
    # Check for processed data
    processed_dir = os.path.join(cbt_data_dir, "processed")
    if not os.path.exists(processed_dir):
        print("❌ CBT processed data not found")
        return False
    
    # Check for embeddings
    embeddings_dir = os.path.join(cbt_data_dir, "embeddings")
    if not os.path.exists(embeddings_dir):
        print("❌ CBT embeddings not found")
        return False
    
    print("✅ CBT data directories found")
    return True

def run_cbt_setup():
    """Run CBT setup process"""
    print("🔧 Setting up CBT system...")
    
    try:
        # Change to CBT_System directory
        original_dir = os.getcwd()
        os.chdir("CBT_System")
        
        # Run setup
        print("📥 Running CBT setup...")
        result = subprocess.run([sys.executable, "setup.py"], 
                              capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✅ CBT setup completed successfully!")
            print("Output:", result.stdout[-200:] if result.stdout else "No output")
        else:
            print("❌ CBT setup failed!")
            print("Error:", result.stderr[-200:] if result.stderr else "No error output")
            return False
            
    except subprocess.TimeoutExpired:
        print("⏰ CBT setup timed out (5 minutes)")
        return False
    except Exception as e:
        print(f"❌ Error running CBT setup: {e}")
        return False
    finally:
        # Change back to original directory
        os.chdir(original_dir)
    
    return True

def test_cbt_system():
    """Test if CBT system is working"""
    print("🧪 Testing CBT system...")
    
    try:
        sys.path.append('CBT_System')
        from CBT_System.integration import CBTIntegration
        
        cbt = CBTIntegration()
        status = cbt.get_cbt_status()
        
        if status['available']:
            print("✅ CBT system is working!")
            print(f"  - Techniques: {status['total_techniques']}")
            print(f"  - Content: {status['total_content']}")
            print(f"  - Categories: {status['categories']}")
            return True
        else:
            print("❌ CBT system not available")
            return False
            
    except Exception as e:
        print(f"❌ CBT test failed: {e}")
        return False

def main():
    print("🌟 CBT Quick Setup Tool")
    print("="*30)
    
    # Check prerequisites
    if not check_cbt_directory():
        print("\n❌ Cannot proceed without CBT_System directory")
        return
    
    # Check if CBT data already exists
    if check_cbt_data():
        print("\n📊 CBT data appears to exist. Testing system...")
        if test_cbt_system():
            print("\n🎉 CBT system is already working!")
            print("You can now run: python quick_demo.py")
            return
        else:
            print("\n⚠️  CBT data exists but system not working. Rebuilding...")
    
    # Set up CBT system
    print("\n🔧 Setting up CBT system...")
    print("This may take a few minutes...")
    
    if run_cbt_setup():
        print("\n🧪 Testing setup...")
        if test_cbt_system():
            print("\n🎉 CBT setup completed successfully!")
            print("You can now run:")
            print("  python quick_demo.py")
            print("  python test_cbt_icd11_demo.py")
        else:
            print("\n❌ Setup completed but system still not working")
            print("Please check the CBT_System logs for errors")
    else:
        print("\n❌ CBT setup failed")
        print("Please check the error messages above")

if __name__ == "__main__":
    main() 