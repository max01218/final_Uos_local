#!/usr/bin/env python3
"""
Start OPRO Integrated System (Optimized, No Emoji)
Runs both FastAPI server and OPRO scheduler, with direct log output and health check.
"""

import subprocess
import time
import signal
import sys
import os
import argparse
from datetime import datetime
import urllib.request

class OPROIntegratedSystem:
    def __init__(self, debug_mode=True):
        self.fastapi_process = None
        self.scheduler_process = None
        self.running = True
        self.debug_mode = debug_mode
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def signal_handler(self, signum, frame):
        print(f"\nReceived signal {signum}, shutting down...")
        self.running = False
        self.cleanup()
        sys.exit(0)
    
    def start_fastapi_server(self):
        print("Starting FastAPI server...")
        try:
            # Ensure environment variables are passed to subprocess
            env = os.environ.copy()
            # Set debug mode based on initialization
            env['SHOW_PROMPT_DEBUG'] = 'true' if self.debug_mode else 'false'
            
            self.fastapi_process = subprocess.Popen(
                [sys.executable, "fastapi_server.py"],
                stdout=None,
                stderr=None,
                text=True,
                env=env
            )
            print("FastAPI server started.")
            debug_status = env.get('SHOW_PROMPT_DEBUG', 'false')
            print(f"Debug output: {'enabled' if debug_status.lower() == 'true' else 'disabled'}")
            return True
        except Exception as e:
            print(f"Failed to start FastAPI server: {e}")
            return False
    
    def wait_for_fastapi(self, timeout=10):
        print("Waiting for FastAPI server to become available...")
        url = "http://localhost:8000/health"
        for i in range(timeout):
            try:
                with urllib.request.urlopen(url, timeout=2) as resp:
                    if resp.status == 200:
                        print("FastAPI health check passed.")
                        return True
            except Exception:
                pass
            time.sleep(1)
        print("FastAPI health check failed: server not available after timeout.")
        return False
    
    def start_opro_scheduler(self):
        print("Starting OPRO scheduler...")
        try:
            # Ensure environment variables are passed to subprocess
            env = os.environ.copy()
            
            self.scheduler_process = subprocess.Popen(
                [sys.executable, "auto_opro_scheduler.py"],
                stdout=None,
                stderr=None,
                text=True,
                env=env
            )
            print("OPRO scheduler started.")
            return True
        except Exception as e:
            print(f"Failed to start OPRO scheduler: {e}")
            return False
    
    def monitor_processes(self):
        while self.running:
            if self.fastapi_process and self.fastapi_process.poll() is not None:
                print("FastAPI server stopped unexpectedly.")
                break
            if self.scheduler_process and self.scheduler_process.poll() is not None:
                print("OPRO scheduler stopped unexpectedly.")
                break
            time.sleep(5)
    
    def cleanup(self):
        print("Cleaning up processes...")
        if self.fastapi_process:
            print("Stopping FastAPI server...")
            self.fastapi_process.terminate()
            try:
                self.fastapi_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.fastapi_process.kill()
        if self.scheduler_process:
            print("Stopping OPRO scheduler...")
            self.scheduler_process.terminate()
            try:
                self.scheduler_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.scheduler_process.kill()
        print("Cleanup completed.")
    
    def run(self):
        print("OPRO Integrated System")
        print("=" * 50)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        if not self.start_fastapi_server():
            print("Cannot start system without FastAPI server.")
            return False
        if not self.wait_for_fastapi():
            print("FastAPI server did not become available. Exiting.")
            self.cleanup()
            return False
        if not self.start_opro_scheduler():
            print("OPRO scheduler failed to start, continuing with FastAPI only.")
        print("\nSystem started successfully!")
        print("Available endpoints:")
        print("   - Health check: http://localhost:8000/health")
        print("   - Main API: http://localhost:8000/api/empathetic_professional")
        print("   - Feedback: http://localhost:8000/api/feedback")
        print("   - Reset conversation: http://localhost:8000/api/reset_conversation")
        print("\nOPRO scheduler will automatically optimize prompts based on user feedback.")
        print("Check interactions.json for collected feedback data.")
        
        debug_status = "enabled" if self.debug_mode else "disabled"
        print(f"\nDebug output: {debug_status}")
        if self.debug_mode:
            print("  - Detailed prompt information will be displayed")
            print("  - RAG retrieval context will be shown")
            print("  - LLM input/output will be logged")
        else:
            print("  - Only basic logging will be shown")
            print("  - To enable debug: python start_opro_integrated_system.py --debug")
        
        print("\nPress Ctrl+C to stop the system.")
        try:
            self.monitor_processes()
        except KeyboardInterrupt:
            print("\nShutdown requested by user.")
        finally:
            self.cleanup()
        return True

def check_dependencies():
    required_files = [
        "fastapi_server.py",
        "auto_opro_scheduler.py",
        "ICD11_OPRO/prompts/optimized_prompt.txt"
    ]
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    if missing_files:
        print("Missing required files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        return False
    print("All required files found.")
    return True

def main():
    parser = argparse.ArgumentParser(description='OPRO Integrated System')
    parser.add_argument('--debug', action='store_true', default=True,
                       help='Enable debug output (default: True)')
    parser.add_argument('--no-debug', action='store_true', 
                       help='Disable debug output')
    
    args = parser.parse_args()
    
    # Determine debug mode
    debug_mode = args.debug and not args.no_debug
    
    print("Checking system dependencies...")
    if not check_dependencies():
        print("\nSystem dependencies not met.")
        print("Please ensure all required files are present.")
        return
    
    print(f"Debug mode: {'enabled' if debug_mode else 'disabled'}")
    system = OPROIntegratedSystem(debug_mode=debug_mode)
    success = system.run()
    if success:
        print("\nSystem shutdown completed successfully.")
    else:
        print("\nSystem encountered errors during startup.")

if __name__ == "__main__":
    main() 