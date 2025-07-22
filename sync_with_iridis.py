#!/usr/bin/env python3
"""
Local to Iridis5 Synchronization Script
Automated OPRO optimization workflow: Local data collection -> Iridis5 optimization -> Local result application
"""

import os
import sys
import json
import subprocess
import time
import logging
from datetime import datetime
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IridisSync:
    def __init__(self, username, iridis_host="iridis5.soton.ac.uk"):
        self.username = username
        self.iridis_host = iridis_host
        self.remote_path = f"~/opro_workspace"
        self.local_opro_path = "ICD11_OPRO"
        
    def upload_test_cases(self, local_file):
        """Upload test cases to Iridis5"""
        try:
            remote_file = f"{self.username}@{self.iridis_host}:{self.remote_path}/ICD11_OPRO/tests/feedback_testcases_latest.json"
            
            cmd = [
                "scp", local_file, remote_file
            ]
            
            logger.info(f"Uploading {local_file} to Iridis5...")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("Test cases uploaded successfully")
                return True
            else:
                logger.error(f"Upload failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Upload error: {e}")
            return False
    
    def submit_opro_job(self):
        """Submit OPRO job to Iridis5"""
        try:
            cmd = [
                "ssh", f"{self.username}@{self.iridis_host}",
                f"cd {self.remote_path} && sbatch opro_job.slurm"
            ]
            
            logger.info("Submitting OPRO job to Iridis5...")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                # Extract job ID
                job_id = result.stdout.strip().split()[-1]
                logger.info(f"Job submitted successfully. Job ID: {job_id}")
                return job_id
            else:
                logger.error(f"Job submission failed: {result.stderr}")
                return None
                
        except Exception as e:
            logger.error(f"Job submission error: {e}")
            return None
    
    def check_job_status(self, job_id):
        """Check job status"""
        try:
            cmd = [
                "ssh", f"{self.username}@{self.iridis_host}",
                f"squeue -j {job_id}"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                output = result.stdout.strip()
                if job_id in output:
                    if "R" in output:
                        return "RUNNING"
                    elif "PD" in output:
                        return "PENDING"
                    elif "CG" in output:
                        return "COMPLETING"
                    else:
                        return "UNKNOWN"
                else:
                    return "COMPLETED"
            else:
                return "ERROR"
                
        except Exception as e:
            logger.error(f"Status check error: {e}")
            return "ERROR"
    
    def download_results(self):
        """Download optimization results"""
        try:
            remote_file = f"{self.username}@{self.iridis_host}:{self.remote_path}/ICD11_OPRO/prompts/optimized_prompt.txt"
            local_file = f"{self.local_opro_path}/prompts/optimized_prompt_iridis.txt"
            
            cmd = [
                "scp", remote_file, local_file
            ]
            
            logger.info("Downloading optimized prompt from Iridis5...")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info(f"Results downloaded to {local_file}")
                
                # Also download summary file
                remote_summary = f"{self.username}@{self.iridis_host}:{self.remote_path}/ICD11_OPRO/prompts/optimized_prompt_summary.json"
                local_summary = f"{self.local_opro_path}/prompts/optimized_prompt_summary.json"
                
                subprocess.run(["scp", remote_summary, local_summary], capture_output=True)
                
                return True
            else:
                logger.error(f"Download failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Download error: {e}")
            return False
    
    def deploy_optimized_prompt(self):
        """Deploy optimized prompt"""
        try:
            iridis_prompt = f"{self.local_opro_path}/prompts/optimized_prompt_iridis.txt"
            current_prompt = f"{self.local_opro_path}/prompts/optimized_prompt.txt"
            
            if os.path.exists(iridis_prompt):
                # Backup current prompt
                if os.path.exists(current_prompt):
                    backup_name = f"optimized_prompt_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                    os.rename(current_prompt, f"{self.local_opro_path}/prompts/{backup_name}")
                
                # Deploy new prompt
                os.rename(iridis_prompt, current_prompt)
                logger.info("Optimized prompt deployed successfully")
                return True
            else:
                logger.error("No Iridis5 optimized prompt found")
                return False
                
        except Exception as e:
            logger.error(f"Deployment error: {e}")
            return False
    
    def run_optimization_cycle(self):
        """Run complete optimization cycle"""
        try:
            # 1. Find latest test cases
            test_files = list(Path("tests").glob("feedback_testcases_*.json"))
            if not test_files:
                logger.error("No test cases found")
                return False
            
            latest_test_file = max(test_files, key=os.path.getctime)
            logger.info(f"Using test file: {latest_test_file}")
            
            # 2. Upload test cases
            if not self.upload_test_cases(str(latest_test_file)):
                return False
            
            # 3. Submit job
            job_id = self.submit_opro_job()
            if not job_id:
                return False
            
            # 4. Wait for job completion
            logger.info("Waiting for job completion...")
            while True:
                status = self.check_job_status(job_id)
                logger.info(f"Job status: {status}")
                
                if status == "COMPLETED":
                    break
                elif status in ["ERROR", "FAILED"]:
                    logger.error("Job failed")
                    return False
                
                time.sleep(30)  # Wait 30 seconds before checking again
            
            # 5. Download results
            if not self.download_results():
                return False
            
            # 6. Deploy new prompt
            if not self.deploy_optimized_prompt():
                return False
            
            logger.info("Optimization cycle completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Optimization cycle error: {e}")
            return False

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Sync with Iridis5 for OPRO optimization")
    parser.add_argument("--username", required=True, help="Iridis5 username")
    parser.add_argument("--setup", action="store_true", help="Setup Iridis5 environment")
    parser.add_argument("--run", action="store_true", help="Run optimization cycle")
    
    args = parser.parse_args()
    
    sync = IridisSync(args.username)
    
    if args.setup:
        logger.info("Please manually run the setup script on Iridis5:")
        logger.info("1. Copy iridis5_setup.sh to Iridis5")
        logger.info("2. Run: bash iridis5_setup.sh")
        logger.info("3. Copy your OPRO code to ~/opro_workspace/")
    
    if args.run:
        success = sync.run_optimization_cycle()
        sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 