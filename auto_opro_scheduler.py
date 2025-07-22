#!/usr/bin/env python3
"""
Automated OPRO Scheduler
Periodically checks for new feedback and runs OPRO optimization
"""

import json
import os
import time
import subprocess
import shutil
from datetime import datetime, timedelta
from typing import Dict, Any, List
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('opro_scheduler.log'),
        logging.StreamHandler()
    ]
)

class OPROScheduler:
    def __init__(self, 
                 interactions_file: str = "interactions.json",
                 opro_dir: str = "ICD11_OPRO",
                 min_new_interactions: int = 10,
                 check_interval_hours: int = 24):
        """
        Initialize OPRO Scheduler
        
        Args:
            interactions_file: Path to interactions.json
            opro_dir: Path to OPRO directory
            min_new_interactions: Minimum new interactions before triggering optimization
            check_interval_hours: Hours between checks
        """
        self.interactions_file = interactions_file
        self.opro_dir = opro_dir
        self.min_new_interactions = min_new_interactions
        self.check_interval_hours = check_interval_hours
        self.last_processed_count = 0
        self.scheduler_state_file = "opro_scheduler_state.json"
        
        # Load scheduler state
        self.load_state()
    
    def load_state(self):
        """Load scheduler state from file"""
        if os.path.exists(self.scheduler_state_file):
            try:
                with open(self.scheduler_state_file, 'r', encoding='utf-8') as f:
                    state = json.load(f)
                    self.last_processed_count = state.get('last_processed_count', 0)
                    logging.info(f"Loaded state: last_processed_count = {self.last_processed_count}")
            except Exception as e:
                logging.error(f"Error loading state: {e}")
                self.last_processed_count = 0
        else:
            self.last_processed_count = 0
    
    def save_state(self):
        """Save scheduler state to file"""
        state = {
            'last_processed_count': self.last_processed_count,
            'last_update': datetime.now().isoformat()
        }
        try:
            with open(self.scheduler_state_file, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logging.error(f"Error saving state: {e}")
    
    def get_interactions_count(self) -> int:
        """Get current number of interactions"""
        if not os.path.exists(self.interactions_file):
            return 0
        
        try:
            with open(self.interactions_file, 'r', encoding='utf-8') as f:
                interactions = json.load(f)
                return len(interactions)
        except Exception as e:
            logging.error(f"Error reading interactions: {e}")
            return 0
    
    def should_run_optimization(self) -> bool:
        """Check if optimization should run"""
        current_count = self.get_interactions_count()
        new_interactions = current_count - self.last_processed_count
        
        logging.info(f"Current interactions: {current_count}, Last processed: {self.last_processed_count}, New: {new_interactions}")
        
        return new_interactions >= self.min_new_interactions
    
    def run_testcase_conversion(self) -> bool:
        """Run the testcase conversion script"""
        try:
            logging.info("Running testcase conversion...")
            result = subprocess.run(
                ['python', 'auto_to_testcases.py'],
                capture_output=True,
                text=True,
                cwd=os.getcwd()
            )
            
            if result.returncode == 0:
                logging.info("Testcase conversion completed successfully")
                logging.info(result.stdout)
                return True
            else:
                logging.error(f"Testcase conversion failed: {result.stderr}")
                return False
                
        except Exception as e:
            logging.error(f"Error running testcase conversion: {e}")
            return False
    
    def run_opro_optimization(self) -> bool:
        """Run OPRO optimization"""
        try:
            logging.info("Running OPRO optimization...")
            
            # Check if headless OPRO script exists
            opro_script = os.path.join(self.opro_dir, "run_opro_headless.py")
            if not os.path.exists(opro_script):
                logging.error(f"OPRO headless script not found: {opro_script}")
                # Fallback to original script
                opro_script = os.path.join(self.opro_dir, "run_opro.py")
                if not os.path.exists(opro_script):
                    logging.error(f"OPRO script not found: {opro_script}")
                    return False
                script_name = "run_opro.py"
            else:
                script_name = "run_opro_headless.py"
            
            # Run the script from the OPRO directory
            result = subprocess.run(
                ['python', script_name],
                capture_output=True,
                text=True,
                cwd=self.opro_dir
            )
            
            if result.returncode == 0:
                logging.info("OPRO optimization completed successfully")
                logging.info(result.stdout)
                return True
            else:
                logging.error(f"OPRO optimization failed: {result.stderr}")
                return False
                
        except Exception as e:
            logging.error(f"Error running OPRO optimization: {e}")
            return False
    
    def deploy_new_prompt(self) -> bool:
        """Deploy the new optimized prompt"""
        try:
            # Find the latest optimized prompt
            optimized_prompt_path = os.path.join(self.opro_dir, "prompts", "optimized_prompt.txt")
            if not os.path.exists(optimized_prompt_path):
                logging.error(f"Optimized prompt not found: {optimized_prompt_path}")
                return False
            
            # Create backup of current prompt
            backup_dir = "prompt_backups"
            os.makedirs(backup_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = os.path.join(backup_dir, f"prompt_backup_{timestamp}.txt")
            
            # Copy current prompt to backup if it exists
            current_prompt_path = os.path.join(self.opro_dir, "prompts", "current_prompt.txt")
            if os.path.exists(current_prompt_path):
                shutil.copy2(current_prompt_path, backup_path)
                logging.info(f"Created backup: {backup_path}")
            
            # Deploy new prompt
            shutil.copy2(optimized_prompt_path, current_prompt_path)
            logging.info(f"Deployed new prompt: {current_prompt_path}")
            
            return True
            
        except Exception as e:
            logging.error(f"Error deploying new prompt: {e}")
            return False
    
    def run_optimization_cycle(self) -> bool:
        """Run complete optimization cycle"""
        logging.info("Starting optimization cycle...")
        
        # Step 1: Convert feedback to test cases
        if not self.run_testcase_conversion():
            logging.error("Testcase conversion failed")
            return False
        
        # Step 2: Run OPRO optimization
        if not self.run_opro_optimization():
            logging.error("OPRO optimization failed")
            return False
        
        # Step 3: Deploy new prompt
        if not self.deploy_new_prompt():
            logging.error("Prompt deployment failed")
            return False
        
        # Update state
        self.last_processed_count = self.get_interactions_count()
        self.save_state()
        
        logging.info("Optimization cycle completed successfully")
        return True
    
    def run_scheduler(self):
        """Run the scheduler loop"""
        logging.info("Starting OPRO Scheduler...")
        logging.info(f"Check interval: {self.check_interval_hours} hours")
        logging.info(f"Min new interactions: {self.min_new_interactions}")
        
        while True:
            try:
                if self.should_run_optimization():
                    logging.info("Triggering optimization cycle...")
                    success = self.run_optimization_cycle()
                    
                    if success:
                        logging.info("Optimization cycle completed successfully")
                    else:
                        logging.error("Optimization cycle failed")
                else:
                    logging.info("No optimization needed")
                
                # Wait for next check
                sleep_seconds = self.check_interval_hours * 3600
                logging.info(f"Sleeping for {self.check_interval_hours} hours...")
                time.sleep(sleep_seconds)
                
            except KeyboardInterrupt:
                logging.info("Scheduler stopped by user")
                break
            except Exception as e:
                logging.error(f"Scheduler error: {e}")
                time.sleep(300)  # Wait 5 minutes before retrying

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="OPRO Automated Scheduler")
    parser.add_argument("--interactions", default="interactions.json", help="Path to interactions.json")
    parser.add_argument("--opro-dir", default="ICD11_OPRO", help="Path to OPRO directory")
    parser.add_argument("--min-interactions", type=int, default=10, help="Minimum new interactions before optimization")
    parser.add_argument("--check-interval", type=int, default=24, help="Check interval in hours")
    parser.add_argument("--run-once", action="store_true", help="Run optimization once and exit")
    
    args = parser.parse_args()
    
    scheduler = OPROScheduler(
        interactions_file=args.interactions,
        opro_dir=args.opro_dir,
        min_new_interactions=args.min_interactions,
        check_interval_hours=args.check_interval
    )
    
    if args.run_once:
        if scheduler.should_run_optimization():
            logging.info("Running optimization once...")
            scheduler.run_optimization_cycle()
        else:
            logging.info("No optimization needed")
    else:
        scheduler.run_scheduler()

if __name__ == "__main__":
    main() 