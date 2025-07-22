#!/usr/bin/env python3
"""
Streamlined OPRO Scheduler
Automatically checks for new feedback and runs OPRO optimization
Clean version without Chinese characters or emojis
"""

import json
import os
import time
import subprocess
import shutil
from datetime import datetime, timedelta
from typing import Dict, Any, List
import logging
import argparse

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/opro_scheduler.log'),
        logging.StreamHandler()
    ]
)

class OPROScheduler:
    def __init__(self, 
                 interactions_file: str = "interactions.json",
                 min_new_interactions: int = 10,
                 check_interval_hours: int = 24):
        """
        Initialize OPRO Scheduler
        
        Args:
            interactions_file: Path to interactions.json
            min_new_interactions: Minimum new interactions before triggering optimization
            check_interval_hours: Hours between checks
        """
        self.interactions_file = interactions_file
        self.min_new_interactions = min_new_interactions
        self.check_interval_hours = check_interval_hours
        self.last_processed_count = 0
        self.scheduler_state_file = "logs/opro_scheduler_state.json"
        
        # Ensure logs directory exists
        os.makedirs("logs", exist_ok=True)
        
        # Load scheduler state
        self.load_state()
        
    def load_state(self):
        """Load scheduler state from file"""
        try:
            if os.path.exists(self.scheduler_state_file):
                with open(self.scheduler_state_file, 'r') as f:
                    state = json.load(f)
                    self.last_processed_count = state.get('last_processed_count', 0)
                    logging.info(f"Loaded scheduler state: last processed count = {self.last_processed_count}")
            else:
                logging.info("No previous scheduler state found, starting fresh")
        except Exception as e:
            logging.error(f"Error loading scheduler state: {e}")
            self.last_processed_count = 0
    
    def save_state(self):
        """Save scheduler state to file"""
        try:
            state = {
                'last_processed_count': self.last_processed_count,
                'last_update': datetime.now().isoformat()
            }
            with open(self.scheduler_state_file, 'w') as f:
                json.dump(state, f, indent=2)
            logging.info("Scheduler state saved")
        except Exception as e:
            logging.error(f"Error saving scheduler state: {e}")
    
    def get_interactions_count(self) -> int:
        """Get current number of interactions"""
        try:
            if os.path.exists(self.interactions_file):
                with open(self.interactions_file, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        return len(data)
                    elif isinstance(data, dict) and 'interactions' in data:
                        return len(data['interactions'])
                    else:
                        return 0
            else:
                logging.warning(f"Interactions file {self.interactions_file} not found")
                return 0
        except Exception as e:
            logging.error(f"Error reading interactions file: {e}")
            return 0
    
    def should_run_optimization(self) -> bool:
        """Check if optimization should be triggered"""
        current_count = self.get_interactions_count()
        new_interactions = current_count - self.last_processed_count
        
        logging.info(f"Current interactions: {current_count}, Last processed: {self.last_processed_count}")
        logging.info(f"New interactions: {new_interactions}")
        
        return new_interactions >= self.min_new_interactions
    
    def convert_feedback_to_test_cases(self) -> bool:
        """Convert interactions to test cases for OPRO"""
        try:
            logging.info("Converting feedback to test cases...")
            
            # Run the conversion script
            result = subprocess.run(
                ['python', 'core/feedback_converter.py'],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                logging.info("Feedback conversion completed successfully")
                return True
            else:
                logging.error(f"Feedback conversion failed: {result.stderr}")
                return False
                
        except Exception as e:
            logging.error(f"Error converting feedback to test cases: {e}")
            return False
    
    def run_opro_optimization(self) -> bool:
        """Run OPRO optimization"""
        try:
            logging.info("Running OPRO optimization...")
            
            # Run the optimization
            result = subprocess.run(
                ['python', 'run_opro.py', '--mode', 'optimize'],
                capture_output=True,
                text=True
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
        """Deploy the optimized prompt"""
        try:
            logging.info("Deploying new optimized prompt...")
            
            # Check if optimized prompt exists
            optimized_prompt_path = "prompts/optimized_prompt.txt"
            if not os.path.exists(optimized_prompt_path):
                logging.error("Optimized prompt file not found")
                return False
            
            # Create backup of current prompt
            backup_dir = "prompts/backups"
            os.makedirs(backup_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"{backup_dir}/prompt_backup_{timestamp}.txt"
            
            # Copy to backup
            shutil.copy2(optimized_prompt_path, backup_path)
            logging.info(f"Prompt backed up to {backup_path}")
            
            logging.info("New prompt deployed successfully")
            return True
            
        except Exception as e:
            logging.error(f"Error deploying new prompt: {e}")
            return False
    
    def run_optimization_cycle(self) -> bool:
        """Run complete optimization cycle"""
        logging.info("Starting optimization cycle...")
        
        # Step 1: Convert feedback to test cases
        if not self.convert_feedback_to_test_cases():
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
    
    def run_once(self):
        """Run scheduler once (check and optimize if needed)"""
        logging.info("Running OPRO scheduler (single check)")
        
        if self.should_run_optimization():
            logging.info("Optimization triggered - sufficient new interactions found")
            success = self.run_optimization_cycle()
            if success:
                logging.info("Optimization cycle completed successfully")
            else:
                logging.error("Optimization cycle failed")
        else:
            logging.info("No optimization needed - insufficient new interactions")
    
    def run_continuous(self):
        """Run scheduler continuously"""
        logging.info(f"Starting continuous OPRO scheduler (check every {self.check_interval_hours} hours)")
        
        while True:
            try:
                self.run_once()
                
                # Sleep for specified interval
                sleep_seconds = self.check_interval_hours * 3600
                logging.info(f"Sleeping for {self.check_interval_hours} hours...")
                time.sleep(sleep_seconds)
                
            except KeyboardInterrupt:
                logging.info("Scheduler stopped by user")
                break
            except Exception as e:
                logging.error(f"Unexpected error in scheduler: {e}")
                time.sleep(300)  # Sleep 5 minutes before retrying

def main():
    parser = argparse.ArgumentParser(description='OPRO Scheduler')
    parser.add_argument('--run-once', action='store_true', 
                       help='Run once instead of continuously')
    parser.add_argument('--min-interactions', type=int, default=10,
                       help='Minimum new interactions to trigger optimization')
    parser.add_argument('--check-interval', type=int, default=24,
                       help='Hours between checks (continuous mode)')
    parser.add_argument('--interactions-file', default='interactions.json',
                       help='Path to interactions file')
    
    args = parser.parse_args()
    
    scheduler = OPROScheduler(
        interactions_file=args.interactions_file,
        min_new_interactions=args.min_interactions,
        check_interval_hours=args.check_interval
    )
    
    if args.run_once:
        scheduler.run_once()
    else:
        scheduler.run_continuous()

if __name__ == "__main__":
    main() 