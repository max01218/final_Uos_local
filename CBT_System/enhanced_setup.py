#!/usr/bin/env python3
"""
Enhanced CBT System Setup and Management
Complete pipeline for enhanced data collection, processing, and vectorization
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import time

# Import enhanced modules
try:
    from enhanced_data_collector import EnhancedCBTDataCollector
    from enhanced_data_processor import EnhancedCBTDataProcessor  
    from enhanced_vectorizer import EnhancedCBTVectorizer
except ImportError as e:
    print(f"Error importing enhanced modules: {e}")
    print("Make sure all enhanced modules are in the CBT_System directory")
    sys.exit(1)

class EnhancedCBTSystemSetup:
    def __init__(self, base_dir="cbt_data"):
        self.base_dir = Path(base_dir)
        self.setup_logging()
        
        # Pipeline configuration
        self.pipeline_config = {
            "collection": {
                "enabled": True,
                "priority_filter": None,  # None = all priorities, 1 = priority 1 only, etc.
                "rate_limit_delay": 5,    # Seconds between source collections
                "max_retries": 3
            },
            "processing": {
                "enabled": True,
                "similarity_threshold": 0.85,  # For deduplication
                "min_quality_score": 60,       # Minimum enhanced quality score
                "enable_classification": True,
                "enable_structure_extraction": True
            },
            "vectorization": {
                "enabled": True,
                "model_strategies": [
                    ("primary", "content_based"),
                    ("clinical", "technique_focused")
                ],
                "create_hierarchical_indices": True,
                "optimize_performance": True
            }
        }
        
    def setup_logging(self):
        """Setup comprehensive logging for the enhanced system"""
        
        # Create logs directory
        logs_dir = self.base_dir / "logs"
        logs_dir.mkdir(exist_ok=True)
        
        # Setup main logger
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        formatter = logging.Formatter(log_format)
        
        # File handler for all logs
        main_log_file = logs_dir / f"enhanced_setup_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(main_log_file)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler for important messages
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(logging.INFO)
        
        # Setup logger
        self.logger = logging.getLogger("EnhancedCBTSetup")
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
    def check_dependencies(self) -> bool:
        """Check if all required dependencies are installed"""
        
        # Package name mappings: pip_name -> import_name
        package_mappings = {
            "requests": "requests",
            "beautifulsoup4": "bs4", 
            "sentence-transformers": "sentence_transformers",
            "faiss-cpu": "faiss",
            "numpy": "numpy",
            "scikit-learn": "sklearn",
            "nltk": "nltk",
            "torch": "torch"
        }
        
        missing_packages = []
        
        for pip_name, import_name in package_mappings.items():
            try:
                __import__(import_name)
                self.logger.debug(f"Package {pip_name} is available")
            except ImportError:
                missing_packages.append(pip_name)
                
        if missing_packages:
            self.logger.error(f"Missing required packages: {missing_packages}")
            print("\nTo install missing packages, run:")
            print(f"pip install {' '.join(missing_packages)}")
            return False
            
        return True
        
    def run_enhanced_collection(self) -> bool:
        """Run enhanced data collection with configuration"""
        
        if not self.pipeline_config["collection"]["enabled"]:
            self.logger.info("Data collection disabled in configuration")
            return True
            
        self.logger.info("Starting enhanced data collection")
        
        try:
            collector = EnhancedCBTDataCollector(str(self.base_dir))
            
            # Get configuration
            priority_filter = self.pipeline_config["collection"]["priority_filter"]
            
            # Run collection
            results = collector.collect_all_enhanced_sources(priority_filter=priority_filter)
            
            # Calculate success metrics
            total_items = sum(len(items) for items in results.values())
            successful_sources = len([k for k, v in results.items() if v])
            
            if total_items > 0:
                self.logger.info(f"Enhanced collection successful: {total_items} items from {successful_sources} sources")
                return True
            else:
                self.logger.warning("No data collected from enhanced sources")
                return False
                
        except Exception as e:
            self.logger.error(f"Enhanced collection failed: {e}")
            return False
            
    def run_enhanced_processing(self) -> bool:
        """Run enhanced data processing with configuration"""
        
        if not self.pipeline_config["processing"]["enabled"]:
            self.logger.info("Data processing disabled in configuration")
            return True
            
        self.logger.info("Starting enhanced data processing")
        
        try:
            processor = EnhancedCBTDataProcessor(str(self.base_dir))
            
            # Run processing
            output_path = processor.process_all_enhanced_data()
            
            if output_path:
                self.logger.info(f"Enhanced processing successful: {output_path}")
                return True
            else:
                self.logger.warning("No data processed")
                return False
                
        except Exception as e:
            self.logger.error(f"Enhanced processing failed: {e}")
            return False
            
    def run_enhanced_vectorization(self) -> bool:
        """Run enhanced vectorization with multiple strategies"""
        
        if not self.pipeline_config["vectorization"]["enabled"]:
            self.logger.info("Vectorization disabled in configuration")
            return True
            
        self.logger.info("Starting enhanced vectorization")
        
        try:
            vectorizer = EnhancedCBTVectorizer(str(self.base_dir))
            
            # Get model strategies from configuration
            model_strategies = self.pipeline_config["vectorization"]["model_strategies"]
            
            success_count = 0
            
            for model_choice, strategy in model_strategies:
                self.logger.info(f"Vectorizing with model: {model_choice}, strategy: {strategy}")
                
                try:
                    result = vectorizer.vectorize_enhanced_data(model_choice, strategy)
                    
                    if result:
                        self.logger.info(f"Vectorization successful: {result}")
                        success_count += 1
                    else:
                        self.logger.warning(f"Vectorization failed for {model_choice}/{strategy}")
                        
                except Exception as e:
                    self.logger.error(f"Vectorization failed for {model_choice}/{strategy}: {e}")
                    
            return success_count > 0
            
        except Exception as e:
            self.logger.error(f"Enhanced vectorization failed: {e}")
            return False
            
    def generate_system_report(self) -> Dict:
        """Generate comprehensive system report"""
        
        report = {
            "generation_timestamp": datetime.now().isoformat(),
            "system_status": "unknown",
            "data_collection": {"status": "not_run", "details": {}},
            "data_processing": {"status": "not_run", "details": {}},
            "vectorization": {"status": "not_run", "details": {}},
            "file_inventory": {},
            "recommendations": []
        }
        
        # Check data collection files
        raw_data_dirs = [
            "raw_data/government",
            "raw_data/professional_orgs", 
            "raw_data/international_orgs",
            "raw_data/clinical_guidelines",
            "raw_data/academic",
            "raw_data/nonprofit"
        ]
        
        collection_files = 0
        for data_dir in raw_data_dirs:
            dir_path = self.base_dir / data_dir
            if dir_path.exists():
                files = list(dir_path.glob("*.json"))
                collection_files += len(files)
                report["file_inventory"][data_dir] = len(files)
                
        report["data_collection"]["details"]["total_files"] = collection_files
        
        if collection_files > 0:
            report["data_collection"]["status"] = "completed"
        else:
            report["data_collection"]["status"] = "no_data"
            report["recommendations"].append("Run data collection to gather CBT resources")
            
        # Check processed data
        processed_dir = self.base_dir / "raw_data" / "processed"
        if processed_dir.exists():
            processed_files = list(processed_dir.glob("cbt_enhanced_processed_*.json"))
            if processed_files:
                report["data_processing"]["status"] = "completed"
                report["data_processing"]["details"]["processed_files"] = len(processed_files)
                
                # Get latest processed file info
                latest_file = max(processed_files, key=lambda f: f.stat().st_mtime)
                try:
                    with open(latest_file, 'r') as f:
                        data = json.load(f)
                        metadata = data.get("metadata", {})
                        report["data_processing"]["details"]["latest_metadata"] = metadata
                except:
                    pass
            else:
                report["data_processing"]["status"] = "no_data"
                report["recommendations"].append("Run data processing to clean and classify data")
        else:
            report["data_processing"]["status"] = "no_data"
            
        # Check vectorization
        embeddings_dir = self.base_dir / "embeddings"
        if embeddings_dir.exists():
            faiss_files = list(embeddings_dir.glob("*.faiss"))
            summary_files = list(embeddings_dir.glob("cbt_index_summary*.json"))
            
            if faiss_files and summary_files:
                report["vectorization"]["status"] = "completed"
                report["vectorization"]["details"]["faiss_indices"] = len(faiss_files)
                
                # Get latest summary
                latest_summary = max(summary_files, key=lambda f: f.stat().st_mtime)
                try:
                    with open(latest_summary, 'r') as f:
                        summary_data = json.load(f)
                        report["vectorization"]["details"]["latest_summary"] = summary_data
                except:
                    pass
            else:
                report["vectorization"]["status"] = "no_data"
                report["recommendations"].append("Run vectorization to create searchable indices")
        else:
            report["vectorization"]["status"] = "no_data"
            
        # Determine overall system status
        statuses = [
            report["data_collection"]["status"],
            report["data_processing"]["status"],
            report["vectorization"]["status"]
        ]
        
        if all(s == "completed" for s in statuses):
            report["system_status"] = "fully_operational"
        elif any(s == "completed" for s in statuses):
            report["system_status"] = "partially_operational"
        else:
            report["system_status"] = "not_operational"
            
        return report
        
    def save_system_report(self, report: Dict) -> str:
        """Save system report to file"""
        
        reports_dir = self.base_dir / "reports"
        reports_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = reports_dir / f"system_report_{timestamp}.json"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
            
        # Also save as latest report
        latest_report_file = reports_dir / "latest_system_report.json"
        with open(latest_report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
            
        return str(report_file)
        
    def run_full_pipeline(self) -> bool:
        """Run the complete enhanced CBT system pipeline"""
        
        self.logger.info("Starting enhanced CBT system full pipeline")
        
        pipeline_success = True
        pipeline_results = {}
        
        # Step 1: Check dependencies
        self.logger.info("Step 1: Checking dependencies")
        if not self.check_dependencies():
            self.logger.error("Dependencies check failed")
            return False
        pipeline_results["dependencies"] = True
        
        # Step 2: Enhanced data collection
        self.logger.info("Step 2: Enhanced data collection")
        collection_success = self.run_enhanced_collection()
        pipeline_results["collection"] = collection_success
        if not collection_success:
            pipeline_success = False
            
        time.sleep(2)  # Brief pause between steps
        
        # Step 3: Enhanced data processing
        self.logger.info("Step 3: Enhanced data processing")
        processing_success = self.run_enhanced_processing()
        pipeline_results["processing"] = processing_success
        if not processing_success:
            pipeline_success = False
            
        time.sleep(2)
        
        # Step 4: Enhanced vectorization
        self.logger.info("Step 4: Enhanced vectorization")
        vectorization_success = self.run_enhanced_vectorization()
        pipeline_results["vectorization"] = vectorization_success
        if not vectorization_success:
            pipeline_success = False
            
        # Step 5: Generate system report
        self.logger.info("Step 5: Generating system report")
        report = self.generate_system_report()
        report_file = self.save_system_report(report)
        pipeline_results["report"] = report_file
        
        # Log final results
        if pipeline_success:
            self.logger.info("Enhanced CBT system pipeline completed successfully")
        else:
            self.logger.warning("Enhanced CBT system pipeline completed with some failures")
            
        self.logger.info(f"Pipeline results: {pipeline_results}")
        self.logger.info(f"System report saved: {report_file}")
        
        return pipeline_success
        
    def run_quick_update(self) -> bool:
        """Run a quick update focusing on high-priority sources"""
        
        self.logger.info("Starting quick update process")
        
        # Temporarily modify config for quick update
        original_config = self.pipeline_config.copy()
        
        # Set to priority 1 sources only
        self.pipeline_config["collection"]["priority_filter"] = 1
        
        # Run pipeline
        success = self.run_full_pipeline()
        
        # Restore original config
        self.pipeline_config = original_config
        
        return success
        
    def test_integration(self) -> bool:
        """Test integration with existing CBT system"""
        
        self.logger.info("Testing integration with existing CBT system")
        
        try:
            # Test if we can load the enhanced data in the original integration
            sys.path.append(str(Path(__file__).parent))
            from integration import CBTIntegration
            
            # Initialize CBT integration
            cbt = CBTIntegration(base_dir=str(self.base_dir))
            
            # Test basic functionality
            status = cbt.get_cbt_status()
            self.logger.info(f"CBT integration status: {status}")
            
            if status.get("available", False):
                # Test search functionality
                test_query = "anxiety coping strategies"
                results = cbt.cbt_kb.search_cbt_techniques(test_query, top_k=3)
                
                self.logger.info(f"Test search returned {len(results)} results")
                return True
            else:
                self.logger.warning("CBT system not available for integration test")
                return False
                
        except Exception as e:
            self.logger.error(f"Integration test failed: {e}")
            return False

def main():
    """Main function with command-line interface"""
    
    parser = argparse.ArgumentParser(description="Enhanced CBT System Setup and Management")
    parser.add_argument("--action", choices=["full", "quick", "collect", "process", "vectorize", "report", "test"], 
                       default="full", help="Action to perform")
    parser.add_argument("--base-dir", default="cbt_data", help="Base directory for CBT data")
    parser.add_argument("--config", help="Configuration file for custom settings")
    
    args = parser.parse_args()
    
    # Initialize system
    setup = EnhancedCBTSystemSetup(args.base_dir)
    
    print("Enhanced CBT System Setup")
    print("=" * 50)
    
    # Load custom configuration if provided
    if args.config and os.path.exists(args.config):
        try:
            with open(args.config, 'r') as f:
                custom_config = json.load(f)
                setup.pipeline_config.update(custom_config)
                print(f"Loaded custom configuration from {args.config}")
        except Exception as e:
            print(f"Failed to load configuration: {e}")
            
    # Execute requested action
    success = False
    
    if args.action == "full":
        print("Running full enhanced pipeline...")
        success = setup.run_full_pipeline()
        
    elif args.action == "quick":
        print("Running quick update (priority 1 sources only)...")
        success = setup.run_quick_update()
        
    elif args.action == "collect":
        print("Running enhanced data collection...")
        success = setup.run_enhanced_collection()
        
    elif args.action == "process":
        print("Running enhanced data processing...")
        success = setup.run_enhanced_processing()
        
    elif args.action == "vectorize":
        print("Running enhanced vectorization...")
        success = setup.run_enhanced_vectorization()
        
    elif args.action == "report":
        print("Generating system report...")
        report = setup.generate_system_report()
        report_file = setup.save_system_report(report)
        print(f"Report saved: {report_file}")
        success = True
        
    elif args.action == "test":
        print("Testing system integration...")
        success = setup.test_integration()
        
    # Print results
    print("\n" + "=" * 50)
    if success:
        print("Operation completed successfully!")
    else:
        print("Operation completed with errors. Check logs for details.")
        
    print("Check the logs directory for detailed information.")

if __name__ == "__main__":
    main() 