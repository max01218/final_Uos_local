#!/usr/bin/env python3
"""
CBT System Setup Script
Orchestrates the complete CBT data collection and integration process
"""

import sys
import os
import subprocess
import logging
from pathlib import Path
from datetime import datetime

class CBTSystemSetup:
    def __init__(self):
        self.setup_logging()
        self.base_dir = Path("cbt_data")
        
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('cbt_setup.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def check_dependencies(self) -> bool:
        """Check if required dependencies are installed"""
        required_packages = [
            'requests',
            'beautifulsoup4',
            'sentence-transformers',
            'faiss-cpu',
            'numpy'
        ]
        
        missing_packages = []
        
        for package in required_packages:
            try:
                if package == 'beautifulsoup4':
                    import bs4
                elif package == 'sentence-transformers':
                    import sentence_transformers
                elif package == 'faiss-cpu':
                    import faiss
                else:
                    __import__(package)
                    
                self.logger.info(f"Found package: {package}")
                
            except ImportError:
                missing_packages.append(package)
                self.logger.warning(f"Missing package: {package}")
                
        if missing_packages:
            self.logger.error(f"Missing required packages: {missing_packages}")
            print("\nTo install missing packages, run:")
            print(f"pip install {' '.join(missing_packages)}")
            return False
            
        return True
        
    def run_data_collection(self) -> bool:
        """Run CBT data collection"""
        self.logger.info("Starting CBT data collection")
        
        try:
            from data_collector import CBTDataCollector
            
            collector = CBTDataCollector()
            results = collector.collect_all_sources()
            
            total_items = sum(len(items) for items in results.values())
            
            if total_items > 0:
                self.logger.info(f"Data collection successful: {total_items} items")
                return True
            else:
                self.logger.warning("No data collected")
                return False
                
        except Exception as e:
            self.logger.error(f"Data collection failed: {e}")
            return False
            
    def run_data_processing(self) -> bool:
        """Run CBT data processing"""
        self.logger.info("Starting CBT data processing")
        
        try:
            from data_processor import CBTDataProcessor
            
            processor = CBTDataProcessor()
            output_path = processor.process_all_data()
            
            if output_path:
                self.logger.info(f"Data processing successful: {output_path}")
                return True
            else:
                self.logger.warning("No data processed")
                return False
                
        except Exception as e:
            self.logger.error(f"Data processing failed: {e}")
            return False
            
    def run_vectorization(self) -> bool:
        """Run CBT data vectorization"""
        self.logger.info("Starting CBT data vectorization")
        
        try:
            from vectorizer import CBTVectorizer
            
            vectorizer = CBTVectorizer()
            success = vectorizer.vectorize_all_data()
            
            if success:
                self.logger.info("Vectorization successful")
                return True
            else:
                self.logger.warning("Vectorization failed")
                return False
                
        except Exception as e:
            self.logger.error(f"Vectorization failed: {e}")
            return False
            
    def test_integration(self) -> bool:
        """Test CBT integration"""
        self.logger.info("Testing CBT integration")
        
        try:
            from integration import CBTIntegration
            
            integration = CBTIntegration()
            status = integration.get_cbt_status()
            
            if status['available']:
                self.logger.info(f"CBT integration test successful")
                self.logger.info(f"Techniques: {status['total_techniques']}")
                self.logger.info(f"Content: {status['total_content']}")
                self.logger.info(f"Categories: {status['categories']}")
                return True
            else:
                self.logger.warning("CBT integration not available")
                return False
                
        except Exception as e:
            self.logger.error(f"Integration test failed: {e}")
            return False
            
    def create_requirements_file(self):
        """Create requirements.txt for CBT system"""
        requirements = [
            "requests>=2.28.0",
            "beautifulsoup4>=4.11.0",
            "sentence-transformers>=2.2.0",
            "faiss-cpu>=1.7.0",
            "numpy>=1.21.0",
            "pathlib",
            "logging"
        ]
        
        requirements_path = Path("requirements.txt")
        
        with open(requirements_path, 'w') as f:
            f.write('\n'.join(requirements))
            
        self.logger.info(f"Created requirements file: {requirements_path}")
        
    def create_usage_guide(self):
        """Create usage guide for CBT system"""
        guide_content = """# CBT System Usage Guide

## Setup Complete

Your CBT (Cognitive Behavioral Therapy) system has been successfully set up and integrated.

## System Components

1. **Data Collection**: Collected CBT resources from public domain sources
2. **Data Processing**: Cleaned and categorized CBT techniques and content
3. **Vectorization**: Created searchable vector database of CBT knowledge
4. **Integration**: Ready to enhance responses with CBT techniques

## Using the CBT System

### Basic Integration:

```python
from integration import CBTIntegration

# Initialize CBT integration
cbt_integration = CBTIntegration()

# Enhance responses with CBT techniques
enhanced_response = cbt_integration.enhance_response_with_cbt(
    user_query="I feel anxious all the time",
    context="relevant context",
    base_response="base response"
)
```

### Testing CBT functionality:

```bash
python integration.py
```

## CBT Features Available

- Cognitive restructuring techniques
- Behavioral activation strategies  
- Exposure therapy guidance
- Problem-solving approaches
- Relaxation techniques
- Psychoeducational content

## Files Created

- `cbt_data/`: Main CBT data directory
- `cbt_data/embeddings/`: Vector database files
- `cbt_data/structured_data/`: Organized CBT content
- `cbt_setup.log`: Setup process log

## Next Steps

1. Integrate CBT enhancement into your existing system endpoints
2. Test with real user queries
3. Monitor CBT technique recommendations
4. Update data periodically by re-running collection scripts

## Troubleshooting

If CBT features are not working:
1. Check that all dependencies are installed
2. Verify that vector database files exist in `cbt_data/embeddings/`
3. Check logs for error messages
4. Re-run setup if needed: `python setup.py`

## Support

For technical issues, check the logs:
- `cbt_setup.log`: Overall setup process
- `cbt_data/collection.log`: Data collection issues  
- `cbt_data/processing.log`: Data processing issues
- `cbt_data/vectorization.log`: Vector database issues
"""
        
        guide_path = Path("USAGE_GUIDE.md")
        
        with open(guide_path, 'w', encoding='utf-8') as f:
            f.write(guide_content)
            
        self.logger.info(f"Created usage guide: {guide_path}")
        
    def run_complete_setup(self) -> bool:
        """Run complete CBT system setup"""
        print("CBT System Setup")
        print("=" * 50)
        print("Setting up Cognitive Behavioral Therapy integration")
        print()
        
        # Check dependencies
        print("Step 1: Checking dependencies...")
        if not self.check_dependencies():
            print("FAILED: Missing dependencies")
            return False
        print("SUCCESS: All dependencies found")
        print()
        
        # Data collection
        print("Step 2: Collecting CBT data from public sources...")
        if not self.run_data_collection():
            print("FAILED: Data collection")
            return False
        print("SUCCESS: Data collection completed")
        print()
        
        # Data processing
        print("Step 3: Processing and cleaning CBT data...")
        if not self.run_data_processing():
            print("FAILED: Data processing")
            return False
        print("SUCCESS: Data processing completed")
        print()
        
        # Vectorization
        print("Step 4: Creating vector database...")
        if not self.run_vectorization():
            print("FAILED: Vectorization")
            return False
        print("SUCCESS: Vector database created")
        print()
        
        # Integration test
        print("Step 5: Testing CBT integration...")
        if not self.test_integration():
            print("FAILED: Integration test")
            return False
        print("SUCCESS: CBT integration ready")
        print()
        
        # Create support files
        print("Step 6: Creating documentation...")
        self.create_requirements_file()
        self.create_usage_guide()
        print("SUCCESS: Documentation created")
        print()
        
        print("CBT SYSTEM SETUP COMPLETE!")
        print("=" * 50)
        print()
        print("Your CBT system is ready to use.")
        print("See USAGE_GUIDE.md for integration instructions.")
        print()
        
        return True

def main():
    """Main setup function"""
    setup = CBTSystemSetup()
    
    try:
        success = setup.run_complete_setup()
        
        if success:
            print("Setup completed successfully!")
            return True
        else:
            print("Setup failed. Check logs for details.")
            return False
            
    except KeyboardInterrupt:
        print("\nSetup interrupted by user")
        return False
    except Exception as e:
        print(f"Setup failed with error: {e}")
        logging.error(f"Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 