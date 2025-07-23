#!/usr/bin/env python3
"""
CBT Data Update Script
Re-processes and re-vectorizes existing data with improved text cleaning
"""

import sys
import os
from pathlib import Path

def update_cbt_data():
    """Update CBT data with improved processing"""
    
    print("CBT Data Update")
    print("=" * 40)
    print("Re-processing data with improved text cleaning...")
    print()
    
    # Check if raw data exists
    cbt_data_dir = Path("cbt_data")
    raw_data_dir = cbt_data_dir / "raw_data" / "government"
    
    if not raw_data_dir.exists() or not list(raw_data_dir.glob("*.json")):
        print("ERROR: No raw data found.")
        print("Please run 'python setup.py' first to collect data.")
        return False
    
    try:
        # Step 1: Re-process data with improved cleaning
        print("Step 1: Re-processing data with improved text cleaning...")
        from data_processor import CBTDataProcessor
        
        processor = CBTDataProcessor()
        output_path = processor.process_all_data()
        
        if not output_path:
            print("FAILED: Data processing")
            return False
        
        print("SUCCESS: Data re-processed with improved cleaning")
        print()
        
        # Step 2: Re-create vector database
        print("Step 2: Re-creating vector database...")
        from vectorizer import CBTVectorizer
        
        vectorizer = CBTVectorizer()
        success = vectorizer.vectorize_all_data()
        
        if not success:
            print("FAILED: Vectorization")
            return False
            
        print("SUCCESS: Vector database updated")
        print()
        
        # Step 3: Test the improvements
        print("Step 3: Testing improvements...")
        from integration import CBTIntegration
        
        integration = CBTIntegration()
        
        # Test the problematic query that had formatting issues
        test_query = "How can I stop negative thinking?"
        recommendations = integration.cbt_kb.get_cbt_recommendation(test_query)
        
        if recommendations['recommended_techniques']:
            response = integration.cbt_kb.format_cbt_response(recommendations, test_query)
            print(f"Sample response preview:")
            print(f"'{response[:200]}...'")
            print()
            
            # Check if text formatting is improved
            if "Researchashown" in response or "    " in response:
                print("WARNING: Text formatting issues may still exist")
            else:
                print("SUCCESS: Text formatting appears improved")
        
        print()
        print("UPDATE COMPLETE!")
        print("=" * 40)
        print("Your CBT system has been updated with improved text processing.")
        
        return True
        
    except Exception as e:
        print(f"Update failed: {e}")
        return False

def main():
    """Main update function"""
    
    # Change to script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    try:
        success = update_cbt_data()
        
        if success:
            print("\nUpdate completed successfully!")
            print("You can now test the improved system with:")
            print("python test_system.py")
        else:
            print("\nUpdate failed. Check the error messages above.")
            
        return success
        
    except Exception as e:
        print(f"Update failed with error: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 