#!/usr/bin/env python3
"""
CBT System Test Script
Tests the complete CBT system functionality
"""

import sys
import os
from pathlib import Path

def test_cbt_system():
    """Test the complete CBT system"""
    
    print("CBT System Test")
    print("=" * 40)
    print()
    
    # Test 1: Check dependencies
    print("Test 1: Checking dependencies...")
    try:
        import requests
        import bs4
        import numpy as np
        print("  Core dependencies: OK")
    except ImportError as e:
        print(f"  Core dependencies: FAILED - {e}")
        return False
        
    try:
        import sentence_transformers
        import faiss
        print("  AI dependencies: OK")
    except ImportError as e:
        print(f"  AI dependencies: MISSING - {e}")
        print("  Install with: pip install sentence-transformers faiss-cpu")
        return False
    
    print()
    
    # Test 2: Data collection
    print("Test 2: Testing data collection...")
    try:
        from data_collector import CBTDataCollector
        collector = CBTDataCollector()
        print("  Data collector initialized: OK")
    except Exception as e:
        print(f"  Data collector: FAILED - {e}")
        return False
    
    print()
    
    # Test 3: Data processing
    print("Test 3: Testing data processing...")
    try:
        from data_processor import CBTDataProcessor
        processor = CBTDataProcessor()
        print("  Data processor initialized: OK")
    except Exception as e:
        print(f"  Data processor: FAILED - {e}")
        return False
    
    print()
    
    # Test 4: Vectorization
    print("Test 4: Testing vectorization...")
    try:
        from vectorizer import CBTVectorizer
        vectorizer = CBTVectorizer()
        print("  Vectorizer initialized: OK")
    except Exception as e:
        print(f"  Vectorizer: FAILED - {e}")
        return False
    
    print()
    
    # Test 5: Integration
    print("Test 5: Testing integration...")
    try:
        from integration import CBTIntegration
        integration = CBTIntegration()
        status = integration.get_cbt_status()
        
        if status['available']:
            print(f"  CBT integration: READY")
            print(f"  Techniques: {status['total_techniques']}")
            print(f"  Content: {status['total_content']}")
        else:
            print("  CBT integration: NOT READY (run setup.py first)")
        
    except Exception as e:
        print(f"  Integration: FAILED - {e}")
        return False
    
    print()
    
    # Test 6: System functionality
    print("Test 6: Testing system functionality...")
    if status['available']:
        try:
            # Test query analysis
            query = "I feel anxious and need help coping"
            recommendations = integration.cbt_kb.get_cbt_recommendation(query)
            
            print(f"  Query analysis: OK")
            print(f"  Detected conditions: {recommendations['query_analysis']['detected_conditions']}")
            print(f"  Techniques found: {len(recommendations['recommended_techniques'])}")
            
            # Test response formatting
            response = integration.cbt_kb.format_cbt_response(recommendations, query)
            print(f"  Response formatting: OK")
            print(f"  Response length: {len(response)} characters")
            
        except Exception as e:
            print(f"  System functionality: FAILED - {e}")
            return False
    else:
        print("  System functionality: SKIPPED (CBT not ready)")
    
    print()
    print("=" * 40)
    
    if status['available']:
        print("ALL TESTS PASSED!")
        print("CBT system is ready for use.")
    else:
        print("SETUP REQUIRED!")
        print("Run 'python setup.py' to set up the CBT system.")
    
    return True

def test_specific_queries():
    """Test specific CBT queries"""
    
    print("\nTesting Specific CBT Queries")
    print("=" * 40)
    
    try:
        from integration import CBTIntegration
        integration = CBTIntegration()
        
        if not integration.get_cbt_status()['available']:
            print("CBT system not available - run setup.py first")
            return False
            
        test_queries = [
            "How can I stop negative thinking?",
            "I feel anxious all the time",
            "What are some coping strategies for depression?",
            "Help me deal with panic attacks",
            "I need relaxation techniques"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\nQuery {i}: {query}")
            
            if integration.should_include_cbt(query):
                recommendations = integration.cbt_kb.get_cbt_recommendation(query)
                response = integration.cbt_kb.format_cbt_response(recommendations, query)
                
                print(f"  CBT relevant: YES")
                print(f"  Techniques found: {len(recommendations['recommended_techniques'])}")
                print(f"  Response preview: {response[:100]}...")
            else:
                print(f"  CBT relevant: NO")
        
        return True
        
    except Exception as e:
        print(f"Query testing failed: {e}")
        return False

def main():
    """Main test function"""
    
    # Change to script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    try:
        # Run basic system tests
        if test_cbt_system():
            print("\nBasic tests completed successfully!")
            
            # Ask if user wants to test specific queries
            while True:
                try:
                    choice = input("\nTest specific CBT queries? (y/n): ").lower().strip()
                    if choice in ['y', 'yes']:
                        test_specific_queries()
                        break
                    elif choice in ['n', 'no']:
                        break
                    else:
                        print("Please enter 'y' or 'n'")
                except KeyboardInterrupt:
                    print("\nTest interrupted by user")
                    break
        
        print("\nTest completed!")
        return True
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 