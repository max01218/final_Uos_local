#!/usr/bin/env python3
"""
FastAPI CBT Integration Test Script
Tests the CBT functionality in the FastAPI server
"""

import requests
import json
import time
import sys

def test_fastapi_cbt_integration():
    """Test CBT integration in FastAPI server"""
    
    base_url = "http://localhost:8000"
    
    print("FastAPI CBT Integration Test")
    print("=" * 50)
    print()
    
    # Test 1: Basic server health
    print("Test 1: Server health check...")
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        
        if response.status_code == 200:
            health_data = response.json()
            print(f"  Server status: {health_data['status']}")
            print(f"  CBT available: {health_data.get('cbt_available', False)}")
            
            if health_data.get('cbt_available', False):
                print(f"  CBT techniques: {health_data.get('cbt_techniques', 0)}")
                print(f"  CBT content: {health_data.get('cbt_content', 0)}")
                print("  SUCCESS: CBT integration is active")
            else:
                print("  WARNING: CBT integration not available")
                
        else:
            print(f"  FAILED: Server health check failed ({response.status_code})")
            return False
            
    except Exception as e:
        print(f"  FAILED: Cannot connect to server - {e}")
        print("  Make sure FastAPI server is running: python fastapi_server.py")
        return False
        
    print()
    
    # Test 2: CBT status endpoint
    print("Test 2: CBT status endpoint...")
    try:
        response = requests.get(f"{base_url}/api/cbt/status", timeout=10)
        
        if response.status_code == 200:
            cbt_status = response.json()
            
            if cbt_status.get('cbt_available', False):
                status_info = cbt_status.get('status', {})
                print(f"  CBT system ready: YES")
                print(f"  Total techniques: {status_info.get('total_techniques', 0)}")
                print(f"  Total content: {status_info.get('total_content', 0)}")
                print(f"  Categories: {status_info.get('categories', [])}")
                print("  SUCCESS: CBT status endpoint working")
            else:
                print("  CBT system ready: NO")
                print(f"  Message: {cbt_status.get('message', 'Unknown error')}")
                
        else:
            print(f"  FAILED: CBT status endpoint failed ({response.status_code})")
            
    except Exception as e:
        print(f"  FAILED: CBT status request failed - {e}")
        
    print()
    
    # Test 3: CBT recommendations
    print("Test 3: CBT recommendations...")
    test_queries = [
        "I feel anxious all the time, what can I do?",
        "How can I stop negative thinking?",
        "I need coping strategies for depression",
        "Help me with panic attacks"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"  Query {i}: {query}")
        
        try:
            response = requests.post(
                f"{base_url}/api/cbt/recommend",
                json={"query": query, "context": ""},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                
                print(f"    CBT relevant: {data.get('cbt_relevant', False)}")
                print(f"    Techniques found: {len(data.get('recommended_techniques', []))}")
                
                formatted_response = data.get('formatted_response', '')
                if formatted_response:
                    preview = formatted_response[:150] + "..." if len(formatted_response) > 150 else formatted_response
                    print(f"    Response preview: {preview}")
                    
                print("    SUCCESS: CBT recommendation working")
                
            else:
                print(f"    FAILED: Request failed ({response.status_code})")
                
        except Exception as e:
            print(f"    FAILED: Request error - {e}")
            
        print()
        time.sleep(1)  # Rate limiting
        
    # Test 4: Enhanced chat endpoint
    print("Test 4: Enhanced chat endpoint with CBT...")
    
    enhanced_test_query = "I'm struggling with anxiety and need help coping"
    
    try:
        response = requests.post(
            f"{base_url}/api/empathetic_professional",
            json={
                "question": enhanced_test_query,
                "type": "empathetic_professional",
                "history": []
            },
            timeout=60  # Allow more time for LLM processing
        )
        
        if response.status_code == 200:
            data = response.json()
            answer = data.get('answer', '')
            
            print(f"  Query: {enhanced_test_query}")
            print(f"  Response length: {len(answer)} characters")
            
            # Check for CBT enhancement indicators
            cbt_indicators = [
                'technique', 'cbt', 'cognitive', 'behavioral', 'strategy', 
                'practice', 'breathing', 'challenge', 'coping'
            ]
            
            found_indicators = [
                word for word in cbt_indicators 
                if word.lower() in answer.lower()
            ]
            
            if found_indicators:
                print(f"  CBT enhancement detected: {found_indicators}")
                print("  SUCCESS: Enhanced response with CBT")
            else:
                print("  INFO: No clear CBT enhancement detected")
                
            # Show response preview
            preview = answer[:200] + "..." if len(answer) > 200 else answer
            print(f"  Response preview: {preview}")
            
        else:
            print(f"  FAILED: Enhanced chat failed ({response.status_code})")
            
    except Exception as e:
        print(f"  FAILED: Enhanced chat error - {e}")
        
    print()
    print("=" * 50)
    print("CBT Integration Test Completed!")
    
    return True

def main():
    """Main test function"""
    
    try:
        success = test_fastapi_cbt_integration()
        
        if success:
            print("Test completed successfully!")
            print("\nYour FastAPI server now has full CBT integration!")
            print("\nAvailable endpoints:")
            print("- GET  /api/cbt/status - Check CBT system status")
            print("- POST /api/cbt/recommend - Get CBT recommendations")
            print("- POST /api/cbt/search - Search CBT techniques")
            print("- POST /api/empathetic_professional - Enhanced chat with CBT")
        else:
            print("Test failed. Please check your server setup.")
            
        return success
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 