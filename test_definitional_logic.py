#!/usr/bin/env python3
"""
Test script to verify definitional question detection logic
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_definitional_detection():
    """Test definitional question detection with various query types"""
    
    # Import the function from the main server
    try:
        from fastapi_server import is_definitional_question
        print("Definitional detection function loaded successfully")
    except ImportError as e:
        print(f"Could not import function: {e}")
        return
    
    # Test cases
    test_cases = [
        # Definitional questions - should return True
        ("what is depression", True, "Definitional question"),
        ("what is anxiety", True, "Definitional question"),
        ("define depression", True, "Definitional question"),
        ("explain anxiety", True, "Definitional question"),
        ("tell me about depression", True, "Definitional question"),
        ("what does depression mean", True, "Definitional question"),
        ("describe anxiety", True, "Definitional question"),
        ("what is the difference between depression and anxiety", True, "Definitional question"),
        ("what are the symptoms of PTSD", True, "Definitional question"),
        ("how is OCD defined", True, "Definitional question"),
        ("what constitutes bipolar disorder", True, "Definitional question"),
        
        # Non-definitional questions - should return False
        ("I feel really sad", False, "Emotional expression"),
        ("I am feeling anxious", False, "Emotional expression"),
        ("how can I improve my mood", False, "How-to question"),
        ("what should I do about my anxiety", False, "How-to question"),
        ("what can I do about my depression", False, "How-to question"),
        ("how do I deal with anxiety", False, "How-to question"),
        ("I need help with depression", False, "Help request"),
        ("can you help me", False, "Help request"),
        ("help me with anxiety", False, "Help request"),
        ("I'm struggling with stress", False, "Personal statement"),
        ("dealing with depression", False, "Personal statement"),
        ("coping with anxiety", False, "Personal statement"),
        ("my friend has depression", False, "Personal statement"),
        ("hello", False, "Social interaction"),
        ("thank you", False, "Social interaction"),
    ]
    
    print("Testing definitional question detection:")
    print("=" * 60)
    
    for query, expected, description in test_cases:
        result = is_definitional_question(query)
        status = "PASS" if result == expected else "FAIL"
        print(f"{status}: '{query}' -> {result} (expected {expected}) - {description}")
    
    print("=" * 60)
    print("Test completed!")

if __name__ == "__main__":
    test_definitional_detection() 