#!/usr/bin/env python3
"""
Test script to verify CBT logic correctly excludes definitional questions
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from CBT_System.integration import CBTIntegration
    print("CBT integration module loaded successfully")
except ImportError as e:
    print(f"CBT integration not available: {e}")
    sys.exit(1)

def test_cbt_logic():
    """Test CBT logic with various query types"""
    
    # Initialize CBT integration
    cbt_integration = CBTIntegration()
    
    # Test cases
    test_cases = [
        # Definitional questions - should return False
        ("what is depression", False, "Definitional question"),
        ("what is anxiety", False, "Definitional question"),
        ("define depression", False, "Definitional question"),
        ("explain anxiety", False, "Definitional question"),
        ("tell me about depression", False, "Definitional question"),
        ("what does depression mean", False, "Definitional question"),
        
        # Emotional expressions - should return True
        ("I feel really sad", True, "Emotional expression"),
        ("I am feeling anxious", True, "Emotional expression"),
        ("I'm feeling depressed", True, "Emotional expression"),
        
        # How-to questions - should return True
        ("how can I improve my mood", True, "How-to question"),
        ("what should I do about my anxiety", True, "How-to question"),
        ("how do I cope with depression", True, "How-to question"),
        
        # Social interactions - should return False
        ("hello", False, "Social interaction"),
        ("thank you", False, "Social interaction"),
        ("goodbye", False, "Social interaction"),
    ]
    
    print("Testing CBT logic:")
    print("=" * 60)
    
    for query, expected, description in test_cases:
        result = cbt_integration.should_include_cbt(query)
        status = "PASS" if result == expected else "FAIL"
        print(f"{status}: '{query}' -> {result} (expected {expected}) - {description}")
    
    print("=" * 60)
    print("Test completed!")

if __name__ == "__main__":
    test_cbt_logic() 