#!/usr/bin/env python3
"""
Test the three fixes:
1. Response truncation fix
2. CBT relevance detection improvement
3. Response strategy optimization
"""

import requests
import json

def test_fixes():
    print("Testing the Three Fixes")
    print("="*50)
    
    # Test cases for the fixes
    test_cases = [
        {
            "name": "How to release anxiety (should trigger CBT and provide advice)",
            "question": "what should I do to release my anxiety",
            "expected_cbt": True,
            "expected_strategy": "provide_advice",
            "expected_complete": True
        },
        {
            "name": "How to manage stress (should trigger CBT and provide advice)",
            "question": "how can I manage my stress",
            "expected_cbt": True,
            "expected_strategy": "provide_advice",
            "expected_complete": True
        },
        {
            "name": "What is anxiety (should not trigger CBT, should provide info)",
            "question": "what is anxiety",
            "expected_cbt": False,
            "expected_strategy": "provide_advice",
            "expected_complete": True
        },
        {
            "name": "Simple greeting (should not trigger CBT, should ask clarification)",
            "question": "hello",
            "expected_cbt": False,
            "expected_strategy": "ask_clarification",
            "expected_complete": True
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n--- Test {i}: {test_case['name']} ---")
        print(f"Question: {test_case['question']}")
        
        try:
            response = requests.post(
                "http://localhost:8000/api/empathetic_professional",
                json={
                    "question": test_case['question'],
                    "type": "empathetic_professional",
                    "history": []
                },
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                answer = data.get('answer', '')
                status = data.get('status', '')
                
                print(f"Status: {status}")
                print(f"Answer length: {len(answer)} characters")
                
                # Test 1: Check if response is complete (not truncated)
                is_complete = not any(incomplete in answer.lower() for incomplete in [
                    'could involve', 'this could', 'might include', 'can include'
                ])
                print(f"Response complete: {is_complete} (expected: {test_case['expected_complete']})")
                
                # Test 2: Check if CBT was triggered (by looking for CBT-specific content)
                cbt_indicators = [
                    'cognitive', 'behavioral', 'technique', 'exercise', 'breathing',
                    'relaxation', 'mindfulness', 'progressive', 'muscle'
                ]
                has_cbt_content = any(indicator in answer.lower() for indicator in cbt_indicators)
                print(f"CBT content detected: {has_cbt_content} (expected: {test_case['expected_cbt']})")
                
                # Test 3: Check response strategy (by analyzing response type)
                if test_case['expected_strategy'] == 'ask_clarification':
                    strategy_correct = any(phrase in answer.lower() for phrase in [
                        'can you tell me more', 'could you share', 'what specifically',
                        'tell me more about', 'can you explain'
                    ])
                elif test_case['expected_strategy'] == 'provide_advice':
                    strategy_correct = any(phrase in answer.lower() for phrase in [
                        'step', 'technique', 'method', 'strategy', 'approach',
                        'try', 'practice', 'exercise', 'breathing'
                    ])
                else:
                    strategy_correct = True
                
                print(f"Strategy correct: {strategy_correct}")
                
                # Show answer preview
                print(f"Answer preview: {answer[:200]}...")
                
                # Overall test result
                test_passed = (
                    is_complete == test_case['expected_complete'] and
                    has_cbt_content == test_case['expected_cbt'] and
                    strategy_correct
                )
                
                if test_passed:
                    print(f"✅ Test {i} PASSED")
                else:
                    print(f"❌ Test {i} FAILED")
                
            else:
                print(f"❌ HTTP Error: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print(f"\n" + "="*50)
    print("Test Complete")

if __name__ == "__main__":
    test_fixes() 