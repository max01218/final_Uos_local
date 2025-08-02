import requests
import json

def test_icd11_retrieval():
    print("Testing ICD-11 Retrieval System")
    print("="*50)
    
    # Test cases that should retrieve ICD-11 information
    test_cases = [
        {
            "name": "What is anxiety",
            "question": "what is anxiety",
            "expected": "should retrieve anxiety disorder information"
        },
        {
            "name": "What is depression", 
            "question": "what is depression",
            "expected": "should retrieve depression disorder information"
        },
        {
            "name": "Depression definition",
            "question": "depression definition",
            "expected": "should retrieve depression definition"
        },
        {
            "name": "Anxiety symptoms",
            "question": "anxiety symptoms",
            "expected": "should retrieve anxiety symptoms"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n--- Test {i}: {test_case['name']} ---")
        print(f"Question: {test_case['question']}")
        print(f"Expected: {test_case['expected']}")
        
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
                context_used = data.get('context_used', '')
                fusion_strategy = data.get('fusion_strategy', '')
                
                print(f"Status: {status}")
                print(f"Fusion strategy: {fusion_strategy}")
                print(f"Context length: {len(context_used)} characters")
                print(f"Answer length: {len(answer)} characters")
                
                # Check if answer contains medical information
                medical_keywords = ['disorder', 'symptoms', 'diagnosis', 'treatment', 'anxiety', 'depression', 'mental']
                has_medical_info = any(keyword in answer.lower() for keyword in medical_keywords)
                
                if has_medical_info:
                    print(f"✅ Answer contains medical information")
                else:
                    print(f"❌ Answer lacks medical information")
                
                # Check if context was retrieved
                if len(context_used) > 100:
                    print(f"✅ Context retrieved successfully")
                    print(f"Context preview: {context_used[:200]}...")
                else:
                    print(f"❌ No context retrieved")
                
                # Check answer quality
                if "not sure how to respond" in answer.lower():
                    print(f"❌ Generic fallback response detected")
                elif len(answer) < 50:
                    print(f"⚠️ Answer too short")
                else:
                    print(f"✅ Answer appears normal")
                
                print(f"Answer: {answer[:200]}...")
                
            else:
                print(f"❌ HTTP Error: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n" + "="*50)
    print("Test Complete")

if __name__ == "__main__":
    test_icd11_retrieval() 