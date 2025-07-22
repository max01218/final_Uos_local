#!/usr/bin/env python3
"""
Convert feedback data from interactions.json to OPRO test cases format
"""

import json
import os
from datetime import datetime
from typing import List, Dict, Any

def load_interactions() -> List[Dict[str, Any]]:
    """Load interactions from interactions.json"""
    interactions_path = "interactions.json"
    
    if not os.path.exists(interactions_path):
        print(f"Warning: {interactions_path} not found. Creating empty file.")
        return []
    
    try:
        with open(interactions_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading interactions.json: {e}")
        return []

def convert_to_testcase(interaction: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a single interaction to OPRO test case format"""
    
    # Calculate overall score (average of all ratings)
    ratings = [
        interaction.get('satisfaction', 0),
        interaction.get('empathy', 0),
        interaction.get('accuracy', 0),
        interaction.get('safety', 0)
    ]
    overall_score = sum(ratings) / len(ratings) if ratings else 0
    
    # Create test case
    test_case = {
        "input": {
            "question": interaction.get('question', ''),
            "tone": interaction.get('tone', 'empathetic_professional')
        },
        "target": {
            "answer": interaction.get('answer', ''),
            "scores": {
                "satisfaction": interaction.get('satisfaction', 0),
                "empathy": interaction.get('empathy', 0),
                "accuracy": interaction.get('accuracy', 0),
                "safety": interaction.get('safety', 0),
                "overall": overall_score
            },
            "comment": interaction.get('comment', '')
        }
    }
    
    return test_case

def filter_quality_interactions(interactions: List[Dict[str, Any]], 
                              min_overall_score: float = 3.0,
                              min_comment_length: int = 10) -> List[Dict[str, Any]]:
    """Filter interactions based on quality criteria"""
    
    quality_interactions = []
    
    for interaction in interactions:
        # Calculate overall score
        ratings = [
            interaction.get('satisfaction', 0),
            interaction.get('empathy', 0),
            interaction.get('accuracy', 0),
            interaction.get('safety', 0)
        ]
        overall_score = sum(ratings) / len(ratings) if ratings else 0
        
        # Check quality criteria
        has_good_score = overall_score >= min_overall_score
        has_meaningful_comment = len(interaction.get('comment', '')) >= min_comment_length
        has_question = bool(interaction.get('question', '').strip())
        has_answer = bool(interaction.get('answer', '').strip())
        
        if has_good_score and has_question and has_answer:
            quality_interactions.append(interaction)
    
    return quality_interactions

def save_testcases(testcases: List[Dict[str, Any]], output_path: str = None):
    """Save test cases to file"""
    
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"ICD11_OPRO/tests/feedback_testcases_{timestamp}.json"
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save test cases
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(testcases, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(testcases)} test cases to {output_path}")
    return output_path

def main():
    """Main function"""
    print("Converting feedback data to OPRO test cases...")
    
    # Load interactions
    interactions = load_interactions()
    print(f"Loaded {len(interactions)} interactions")
    
    if not interactions:
        print("No interactions found. Exiting.")
        return
    
    # Filter quality interactions
    quality_interactions = filter_quality_interactions(interactions)
    print(f"Found {len(quality_interactions)} quality interactions")
    
    # Convert to test cases
    testcases = []
    for interaction in quality_interactions:
        testcase = convert_to_testcase(interaction)
        testcases.append(testcase)
    
    # Save test cases
    output_path = save_testcases(testcases)
    
    # Print summary
    print("\nSummary:")
    print(f"- Total interactions: {len(interactions)}")
    print(f"- Quality interactions: {len(quality_interactions)}")
    print(f"- Test cases created: {len(testcases)}")
    print(f"- Output file: {output_path}")
    
    # Print some examples
    if testcases:
        print("\nExample test case:")
        example = testcases[0]
        print(f"Question: {example['input']['question'][:100]}...")
        print(f"Tone: {example['input']['tone']}")
        print(f"Overall Score: {example['target']['scores']['overall']:.2f}")
        if example['target']['comment']:
            print(f"Comment: {example['target']['comment'][:100]}...")

if __name__ == "__main__":
    main() 