#!/usr/bin/env python3
"""
Feedback to Test Cases Converter
Converts user interactions and feedback into OPRO test cases
Clean version without Chinese characters or emojis
"""

import json
import os
from datetime import datetime
from typing import List, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeedbackConverter:
    def __init__(self, interactions_file: str = "interactions.json"):
        self.interactions_file = interactions_file
        self.test_cases_dir = "tests"
        os.makedirs(self.test_cases_dir, exist_ok=True)
    
    def load_interactions(self) -> List[Dict[str, Any]]:
        """Load interactions from JSON file"""
        try:
            if not os.path.exists(self.interactions_file):
                logger.warning(f"Interactions file {self.interactions_file} not found")
                return []
            
            with open(self.interactions_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            if isinstance(data, list):
                return data
            elif isinstance(data, dict) and 'interactions' in data:
                return data['interactions']
            else:
                logger.warning("Unexpected interactions file format")
                return []
                
        except Exception as e:
            logger.error(f"Error loading interactions: {e}")
            return []
    
    def filter_quality_interactions(self, interactions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter interactions based on quality criteria"""
        filtered = []
        
        for interaction in interactions:
            # Check if interaction has required fields
            if not all(key in interaction for key in ['question', 'answer']):
                continue
            
            # Check for feedback scores if available
            if 'feedback' in interaction:
                feedback = interaction['feedback']
                
                # Calculate average score if multiple ratings
                if isinstance(feedback, dict):
                    scores = []
                    for key in ['satisfaction', 'empathy', 'accuracy', 'safety']:
                        if key in feedback and isinstance(feedback[key], (int, float)):
                            scores.append(feedback[key])
                    
                    if scores:
                        avg_score = sum(scores) / len(scores)
                        # Only include interactions with average score >= 3.0
                        if avg_score >= 3.0:
                            filtered.append(interaction)
                else:
                    # Simple rating system
                    if isinstance(feedback, (int, float)) and feedback >= 3:
                        filtered.append(interaction)
            else:
                # Include interactions without feedback for diversity
                filtered.append(interaction)
        
        logger.info(f"Filtered {len(filtered)} quality interactions from {len(interactions)} total")
        return filtered
    
    def convert_to_test_cases(self, interactions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert interactions to OPRO test cases format"""
        test_cases = []
        
        for i, interaction in enumerate(interactions):
            # Create test case
            test_case = {
                "id": f"feedback_{i+1}",
                "input": {
                    "question": interaction.get('question', ''),
                    "context": interaction.get('context', ''),
                    "history": interaction.get('history', [])
                },
                "expected_qualities": {
                    "empathy": True,
                    "relevance": True,
                    "safety": True,
                    "professionalism": True
                },
                "metadata": {
                    "source": "user_feedback",
                    "timestamp": interaction.get('timestamp', ''),
                    "tone": interaction.get('tone', 'empathetic_professional')
                }
            }
            
            # Add feedback scores if available
            if 'feedback' in interaction:
                feedback = interaction['feedback']
                if isinstance(feedback, dict):
                    test_case['user_feedback'] = feedback
                else:
                    test_case['user_feedback'] = {'overall_rating': feedback}
            
            # Add expected response characteristics based on good examples
            if 'answer' in interaction:
                answer = interaction['answer']
                test_case['reference_response'] = answer
                
                # Analyze response characteristics
                response_length = len(answer.split())
                has_question = '?' in answer
                
                test_case['expected_qualities'].update({
                    "appropriate_length": 50 <= response_length <= 200,
                    "includes_followup": has_question,
                    "concise": response_length <= 150
                })
            
            test_cases.append(test_case)
        
        return test_cases
    
    def save_test_cases(self, test_cases: List[Dict[str, Any]]):
        """Save test cases to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"feedback_testcases_{timestamp}.json"
        filepath = os.path.join(self.test_cases_dir, filename)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump({
                    "metadata": {
                        "generated_at": datetime.now().isoformat(),
                        "source": "user_feedback_conversion",
                        "total_cases": len(test_cases)
                    },
                    "test_cases": test_cases
                }, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved {len(test_cases)} test cases to {filepath}")
            
            # Also save as latest for scheduler
            latest_filepath = os.path.join(self.test_cases_dir, "feedback_testcases_latest.json")
            with open(latest_filepath, 'w', encoding='utf-8') as f:
                json.dump({
                    "metadata": {
                        "generated_at": datetime.now().isoformat(),
                        "source": "user_feedback_conversion",
                        "total_cases": len(test_cases)
                    },
                    "test_cases": test_cases
                }, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Also saved as latest: {latest_filepath}")
            
        except Exception as e:
            logger.error(f"Error saving test cases: {e}")
    
    def convert_all(self):
        """Main conversion process"""
        logger.info("Starting feedback to test cases conversion...")
        
        # Load interactions
        interactions = self.load_interactions()
        if not interactions:
            logger.warning("No interactions found to convert")
            return
        
        # Filter quality interactions
        quality_interactions = self.filter_quality_interactions(interactions)
        if not quality_interactions:
            logger.warning("No quality interactions found after filtering")
            return
        
        # Convert to test cases
        test_cases = self.convert_to_test_cases(quality_interactions)
        
        # Save test cases
        self.save_test_cases(test_cases)
        
        logger.info("Conversion completed successfully")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert feedback to test cases')
    parser.add_argument('--input', default='interactions.json', 
                       help='Input interactions file')
    
    args = parser.parse_args()
    
    converter = FeedbackConverter(interactions_file=args.input)
    converter.convert_all()

if __name__ == "__main__":
    main() 