import re
import json
from typing import Dict, List, Tuple, Optional

class PromptEvaluator:
    def __init__(self):
        self.tone_keywords = {
            "professional": [
                "clinical", "evidence-based", "therapeutic", "assessment", 
                "intervention", "professional", "licensed", "psychologist"
            ],
            "empathetic": [
                "empathy", "understanding", "compassion", "caring", 
                "supportive", "warm", "gentle", "kind"
            ],
            "caring": [
                "care", "support", "help", "concern", "worry", 
                "comfort", "reassurance", "safety"
            ],
            "unsafe": [
                "diagnose", "prescribe", "medication", "treatment plan",
                "medical advice", "cure", "guarantee", "promise"
            ]
        }
        
        self.structure_requirements = [
            "validate",
            "question",
            "technique",
            "crisis",
            "step 1",
            "step 2", 
            "step 3",
            "step 4"
        ]
    
    def classify_prompt_tone(self, prompt: str) -> str:
        """
        Classify the tone of a prompt based on keyword analysis
        """
        prompt_lower = prompt.lower()
        
        # Count keyword occurrences for each tone
        tone_scores = {}
        for tone, keywords in self.tone_keywords.items():
            score = sum(1 for keyword in keywords if keyword in prompt_lower)
            tone_scores[tone] = score
        
        # Check for unsafe content
        if tone_scores.get("unsafe", 0) > 0:
            return "unsafe"
        
        # Determine primary tone
        primary_tone = max(tone_scores.items(), key=lambda x: x[1])
        
        if primary_tone[1] == 0:
            return "off-tone"
        
        return primary_tone[0]
    
    def evaluate_structure_compliance(self, prompt: str) -> Dict[str, bool]:
        """
        Evaluate if prompt follows required clinical structure
        """
        prompt_lower = prompt.lower()
        
        compliance = {}
        for requirement in self.structure_requirements:
            compliance[requirement] = requirement in prompt_lower
        
        return compliance
    
    def is_prompt_acceptable(self, prompt: str) -> Tuple[bool, str, Dict]:
        """
        Comprehensive evaluation of prompt quality
        """
        tone = self.classify_prompt_tone(prompt)
        structure = self.evaluate_structure_compliance(prompt)
        
        # Check if all required structure elements are present
        structure_complete = all(structure.values())
        
        # Determine acceptability
        acceptable = (
            tone not in ["unsafe", "off-tone"] and 
            structure_complete
        )
        
        return acceptable, tone, structure
    
    def get_evaluation_report(self, prompt: str) -> Dict:
        """
        Generate comprehensive evaluation report
        """
        acceptable, tone, structure = self.is_prompt_acceptable(prompt)
        
        return {
            "acceptable": acceptable,
            "tone": tone,
            "structure_compliance": structure,
            "structure_score": sum(structure.values()) / len(structure),
            "prompt_length": len(prompt),
            "paragraph_count": prompt.count('\n\n') + 1
        } 