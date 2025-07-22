"""
Prompt Evaluation Module for ICD-11 OPRO System

This module provides comprehensive evaluation capabilities for mental health prompts.
"""

import json
import os
import re
import time
import logging
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import numpy as np
from textblob import TextBlob
from openai import OpenAI
import anthropic

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class EvaluationResult:
    """Stores evaluation results for a prompt"""
    prompt_id: str
    overall_score: float
    dimension_scores: Dict[str, float]
    test_case_results: List[Dict[str, Any]]
    timestamp: str
    evaluation_method: str
    feedback: str = ""

class PromptEvaluator:
    """Comprehensive prompt evaluation system"""
    
    def __init__(self, config_path: str = "config.json"):
        """Initialize the evaluator with configuration"""
        self.config = self._load_config(config_path)
        self.test_cases = self._load_test_cases()
        self.evaluation_criteria = self._load_evaluation_criteria()
        
        # Setup API clients
        self._setup_api_clients()
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"Configuration file {config_path} not found!")
            raise
    
    def _load_test_cases(self) -> List[Dict[str, Any]]:
        """Load test cases from JSON file"""
        test_file = "tests/test_cases.json"
        try:
            with open(test_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('test_cases', [])
        except FileNotFoundError:
            logger.error(f"Test cases file {test_file} not found!")
            return []
    
    def _load_evaluation_criteria(self) -> Dict[str, Any]:
        """Load evaluation criteria from test cases file"""
        test_file = "tests/test_cases.json"
        try:
            with open(test_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('evaluation_criteria', {})
        except FileNotFoundError:
            logger.error(f"Test cases file {test_file} not found!")
            return {}
    
    def _setup_api_clients(self):
        """Setup API clients for LLM evaluation"""
        try:
            # Setup OpenAI client
            if self.config["api_settings"]["openai_api_key"]:
                self.openai_client = OpenAI(
                    api_key=self.config["api_settings"]["openai_api_key"]
                )
            else:
                self.openai_client = OpenAI()  # Uses OPENAI_API_KEY environment variable
            
            # Setup Anthropic client
            if self.config["api_settings"]["anthropic_api_key"]:
                self.anthropic_client = anthropic.Anthropic(
                    api_key=self.config["api_settings"]["anthropic_api_key"]
                )
        except Exception as e:
            logger.warning(f"API client setup warning: {e}")
    
    def evaluate_prompt(self, prompt: str, evaluation_method: str = "comprehensive") -> EvaluationResult:
        """
        Evaluate a prompt using specified method
        
        Args:
            prompt: The prompt to evaluate
            evaluation_method: "comprehensive", "fast", or "llm_based"
        """
        logger.info(f"🔍 Evaluating prompt using {evaluation_method} method...")
        
        if evaluation_method == "comprehensive":
            return self._comprehensive_evaluation(prompt)
        elif evaluation_method == "fast":
            return self._fast_evaluation(prompt)
        elif evaluation_method == "llm_based":
            return self._llm_based_evaluation(prompt)
        else:
            raise ValueError(f"Unknown evaluation method: {evaluation_method}")
    
    def _comprehensive_evaluation(self, prompt: str) -> EvaluationResult:
        """Perform comprehensive evaluation including test cases"""
        logger.info("📊 Running comprehensive evaluation...")
        
        # Initialize scoring
        dimension_scores = {}
        test_case_results = []
        
        # Evaluate each dimension
        dimension_scores['relevance'] = self._evaluate_relevance(prompt)
        dimension_scores['empathy'] = self._evaluate_empathy(prompt)
        dimension_scores['accuracy'] = self._evaluate_accuracy(prompt)
        dimension_scores['safety'] = self._evaluate_safety(prompt)
        
        # Run test cases
        for test_case in self.test_cases[:5]:  # Limit to 5 test cases for faster evaluation
            result = self._run_test_case(prompt, test_case)
            test_case_results.append(result)
        
        # Calculate overall score
        weights = self.config['evaluation']['weights']
        overall_score = sum(
            dimension_scores[dim] * weight 
            for dim, weight in weights.items()
        )
        
        # Adjust score based on test case performance
        if test_case_results:
            test_case_avg = np.mean([r['score'] for r in test_case_results])
            overall_score = (overall_score + test_case_avg) / 2
        
        return EvaluationResult(
            prompt_id=f"eval_{int(time.time())}",
            overall_score=overall_score,
            dimension_scores=dimension_scores,
            test_case_results=test_case_results,
            timestamp=datetime.now().isoformat(),
            evaluation_method="comprehensive"
        )
    
    def _fast_evaluation(self, prompt: str) -> EvaluationResult:
        """Perform fast evaluation using heuristics"""
        logger.info("⚡ Running fast evaluation...")
        
        # Quick heuristic-based scoring
        dimension_scores = {
            'relevance': self._quick_relevance_score(prompt),
            'empathy': self._quick_empathy_score(prompt),
            'accuracy': self._quick_accuracy_score(prompt),
            'safety': self._quick_safety_score(prompt)
        }
        
        # Calculate overall score
        weights = self.config['evaluation']['weights']
        overall_score = sum(
            dimension_scores[dim] * weight 
            for dim, weight in weights.items()
        )
        
        return EvaluationResult(
            prompt_id=f"eval_{int(time.time())}",
            overall_score=overall_score,
            dimension_scores=dimension_scores,
            test_case_results=[],
            timestamp=datetime.now().isoformat(),
            evaluation_method="fast"
        )
    
    def _llm_based_evaluation(self, prompt: str) -> EvaluationResult:
        """Perform LLM-based evaluation"""
        logger.info("🤖 Running LLM-based evaluation...")
        
        evaluation_prompt = self._create_evaluation_prompt(prompt)
        
        try:
            response = self._call_evaluation_llm(evaluation_prompt)
            scores = self._parse_llm_evaluation(response)
            
            # Calculate overall score
            weights = self.config['evaluation']['weights']
            overall_score = sum(
                scores[dim] * weight 
                for dim, weight in weights.items()
                if dim in scores
            )
            
            return EvaluationResult(
                prompt_id=f"eval_{int(time.time())}",
                overall_score=overall_score,
                dimension_scores=scores,
                test_case_results=[],
                timestamp=datetime.now().isoformat(),
                evaluation_method="llm_based",
                feedback=response
            )
            
        except Exception as e:
            logger.error(f"LLM evaluation failed: {e}")
            return self._fast_evaluation(prompt)  # Fallback to fast evaluation
    
    def _evaluate_relevance(self, prompt: str) -> float:
        """Evaluate prompt relevance"""
        score = 5.0  # Base score
        
        # Check for key relevance components
        relevance_indicators = [
            'question', 'context', 'response', 'user', 'information',
            'relevant', 'answer', 'address', 'concern'
        ]
        
        for indicator in relevance_indicators:
            if indicator.lower() in prompt.lower():
                score += 0.3
        
        # Check for structure indicators
        if 'guidelines' in prompt.lower():
            score += 0.5
        if 'instructions' in prompt.lower():
            score += 0.5
        
        # Penalty for too generic
        if len(prompt.split()) < 50:
            score -= 1.0
        
        return min(max(score, 0), 10)
    
    def _evaluate_empathy(self, prompt: str) -> float:
        """Evaluate prompt empathy"""
        score = 5.0  # Base score
        
        # Empathy keywords and phrases
        empathy_indicators = [
            'compassionate', 'understanding', 'supportive', 'caring',
            'empathetic', 'validate', 'acknowledge', 'warm',
            'non-judgmental', 'gentle', 'respectful', 'sensitive'
        ]
        
        for indicator in empathy_indicators:
            if indicator.lower() in prompt.lower():
                score += 0.4
        
        # Check for emotional awareness
        emotional_awareness = [
            'feelings', 'emotions', 'pain', 'suffering', 'distress',
            'experience', 'struggle', 'difficult'
        ]
        
        for term in emotional_awareness:
            if term.lower() in prompt.lower():
                score += 0.3
        
        # Bonus for crisis sensitivity
        if 'crisis' in prompt.lower() or 'emergency' in prompt.lower():
            score += 1.0
        
        return min(max(score, 0), 10)
    
    def _evaluate_accuracy(self, prompt: str) -> float:
        """Evaluate prompt accuracy and professionalism"""
        score = 5.0  # Base score
        
        # Professional and accuracy indicators
        accuracy_indicators = [
            'icd-11', 'evidence-based', 'clinical', 'professional',
            'medical', 'diagnosis', 'treatment', 'therapy',
            'research', 'scientific', 'accurate', 'reliable'
        ]
        
        for indicator in accuracy_indicators:
            if indicator.lower() in prompt.lower():
                score += 0.4
        
        # Check for disclaimer presence
        disclaimer_terms = ['disclaimer', 'professional advice', 'medical advice', 'qualified']
        for term in disclaimer_terms:
            if term.lower() in prompt.lower():
                score += 0.5
        
        # Check for boundaries
        boundary_terms = ['boundaries', 'scope', 'limitations', 'cannot diagnose']
        for term in boundary_terms:
            if term.lower() in prompt.lower():
                score += 0.3
        
        return min(max(score, 0), 10)
    
    def _evaluate_safety(self, prompt: str) -> float:
        """Evaluate prompt safety protocols"""
        score = 5.0  # Base score
        
        # Safety indicators
        safety_indicators = [
            'safety', 'crisis', 'emergency', 'suicide', 'self-harm',
            'immediate help', 'professional help', 'emergency services',
            'hotline', 'urgent', 'danger'
        ]
        
        for indicator in safety_indicators:
            if indicator.lower() in prompt.lower():
                score += 0.5
        
        # Crisis response elements
        crisis_elements = [
            'crisis resources', 'emergency contact', 'immediate safety',
            'professional intervention', 'safety plan'
        ]
        
        for element in crisis_elements:
            if element.lower() in prompt.lower():
                score += 0.6
        
        # Mandatory crisis statement check
        crisis_keywords = ['自殺', '緊急', '立即', '危險']
        for keyword in crisis_keywords:
            if keyword in prompt:
                score += 1.0
                break
        
        return min(max(score, 0), 10)
    
    def _quick_relevance_score(self, prompt: str) -> float:
        """Quick relevance scoring using simple heuristics"""
        keywords = ['question', 'context', 'response', 'guidelines']
        score = 5.0 + sum(1 for kw in keywords if kw.lower() in prompt.lower())
        return min(score, 10)
    
    def _quick_empathy_score(self, prompt: str) -> float:
        """Quick empathy scoring using simple heuristics"""
        keywords = ['compassionate', 'supportive', 'understanding', 'caring']
        score = 5.0 + sum(1 for kw in keywords if kw.lower() in prompt.lower())
        return min(score, 10)
    
    def _quick_accuracy_score(self, prompt: str) -> float:
        """Quick accuracy scoring using simple heuristics"""
        keywords = ['professional', 'evidence', 'clinical', 'icd-11']
        score = 5.0 + sum(1 for kw in keywords if kw.lower() in prompt.lower())
        return min(score, 10)
    
    def _quick_safety_score(self, prompt: str) -> float:
        """Quick safety scoring using simple heuristics"""
        keywords = ['safety', 'crisis', 'emergency', 'professional help']
        score = 5.0 + sum(1 for kw in keywords if kw.lower() in prompt.lower())
        return min(score, 10)
    
    def _run_test_case(self, prompt: str, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single test case against the prompt"""
        logger.info(f"Running test case: {test_case['id']}")
        
        # Format the prompt with test case data
        formatted_prompt = prompt.format(
            context=test_case['context'],
            question=test_case['question']
        )
        
        # Simulate response generation (in practice, this would call your LLM)
        simulated_response = f"Response to: {test_case['question']}"
        
        # Evaluate the response
        score = self._evaluate_test_response(simulated_response, test_case)
        
        return {
            'test_case_id': test_case['id'],
            'score': score,
            'category': test_case['category'],
            'crisis_level': test_case['crisis_level'],
            'response': simulated_response,
            'expected_aspects': test_case['expected_aspects']
        }
    
    def _evaluate_test_response(self, response: str, test_case: Dict[str, Any]) -> float:
        """Evaluate a test response"""
        score = 5.0  # Base score
        
        # Check for expected aspects
        expected_aspects = test_case.get('expected_aspects', [])
        for aspect in expected_aspects:
            if any(keyword in response.lower() for keyword in aspect.split()):
                score += 1.0
        
        # Crisis handling bonus
        if test_case['crisis_level'] == 'high':
            crisis_terms = ['emergency', 'immediate', 'crisis', 'help']
            if any(term in response.lower() for term in crisis_terms):
                score += 2.0
        
        return min(score, 10)
    
    def _create_evaluation_prompt(self, prompt: str) -> str:
        """Create an evaluation prompt for LLM-based evaluation"""
        evaluation_prompt = f"""
Please evaluate the following mental health support prompt across four dimensions:

PROMPT TO EVALUATE:
{prompt}

EVALUATION CRITERIA:
1. RELEVANCE (0-10): How well does the prompt guide relevant, on-topic responses?
2. EMPATHY (0-10): How well does the prompt encourage empathetic, supportive responses?
3. ACCURACY (0-10): How well does the prompt ensure clinical accuracy and professionalism?
4. SAFETY (0-10): How well does the prompt handle crisis situations and safety protocols?

Please provide scores for each dimension and brief reasoning.

FORMAT YOUR RESPONSE AS:
RELEVANCE: X/10 - [brief reasoning]
EMPATHY: X/10 - [brief reasoning]
ACCURACY: X/10 - [brief reasoning]
SAFETY: X/10 - [brief reasoning]

OVERALL FEEDBACK: [summary of strengths and areas for improvement]
"""
        return evaluation_prompt
    
    def _call_evaluation_llm(self, evaluation_prompt: str) -> str:
        """Call LLM for evaluation"""
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": evaluation_prompt}],
                temperature=0.3,
                max_tokens=1000
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM evaluation call failed: {e}")
            raise
    
    def _parse_llm_evaluation(self, response: str) -> Dict[str, float]:
        """Parse LLM evaluation response"""
        scores = {}
        
        # Extract scores using regex
        patterns = {
            'relevance': r'RELEVANCE:\s*(\d+(?:\.\d+)?)',
            'empathy': r'EMPATHY:\s*(\d+(?:\.\d+)?)', 
            'accuracy': r'ACCURACY:\s*(\d+(?:\.\d+)?)',
            'safety': r'SAFETY:\s*(\d+(?:\.\d+)?)'
        }
        
        for dimension, pattern in patterns.items():
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                scores[dimension] = float(match.group(1))
            else:
                scores[dimension] = 5.0  # Default score
        
        return scores
    
    def batch_evaluate_prompts(self, prompts: List[str], method: str = "fast") -> List[EvaluationResult]:
        """Evaluate multiple prompts in batch"""
        logger.info(f"📋 Batch evaluating {len(prompts)} prompts...")
        
        results = []
        for i, prompt in enumerate(prompts):
            logger.info(f"Evaluating prompt {i+1}/{len(prompts)}")
            result = self.evaluate_prompt(prompt, method)
            results.append(result)
            
            # Rate limiting
            time.sleep(0.5)
        
        return results
    
    def save_evaluation_results(self, results: List[EvaluationResult], filename: str = None):
        """Save evaluation results to file"""
        if filename is None:
            filename = f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        filepath = os.path.join("prompts", filename)
        
        # Convert results to dictionaries
        results_data = []
        for result in results:
            result_dict = {
                'prompt_id': result.prompt_id,
                'overall_score': result.overall_score,
                'dimension_scores': result.dimension_scores,
                'test_case_results': result.test_case_results,
                'timestamp': result.timestamp,
                'evaluation_method': result.evaluation_method,
                'feedback': result.feedback
            }
            results_data.append(result_dict)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Evaluation results saved to {filepath}")

def evaluate_prompt_file(prompt_file: str, method: str = "comprehensive") -> EvaluationResult:
    """Convenience function to evaluate a prompt from file"""
    if not os.path.exists(prompt_file):
        raise FileNotFoundError(f"Prompt file {prompt_file} not found")
    
    with open(prompt_file, 'r', encoding='utf-8') as f:
        prompt = f.read()
    
    evaluator = PromptEvaluator()
    return evaluator.evaluate_prompt(prompt, method)

if __name__ == "__main__":
    # Example usage
    evaluator = PromptEvaluator()
    
    # Evaluate optimized prompt if it exists
    optimized_prompt_file = "prompts/optimized_prompt.txt"
    if os.path.exists(optimized_prompt_file):
        result = evaluate_prompt_file(optimized_prompt_file)
        print(f"Evaluation complete! Overall score: {result.overall_score:.2f}")
        print(f"Dimension scores: {result.dimension_scores}")
    else:
        print("No optimized prompt found. Please run OPRO optimization first.") 