"""
OPRO (Optimization by PROmpting) Core Module

This module implements the core OPRO optimization logic for mental health prompts.
Clean version without Chinese characters or emojis.
"""

import json
import os
import random
import re
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
from tqdm import tqdm

# Local LLM support using Llama 3
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    import torch
    _llama3_pipeline = None
    
    def call_local_llm(prompt, max_new_tokens=512, temperature=0.7):
        global _llama3_pipeline
        if _llama3_pipeline is None:
            print("Loading local Llama 3 model for OPRO optimization...")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
            
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                padding_side="left"
            )
            
            # Set pad_token to eos_token for Llama 3
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map=device,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                low_cpu_mem_usage=True
            )
            
            _llama3_pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                return_full_text=False
            )
            print("Llama 3 model loaded successfully for OPRO!")
        
        # Use Llama 3 instruct format
        formatted_prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        result = _llama3_pipeline(formatted_prompt)
        return result[0]['generated_text'] if result else ""
        
except ImportError:
    def call_local_llm(prompt, max_new_tokens=512, temperature=0.7):
        raise RuntimeError("transformers or torch not installed, cannot use local Llama 3 model")

# Configure logging
logging.basicConfig(level=logging.INFO) 
logger = logging.getLogger(__name__)

@dataclass
class PromptCandidate:
    """Represents a prompt candidate with metadata"""
    content: str
    score: float
    iteration: int
    parent_id: Optional[str] = None
    generation_method: str = "seed"
    timestamp: str = None
    evaluation_details: Dict[str, float] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
        if self.evaluation_details is None:
            self.evaluation_details = {}

@dataclass
class OptimizationResult:
    """Stores the result of an optimization run"""
    best_prompt: PromptCandidate
    optimization_history: List[PromptCandidate]
    total_iterations: int
    improvement_achieved: float
    final_score: float
    time_elapsed: float

class OPROOptimizer:
    """Main OPRO optimizer for mental health prompts"""
    
    def __init__(self, config_path: str = "config/config.json"):
        """Initialize the OPRO optimizer with configuration"""
        self.config = self._load_config(config_path)
        self.optimization_history = []
        self.current_best = None
        self.iteration_count = 0
        self.last_evaluation_details = {}
        
        # Load seed prompts
        self.seed_prompts = self._load_seed_prompts()
        
        # Create output directories
        os.makedirs("prompts", exist_ok=True)
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"Configuration file {config_path} not found!")
            # Return default config
            return self._get_default_config()
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in configuration file: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "opro_settings": {
                "max_iterations": 5,
                "improvement_threshold": 0.05,
                "early_stopping_patience": 2,
                "temperature": 0.7,
                "max_tokens": 512
            },
            "evaluation": {
                "weights": {
                    "relevance": 0.25,
                    "empathy": 0.30,
                    "accuracy": 0.25,
                    "safety": 0.20
                },
                "score_range": [0, 10],
                "passing_threshold": 7.0
            }
        }
    
    def _load_seed_prompts(self) -> List[str]:
        """Load all seed prompts from the prompts directory"""
        seed_prompts = []
        seed_dir = "prompts/seeds"
        
        if not os.path.exists(seed_dir):
            logger.warning(f"Seed prompts directory {seed_dir} not found!")
            # Return default seed prompt
            return [self._get_default_seed_prompt()]
        
        for filename in os.listdir(seed_dir):
            if filename.endswith('.txt'):
                filepath = os.path.join(seed_dir, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                        if content:
                            seed_prompts.append(content)
                            logger.info(f"Loaded seed prompt from {filename}")
                except Exception as e:
                    logger.error(f"Error loading seed prompt {filename}: {e}")
        
        if not seed_prompts:
            logger.warning("No seed prompts found, using default")
            seed_prompts = [self._get_default_seed_prompt()]
        
        return seed_prompts
    
    def _get_default_seed_prompt(self) -> str:
        """Get default seed prompt"""
        return """You are a professional mental health advisor. Provide empathetic and evidence-based responses.

MEDICAL CONTEXT:
{context}

CONVERSATION HISTORY:
{history}

USER QUESTION: {question}

INSTRUCTIONS:
- Respond with empathy and understanding
- Reference medical context when relevant
- Keep responses concise and supportive
- Ask thoughtful follow-up questions
- Maintain professional standards

RESPONSE:"""

    def _evaluate_prompt(self, prompt: str) -> float:
        """Evaluate a prompt using LLM-based scoring"""
        try:
            return self._evaluate_prompt_llm(prompt)
        except Exception as e:
            print(f"DEBUG: LLM evaluation failed: {e}")
            print("DEBUG: Falling back to rule-based evaluation")
            return self._evaluate_prompt_heuristic(prompt)
    
    def _evaluate_prompt_llm(self, prompt: str) -> float:
        """Evaluate a prompt using LLM"""
        evaluation_instruction = f"""
You are an expert in prompt engineering for mental health chatbots. Please evaluate the following prompt on a scale of 0-10.

PROMPT TO EVALUATE:
{prompt}

EVALUATION CRITERIA:
1. EMPATHY (25%): Does it encourage empathetic, compassionate responses?
2. PROFESSIONALISM (20%): Does it maintain professional medical standards?
3. CLARITY (20%): Are the instructions clear and well-structured?
4. SAFETY (20%): Does it include appropriate safety guidelines?
5. EFFECTIVENESS (15%): Will it produce helpful, actionable responses?

SCORING SCALE:
- 9-10: Excellent prompt that excels in all criteria
- 7-8: Good prompt with strong performance in most areas
- 5-6: Average prompt with some strengths but room for improvement
- 3-4: Below average prompt with significant weaknesses
- 1-2: Poor prompt with major issues
- 0: Completely inadequate prompt

Please provide your evaluation in exactly this format:
SCORE: [number between 0-10]
REASON: [brief explanation of the score]

Your evaluation:"""
        
        print("DEBUG: Using LLM to evaluate prompt...")
        response = call_local_llm(evaluation_instruction, max_new_tokens=256, temperature=0.3)
        
        print(f"DEBUG: LLM evaluation response: {response[:200]}...")
        
        # Parse the score from response
        import re
        score_match = re.search(r'SCORE:\s*([0-9]*\.?[0-9]+)', response)
        
        if score_match:
            score = float(score_match.group(1))
            score = max(0.0, min(10.0, score))  # Clamp between 0-10
            
            # Extract reason if available
            reason_match = re.search(r'REASON:\s*(.+)', response, re.DOTALL)
            reason = reason_match.group(1).strip() if reason_match else "No reason provided"
            
            print(f"DEBUG: LLM Score: {score:.1f}")
            print(f"DEBUG: LLM Reason: {reason[:100]}...")
            
            # Store evaluation details for later use
            self.last_evaluation_details = {
                "evaluation_method": "llm",
                "score": score,
                "reason": reason,
                "llm_response": response
            }
            
            return score
        else:
            print("DEBUG: Could not parse score from LLM response")
            raise ValueError("Cannot parse LLM evaluation score")
    
    def _evaluate_prompt_heuristic(self, prompt: str) -> float:
        """Evaluate a prompt using heuristic rules (fallback method)"""
        score = 0.0
        
        # Basic quality checks
        if len(prompt) > 100:
            score += 1.0
        if "empathy" in prompt.lower() or "understanding" in prompt.lower():
            score += 2.0
        if "medical" in prompt.lower() or "context" in prompt.lower():
            score += 1.5
        if "professional" in prompt.lower():
            score += 1.0
        if "question" in prompt.lower() and "response" in prompt.lower():
            score += 1.5
        
        # Penalize overly long prompts
        if len(prompt) > 2000:
            score -= 1.0
        
        final_score = min(score, 10.0)
        
        # Store evaluation details
        self.last_evaluation_details = {
            "evaluation_method": "heuristic",
            "score": final_score,
            "reason": "Rule-based scoring using keyword matching and length checks"
        }
            
        return final_score
    
    def _generate_prompt_variants(self, base_prompt: str, num_variants: int = 3) -> List[str]:
        """Generate prompt variants using local LLM"""
        variants = []
        
        optimization_instruction = f"""
You are an expert prompt engineer. Your task is to improve the following prompt for a mental health chatbot.

CURRENT PROMPT:
{base_prompt}

Please generate {num_variants} improved variants that:
1. Maintain empathetic and professional tone
2. Include clear instructions for context usage
3. Specify response format and length
4. Include safety guidelines
5. Are concise but comprehensive

Generate each variant separately, starting with "VARIANT:" 
"""
        
        try:
            print("DEBUG: Calling LLM to generate variants...")
            response = call_local_llm(optimization_instruction, max_new_tokens=1024, temperature=0.8)
            
            print(f"DEBUG: LLM Response received (length: {len(response)})")
            print(f"DEBUG: First 200 chars: {response[:200]}...")
            
            # Parse variants from response - try multiple patterns
            variants_found = []
            
            # Method 1: Try splitting by "VARIANT:" (original method)
            if "VARIANT:" in response:
                variant_sections = response.split("VARIANT:")
                print(f"DEBUG: Method 1 - Found {len(variant_sections)} sections by 'VARIANT:'")
                for i, section in enumerate(variant_sections[1:]):
                    variant = section.strip()
                    if variant and len(variant) > 50:
                        variants_found.append(variant)
            
            # Method 2: Try splitting by "**VARIANT" (Llama's actual format)
            elif "**VARIANT" in response:
                import re
                # Split by **VARIANT 1**, **VARIANT 2**, etc.
                variant_sections = re.split(r'\*\*VARIANT\s+\d+\*\*', response)
                print(f"DEBUG: Method 2 - Found {len(variant_sections)} sections by '**VARIANT N**'")
                
                # More detailed debugging
                for i, section in enumerate(variant_sections):
                    print(f"DEBUG: Section {i} preview: {section[:100]}...")
                
                for i, section in enumerate(variant_sections[1:]):  # Skip intro text
                    variant = section.strip()
                    print(f"DEBUG: Processing section {i+1}, length: {len(variant)}")
                    
                    # Clean up the variant text
                    # Remove any leading/trailing explanatory text
                    lines = variant.split('\n')
                    cleaned_lines = []
                    
                    for line in lines:
                        line = line.strip()
                        if line and not line.startswith('**') and not line.startswith('---'):
                            cleaned_lines.append(line)
                    
                    cleaned_variant = '\n'.join(cleaned_lines).strip()
                    
                    if cleaned_variant and len(cleaned_variant) > 50:
                        variants_found.append(cleaned_variant)
                        print(f"DEBUG: Method 2 - Added variant {len(variants_found)}: {cleaned_variant[:100]}...")
                    else:
                        print(f"DEBUG: Method 2 - Skipped section {i+1} (too short: {len(cleaned_variant)})")
                        
                # If still no variants, try a more aggressive approach
                if not variants_found:
                    print("DEBUG: Method 2b - Trying more aggressive splitting")
                    # Try to split by any **VARIANT pattern
                    all_variants = re.split(r'\*\*VARIANT[^*]*\*\*', response)
                    for i, variant in enumerate(all_variants[1:]):  # Skip first section
                        cleaned = variant.strip()
                        if len(cleaned) > 100:  # Lowered threshold
                            variants_found.append(cleaned)
                            print(f"DEBUG: Method 2b - Added variant {len(variants_found)}")
            
            # Method 3: Try splitting by numbered lists (1., 2., 3.)
            elif re.search(r'\d+\.\s+', response):
                # Split by "1. ", "2. ", "3. "
                variant_sections = re.split(r'\d+\.\s+', response)
                print(f"DEBUG: Method 3 - Found {len(variant_sections)} sections by numbered list")
                for i, section in enumerate(variant_sections[1:]):  # Skip intro text
                    variant = section.strip()
                    if variant and len(variant) > 50:
                        variants_found.append(variant)
                        print(f"DEBUG: Method 3 - Added variant {len(variants_found)}: {variant[:100]}...")
            
            # Method 4: If no clear pattern, try to extract by keywords
            else:
                print("DEBUG: Method 4 - No clear pattern found, trying keyword extraction")
                lines = response.split('\n')
                current_variant = ""
                collecting = False
                
                for line in lines:
                    if any(keyword in line.lower() for keyword in ['variant', 'version', 'option', 'prompt']):
                        if current_variant and len(current_variant) > 50:
                            variants_found.append(current_variant.strip())
                        current_variant = ""
                        collecting = True
                    elif collecting:
                        current_variant += line + "\n"
                
                # Add the last variant
                if current_variant and len(current_variant) > 50:
                    variants_found.append(current_variant.strip())
                
                print(f"DEBUG: Method 4 - Extracted {len(variants_found)} variants by keyword detection")
            
            # Use the found variants
            variants = variants_found
            print(f"DEBUG: Total variants collected: {len(variants)}")
                    
        except Exception as e:
            print(f"DEBUG: LLM call failed with error: {e}")
            logger.error(f"Error generating variants: {e}")
            # Fallback: simple modifications
            print("DEBUG: Using fallback simple variants...")
            variants = self._generate_simple_variants(base_prompt, num_variants)
            print(f"DEBUG: Fallback generated {len(variants)} variants")
        
        return variants[:num_variants]
    
    def _generate_simple_variants(self, base_prompt: str, num_variants: int) -> List[str]:
        """Generate simple variants as fallback"""
        variants = []
        
        # Simple modifications
        modifications = [
            lambda p: p.replace("professional", "compassionate professional"),
            lambda p: p.replace("RESPONSE:", "EMPATHETIC RESPONSE:"),
            lambda p: p.replace("evidence-based", "evidence-based and empathetic")
        ]
        
        for i, mod in enumerate(modifications[:num_variants]):
            try:
                variant = mod(base_prompt)
                variants.append(variant)
            except:
                variants.append(base_prompt)  # Fallback to original
                
        return variants
    
    def optimize_prompts(self) -> OptimizationResult:
        """Main optimization loop"""
        start_time = time.time()
        logger.info("Starting OPRO optimization...")
        
        # Initialize with best seed prompt
        best_score = -1
        for seed_prompt in self.seed_prompts:
            score = self._evaluate_prompt(seed_prompt)
            candidate = PromptCandidate(
                content=seed_prompt,
                score=score,
                iteration=0,
                generation_method="seed",
                evaluation_details=getattr(self, 'last_evaluation_details', {})
            )
            self.optimization_history.append(candidate)
            
            if score > best_score:
                best_score = score
                self.current_best = candidate
        
        initial_score = self.current_best.score
        logger.info(f"Initial best score: {initial_score:.3f}")
        
        # Optimization loop
        no_improvement_count = 0
        max_iterations = self.config["opro_settings"]["max_iterations"]
        improvement_threshold = self.config["opro_settings"]["improvement_threshold"]
        patience = self.config["opro_settings"]["early_stopping_patience"]
        
        for iteration in range(1, max_iterations + 1):
            self.iteration_count = iteration
            logger.info(f"Iteration {iteration}/{max_iterations}")
            
            # Generate variants
            variants = self._generate_prompt_variants(self.current_best.content)
            
            # Evaluate variants
            iteration_best_score = self.current_best.score
            for variant in variants:
                score = self._evaluate_prompt(variant)
                candidate = PromptCandidate(
                    content=variant,
                    score=score,
                    iteration=iteration,
                    parent_id=f"iter_{iteration-1}_best",
                    generation_method="llm_variant",
                    evaluation_details=getattr(self, 'last_evaluation_details', {})
                )
                self.optimization_history.append(candidate)
                
                # Update best if improved
                if score > self.current_best.score + improvement_threshold:
                    self.current_best = candidate
                    iteration_best_score = score
                    no_improvement_count = 0
                    logger.info(f"New best score: {score:.3f} (improvement: {score - self.current_best.score:.3f})")
            
            # Early stopping check
            if iteration_best_score <= self.current_best.score:
                no_improvement_count += 1
                if no_improvement_count >= patience:
                    logger.info(f"Early stopping: no improvement for {patience} iterations")
                    break
        
        # Save results
        self._save_results()
        
        end_time = time.time()
        time_elapsed = end_time - start_time
        improvement_achieved = self.current_best.score - initial_score
        
        result = OptimizationResult(
            best_prompt=self.current_best,
            optimization_history=self.optimization_history,
            total_iterations=self.iteration_count,
            improvement_achieved=improvement_achieved,
            final_score=self.current_best.score,
            time_elapsed=time_elapsed
        )
        
        logger.info(f"Optimization completed! Final score: {self.current_best.score:.3f}")
        logger.info(f"Total improvement: {improvement_achieved:.3f}")
        
        return result
    
    def _save_results(self):
        """Save optimization results to files"""
        # Save optimized prompt
        with open("prompts/optimized_prompt.txt", "w", encoding="utf-8") as f:
            f.write(self.current_best.content)
        
        # Save optimization history
        history_data = {
            "optimization_run": {
                "timestamp": datetime.now().isoformat(),
                "total_iterations": self.iteration_count,
                "final_score": self.current_best.score
            },
            "history": [asdict(candidate) for candidate in self.optimization_history]
        }
        
        with open("prompts/optimization_history.json", "w", encoding="utf-8") as f:
            json.dump(history_data, f, indent=2, ensure_ascii=False)
        
        logger.info("Results saved to prompts/ directory") 