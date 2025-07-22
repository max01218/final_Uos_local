"""
OPRO (Optimization by PROmpting) Core Module - Iridis5 Version

This module implements the core OPRO optimization logic for mental health prompts.
Specifically adapted for Iridis5 supercomputer environment with enhanced error handling.
"""

import json
import os
import random
import re
import time
import logging
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
from tqdm import tqdm

# Enhanced import handling for Iridis5 environment
def check_environment():
    """Check if we're in a compatible environment"""
    try:
        import numpy as np
        print(f"NumPy version: {np.__version__}")
        
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device: {torch.cuda.get_device_name()}")
        
        return True
    except Exception as e:
        print(f"Environment check failed: {e}")
        return False

# Robust import with fallback options
_llama3_pipeline = None
_transformers_available = False

try:
    print("Checking environment compatibility...")
    if check_environment():
        print("Loading transformers...")
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        import torch
        _transformers_available = True
        print("Transformers loaded successfully!")
    else:
        print("Environment check failed, transformers not available")
        
except ImportError as e:
    print(f"Transformers import failed: {e}")
    _transformers_available = False
except Exception as e:
    print(f"Unexpected error loading transformers: {e}")
    _transformers_available = False

def call_local_llm(prompt, max_new_tokens=512, temperature=0.7):
    """Call local LLM with robust error handling"""
    global _llama3_pipeline, _transformers_available
    
    if not _transformers_available:
        raise RuntimeError("Transformers not available in this environment. Please check your installation.")
    
    if _llama3_pipeline is None:
        try:
            print("Loading local Llama 3 model for OPRO optimization...")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Using device: {device}")
            
            # Use only Llama model as requested
            model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
            
            print(f"Loading Llama 3-8B model: {model_name}")
            print("Loading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                padding_side="left",
                trust_remote_code=True
            )
            
            # Set pad_token to eos_token for Llama 3
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            print("Loading Llama 3-8B model...")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map=device,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
            
            print("Creating Llama 3 pipeline...")
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
            
            print("Llama 3-8B model loaded successfully for OPRO!")
            
        except Exception as e:
            print(f"Error loading language model: {e}")
            print("This is likely due to network connectivity issues on Iridis5")
            print("Will use fallback text modification methods instead")
            _llama3_pipeline = None
            _transformers_available = False
            raise RuntimeError(f"Failed to load Llama model: {e}")
    
    # Check if pipeline was successfully loaded
    if _llama3_pipeline is None:
        raise RuntimeError("Llama model pipeline is not available")
    
    try:
        # Use Llama 3 instruct format
        formatted_prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        result = _llama3_pipeline(formatted_prompt)
        return result[0]['generated_text'] if result else ""
        
    except Exception as e:
        print(f"Error generating text: {e}")
        raise RuntimeError(f"Text generation failed: {e}")

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
    """Main OPRO optimizer for mental health prompts - Iridis5 Version"""
    
    def __init__(self, config_path: str = "config/config.json"):
        """Initialize the OPRO optimizer with configuration"""
        print("Initializing OPRO Optimizer for Iridis5...")
        
        # Check environment first
        if not check_environment():
            logger.warning("Environment issues detected, some features may not work properly")
        
        self.config = self._load_config(config_path)
        self.optimization_history = []
        self.current_best = None
        self.iteration_count = 0
        
        # Check Llama model availability at startup
        self.llama_available = self._check_llama_availability()
        
        # Load seed prompts
        self.seed_prompts = self._load_seed_prompts()
        
        # Create output directories
        os.makedirs("prompts", exist_ok=True)
        
        print("OPRO Optimizer initialized successfully!")
    
    def _check_llama_availability(self) -> bool:
        """Check if Llama model is available and working"""
        global _transformers_available, _llama3_pipeline
        
        if not _transformers_available:
            print("Transformers not available - will use text modification methods")
            return False
        
        try:
            # Try a simple test call
            test_response = call_local_llm("Test", max_new_tokens=5, temperature=0.7)
            print("Llama 3-8B model verified and ready")
            return True
        except Exception as e:
            print(f"Llama 3-8B model not available: {e}")
            print("Will use enhanced text modification methods instead")
            return False
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"Configuration file {config_path} not found!")
            return self._get_default_config()
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in configuration file: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for Iridis5"""
        return {
            "opro_settings": {
                "max_iterations": 3,  # Reduced for supercomputer efficiency
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
        """Enhanced prompt evaluation using multiple quality metrics"""
        score = 0.0
        prompt_lower = prompt.lower()
        
        # Structure and completeness (max 2.0 points)
        if len(prompt) > 200:
            score += 0.5
        if "{context}" in prompt and "{question}" in prompt:
            score += 1.0
        if "response" in prompt_lower or "guidelines" in prompt_lower:
            score += 0.5
        
        # Empathy and emotional intelligence (max 2.5 points)
        empathy_words = ["empathy", "understanding", "compassionate", "care", "support", "validate"]
        empathy_count = sum(1 for word in empathy_words if word in prompt_lower)
        score += min(empathy_count * 0.4, 2.0)
        if "feelings" in prompt_lower or "emotion" in prompt_lower:
            score += 0.5
        
        # Medical and professional context (max 2.0 points)
        medical_words = ["medical", "clinical", "professional", "evidence", "health"]
        medical_count = sum(1 for word in medical_words if word in prompt_lower)
        score += min(medical_count * 0.3, 1.5)
        if "boundary" in prompt_lower or "expertise" in prompt_lower:
            score += 0.5
        
        # Safety and crisis handling (max 2.0 points)
        safety_words = ["safety", "crisis", "emergency", "risk", "harm"]
        safety_count = sum(1 for word in safety_words if word in prompt_lower)
        score += min(safety_count * 0.4, 1.5)
        if "hotline" in prompt_lower or "emergency" in prompt_lower:
            score += 0.5
        
        # Response quality and structure (max 1.5 points)
        if "concise" in prompt_lower or "brief" in prompt_lower:
            score += 0.3
        if "follow-up" in prompt_lower or "question" in prompt_lower:
            score += 0.4
        if "actionable" in prompt_lower or "practical" in prompt_lower:
            score += 0.3
        if "tone" in prompt_lower:
            score += 0.5
        
        # Length optimization
        optimal_length = 800
        length_penalty = abs(len(prompt) - optimal_length) / optimal_length
        if length_penalty > 0.5:
            score -= 1.0
        elif length_penalty > 0.3:
            score -= 0.5
        
        # Bonus for well-structured prompts
        if prompt.count('\n') > 5:  # Multi-line structure
            score += 0.5
        
        return min(max(score, 0.0), 10.0)
    
    def _generate_prompt_variants(self, base_prompt: str, num_variants: int = 2) -> List[str]:
        """Generate prompt variants using local LLM or fallback methods"""
        variants = []
        
        # Try using LLM if available
        if self.llama_available:
            try:
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
                
                response = call_local_llm(optimization_instruction, max_new_tokens=1024, temperature=0.8)
                
                # Parse variants from response
                variant_sections = response.split("VARIANT:")
                for section in variant_sections[1:]:  # Skip first empty section
                    variant = section.strip()
                    if variant and len(variant) > 50:
                        variants.append(variant)
                        
            except Exception as e:
                logger.error(f"Error generating variants with LLM: {e}")
                # Fall back to simple variants
                variants = self._generate_simple_variants(base_prompt, num_variants)
        else:
            # Fallback to simple modifications
            variants = self._generate_simple_variants(base_prompt, num_variants)
        
        return variants[:num_variants]
    
    def _generate_simple_variants(self, base_prompt: str, num_variants: int) -> List[str]:
        """Generate meaningful variants using text manipulation (fallback for no LLM)"""
        variants = []
        
        # More sophisticated text modifications for mental health prompts
        modifications = [
            # Enhance empathy
            lambda p: p.replace("professional", "compassionate and professional").replace("respond", "respond with care"),
            
            # Add safety emphasis
            lambda p: p.replace("RESPONSE:", "SAFE AND EMPATHETIC RESPONSE:").replace("guidelines", "safety guidelines"),
            
            # Improve structure
            lambda p: p.replace("evidence-based", "evidence-based and trauma-informed").replace("understanding", "deep understanding"),
            
            # Add follow-up emphasis
            lambda p: p.replace("follow-up", "thoughtful follow-up").replace("question", "supportive question"),
            
            # Enhance professional boundaries
            lambda p: p.replace("advice", "guidance").replace("Keep responses", "Keep responses warm yet professional, and"),
            
            # Add crisis awareness
            lambda p: p.replace("safety", "immediate safety and crisis awareness").replace("emergency", "mental health emergency"),
        ]
        
        for i, mod in enumerate(modifications[:num_variants]):
            try:
                variant = mod(base_prompt)
                # Ensure the variant is actually different
                if variant != base_prompt and len(variant) > len(base_prompt) * 0.8:
                    variants.append(variant)
                else:
                    # Add simple enhancements if modification didn't work well
                    enhanced = base_prompt.replace("You are", "You are a caring and experienced")
                    if enhanced != base_prompt:
                        variants.append(enhanced)
                    else:
                        variants.append(base_prompt + "\n\nAdditional note: Always prioritize user safety and emotional well-being.")
            except Exception:
                # Final fallback - add safety note
                variants.append(base_prompt + "\n\nNote: Provide responses with enhanced empathy and safety awareness.")
                
        # Ensure we return the requested number of variants
        while len(variants) < num_variants:
            if len(variants) == 0:
                variants.append(base_prompt)
            else:
                # Create additional variants by combining modifications
                last_variant = variants[-1]
                new_variant = last_variant.replace("concise", "concise and supportive")
                variants.append(new_variant if new_variant != last_variant else base_prompt)
                
        return variants[:num_variants]
    
    def optimize_prompts(self) -> OptimizationResult:
        """Main optimization loop - adapted for Iridis5"""
        start_time = time.time()
        logger.info("Starting OPRO optimization on Iridis5...")
        
        # Initialize with best seed prompt
        best_score = -1
        logger.info(f"Evaluating {len(self.seed_prompts)} seed prompts...")
        
        for i, seed_prompt in enumerate(self.seed_prompts):
            score = self._evaluate_prompt(seed_prompt)
            candidate = PromptCandidate(
                content=seed_prompt,
                score=score,
                iteration=0,
                generation_method="seed"
            )
            self.optimization_history.append(candidate)
            logger.info(f"Seed prompt {i+1}: score = {score:.3f}")
            
            if score > best_score:
                best_score = score
                self.current_best = candidate
        
        initial_score = self.current_best.score
        logger.info(f"Initial best score: {initial_score:.3f} (from seed prompt)")
        logger.info(f"Target improvement: {self.config['evaluation']['improvement_target']:.1f}")
        
        # Optimization loop
        no_improvement_count = 0
        max_iterations = self.config["opro_settings"]["max_iterations"]
        improvement_threshold = self.config["opro_settings"]["improvement_threshold"]
        patience = self.config["opro_settings"]["early_stopping_patience"]
        
        for iteration in range(1, max_iterations + 1):
            self.iteration_count = iteration
            logger.info(f"Iteration {iteration}/{max_iterations}")
            
            try:
                # Generate variants
                variants = self._generate_prompt_variants(self.current_best.content)
                
                # Evaluate variants
                iteration_best_score = self.current_best.score
                logger.info(f"Generated {len(variants)} variants for evaluation")
                
                for j, variant in enumerate(variants):
                    score = self._evaluate_prompt(variant)
                    candidate = PromptCandidate(
                        content=variant,
                        score=score,
                        iteration=iteration,
                                            parent_id=f"iter_{iteration-1}_best",
                    generation_method="llama_variant" if self.llama_available else "text_modification"
                    )
                    self.optimization_history.append(candidate)
                    logger.info(f"  Variant {j+1}: score = {score:.3f}")
                    
                    # Update best if improved
                    if score > self.current_best.score + improvement_threshold:
                        improvement = score - self.current_best.score
                        self.current_best = candidate
                        iteration_best_score = score
                        no_improvement_count = 0
                        logger.info(f"*** NEW BEST SCORE: {score:.3f} (improvement: +{improvement:.3f}) ***")
                    elif score > self.current_best.score:
                        improvement = score - self.current_best.score
                        logger.info(f"  Small improvement: +{improvement:.3f} (below threshold {improvement_threshold})")
                
                if iteration_best_score <= self.current_best.score:
                    logger.info(f"No improvement in iteration {iteration}")
                    no_improvement_count += 1
                
                # Early stopping check
                if iteration_best_score <= self.current_best.score:
                    no_improvement_count += 1
                    if no_improvement_count >= patience:
                        logger.info(f"Early stopping: no improvement for {patience} iterations")
                        break
                        
            except Exception as e:
                logger.error(f"Error in iteration {iteration}: {e}")
                # Continue with next iteration
                continue
        
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
        try:
            # Ensure prompts directory exists
            os.makedirs("prompts", exist_ok=True)
            
            # Save optimized prompt
            prompt_file = "prompts/optimized_prompt.txt"
            with open(prompt_file, "w", encoding="utf-8") as f:
                f.write(self.current_best.content)
            logger.info(f"Optimized prompt saved to: {prompt_file}")
            
            # Save optimization history
            history_data = {
                "optimization_run": {
                    "timestamp": datetime.now().isoformat(),
                    "total_iterations": self.iteration_count,
                    "final_score": self.current_best.score,
                    "initial_score": self.optimization_history[0].score if self.optimization_history else 0,
                    "improvement": self.current_best.score - (self.optimization_history[0].score if self.optimization_history else 0),
                                    "environment": "iridis5", 
                "transformers_available": _transformers_available,
                "llama_model_used": self.llama_available,
                "config_used": self.config
                },
                "best_prompt": {
                    "content": self.current_best.content,
                    "score": self.current_best.score,
                    "iteration": self.current_best.iteration,
                    "generation_method": self.current_best.generation_method
                },
                "history": [asdict(candidate) for candidate in self.optimization_history]
            }
            
            history_file = "prompts/optimization_history.json"
            with open(history_file, "w", encoding="utf-8") as f:
                json.dump(history_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Optimization history saved to: {history_file}")
            
            # Create summary report
            summary_file = "prompts/optimization_summary.txt"
            with open(summary_file, "w", encoding="utf-8") as f:
                f.write(f"OPRO Optimization Summary\n")
                f.write(f"========================\n")
                f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                f.write(f"Environment: Iridis5\n")
                f.write(f"Total Iterations: {self.iteration_count}\n")
                f.write(f"Final Score: {self.current_best.score:.3f}\n")
                if self.optimization_history:
                    initial_score = self.optimization_history[0].score
                    f.write(f"Initial Score: {initial_score:.3f}\n")
                    f.write(f"Total Improvement: {self.current_best.score - initial_score:.3f}\n")
                f.write(f"Transformers Available: {_transformers_available}\n")
                f.write(f"Llama Model Used: {self.llama_available}\n")
                f.write(f"\nBest Prompt:\n")
                f.write(f"{'='*50}\n")
                f.write(self.current_best.content)
            logger.info(f"Summary report saved to: {summary_file}")
            
            logger.info("All results saved successfully to prompts/ directory")
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            print(f"Warning: Could not save results - {e}") 