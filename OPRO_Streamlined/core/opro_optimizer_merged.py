"""
OPRO (Optimization by PROmpting) Core Module - Merged Version

This module combines the complete functionality of ICD11_OPRO with the optimized features of OPRO_Streamlined.
Features:
- Complete OPRO optimization logic from ICD11_OPRO
- Robust LLaMA integration from OPRO_Streamlined
- Enhanced error handling and memory management
- Comprehensive evaluation system
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

# Enhanced LLaMA integration with robust error handling
_llama3_pipeline = None
_transformers_available = False

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
try:
    print("Checking environment compatibility...")
    if check_environment():
        print("Loading transformers...")
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
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
    """Enhanced local LLM call with robust error handling and quantization support"""
    global _llama3_pipeline, _transformers_available
    
    if not _transformers_available:
        raise RuntimeError("Transformers not available in this environment. Please check your installation.")
    
    if _llama3_pipeline is None:
        try:
            print("Loading local Llama 3 model for OPRO optimization...")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Using device: {device}")
            
            # Try models in order of preference
            models_to_try = [
                "meta-llama/Meta-Llama-3-8B-Instruct"
            ]
            
            for model_name in models_to_try:
                try:
                    print(f"Attempting to load {model_name}...")
                    
                    # Configure quantization for memory efficiency
                    quantization_config = None
                    if device == "cuda":
                        try:
                            import bitsandbytes
                            quantization_config = BitsAndBytesConfig(
                                load_in_4bit=True,
                                bnb_4bit_compute_dtype=torch.float16,
                                bnb_4bit_use_double_quant=True,
                                bnb_4bit_quant_type="nf4"
                            )
                            print("Using 4-bit quantization for memory efficiency")
                        except ImportError:
                            print("BitsAndBytes not available, using full precision")
                    
                    # Load tokenizer
                    tokenizer = AutoTokenizer.from_pretrained(
                        model_name,
                        padding_side="left",
                        trust_remote_code=True
                    )
                    
                    # Set pad_token to eos_token for Llama 3
                    if tokenizer.pad_token is None:
                        tokenizer.pad_token = tokenizer.eos_token
                    
                    # Load model
                    model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        quantization_config=quantization_config,
                        device_map=device,
                        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                        low_cpu_mem_usage=True,
                        trust_remote_code=True
                    )
                    
                    # Create pipeline
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
                    
                    # Check GPU memory usage
                    if torch.cuda.is_available():
                        allocated = torch.cuda.memory_allocated() / (1024**3)
                        print(f"Successfully loaded {model_name}")
                        print(f"GPU Memory used: {allocated:.2f}GB")
                    
                    break
                    
                except Exception as e:
                    print(f"Failed to load {model_name}: {e}")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
            
            if _llama3_pipeline is None:
                raise RuntimeError("All model loading attempts failed")
                
        except Exception as e:
            print(f"Error loading language model: {e}")
            raise RuntimeError(f"Failed to load Llama model: {e}")
    
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
    """
    Enhanced OPRO optimizer combining complete functionality with optimized performance
    """
    
    def __init__(self, config_path: str = "config/config.json"):
        """Initialize the optimizer with enhanced configuration"""
        self.config = self._load_config(config_path)
        self.optimization_history = []
        self.current_iteration = 0
        self.best_score = 0.0
        self.start_time = datetime.now()
        
        # Enhanced LLaMA availability check
        self.llama_available = self._check_llama_availability()
        
        # Load seed prompts
        self.seed_prompts = self._load_seed_prompts()
        
        logger.info(f"OPRO Optimizer initialized with {len(self.seed_prompts)} seed prompts")
        logger.info(f"LLaMA availability: {self.llama_available}")
    
    def _check_llama_availability(self) -> bool:
        """Check if LLaMA model is available"""
        try:
            if not _transformers_available:
                return False
            
            # Test with a simple prompt
            test_result = call_local_llm("Hello", max_new_tokens=10)
            return len(test_result) > 0
        except Exception as e:
            logger.warning(f"LLaMA availability check failed: {e}")
            return False
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration with fallback to defaults"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            logger.info(f"Configuration loaded from {config_path}")
            return config
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "optimization": {
                "max_iterations": 10,
                "population_size": 5,
                "mutation_rate": 0.3,
                "crossover_rate": 0.7,
                "elite_size": 2
            },
            "evaluation": {
                "criteria": ["clarity", "empathy", "professionalism", "effectiveness"],
                "weights": [0.25, 0.25, 0.25, 0.25],
                "max_score": 10.0
            },
            "llm": {
                "model_name": "meta-llama/Meta-Llama-3-8B-Instruct",
                "max_tokens": 512,
                "temperature": 0.7,
                "top_p": 0.9,
                "repetition_penalty": 1.1
            },
            "output": {
                "save_history": True,
                "save_optimized_prompt": True,
                "history_file": "prompts/optimization_history.json",
                "prompt_file": "prompts/optimized_prompt.txt"
            },
            "seeds": {
                "directory": "prompts/seeds/",
                "default_prompt": "You are a mental health professional providing support and guidance."
            }
        }
    
    def _load_seed_prompts(self) -> List[str]:
        """Load seed prompts with enhanced error handling"""
        try:
            seeds_dir = self.config.get("seeds", {}).get("directory", "prompts/seeds/")
            if os.path.exists(seeds_dir):
                seed_files = [f for f in os.listdir(seeds_dir) if f.endswith('.txt')]
                prompts = []
                for seed_file in seed_files:
                    with open(os.path.join(seeds_dir, seed_file), 'r', encoding='utf-8') as f:
                        prompts.append(f.read().strip())
                logger.info(f"Loaded {len(prompts)} seed prompts from {seeds_dir}")
                return prompts
        except Exception as e:
            logger.warning(f"Failed to load seed prompts: {e}")
        
        # Fallback to default prompt
        default_prompt = self.config.get("seeds", {}).get("default_prompt", 
            "You are a mental health professional providing support and guidance.")
        logger.info("Using default seed prompt")
        return [default_prompt]
    
    def _evaluate_prompt(self, prompt: str) -> float:
        """Enhanced prompt evaluation with comprehensive scoring"""
        try:
            # Multi-dimensional evaluation
            scores = {
                'clarity': self._score_clarity(prompt),
                'empathy': self._score_empathy(prompt),
                'professionalism': self._score_professionalism(prompt),
                'effectiveness': self._score_effectiveness(prompt)
            }
            
            # Weighted average
            weights = self.config.get("evaluation", {}).get("weights", [0.25, 0.25, 0.25, 0.25])
            total_score = sum(score * weight for score, weight in zip(scores.values(), weights))
            
            logger.debug(f"Prompt evaluation scores: {scores}, Total: {total_score:.3f}")
            return total_score
            
        except Exception as e:
            logger.error(f"Prompt evaluation failed: {e}")
            return 0.0
    
    def _score_clarity(self, prompt: str) -> float:
        """Score prompt clarity"""
        # Implementation of clarity scoring logic
        clarity_indicators = ['clear', 'understandable', 'simple', 'direct']
        score = sum(1 for indicator in clarity_indicators if indicator in prompt.lower())
        return min(score / len(clarity_indicators) * 10, 10.0)
    
    def _score_empathy(self, prompt: str) -> float:
        """Score prompt empathy"""
        empathy_indicators = ['empathy', 'understanding', 'support', 'care', 'compassion']
        score = sum(1 for indicator in empathy_indicators if indicator in prompt.lower())
        return min(score / len(empathy_indicators) * 10, 10.0)
    
    def _score_professionalism(self, prompt: str) -> float:
        """Score prompt professionalism"""
        professional_indicators = ['professional', 'clinical', 'medical', 'therapeutic']
        score = sum(1 for indicator in professional_indicators if indicator in prompt.lower())
        return min(score / len(professional_indicators) * 10, 10.0)
    
    def _score_effectiveness(self, prompt: str) -> float:
        """Score prompt effectiveness"""
        effectiveness_indicators = ['effective', 'helpful', 'beneficial', 'therapeutic']
        score = sum(1 for indicator in effectiveness_indicators if indicator in prompt.lower())
        return min(score / len(effectiveness_indicators) * 10, 10.0)
    
    def _generate_prompt_variants(self, base_prompt: str, num_variants: int = 3) -> List[str]:
        """Generate prompt variants using LLaMA or fallback methods"""
        variants = []
        
        if self.llama_available:
            try:
                # Use LLaMA for variant generation
                meta_prompt = self._create_meta_optimization_prompt(base_prompt)
                response = call_local_llm(meta_prompt, max_new_tokens=512)
                parsed_variants = self._parse_meta_llm_response(response)
                variants.extend(parsed_variants[:num_variants])
                logger.info(f"Generated {len(parsed_variants)} variants using LLaMA")
            except Exception as e:
                logger.warning(f"LLaMA variant generation failed: {e}")
        
        # Fallback to offline methods if needed
        if len(variants) < num_variants:
            offline_variants = self._generate_offline_variants(base_prompt, num_variants - len(variants))
            variants.extend(offline_variants)
        
        return variants[:num_variants]
    
    def _create_meta_optimization_prompt(self, base_prompt: str) -> str:
        """Create meta-optimization prompt for LLaMA"""
        return f"""
You are an expert at optimizing prompts for mental health professionals. 
Given the following base prompt, generate 3 improved variants that are:
1. More empathetic and supportive
2. More professional and clinical
3. More effective and actionable

Base prompt: {base_prompt}

Please generate 3 variants, each separated by "---VARIANT---":

Variant 1:
"""

    def _parse_meta_llm_response(self, response: str) -> List[str]:
        """Parse LLaMA response to extract variants"""
        try:
            # Split by variant separator
            variants = response.split("---VARIANT---")
            # Clean up each variant
            cleaned_variants = []
            for variant in variants:
                variant = variant.strip()
                if variant and len(variant) > 10:  # Minimum length check
                    cleaned_variants.append(variant)
            return cleaned_variants
        except Exception as e:
            logger.error(f"Failed to parse LLaMA response: {e}")
            return []
    
    def _generate_offline_variants(self, base_prompt: str, num_variants: int) -> List[str]:
        """Generate variants using offline methods"""
        variants = []
        
        # Empathy enhancement
        empathy_variant = self._create_empathy_enhanced_variant(base_prompt)
        variants.append(empathy_variant)
        
        # Safety enhancement
        safety_variant = self._create_safety_enhanced_variant(base_prompt)
        variants.append(safety_variant)
        
        # Professional enhancement
        professional_variant = self._create_professional_enhanced_variant(base_prompt)
        variants.append(professional_variant)
        
        return variants[:num_variants]
    
    def _create_empathy_enhanced_variant(self, base_prompt: str) -> str:
        """Create empathy-enhanced variant"""
        empathy_phrases = [
            "with empathy and understanding",
            "in a supportive and caring manner",
            "with compassion and patience"
        ]
        return f"{base_prompt} {random.choice(empathy_phrases)}."
    
    def _create_safety_enhanced_variant(self, base_prompt: str) -> str:
        """Create safety-enhanced variant"""
        safety_phrases = [
            "while ensuring safety and crisis awareness",
            "with appropriate safety protocols",
            "maintaining professional boundaries and safety"
        ]
        return f"{base_prompt} {random.choice(safety_phrases)}."
    
    def _create_professional_enhanced_variant(self, base_prompt: str) -> str:
        """Create professional-enhanced variant"""
        professional_phrases = [
            "using evidence-based approaches",
            "following clinical best practices",
            "with professional expertise and experience"
        ]
        return f"{base_prompt} {random.choice(professional_phrases)}."
    
    def optimize_prompts(self) -> OptimizationResult:
        """Main optimization loop with enhanced functionality"""
        logger.info("Starting OPRO optimization...")
        
        # Initialize with seed prompts
        self._initialize_with_seeds()
        
        max_iterations = self.config.get("optimization", {}).get("max_iterations", 10)
        
        for iteration in range(max_iterations):
            self.current_iteration = iteration + 1
            logger.info(f"Starting iteration {self.current_iteration}/{max_iterations}")
            
            # Generate variants from best prompts
            variants = self._generate_prompt_variants_from_best()
            
            # Evaluate variants
            for variant in variants:
                score = self._evaluate_prompt(variant)
                candidate = PromptCandidate(
                    content=variant,
                    score=score,
                    iteration=self.current_iteration,
                    generation_method="llama" if self.llama_available else "offline"
                )
                self.optimization_history.append(candidate)
                
                # Update best score
                if score > self.best_score:
                    self.best_score = score
                    logger.info(f"New best score: {score:.3f}")
            
            # Check for early stopping
            if self._should_stop_early():
                logger.info("Early stopping triggered")
                break
        
        # Calculate results
        best_prompt = max(self.optimization_history, key=lambda x: x.score)
        improvement = self.best_score - self.optimization_history[0].score if self.optimization_history else 0
        time_elapsed = (datetime.now() - self.start_time).total_seconds()
        
        result = OptimizationResult(
            best_prompt=best_prompt,
            optimization_history=self.optimization_history,
            total_iterations=self.current_iteration,
            improvement_achieved=improvement,
            final_score=self.best_score,
            time_elapsed=time_elapsed
        )
        
        # Save results
        self._save_results(result)
        
        logger.info(f"Optimization completed. Final score: {self.best_score:.3f}, Improvement: {improvement:.3f}")
        return result
    
    def _initialize_with_seeds(self):
        """Initialize optimization with seed prompts"""
        for i, seed_prompt in enumerate(self.seed_prompts):
            score = self._evaluate_prompt(seed_prompt)
            candidate = PromptCandidate(
                content=seed_prompt,
                score=score,
                iteration=0,
                generation_method="seed"
            )
            self.optimization_history.append(candidate)
            
            if score > self.best_score:
                self.best_score = score
    
    def _generate_prompt_variants_from_best(self) -> List[str]:
        """Generate variants from the best prompts"""
        # Get top prompts
        top_prompts = sorted(self.optimization_history, key=lambda x: x.score, reverse=True)[:3]
        variants = []
        
        for prompt_candidate in top_prompts:
            prompt_variants = self._generate_prompt_variants(prompt_candidate.content, num_variants=2)
            variants.extend(prompt_variants)
        
        return variants
    
    def _should_stop_early(self) -> bool:
        """Check if optimization should stop early"""
        if len(self.optimization_history) < 5:
            return False
        
        # Check for no improvement in last 3 iterations
        recent_scores = [c.score for c in self.optimization_history[-3:]]
        if len(recent_scores) >= 3 and max(recent_scores) <= self.best_score:
            return True
        
        return False
    
    def _save_results(self, result: OptimizationResult):
        """Save optimization results"""
        try:
            # Save optimization history
            history_file = self.config.get("output", {}).get("history_file", "prompts/optimization_history.json")
            os.makedirs(os.path.dirname(history_file), exist_ok=True)
            
            history_data = {
                'timestamp': datetime.now().isoformat(),
                'optimization_history': [asdict(candidate) for candidate in result.optimization_history],
                'final_score': result.final_score,
                'improvement': result.improvement_achieved,
                'iterations': result.total_iterations,
                'time_elapsed': result.time_elapsed
            }
            
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(history_data, f, indent=2, ensure_ascii=False)
            
            # Save best prompt
            prompt_file = self.config.get("output", {}).get("prompt_file", "prompts/optimized_prompt.txt")
            os.makedirs(os.path.dirname(prompt_file), exist_ok=True)
            
            with open(prompt_file, 'w', encoding='utf-8') as f:
                f.write(result.best_prompt.content)
            
            logger.info(f"Results saved to {history_file} and {prompt_file}")
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}") 