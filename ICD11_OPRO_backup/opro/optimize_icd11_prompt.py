"""
ICD-11 OPRO (Optimization by PROmpting) Core Module

This module implements the core OPRO optimization logic for mental health prompts.
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
import anthropic
from tqdm import tqdm

# ====== 使用Llama 3模型進行OPRO優化 ======
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    import torch
    _llama3_pipeline = None
    
    def call_local_llm(prompt, max_new_tokens=512, temperature=0.7):
        global _llama3_pipeline
        if _llama3_pipeline is None:
            print("Loading local Llama 3 model for OPRO optimization...")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            
            # 使用Llama 3模型
            model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
            
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                padding_side="left"
            )
            
            # 設置pad_token為eos_token（Llama 3需要）
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
        
        # 使用Llama 3的instruct格式
        formatted_prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        result = _llama3_pipeline(formatted_prompt)
        return result[0]['generated_text'] if result else ""
        
except ImportError:
    def call_local_llm(prompt, max_new_tokens=512, temperature=0.7):
        raise RuntimeError("transformers 或 torch 未安裝，無法使用本地Llama 3模型")

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
    """Main OPRO optimizer for ICD-11 mental health prompts"""
    
    def __init__(self, config_path: str = "config.json"):
        """Initialize the OPRO optimizer with configuration"""
        self.config = self._load_config(config_path)
        self.optimization_history = []
        self.current_best = None
        self.iteration_count = 0
        
        # Initialize API clients
        self._setup_api_clients()
        
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
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in configuration file: {e}")
            raise
    
    
    def _load_seed_prompts(self) -> List[str]:
        """Load all seed prompts from the seed_prompts directory"""
        seed_prompts = []
        seed_dir = "opro/seed_prompts"
        
        if not os.path.exists(seed_dir):
            logger.error(f"Seed prompts directory {seed_dir} not found!")
            return []
        
        for filename in os.listdir(seed_dir):
            if filename.endswith('.txt'):
                filepath = os.path.join(seed_dir, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                        if content:
                            seed_prompts.append(content)
                            logger.info(f"Loaded seed prompt: {filename}")
                except Exception as e:
                    logger.error(f"Failed to load seed prompt {filename}: {e}")
        
        return seed_prompts
    
    def optimize_prompts(self) -> OptimizationResult:
        """Main optimization loop using OPRO methodology"""
        logger.info("🚀 Starting OPRO optimization process...")
        start_time = time.time()
        
        # Initialize with seed prompts
        self._initialize_with_seeds()
        
        # Run optimization iterations
        for iteration in range(self.config["opro_settings"]["max_iterations"]):
            self.iteration_count = iteration + 1
            logger.info(f"Running iteration {self.iteration_count}/{self.config['opro_settings']['max_iterations']}")
            
            # Generate new prompt candidates
            new_candidates = self._generate_prompt_variants()
            
            # Evaluate candidates
            for candidate in new_candidates:
                score = self._evaluate_prompt_candidate(candidate)
                candidate.score = score
                candidate.iteration = self.iteration_count
                
                # Update best prompt if improved
                if self.current_best is None or score > self.current_best.score:
                    self.current_best = candidate
                    logger.info(f"✨ New best score: {score:.3f} (improvement: {score - (self.optimization_history[-1].score if self.optimization_history else 0):.3f})")
                
                self.optimization_history.append(candidate)
            
            # Check for early stopping
            if self._should_stop_early():
                logger.info("🛑 Early stopping criteria met")
                break
            
            # Rate limiting
            time.sleep(self.config["api_settings"]["rate_limit_delay"])
        
        end_time = time.time()
        time_elapsed = end_time - start_time
        
        # Save results
        result = OptimizationResult(
            best_prompt=self.current_best,
            optimization_history=self.optimization_history,
            total_iterations=self.iteration_count,
            improvement_achieved=self._calculate_improvement(),
            final_score=self.current_best.score if self.current_best else 0,
            time_elapsed=time_elapsed
        )
        
        self._save_optimization_results(result)
        
        logger.info(f"Optimization completed! Best score: {result.final_score:.3f}")
        return result
    
    def _initialize_with_seeds(self):
        """Initialize optimization with seed prompts"""
        logger.info("🌱 Evaluating seed prompts...")
        
        for i, seed_prompt in enumerate(self.seed_prompts):
            candidate = PromptCandidate(
                content=seed_prompt,
                score=0.0,
                iteration=0,
                generation_method=f"seed_{i+1}"
            )
            
            score = self._evaluate_prompt_candidate(candidate)
            candidate.score = score
            
            if self.current_best is None or score > self.current_best.score:
                self.current_best = candidate
            
            self.optimization_history.append(candidate)
            logger.info(f"Seed {i+1} score: {score:.3f}")
    
    def _generate_prompt_variants(self) -> List[PromptCandidate]:
        """Generate new prompt variants using meta-LLM or offline mode"""
        logger.info("🔄 Generating prompt variants...")
        
        # 檢查是否為離線模式
        if (self.config["opro_settings"].get("use_offline_mode", False) or 
            self.config["opro_settings"]["meta_llm_model"] == "offline"):
            return self._generate_offline_prompt_variants()
        
        # Create meta-prompt for optimization
        meta_prompt = self._create_meta_optimization_prompt()
        
        try:
            # Generate variants using meta-LLM
            response = self._call_meta_llm(meta_prompt)
            variants = self._parse_meta_llm_response(response)
            
            # Create candidate objects
            candidates = []
            for i, variant in enumerate(variants):
                candidate = PromptCandidate(
                    content=variant,
                    score=0.0,
                    iteration=self.iteration_count,
                    parent_id=self.current_best.content[:50] if self.current_best else None,
                    generation_method="meta_llm_generation"
                )
                candidates.append(candidate)
            
            logger.info(f"Generated {len(candidates)} prompt variants")
            return candidates
            
        except Exception as e:
            logger.error(f"Failed to generate variants: {e}")
            logger.info("🔄 Falling back to offline mode")
            return self._generate_offline_prompt_variants()
    
    def _create_meta_optimization_prompt(self) -> str:
        """Create the meta-prompt for optimization"""
        
        # Get recent optimization history
        recent_history = self.optimization_history[-5:] if len(self.optimization_history) >= 5 else self.optimization_history
        
        # Format performance analysis
        performance_analysis = self._analyze_recent_performance(recent_history)
        
        meta_prompt = f"""
You are an expert prompt engineer specializing in mental health AI systems. Your task is to optimize prompts for an ICD-11 based mental health support chatbot.

CURRENT BEST PROMPT (Score: {self.current_best.score:.3f}):
{self.current_best.content if self.current_best else "No current best"}

PERFORMANCE ANALYSIS:
{performance_analysis}

OPTIMIZATION OBJECTIVES:
1. Improve empathy and emotional connection (weight: {self.config['evaluation']['weights']['empathy']})
2. Enhance clinical accuracy and ICD-11 compliance (weight: {self.config['evaluation']['weights']['accuracy']})
3. Increase response relevance (weight: {self.config['evaluation']['weights']['relevance']})
4. Strengthen safety protocols (weight: {self.config['evaluation']['weights']['safety']})

REQUIREMENTS:
- Length: {self.config['prompt_structure']['min_prompt_length']}-{self.config['prompt_structure']['max_prompt_length']} characters
- Must include: {', '.join(self.config['prompt_structure']['required_sections'])}
- Must handle crisis situations appropriately
- Must include safety disclaimers

Please generate 3 improved prompt variants that address the performance gaps identified above. Each variant should:
1. Build upon the strengths of the current best prompt
2. Address specific weaknesses identified in the analysis
3. Experiment with different approaches (e.g., structure, tone, instructions)

Format your response as:

VARIANT 1:
[Full prompt text here]

VARIANT 2:
[Full prompt text here]

VARIANT 3:
[Full prompt text here]

OPTIMIZATION RATIONALE:
[Brief explanation of the changes made and expected improvements]
"""
        return meta_prompt
    
    def _analyze_recent_performance(self, recent_history: List[PromptCandidate]) -> str:
        """Analyze recent performance to guide optimization"""
        if not recent_history:
            return "No performance history available yet."
        
        scores = [candidate.score for candidate in recent_history]
        avg_score = sum(scores) / len(scores)
        
        # Analyze evaluation details if available
        evaluation_breakdown = {}
        for candidate in recent_history:
            if candidate.evaluation_details:
                for metric, score in candidate.evaluation_details.items():
                    if metric not in evaluation_breakdown:
                        evaluation_breakdown[metric] = []
                    evaluation_breakdown[metric].append(score)
        
        analysis = f"Average recent score: {avg_score:.3f}\n"
        
        if evaluation_breakdown:
            analysis += "Performance by metric:\n"
            for metric, scores in evaluation_breakdown.items():
                avg_metric_score = sum(scores) / len(scores)
                analysis += f"- {metric}: {avg_metric_score:.3f}\n"
        
        # Identify trends
        if len(scores) >= 2:
            trend = "improving" if scores[-1] > scores[0] else "declining" if scores[-1] < scores[0] else "stable"
            analysis += f"Recent trend: {trend}\n"
        
        return analysis
    
    
    def _parse_meta_llm_response(self, response: str) -> List[str]:
        """Parse meta-LLM response to extract prompt variants"""
        variants = []
        
        # Look for VARIANT patterns
        variant_pattern = r'VARIANT\s+\d+:\s*(.*?)(?=VARIANT\s+\d+:|OPTIMIZATION RATIONALE:|$)'
        matches = re.findall(variant_pattern, response, re.DOTALL | re.IGNORECASE)
        
        for match in matches:
            variant = match.strip()
            if len(variant) >= self.config['prompt_structure']['min_prompt_length']:
                variants.append(variant)
        
        # Fallback: split by common delimiters if pattern matching fails
        if not variants:
            potential_variants = response.split('\n\n')
            for variant in potential_variants:
                if len(variant.strip()) >= self.config['prompt_structure']['min_prompt_length']:
                    variants.append(variant.strip())
        
        return variants[:3]  # Limit to 3 variants
    
    def _generate_offline_variants(self) -> str:
        """生成離線模式的變體建議"""
        if not self.current_best:
            return "VARIANT 1:\n使用更具同理心的語言來回應用戶的心理健康問題。強調理解和支持。\n\nVARIANT 2:\n增強安全性檢查，對危機情況有更明確的應對指導。\n\nVARIANT 3:\n提高醫學準確性，確保符合 ICD-11 標準的專業建議。"
        
        # 根據當前最佳提示詞的特徵生成改進建議
        current_content = self.current_best.content.lower()
        variants = []
        
        # 變體1：增強同理心
        if "understanding" not in current_content and "compassionate" not in current_content:
            variants.append("增加更多體現同理心和理解的詞彙，如'理解您的感受'、'這確實不容易'等表達。")
        else:
            variants.append("進一步強化同理心表達，使用更溫暖和支持性的語言。")
        
        # 變體2：安全性優化
        if "crisis" not in current_content or "emergency" not in current_content:
            variants.append("添加更明確的危機處理協議，包括緊急聯繫方式和專業轉介指導。")
        else:
            variants.append("優化危機應對流程，確保能夠快速識別和處理高危情況。")
        
        # 變體3：專業準確性
        if "icd-11" not in current_content:
            variants.append("加強與 ICD-11 標準的一致性，使用更精確的醫學術語和分類。")
        else:
            variants.append("進一步提升專業性，確保醫學信息的準確性和權威性。")
        
        # 格式化回應
        response = ""
        for i, variant in enumerate(variants, 1):
            response += f"VARIANT {i}:\n{variant}\n\n"
        
        return response.strip()
    
    def _generate_offline_prompt_variants(self) -> List[PromptCandidate]:
        """生成離線模式的實際提示詞變體"""
        logger.info("📴 Using offline prompt generation")
        
        if not self.current_best:
            # 如果沒有當前最佳提示詞，使用預設模板
            return self._generate_default_variants()
        
        base_prompt = self.current_best.content
        variants = []
        
        # 變體1：增強同理心版本
        empathy_variant = self._create_empathy_enhanced_variant(base_prompt)
        if empathy_variant:
            variants.append(PromptCandidate(
                content=empathy_variant,
                score=0.0,
                iteration=self.iteration_count,
                parent_id=self.current_best.content[:50],
                generation_method="offline_empathy_enhancement"
            ))
        
        # 變體2：安全性增強版本
        safety_variant = self._create_safety_enhanced_variant(base_prompt)
        if safety_variant:
            variants.append(PromptCandidate(
                content=safety_variant,
                score=0.0,
                iteration=self.iteration_count,
                parent_id=self.current_best.content[:50],
                generation_method="offline_safety_enhancement"
            ))
        
        # 變體3：專業性增強版本
        professional_variant = self._create_professional_enhanced_variant(base_prompt)
        if professional_variant:
            variants.append(PromptCandidate(
                content=professional_variant,
                score=0.0,
                iteration=self.iteration_count,
                parent_id=self.current_best.content[:50],
                generation_method="offline_professional_enhancement"
            ))
        
        logger.info(f"Generated {len(variants)} offline variants")
        return variants
    
    def _generate_default_variants(self) -> List[PromptCandidate]:
        """生成預設的提示詞變體"""
        default_prompts = [
            """你是一個專業的心理健康支持AI助手，基於ICD-11標準提供協助。

角色定義：
- 以同理心和理解回應用戶的心理健康問題
- 提供基於證據的信息和支持
- 優先考慮用戶安全和福祉

回應指導：
1. 始終以溫暖、非評判的態度回應
2. 承認用戶的感受並提供驗證
3. 提供實用的應對策略和資源
4. 在適當時建議尋求專業幫助

安全協議：
- 如果檢測到自殺或自傷風險，立即提供危機資源
- 始終包含專業醫療免責聲明
- 鼓勵尋求專業心理健康服務

輸出格式：
- 使用清晰、易懂的語言
- 提供結構化的回應
- 包含具體的建議和資源""",

            """作為ICD-11認證的心理健康支持AI，我致力於提供專業、同理心的協助。

我的角色：
- 理解和驗證您的感受
- 提供準確的心理健康信息
- 協助您理解您的經歷
- 引導您獲得適當的專業幫助

回應原則：
1. 同理心第一：我會傾聽並理解您的感受
2. 安全為重：任何危機情況都會優先處理
3. 專業準確：所有信息都基於ICD-11標準
4. 賦能支持：幫助您找到自己的力量和資源

危機處理：
如果您有傷害自己的想法，請立即：
- 聯繫當地緊急服務
- 撥打心理健康熱線
- 前往最近的急診室

重要聲明：此服務不能替代專業醫療建議。如需診斷或治療，請諮詢合格的心理健康專業人員。""",

            """您好，我是您的心理健康支持夥伴，遵循ICD-11標準為您提供協助。

我的承諾：
- 創造一個安全、非評判的空間
- 提供基於科學證據的支持
- 尊重您的經歷和感受
- 協助您建立應對策略

服務範圍：
✓ 情緒支持和理解
✓ 心理健康教育
✓ 應對技巧指導
✓ 資源和轉介建議
✓ 危機干預支持

安全保障：
- 24/7危機檢測和回應
- 專業轉介指導
- 保密性保護
- 以用戶安全為最高優先

請記住：雖然我能提供支持和信息，但無法替代專業心理健康服務。如果您需要診斷、治療計劃或處方藥物，請諮詢合格的心理健康專業人員。

讓我們一起開始您的康復之旅。"""
        ]
        
        variants = []
        for i, prompt in enumerate(default_prompts):
            variants.append(PromptCandidate(
                content=prompt,
                score=0.0,
                iteration=self.iteration_count,
                generation_method=f"default_variant_{i+1}"
            ))
        
        return variants
    
    def _create_empathy_enhanced_variant(self, base_prompt: str) -> str:
        """創建同理心增強的變體"""
        # 添加更多同理心元素
        empathy_additions = [
            "我理解這對您來說一定很困難。",
            "您的感受是完全可以理解的。",
            "我會陪伴您度過這個艱難時期。",
            "每個人的康復之路都是獨特的，我們會以您的步調前進。"
        ]
        
        # 在適當位置插入同理心語句
        enhanced_prompt = base_prompt
        if "回應" in enhanced_prompt or "Response" in enhanced_prompt:
            enhanced_prompt += f"\n\n同理心重點：\n- {empathy_additions[0]}\n- 始終以理解和支持的態度回應"
        
        return enhanced_prompt
    
    def _create_safety_enhanced_variant(self, base_prompt: str) -> str:
        """Create safety-enhanced variant"""
        safety_addition = """

Enhanced Safety Protocol:
WARNING - Crisis Alert: If user expresses suicidal or self-harm intentions:
1. Express immediate concern and support
2. Provide emergency contact resources
3. Encourage seeking immediate professional help
4. Continuously monitor dialogue for risk signals

EMERGENCY RESOURCES:
- Crisis Hotline: Available 24/7 in your region
- Emergency Medical Services: Call local emergency number
- Mental Health Support: Contact local mental health services

Professional Disclaimer: This AI assistant cannot replace professional medical diagnosis or treatment. All serious mental health issues should consult qualified professionals."""
        
        return base_prompt + safety_addition
    
    def _create_professional_enhanced_variant(self, base_prompt: str) -> str:
        """Create professional-enhanced variant"""
        professional_addition = """

ICD-11 Compliance Requirements:
- Use standardized mental health terminology
- Follow evidence-based practice principles
- Provide accurate diagnostic reference information
- Maintain professional boundaries and ethical standards

Quality Assurance:
- All information based on latest ICD-11 classification
- Recommendations comply with international best practice guidelines
- Regular updates to reflect latest research evidence
- Strict adherence to privacy and confidentiality requirements

Professional Development: Continuous learning and improvement to ensure highest quality mental health support services."""
        
        return base_prompt + professional_addition
    
    def _generate_fallback_variants(self) -> List[PromptCandidate]:
        """Generate fallback variants using simple mutations"""
        logger.info("Generating fallback variants...")
        
        if not self.current_best:
            return []
        
        variants = []
        base_prompt = self.current_best.content
        
        # Simple mutation strategies
        mutations = [
            self._add_emphasis_mutation(base_prompt),
            self._reorder_sections_mutation(base_prompt),
            self._enhance_safety_mutation(base_prompt)
        ]
        
        for i, mutation in enumerate(mutations):
            if mutation:
                candidate = PromptCandidate(
                    content=mutation,
                    score=0.0,
                    iteration=self.iteration_count,
                    parent_id=self.current_best.content[:50],
                    generation_method=f"fallback_mutation_{i+1}"
                )
                variants.append(candidate)
        
        return variants
    
    def _add_emphasis_mutation(self, prompt: str) -> str:
        """Add emphasis to key sections"""
        # Simple emphasis addition
        emphasized = prompt.replace("GUIDELINES:", "**CRITICAL GUIDELINES:**")
        emphasized = emphasized.replace("SAFETY:", "**URGENT SAFETY:**")
        return emphasized
    
    def _reorder_sections_mutation(self, prompt: str) -> str:
        """Reorder sections for better flow"""
        # This is a simple example - in practice, this would be more sophisticated
        return prompt  # For now, return unchanged
    
    def _enhance_safety_mutation(self, prompt: str) -> str:
        """Enhance safety protocols"""
        safety_addition = "\nSAFETY PRIORITY: Always prioritize user safety above all other considerations."
        return prompt + safety_addition
    
    def _evaluate_prompt_candidate(self, candidate: PromptCandidate) -> float:
        """Evaluate a prompt candidate using the evaluation system"""
        logger.info(f"Evaluating prompt candidate (method: {candidate.generation_method})")
        
        # 使用快速評估方法（基於關鍵詞和啟發式規則）
        evaluation_details = {}
        
        # Relevance scoring
        relevance_score = self._score_relevance(candidate.content)
        evaluation_details['relevance'] = relevance_score
        
        # Empathy scoring  
        empathy_score = self._score_empathy(candidate.content)
        evaluation_details['empathy'] = empathy_score
        
        # Accuracy scoring
        accuracy_score = self._score_accuracy(candidate.content)
        evaluation_details['accuracy'] = accuracy_score
        
        # Safety scoring
        safety_score = self._score_safety(candidate.content)
        evaluation_details['safety'] = safety_score
        
        # Calculate weighted total
        weights = self.config['evaluation']['weights']
        score = (
            relevance_score * weights['relevance'] +
            empathy_score * weights['empathy'] +
            accuracy_score * weights['accuracy'] +
            safety_score * weights['safety']
        )
        
        candidate.evaluation_details = evaluation_details
        
        logger.info(f"Evaluation complete. Score: {score:.3f} (R:{relevance_score:.1f} E:{empathy_score:.1f} A:{accuracy_score:.1f} S:{safety_score:.1f})")
        return score
    
    def _score_relevance(self, prompt: str) -> float:
        """Score prompt relevance"""
        score = 5.0  # Base score
        
        # Check for key relevance indicators
        if "question" in prompt.lower():
            score += 1.0
        if "context" in prompt.lower():
            score += 1.0
        if "response" in prompt.lower():
            score += 1.0
        
        return min(score, 10.0)
    
    def _score_empathy(self, prompt: str) -> float:
        """Score prompt empathy"""
        score = 5.0  # Base score
        
        empathy_keywords = ["compassionate", "understanding", "supportive", "validat", "empathy", "care"]
        for keyword in empathy_keywords:
            if keyword in prompt.lower():
                score += 0.5
        
        return min(score, 10.0)
    
    def _score_accuracy(self, prompt: str) -> float:
        """Score prompt accuracy"""
        score = 5.0  # Base score
        
        accuracy_keywords = ["icd-11", "evidence", "professional", "clinical", "accurate"]
        for keyword in accuracy_keywords:
            if keyword in prompt.lower():
                score += 0.5
        
        return min(score, 10.0)
    
    def _score_safety(self, prompt: str) -> float:
        """Score prompt safety"""
        score = 5.0  # Base score
        
        safety_keywords = ["safety", "crisis", "emergency", "professional help", "disclaimer"]
        for keyword in safety_keywords:
            if keyword in prompt.lower():
                score += 0.5
        
        return min(score, 10.0)
    
    def _should_stop_early(self) -> bool:
        """Check if early stopping criteria are met"""
        if len(self.optimization_history) < self.config["opro_settings"]["early_stopping_patience"]:
            return False
        
        # Check if improvement has stagnated
        recent_scores = [h.score for h in self.optimization_history[-self.config["opro_settings"]["early_stopping_patience"]:]]
        improvement = max(recent_scores) - min(recent_scores)
        
        return improvement < self.config["opro_settings"]["improvement_threshold"]
    
    def _calculate_improvement(self) -> float:
        """Calculate total improvement achieved"""
        if len(self.optimization_history) < 2:
            return 0.0
        
        initial_score = min([h.score for h in self.optimization_history[:3]])
        final_score = self.current_best.score if self.current_best else 0
        
        return final_score - initial_score
    
    def _save_optimization_results(self, result: OptimizationResult):
        """Save optimization results to files"""
        logger.info("💾 Saving optimization results...")
        
        # Save best prompt
        with open("prompts/optimized_prompt.txt", "w", encoding="utf-8") as f:
            f.write(result.best_prompt.content)
        
        # Save optimization history
        history_data = {
            "optimization_completed": datetime.now().isoformat(),
            "final_score": result.final_score,
            "improvement_achieved": result.improvement_achieved,
            "total_iterations": result.total_iterations,
            "time_elapsed": result.time_elapsed,
            "candidates": [asdict(candidate) for candidate in result.optimization_history]
        }
        
        with open("prompts/optimization_history.json", "w", encoding="utf-8") as f:
            json.dump(history_data, f, indent=2, ensure_ascii=False)
        
        logger.info("Results saved successfully!")
    
    def load_optimization_history(self) -> List[PromptCandidate]:
        """Load previous optimization history"""
        history_file = "prompts/optimization_history.json"
        
        if not os.path.exists(history_file):
            return []
        
        try:
            with open(history_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            candidates = []
            for candidate_data in data.get('candidates', []):
                candidate = PromptCandidate(**candidate_data)
                candidates.append(candidate)
            
            return candidates
            
        except Exception as e:
            logger.error(f"Failed to load optimization history: {e}")
            return [] 