#!/usr/bin/env python3
"""
Intelligent Fusion System for CBT and ICD-11
Multi-modal fusion with hierarchical decision trees and intelligent routing
"""

import json
import logging
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import re

# Import our enhanced RAG system
try:
    from enhanced_rag_retriever import EnhancedRAGRetriever, QueryIntent, RetrievalContext, RetrievalResult
except ImportError:
    print("Warning: Enhanced RAG retriever not available")

class ProblemType(Enum):
    """Primary problem type classification"""
    SYMPTOM_INQUIRY = "symptom_inquiry"
    EMOTIONAL_SUPPORT = "emotional_support"  
    SKILL_LEARNING = "skill_learning"
    CRISIS_INTERVENTION = "crisis_intervention"
    INFORMATION_SEEKING = "information_seeking"
    MIXED_COMPLEX = "mixed_complex"

class FusionStrategy(Enum):
    """Different fusion strategies"""
    ICD11_PRIMARY = "icd11_primary"
    CBT_PRIMARY = "cbt_primary"
    BALANCED_FUSION = "balanced_fusion"
    DYNAMIC_ROUTING = "dynamic_routing"
    SAFETY_FIRST = "safety_first"

@dataclass
class FusionContext:
    """Context for fusion decisions"""
    problem_type: ProblemType = None
    urgency_level: int = 1
    user_history: List[Dict] = None
    session_stage: str = "initial"  # initial, ongoing, deep
    preferred_approach: str = None  # cbt, medical, integrated
    safety_concern: bool = False

@dataclass
class FusedResponse:
    """Integrated response from multiple knowledge bases"""
    primary_content: str
    supporting_content: str
    confidence: float
    fusion_strategy: FusionStrategy
    source_breakdown: Dict[str, float]
    reasoning: str
    follow_up_suggestions: List[str]
    safety_notes: List[str]
    metadata: Dict

class ProblemTypeClassifier:
    """Classify the primary problem type for routing decisions"""
    
    def __init__(self):
        self.classification_patterns = {
            ProblemType.SYMPTOM_INQUIRY: {
                "patterns": [
                    r"symptoms|signs|experiencing|feeling|have been",
                    r"diagnosis|disorder|condition|illness|disease",
                    r"what's wrong|what is happening|am i|do i have"
                ],
                "weight": 1.0
            },
            ProblemType.EMOTIONAL_SUPPORT: {
                "patterns": [
                    r"feeling|feel|emotions|emotional|upset|sad|lonely|scared",
                    r"support|comfort|understand|listen|talk|help me",
                    r"overwhelmed|stressed|anxious|depressed|hopeless"
                ],
                "weight": 1.0
            },
            ProblemType.SKILL_LEARNING: {
                "patterns": [
                    r"how to|teach me|learn|technique|strategy|skill|method",
                    r"steps|guide|instruction|practice|exercise|training",
                    r"coping|management|techniques|strategies|tools"
                ],
                "weight": 1.0
            },
            ProblemType.CRISIS_INTERVENTION: {
                "patterns": [
                    r"suicide|kill myself|hurt myself|end it all|can't go on",
                    r"crisis|emergency|urgent|desperate|breaking point",
                    r"immediate help|right now|emergency|can't cope"
                ],
                "weight": 2.0  # Higher weight for crisis
            },
            ProblemType.INFORMATION_SEEKING: {
                "patterns": [
                    r"what is|tell me about|explain|information|definition",
                    r"research|studies|facts|statistics|causes|treatment",
                    r"learn about|understand|knowledge"
                ],
                "weight": 1.0
            }
        }
        
    def classify_problem_type(self, query: str, context: Dict = None) -> Tuple[ProblemType, float]:
        """Classify the primary problem type"""
        query_lower = query.lower()
        type_scores = {}
        
        for problem_type, config in self.classification_patterns.items():
            score = 0
            weight = config["weight"]
            
            for pattern in config["patterns"]:
                matches = len(re.findall(pattern, query_lower))
                score += matches * weight
                
            if score > 0:
                type_scores[problem_type] = score
                
        if type_scores:
            best_type = max(type_scores.items(), key=lambda x: x[1])
            confidence = min(best_type[1] / len(query.split()), 1.0)
            
            # Check for mixed/complex problems
            if len([s for s in type_scores.values() if s > 0]) >= 2:
                return ProblemType.MIXED_COMPLEX, confidence * 0.8
                
            return best_type[0], confidence
        else:
            return ProblemType.INFORMATION_SEEKING, 0.1

class DecisionTree:
    """Hierarchical decision tree for routing"""
    
    def __init__(self):
        self.decision_rules = {
            # Crisis intervention - always priority
            ProblemType.CRISIS_INTERVENTION: {
                "strategy": FusionStrategy.SAFETY_FIRST,
                "cbt_weight": 0.8,
                "icd11_weight": 0.2,
                "reasoning": "Crisis detected - prioritize immediate coping strategies"
            },
            
            # Symptom inquiry - medical knowledge primary
            ProblemType.SYMPTOM_INQUIRY: {
                "strategy": FusionStrategy.ICD11_PRIMARY,
                "cbt_weight": 0.3,
                "icd11_weight": 0.7,
                "reasoning": "Symptom focus - medical knowledge with therapeutic support"
            },
            
            # Emotional support - therapeutic approach
            ProblemType.EMOTIONAL_SUPPORT: {
                "strategy": FusionStrategy.CBT_PRIMARY,
                "cbt_weight": 0.8,
                "icd11_weight": 0.2,
                "reasoning": "Emotional support - therapeutic techniques with context"
            },
            
            # Skill learning - CBT focused
            ProblemType.SKILL_LEARNING: {
                "strategy": FusionStrategy.CBT_PRIMARY,
                "cbt_weight": 0.9,
                "icd11_weight": 0.1,
                "reasoning": "Skill learning - practical CBT techniques prioritized"
            },
            
            # Information seeking - balanced approach
            ProblemType.INFORMATION_SEEKING: {
                "strategy": FusionStrategy.BALANCED_FUSION,
                "cbt_weight": 0.5,
                "icd11_weight": 0.5,
                "reasoning": "Information request - comprehensive integrated response"
            },
            
            # Mixed/complex - dynamic routing
            ProblemType.MIXED_COMPLEX: {
                "strategy": FusionStrategy.DYNAMIC_ROUTING,
                "cbt_weight": 0.6,
                "icd11_weight": 0.4,
                "reasoning": "Complex query - adaptive multi-modal approach"
            }
        }
        
    def get_routing_decision(self, problem_type: ProblemType, context: FusionContext) -> Dict:
        """Get routing decision based on problem type and context"""
        
        base_decision = self.decision_rules.get(problem_type, self.decision_rules[ProblemType.INFORMATION_SEEKING])
        decision = base_decision.copy()
        
        # Adjust based on context
        if context.urgency_level >= 4:
            # High urgency - boost CBT practical strategies
            decision["cbt_weight"] = min(decision["cbt_weight"] + 0.2, 1.0)
            decision["icd11_weight"] = max(decision["icd11_weight"] - 0.2, 0.0)
            decision["reasoning"] += " (adjusted for urgency)"
            
        if context.session_stage == "deep":
            # Deep conversation - more balanced approach
            decision["cbt_weight"] = (decision["cbt_weight"] + 0.5) / 2
            decision["icd11_weight"] = (decision["icd11_weight"] + 0.5) / 2
            decision["reasoning"] += " (adjusted for session depth)"
            
        if context.safety_concern:
            # Safety concern - prioritize immediate help
            decision["strategy"] = FusionStrategy.SAFETY_FIRST
            decision["cbt_weight"] = 0.9
            decision["icd11_weight"] = 0.1
            decision["reasoning"] = "Safety concern detected - immediate intervention prioritized"
            
        return decision

class ResponseTemplateEngine:
    """Generate structured responses based on fusion strategy"""
    
    def __init__(self):
        self.templates = {
            FusionStrategy.ICD11_PRIMARY: {
                "structure": ["understanding", "medical_context", "cbt_support", "resources"],
                "intro": "I understand you're experiencing {symptoms}. Let me provide some medical context and supportive strategies.",
                "transition": "From a therapeutic perspective,"
            },
            
            FusionStrategy.CBT_PRIMARY: {
                "structure": ["empathy", "cbt_techniques", "medical_context", "practice"],
                "intro": "I hear that you're {emotional_state}. There are effective techniques that can help.",
                "transition": "It's also helpful to understand that"
            },
            
            FusionStrategy.BALANCED_FUSION: {
                "structure": ["acknowledgment", "integrated_explanation", "practical_steps", "follow_up"],
                "intro": "Thank you for sharing this with me. Let me provide a comprehensive understanding.",
                "transition": "Connecting this information with practical approaches,"
            },
            
            FusionStrategy.SAFETY_FIRST: {
                "structure": ["immediate_support", "safety_resources", "coping_strategies", "professional_help"],
                "intro": "I'm concerned about your safety and want to help immediately.",
                "transition": "Right now, let's focus on"
            },
            
            FusionStrategy.DYNAMIC_ROUTING: {
                "structure": ["comprehensive_response", "multiple_perspectives", "personalized_guidance"],
                "intro": "This seems like a complex situation with multiple aspects to consider.",
                "transition": "Looking at this from different angles,"
            }
        }
        
    def generate_structured_response(self, 
                                   fusion_strategy: FusionStrategy,
                                   cbt_content: str,
                                   icd11_content: str,
                                   context: FusionContext,
                                   weights: Dict[str, float]) -> FusedResponse:
        """Generate a structured, integrated response"""
        
        template = self.templates.get(fusion_strategy, self.templates[FusionStrategy.BALANCED_FUSION])
        
        # Extract key information
        symptoms = self._extract_symptoms(cbt_content + " " + icd11_content)
        emotional_state = self._extract_emotional_state(cbt_content + " " + icd11_content)
        
        # Build response sections
        response_sections = []
        
        if fusion_strategy == FusionStrategy.ICD11_PRIMARY:
            response_sections = self._build_medical_primary_response(
                cbt_content, icd11_content, symptoms, emotional_state, template
            )
        elif fusion_strategy == FusionStrategy.CBT_PRIMARY:
            response_sections = self._build_therapeutic_primary_response(
                cbt_content, icd11_content, symptoms, emotional_state, template
            )
        elif fusion_strategy == FusionStrategy.SAFETY_FIRST:
            response_sections = self._build_safety_first_response(
                cbt_content, icd11_content, context, template
            )
        else:
            response_sections = self._build_balanced_response(
                cbt_content, icd11_content, weights, template
            )
            
        # Combine sections
        primary_content = "\n\n".join(response_sections)
        
        # Generate supporting content and metadata
        supporting_content = self._generate_supporting_content(cbt_content, icd11_content, fusion_strategy)
        follow_up_suggestions = self._generate_follow_up_suggestions(fusion_strategy, context)
        safety_notes = self._generate_safety_notes(context)
        
        # If both contents are empty due to filtering, return a graceful fallback
        if not cbt_content and not icd11_content:
            return self._generate_fallback_response("", context, fusion_strategy)

        return FusedResponse(
            primary_content=primary_content,
            supporting_content=supporting_content,
            confidence=self._calculate_fusion_confidence(weights),
            fusion_strategy=fusion_strategy,
            source_breakdown=weights,
            reasoning=f"Used {fusion_strategy.value} approach based on problem classification",
            follow_up_suggestions=follow_up_suggestions,
            safety_notes=safety_notes,
            metadata={
                "problem_type": context.problem_type.value if context.problem_type else None,
                "urgency_level": context.urgency_level,
                "session_stage": context.session_stage
            }
        )
        
    def _extract_symptoms(self, content: str) -> str:
        """Extract symptom mentions from content"""
        symptom_patterns = [
            r"anxiety|anxious|worried|nervous|panic|fear",
            r"depression|depressed|sad|down|hopeless",
            r"stress|stressed|overwhelmed|pressure",
            r"sleep|insomnia|tired|fatigue",
            r"anger|angry|irritated|frustrated"
        ]
        
        found_symptoms = []
        content_lower = content.lower()
        
        for pattern in symptom_patterns:
            matches = re.findall(pattern, content_lower)
            found_symptoms.extend(matches)
            
        return ", ".join(list(set(found_symptoms))[:3]) if found_symptoms else "these concerns"
        
    def _extract_emotional_state(self, content: str) -> str:
        """Extract emotional state from content"""
        emotional_patterns = {
            "overwhelmed": r"overwhelmed|too much|can't handle",
            "anxious": r"anxious|worried|nervous|scared",
            "sad": r"sad|down|depressed|low|empty",
            "frustrated": r"frustrated|angry|annoyed|irritated",
            "confused": r"confused|don't understand|unclear"
        }
        
        content_lower = content.lower()
        
        for emotion, pattern in emotional_patterns.items():
            if re.search(pattern, content_lower):
                return emotion
                
        return "struggling"
        
    def _build_medical_primary_response(self, cbt_content: str, icd11_content: str, 
                                      symptoms: str, emotional_state: str, template: Dict) -> List[str]:
        """Build response with medical context as primary"""
        sections = []
        
        # Understanding section
        intro = template["intro"].format(symptoms=symptoms)
        sections.append(intro)
        
        # Medical context (primary)
        if icd11_content:
            medical_section = f"From a medical perspective: {icd11_content[:300]}..."
            sections.append(medical_section)
            
        # CBT support (secondary)
        if cbt_content:
            transition = template["transition"]
            cbt_section = f"{transition} {cbt_content[:200]}..."
            sections.append(cbt_section)
            
        return sections
        
    def _build_therapeutic_primary_response(self, cbt_content: str, icd11_content: str,
                                          symptoms: str, emotional_state: str, template: Dict) -> List[str]:
        """Build response with therapeutic approach as primary"""
        sections = []
        
        # Empathy section
        intro = template["intro"].format(emotional_state=emotional_state)
        sections.append(intro)
        
        # CBT techniques (primary)
        if cbt_content:
            cbt_section = f"Here are some effective approaches: {cbt_content[:300]}..."
            sections.append(cbt_section)
            
        # Medical context (supporting)
        if icd11_content:
            transition = template["transition"]
            medical_section = f"{transition} {icd11_content[:150]}..."
            sections.append(medical_section)
            
        return sections
        
    def _build_safety_first_response(self, cbt_content: str, icd11_content: str,
                                   context: FusionContext, template: Dict) -> List[str]:
        """Build response prioritizing safety and immediate help"""
        sections = []
        
        # Immediate support
        sections.append(template["intro"])
        
        # Safety resources
        sections.append("If you're having thoughts of suicide or self-harm, please reach out for immediate help:")
        sections.append("- Crisis Text Line: Text HOME to 741741")
        sections.append("- National Suicide Prevention Lifeline: 988")
        sections.append("- Emergency Services: 911")
        
        # Immediate coping strategies
        if cbt_content:
            sections.append(f"Right now, try these immediate coping strategies: {cbt_content[:200]}...")
            
        return sections
        
    def _build_balanced_response(self, cbt_content: str, icd11_content: str,
                               weights: Dict[str, float], template: Dict) -> List[str]:
        """Build balanced integrated response"""
        sections = []
        
        # Acknowledgment
        sections.append(template["intro"])
        
        # Determine content order based on weights
        if weights.get("cbt", 0) > weights.get("icd11", 0):
            # CBT first
            if cbt_content:
                sections.append(f"From a therapeutic standpoint: {cbt_content[:250]}...")
            if icd11_content:
                sections.append(f"The medical understanding is: {icd11_content[:200]}...")
        else:
            # Medical first
            if icd11_content:
                sections.append(f"From a medical perspective: {icd11_content[:250]}...")
            if cbt_content:
                sections.append(f"Therapeutically, we can approach this by: {cbt_content[:200]}...")
                
        return sections
        
    def _generate_supporting_content(self, cbt_content: str, icd11_content: str, 
                                   fusion_strategy: FusionStrategy) -> str:
        """Generate supporting content"""
        if fusion_strategy == FusionStrategy.CBT_PRIMARY and icd11_content:
            return f"Additional medical context: {icd11_content[:300]}..."
        elif fusion_strategy == FusionStrategy.ICD11_PRIMARY and cbt_content:
            return f"Therapeutic approaches: {cbt_content[:300]}..."
        else:
            return "For more comprehensive information, both medical and therapeutic perspectives are available."
            
    def _generate_follow_up_suggestions(self, fusion_strategy: FusionStrategy, 
                                      context: FusionContext) -> List[str]:
        """Generate follow-up suggestions"""
        suggestions = []
        
        if fusion_strategy == FusionStrategy.SAFETY_FIRST:
            suggestions.extend([
                "Would you like to talk about what's been leading to these feelings?",
                "Can I help you identify someone you trust to talk to?",
                "Would you like some grounding techniques to use right now?"
            ])
        elif fusion_strategy == FusionStrategy.CBT_PRIMARY:
            suggestions.extend([
                "Would you like to practice this technique together?",
                "Can I help you create a plan for using these strategies?",
                "Would you like to explore what triggers these feelings?"
            ])
        elif fusion_strategy == FusionStrategy.ICD11_PRIMARY:
            suggestions.extend([
                "Would you like to discuss treatment options?",
                "Can I help you understand these symptoms better?",
                "Would you like information about professional resources?"
            ])
        else:
            suggestions.extend([
                "What aspect would you like to explore further?",
                "Would you like practical strategies or more information?",
                "How can I best support you moving forward?"
            ])
            
        return suggestions
        
    def _generate_safety_notes(self, context: FusionContext) -> List[str]:
        """Generate safety notes if needed"""
        notes = []
        
        if context.urgency_level >= 4:
            notes.append("High urgency detected - consider professional support")
            
        if context.safety_concern:
            notes.append("Safety concern identified - crisis resources provided")
            
        return notes
        
    def _calculate_fusion_confidence(self, weights: Dict[str, float]) -> float:
        """Calculate confidence in the fusion result"""
        total_weight = sum(weights.values())
        if total_weight == 0:
            return 0.0
            
        # Higher confidence when weights are more balanced
        balance_score = 1.0 - abs(weights.get("cbt", 0) - weights.get("icd11", 0))
        weight_strength = min(total_weight, 1.0)
        
        return (balance_score + weight_strength) / 2

class IntelligentFusionSystem:
    """Main intelligent fusion system coordinating CBT and ICD-11"""
    
    def __init__(self, config: Dict = None):
        # Merge config with defaults to ensure all required keys exist
        default_config = self._default_config()
        if config and isinstance(config, dict) and len(config) > 0:
            # Deep merge config with defaults
            merged_config = default_config.copy()
            merged_config.update(config)
            self.config = merged_config
        else:
            self.config = default_config
            
        self.problem_classifier = ProblemTypeClassifier()
        self.decision_tree = DecisionTree()
        self.template_engine = ResponseTemplateEngine()
        self.enhanced_retriever = None
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize retriever if available
        try:
            self.enhanced_retriever = EnhancedRAGRetriever()
        except:
            self.logger.warning("Enhanced RAG retriever not available")
            
    def _default_config(self) -> Dict:
        """Default configuration for fusion system"""
        return {
            "enable_safety_monitoring": True,
            "enable_context_awareness": True,
            "min_confidence_threshold": 0.1,
            "max_response_length": 500,
            "enable_follow_up_suggestions": True,
            "safety_keywords": ["suicide", "kill myself", "hurt myself", "end it all"],
            "context_memory_size": 5
        }
        
    def initialize_knowledge_bases(self, cbt_path: str = None, icd11_path: str = None):
        """Initialize knowledge bases for retrieval"""
        if self.enhanced_retriever:
            self.enhanced_retriever.load_knowledge_bases(cbt_path, icd11_path)
            
    def fuse_response(self, query: str, context: FusionContext = None) -> FusedResponse:
        """Main fusion method - integrate CBT and ICD-11 responses"""
        
        context = context or FusionContext()
        
        # Step 1: Classify problem type
        problem_type, type_confidence = self.problem_classifier.classify_problem_type(query)
        context.problem_type = problem_type
        
        # Step 2: Safety monitoring
        if self.config["enable_safety_monitoring"]:
            context.safety_concern = self._detect_safety_concern(query)
            
        # Step 3: Get routing decision
        routing_decision = self.decision_tree.get_routing_decision(problem_type, context)
        fusion_strategy = routing_decision["strategy"]
        
        # Step 4: Retrieve from knowledge bases
        retrieval_results = self._retrieve_from_knowledge_bases(query, context, routing_decision)
        
        # Step 5: Extract content by source
        cbt_content = self._extract_source_content(retrieval_results, "cbt")
        icd11_content = self._extract_source_content(retrieval_results, "icd11")
        
        # Step 6: Calculate final weights
        weights = {
            "cbt": routing_decision["cbt_weight"],
            "icd11": routing_decision["icd11_weight"]
        }
        
        # Adjust weights based on content availability
        if not cbt_content and icd11_content:
            weights = {"cbt": 0.0, "icd11": 1.0}
        elif not icd11_content and cbt_content:
            weights = {"cbt": 1.0, "icd11": 0.0}
        elif not cbt_content and not icd11_content:
            # Fallback response
            return self._generate_fallback_response(query, context, fusion_strategy)
            
        # Step 7: Generate fused response
        fused_response = self.template_engine.generate_structured_response(
            fusion_strategy, cbt_content, icd11_content, context, weights
        )
        
        # Step 8: Post-process and validate
        fused_response = self._post_process_response(fused_response, query, context)
        
        return fused_response
        
    def _detect_safety_concern(self, query: str) -> bool:
        """Detect safety concerns in the query"""
        query_lower = query.lower()
        
        for keyword in self.config["safety_keywords"]:
            if keyword in query_lower:
                return True
                
        # Additional pattern-based detection
        safety_patterns = [
            r"can't go on|breaking point|end it all",
            r"hurt myself|harm myself|kill myself",
            r"not worth living|want to die"
        ]
        
        for pattern in safety_patterns:
            if re.search(pattern, query_lower):
                return True
                
        return False
        
    def _retrieve_from_knowledge_bases(self, query: str, context: FusionContext, 
                                     routing_decision: Dict) -> List[RetrievalResult]:
        """Retrieve from both knowledge bases using enhanced retriever"""
        
        if not self.enhanced_retriever:
            return []
            
        try:
            # Create retrieval context
            retrieval_context = RetrievalContext(
                urgency_level=context.urgency_level,
                domain_focus="both"
            )
            
            # Retrieve results
            results = self.enhanced_retriever.retrieve(
                query, retrieval_context, top_k=8
            )
            
            return results
            
        except Exception as e:
            self.logger.error(f"Retrieval failed: {e}")
            return []
            
    def _extract_source_content(self, results: List[RetrievalResult], source: str) -> str:
        """Extract content from specific source"""
        source_results = [r for r in results if r.source == source]
        
        if not source_results:
            return ""
            
        # Combine top results from this source
        combined_content = []
        # Allow slightly below-threshold results to still contribute minimally
        threshold = float(self.config.get("min_confidence_threshold", 0.1))
        for result in source_results[:3]:  # Top 3 results
            if result.relevance_score >= threshold:
                combined_content.append(result.content)
            elif result.relevance_score >= max(0.05, threshold * 0.6):
                # Include attenuated content to avoid full fallback
                combined_content.append(result.content[:150])
                
        return " ".join(combined_content)
        
    def _generate_fallback_response(self, query: str, context: FusionContext, 
                                  fusion_strategy: FusionStrategy) -> FusedResponse:
        """Generate fallback response when no content is available"""
        
        fallback_messages = {
            FusionStrategy.SAFETY_FIRST: "I'm concerned about your safety. Please reach out to a crisis helpline or emergency services for immediate support.",
            FusionStrategy.CBT_PRIMARY: "I understand you're looking for coping strategies. While I don't have specific techniques to share right now, focusing on breathing and grounding can be helpful.",
            FusionStrategy.ICD11_PRIMARY: "I understand you're seeking information about your symptoms. For accurate medical guidance, please consult with a healthcare professional.",
            FusionStrategy.BALANCED_FUSION: "I want to help you with your concerns. While I don't have comprehensive information available right now, please know that support is available."
        }
        
        fallback_content = fallback_messages.get(fusion_strategy, fallback_messages[FusionStrategy.BALANCED_FUSION])
        
        return FusedResponse(
            primary_content=fallback_content,
            supporting_content="",
            confidence=0.2,
            fusion_strategy=fusion_strategy,
            source_breakdown={"fallback": 1.0},
            reasoning="Fallback response - limited content available",
            follow_up_suggestions=["Would you like to rephrase your question?", "Can I help you with something more specific?"],
            safety_notes=["Limited information available - consider professional consultation"] if context.safety_concern else [],
            metadata={"fallback": True}
        )
        
    def _post_process_response(self, response: FusedResponse, query: str, 
                             context: FusionContext) -> FusedResponse:
        """Post-process the fused response"""
        
        # Trim content if too long
        if len(response.primary_content) > self.config["max_response_length"]:
            response.primary_content = response.primary_content[:self.config["max_response_length"]] + "..."
            
        # Add safety notes if needed
        if context.safety_concern and not response.safety_notes:
            response.safety_notes.append("Safety concern detected - please consider professional support")
            
        # Enhance follow-up suggestions based on query
        if self.config["enable_follow_up_suggestions"] and len(response.follow_up_suggestions) < 2:
            response.follow_up_suggestions.extend([
                "Is there anything specific you'd like to know more about?",
                "How else can I support you today?"
            ])
            
        return response

if __name__ == "__main__":
    # Example usage
    fusion_system = IntelligentFusionSystem()
    
    # Example queries
    test_queries = [
        "I'm feeling very anxious and need coping strategies",
        "What are the symptoms of depression?",
        "I've been having thoughts of suicide",
        "How can I challenge my negative thoughts?",
        "I need information about anxiety disorders"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        context = FusionContext(urgency_level=2)
        
        response = fusion_system.fuse_response(query, context)
        
        print(f"Strategy: {response.fusion_strategy.value}")
        print(f"Response: {response.primary_content[:200]}...")
        print(f"Confidence: {response.confidence:.3f}")
        print(f"Reasoning: {response.reasoning}") 