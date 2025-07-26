#!/usr/bin/env python3
"""
CBT Integration Module
Integrates CBT functionality with existing systems
"""

import json
import os
import pickle
import random
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

# Try to import required libraries
try:
    import faiss
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    print(f"Missing dependencies: {e}")
    faiss = None
    SentenceTransformer = None

class CBTKnowledgeBase:
    def __init__(self, base_dir="cbt_data"):
        self.base_dir = Path(base_dir)
        self.embedding_model = None
        self.index = None
        self.metadata = []
        self.setup_logging()
        
        # Load CBT knowledge base
        self.load_cbt_knowledge()
        
    def setup_logging(self):
        """Setup logging configuration"""
        self.logger = logging.getLogger("cbt_integration")
        
    def load_cbt_knowledge(self) -> bool:
        """Load CBT vector index and metadata"""
        try:
            embeddings_dir = self.base_dir / "embeddings"
            
            # Check if CBT index exists
            index_path = embeddings_dir / "cbt_index.faiss"
            metadata_path = embeddings_dir / "cbt_metadata.pkl"
            summary_path = embeddings_dir / "cbt_index_summary.json"
            
            if not (index_path.exists() and metadata_path.exists()):
                self.logger.warning("CBT knowledge base not found")
                return False
                
            # Load summary info
            with open(summary_path, 'r', encoding='utf-8') as f:
                summary = json.load(f)
                
            model_name = summary.get('embedding_model', 'all-MiniLM-L6-v2')
            
            # Load embedding model
            if SentenceTransformer:
                self.embedding_model = SentenceTransformer(model_name)
                self.logger.info(f"Loaded CBT embedding model: {model_name}")
            else:
                self.logger.error("SentenceTransformer not available")
                return False
                
            # Load FAISS index
            if faiss:
                self.index = faiss.read_index(str(index_path))
                self.logger.info(f"Loaded CBT FAISS index with {self.index.ntotal} vectors")
            else:
                self.logger.error("FAISS not available")
                return False
                
            # Load metadata
            with open(metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
                
            self.logger.info(f"Loaded CBT metadata: {len(self.metadata)} items")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load CBT knowledge base: {e}")
            return False
            
    def search_cbt_techniques(self, query: str, top_k: int = 5, category_filter: Optional[str] = None) -> List[Dict]:
        """Search for CBT techniques and information"""
        if not self.index or not self.embedding_model:
            self.logger.warning("CBT knowledge base not available")
            return []
            
        try:
            # Create query embedding
            query_embedding = self.embedding_model.encode([query])
            query_embedding = query_embedding.astype('float32')
            faiss.normalize_L2(query_embedding)
            
            # Search in index
            search_k = top_k * 3  # Get more results for filtering
            scores, indices = self.index.search(query_embedding, search_k)
            
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.metadata):
                    result = self.metadata[idx].copy()
                    result['relevance_score'] = float(score)
                    
                    # Apply category filter if specified
                    if category_filter and result.get('category') != category_filter:
                        continue
                        
                    results.append(result)
                    
                    if len(results) >= top_k:
                        break
                        
            return results
            
        except Exception as e:
            self.logger.error(f"CBT search failed: {e}")
            return []
            
    def get_cbt_recommendation(self, user_query: str, context: str = "") -> Dict:
        """Get CBT technique recommendations based on user query"""
        
        # Analyze query for key indicators
        query_analysis = self.analyze_user_query(user_query)
        
        # Search for relevant CBT techniques
        indicators = query_analysis['indicators']
        if indicators:
            search_query = f"{user_query} {' '.join(indicators)}"
        else:
            # If no specific indicators, add generic anxiety/mental health terms
            search_query = f"{user_query} anxiety stress coping mental health"
        search_results = self.search_cbt_techniques(search_query, top_k=8)
        
        # Filter by technique type
        technique_results = [r for r in search_results if r.get('type') == 'cbt_technique']
        content_results = [r for r in search_results if r.get('type') == 'main_content']
        
        return {
            'query_analysis': query_analysis,
            'recommended_techniques': technique_results[:2],
            'supporting_content': content_results[:2],
            'total_found': len(search_results)
        }
        
    def analyze_user_query(self, query: str) -> Dict:
        """Analyze user query to identify relevant CBT indicators"""
        query_lower = query.lower()
        
        # CBT condition indicators
        condition_indicators = {
            'depression': ['sad', 'depressed', 'hopeless', 'worthless', 'empty', 'down'],
            'anxiety': ['anxious', 'anxiety', 'worried', 'nervous', 'panic', 'fear', 'stress', 'struggling'],
            'trauma': ['trauma', 'ptsd', 'flashback', 'nightmare', 'abuse'],
            'ocd': ['obsessive', 'compulsive', 'ritual', 'checking', 'washing'],
            'phobia': ['phobia', 'afraid', 'scared', 'avoidance'],
            'anger': ['angry', 'rage', 'furious', 'irritated', 'mad']
        }
        
        # CBT technique indicators
        technique_indicators = {
            'cognitive_restructuring': ['thoughts', 'thinking', 'believe', 'assume', 'negative thinking'],
            'behavioral_activation': ['activity', 'behavior', 'action', 'doing'],
            'exposure': ['avoid', 'escape', 'fear', 'exposure'],
            'relaxation': ['tense', 'relaxation', 'calm', 'breathe', 'breathing'],
            'problem_solving': ['problem', 'solution', 'decision', 'choice'],
            'coping_strategies': ['coping', 'cope', 'help', 'struggling', 'struggle', 'need help', 'strategies']
        }
        
        detected_conditions = []
        detected_techniques = []
        confidence_scores = {}
        
        # Check for condition indicators
        for condition, keywords in condition_indicators.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            if score > 0:
                detected_conditions.append(condition)
                confidence_scores[condition] = score / len(keywords)
                
        # Check for technique indicators  
        for technique, keywords in technique_indicators.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            if score > 0:
                detected_techniques.append(technique)
                confidence_scores[technique] = score / len(keywords)
                
        return {
            'detected_conditions': detected_conditions,
            'suggested_techniques': detected_techniques,
            'confidence_scores': confidence_scores,
            'indicators': detected_conditions + detected_techniques
        }
        
    def format_cbt_response(self, recommendations: Dict, user_query: str, context: str = "") -> str:
        """Format CBT recommendations with context-aware, natural response templates"""
        
        if not recommendations['recommended_techniques']:
            return self._get_contextual_fallback_response(user_query, context)
        
        # Analyze conversational context
        conversation_state = self._analyze_conversation_state(context, user_query)
        user_tone = self._detect_user_communication_style(user_query, context)
        
        # Get primary technique and condition
        techniques = recommendations['recommended_techniques']
        primary_technique = techniques[0] if techniques else {}
        technique_category = primary_technique.get('category', 'general_support')
        conditions = recommendations['query_analysis']['detected_conditions']
        primary_condition = conditions[0] if conditions else 'general_concern'
        
        # Generate contextual response
        response = self._generate_contextual_response(
            technique_category=technique_category,
            primary_condition=primary_condition,
            conversation_state=conversation_state,
            user_tone=user_tone,
            user_query=user_query
        )
        
        return response
        
    def _analyze_conversation_state(self, context: str, user_query: str) -> Dict:
        """Analyze the current state of conversation for contextual response"""
        state = {
            'stage': 'initial',  # initial, ongoing, follow_up, progress_check
            'user_progress': 'unknown',  # improving, stable, struggling, unknown
            'interaction_count': 0,
            'previous_topics': [],
            'emotional_trajectory': 'neutral'  # improving, declining, stable, mixed
        }
        
        if not context:
            return state
            
        context_lower = context.lower()
        
        # Determine conversation stage
        if 'recent conversation:' in context_lower:
            interaction_patterns = context_lower.count('user:')
            if interaction_patterns >= 4:
                state['stage'] = 'ongoing'
                state['interaction_count'] = interaction_patterns
            elif interaction_patterns >= 2:
                state['stage'] = 'follow_up'
                state['interaction_count'] = interaction_patterns
            else:
                state['stage'] = 'initial'
        
        # Analyze user progress
        progress_indicators = {
            'improving': ['better', 'calmer', 'helped', 'working', 'trying', 'good'],
            'struggling': ['worse', 'harder', 'difficult', 'still', 'same', 'stuck'],
            'stable': ['okay', 'managing', 'continue', 'more']
        }
        
        for progress_type, keywords in progress_indicators.items():
            if any(keyword in context_lower for keyword in keywords):
                state['user_progress'] = progress_type
                break
        
        # Track emotional trajectory
        emotional_words = {
            'positive': ['better', 'calmer', 'peaceful', 'hopeful', 'good'],
            'negative': ['worse', 'anxious', 'stressed', 'overwhelmed', 'bad'],
            'neutral': ['okay', 'same', 'still', 'continue']
        }
        
        for emotion_type, words in emotional_words.items():
            if any(word in context_lower for word in words):
                state['emotional_trajectory'] = emotion_type
                break
                
        return state
    
    def _detect_user_communication_style(self, user_query: str, context: str) -> Dict:
        """Detect user's communication preferences and style"""
        style = {
            'formality': 'neutral',  # formal, casual, neutral
            'directness': 'moderate',  # direct, gentle, moderate
            'question_preference': 'balanced',  # question_heavy, statement_heavy, balanced
            'detail_level': 'moderate'  # brief, detailed, moderate
        }
        
        query_lower = user_query.lower()
        full_context = f"{context} {user_query}".lower()
        
        # Analyze formality
        formal_indicators = ['please', 'could you', 'would you', 'i would like', 'assistance']
        casual_indicators = ['help me', 'can you', 'what should i', 'how do i', 'i need']
        
        formal_count = sum(1 for indicator in formal_indicators if indicator in full_context)
        casual_count = sum(1 for indicator in casual_indicators if indicator in full_context)
        
        if formal_count > casual_count:
            style['formality'] = 'formal'
        elif casual_count > formal_count:
            style['formality'] = 'casual'
        
        # Analyze directness preference
        if any(pattern in query_lower for pattern in ['just tell me', 'what should i do', 'give me']):
            style['directness'] = 'direct'
        elif any(pattern in query_lower for pattern in ['maybe', 'i think', 'perhaps', 'wondering']):
            style['directness'] = 'gentle'
        
        # Analyze response length preference
        if len(user_query.split()) <= 5:
            style['detail_level'] = 'brief'
        elif len(user_query.split()) >= 15:
            style['detail_level'] = 'detailed'
            
        return style
    
    def _generate_contextual_response(self, technique_category: str, primary_condition: str, 
                                     conversation_state: Dict, user_tone: Dict, user_query: str) -> str:
        """Generate contextually appropriate response using template pools"""
        
        # Select appropriate acknowledgment
        acknowledgment = self._select_contextual_acknowledgment(
            primary_condition, conversation_state, user_tone
        )
        
        # Select technique-specific guidance
        guidance = self._select_contextual_guidance(
            technique_category, conversation_state, user_tone, user_query
        )
        
        # Combine with natural flow
        return f"{acknowledgment} {guidance}"
    
    def _select_contextual_acknowledgment(self, condition: str, state: Dict, tone: Dict) -> str:
        """Select appropriate acknowledgment based on context"""
        
        # Acknowledgment template pools
        acknowledgments = {
            'anxiety': {
                'initial': [
                    "It sounds like you're dealing with some anxiety.",
                    "I can hear that anxiety is affecting you.", 
                    "It takes courage to reach out about anxiety.",
                    "I understand you're experiencing anxious feelings."
                ],
                'follow_up': {
                    'improving': [
                        "I'm glad to hear you're making progress with your anxiety.",
                        "It sounds like you're finding ways to manage those anxious feelings.",
                        "Thank you for sharing how things are going."
                    ],
                    'struggling': [
                        "I understand you're still working through these anxious feelings.",
                        "Anxiety can be persistent, and that's really challenging.",
                        "Thank you for continuing to share what you're experiencing."
                    ],
                    'stable': [
                        "It sounds like you're managing things as best you can.",
                        "Thank you for the update on how you're doing.",
                        "I appreciate you sharing where you're at right now."
                    ]
                }
            },
            'depression': {
                'initial': [
                    "I hear that you're going through a difficult time.",
                    "It sounds like you're dealing with some heavy feelings.",
                    "Thank you for trusting me with what you're experiencing.",
                    "I can sense that things feel overwhelming right now."
                ],
                'follow_up': {
                    'improving': [
                        "I'm glad to hear there are some positive changes.",
                        "It sounds like you're finding small steps forward.",
                        "Thank you for sharing this progress with me."
                    ],
                    'struggling': [
                        "I understand these feelings are still very present.",
                        "Depression can feel so persistent, and that's exhausting.",
                        "Thank you for continuing to reach out even when it's hard."
                    ]
                }
            },
            'stress': {
                'initial': [
                    "It sounds like you're under a lot of pressure right now.",
                    "I can hear that stress is really impacting you.",
                    "It makes sense that you're feeling overwhelmed.",
                    "Thank you for sharing what's been weighing on you."
                ],
                'follow_up': {
                    'improving': [
                        "I'm glad to hear you're finding ways to manage the stress.",
                        "It sounds like some strategies are starting to help.",
                        "Thank you for updating me on how things are going."
                    ]
                }
            },
            'general_concern': {
                'initial': [
                    "I appreciate you reaching out for support.",
                    "Thank you for sharing what's on your mind.",
                    "I'm here to help with whatever you're going through.",
                    "It sounds like you're dealing with something challenging."
                ]
            }
        }
        
        # Select based on context
        condition_templates = acknowledgments.get(condition, acknowledgments['general_concern'])
        
        if state['stage'] in ['follow_up', 'ongoing'] and isinstance(condition_templates.get('follow_up'), dict):
            progress_templates = condition_templates['follow_up'].get(state['user_progress'], 
                                                                     condition_templates['follow_up'].get('stable', []))
            if progress_templates:
                return self._select_by_tone(progress_templates, tone)
        
        initial_templates = condition_templates.get('initial', condition_templates)
        if isinstance(initial_templates, list):
            return self._select_by_tone(initial_templates, tone)
        
        return "Thank you for sharing what you're experiencing."
    
    def _select_contextual_guidance(self, technique_category: str, state: Dict, tone: Dict, user_query: str) -> str:
        """Select appropriate guidance based on technique and context"""
        
        guidance_templates = {
            'relaxation_techniques': {
                'initial': [
                    "Would you like to try a breathing exercise, or are you more interested in other relaxation approaches?",
                    "I can guide you through a simple technique right now. What sounds most helpful - breathing, muscle relaxation, or visualization?",
                    "Let's find something that feels right for you. Have you tried any relaxation methods before?",
                    "There are several approaches we could explore. What kind of situation are you hoping to feel calmer in?"
                ],
                'follow_up': [
                    "How did the last technique work for you? Should we try something similar or explore a different approach?",
                    "Building on what we talked about before, what would be most helpful right now?",
                    "Since you've been practicing, what aspects would you like to focus on today?"
                ]
            },
            'cognitive_restructuring': {
                'initial': [
                    "I'd like to help you explore your thoughts around this. What goes through your mind when these feelings are strongest?",
                    "Sometimes our thoughts can intensify difficult feelings. What thoughts have you noticed?",
                    "Let's look at how you're thinking about this situation. What story is your mind telling you?",
                    "I'm curious about your perspective on this. What thoughts come up for you?"
                ],
                'follow_up': [
                    "Let's continue examining those thought patterns. What have you noticed since we last talked?",
                    "How has your thinking about this shifted, if at all?",
                    "What thoughts are you working with today?"
                ]
            },
            'behavioral_activation': {
                'initial': [
                    "Sometimes small actions can make a big difference. What's one thing that usually helps you feel a bit better?",
                    "Let's think about activities that might help. What used to bring you some sense of accomplishment or pleasure?",
                    "I'm wondering about your daily routine. What feels manageable to focus on right now?",
                    "What's one small step you could take today that might help?"
                ]
            },
            'problem_solving': {
                'initial': [
                    "Let's break this down into manageable pieces. What feels like the most pressing part right now?",
                    "I'd like to help you work through this step by step. Where would you like to start?",
                    "What aspect of this situation feels most within your control?",
                    "Let's focus on what you can influence. What comes to mind first?"
                ]
            },
            'general_support': {
                'initial': [
                    "What would feel most helpful to focus on right now?",
                    "I'm wondering what kind of support would be most useful for you today.",
                    "What's your sense of what might help you feel even a little bit better?",
                    "Where would you like to start in addressing this?"
                ]
            }
        }
        
        # Select templates based on technique
        technique_templates = guidance_templates.get(technique_category, guidance_templates['general_support'])
        
        # Choose based on conversation stage
        if state['stage'] in ['follow_up', 'ongoing'] and 'follow_up' in technique_templates:
            templates = technique_templates['follow_up']
        else:
            templates = technique_templates['initial']
        
        return self._select_by_tone(templates, tone)
    
    def _select_by_tone(self, templates: list, tone: Dict) -> str:
        """Select template based on user's communication style"""
        
        # For now, randomly select from appropriate templates
        # Could be enhanced with ML-based selection in the future
        return random.choice(templates)
    
    def _get_contextual_fallback_response(self, user_query: str, context: str) -> str:
        """Generate contextual fallback when no specific techniques are recommended"""
        
        conversation_state = self._analyze_conversation_state(context, user_query)
        user_tone = self._detect_user_communication_style(user_query, context)
        
        fallback_templates = {
            'initial': [
                "I'm here to support you. What's been most on your mind lately?",
                "Thank you for reaching out. What would you like to explore today?",
                "I'd like to understand what you're going through. Can you tell me more?",
                "What brings you here today? I'm listening."
            ],
            'follow_up': [
                "What would be most helpful to focus on in our conversation today?",
                "How are things feeling for you right now?",
                "What's been on your mind since we last talked?",
                "Where would you like to start today?"
            ]
        }
        
        if conversation_state['stage'] in ['follow_up', 'ongoing']:
            templates = fallback_templates['follow_up']
        else:
            templates = fallback_templates['initial']
            
        return self._select_by_tone(templates, user_tone)
    
    def _is_followup_conversation(self, context: str, user_query: str) -> bool:
        """Check if this is a follow-up in an ongoing conversation"""
        if not context:
            return False
            
        context_lower = context.lower()
        query_lower = user_query.lower()
        
        # Indicators of follow-up conversation
        followup_indicators = [
            'recent conversation:', 'assistant asked:', 'user\'s recent emotional state:',
            'user\'s expressed goals:'
        ]
        
        # Short responses that suggest continuation
        short_responses = ['yes', 'no', 'better', 'worse', 'same', 'okay', 'good', 'bad']
        
        has_context = any(indicator in context_lower for indicator in followup_indicators)
        is_short_response = len(user_query.split()) <= 3 and any(word in query_lower for word in short_responses)
        
        return has_context or is_short_response
        
    def _analyze_user_progress(self, context: str) -> str:
        """Analyze user's progress from context"""
        if not context:
            return ""
            
        context_lower = context.lower()
        
        # Look for progress indicators
        progress_indicators = {
            'improved': ['better', 'calmer', 'improved', 'good', 'peaceful'],
            'same': ['same', 'still', 'no change', 'unchanged'],
            'worse': ['worse', 'harder', 'difficult', 'bad', 'terrible']
        }
        
        # Check emotional state from context
        if 'user\'s recent emotional state:' in context_lower:
            emotional_part = context_lower.split('user\'s recent emotional state:')[1].split('\n')[0]
            
            for progress_type, keywords in progress_indicators.items():
                if any(keyword in emotional_part for keyword in keywords):
                    return progress_type
                    
        # Check in recent conversation
        if 'recent conversation:' in context_lower:
            recent_part = context_lower.split('recent conversation:')[1]
            
            for progress_type, keywords in progress_indicators.items():
                if any(keyword in recent_part for keyword in keywords):
                    return progress_type
        
        return ""
    
    def extract_key_points(self, text: str, max_length: int = 150) -> str:
        """Extract key points from CBT technique text"""
        if not text:
            return ""
            
        # Remove "CBT Technique:" prefix if present
        text = text.replace('CBT Technique:', '').strip()
        
        # Clean up any formatting issues
        text = re.sub(r'\s+', ' ', text)  # Multiple spaces to single space
        text = re.sub(r'([a-zA-Z])([A-Z])', r'\1 \2', text)  # Fix concatenated words
        
        # Split into sentences
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        
        # Find most informative sentences (containing action words)
        action_words = ['practice', 'try', 'identify', 'challenge', 'focus', 'write', 'schedule', 'engage', 'use', 'apply', 'learn', 'develop']
        
        key_sentences = []
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(word in sentence_lower for word in action_words) and len(sentence) > 10:
                key_sentences.append(sentence)
                
        # If no action sentences, take first meaningful sentences
        if not key_sentences:
            key_sentences = [s for s in sentences if len(s) > 20][:2]
            
        # Combine and truncate
        if key_sentences:
            result = '. '.join(key_sentences[:2])
            if not result.endswith('.'):
                result += '.'
        else:
            result = text[:max_length] if len(text) > max_length else text
            
        if len(result) > max_length:
            # Cut at word boundary
            result = result[:max_length].rsplit(' ', 1)[0] + '...'
            
        return result
        
    def should_include_cbt(self, query: str) -> bool:
        """Determine if query would benefit from CBT techniques"""
        query_lower = query.lower()
        
        # Exclude simple social interactions that don't need CBT
        social_exclusions = [
            'thank you', 'thanks', 'hello', 'hi', 'good morning', 'good afternoon',
            'goodbye', 'bye', 'see you', 'nice talking', 'have a good day',
            'you too', 'same to you', 'no problem', 'you\'re welcome'
        ]
        
        # If it's a simple social interaction, don't use CBT
        if any(social in query_lower for social in social_exclusions):
            return False
        
        # Keywords that suggest CBT might be helpful - more specific
        cbt_indicators = [
            'how can i cope', 'what should i do about', 'help me with', 'strategies for',
            'coping with', 'deal with my', 'manage my', 'overcome my',
            'techniques for', 'methods for', 'ways to cope', 'exercises for', 'i need help with',
            'struggling with', 'i struggle with', 'i need help dealing'
        ]
        
        # Mental health conditions that commonly use CBT
        condition_keywords = [
            'anxiety', 'anxious', 'depression', 'depressed', 'stress', 'stressed',
            'worry', 'worried', 'panic', 'fear', 'afraid', 'nervous',
            'thoughts', 'thinking', 'behavior', 'mood', 'overwhelmed',
            'sad', 'down', 'upset', 'tense'
        ]
        
        # Direct CBT technique requests
        direct_cbt_requests = [
            'cbt', 'cognitive behavioral', 'cognitive behaviour',
            'thought challenging', 'breathing exercises', 'relaxation techniques',
            'mindfulness', 'meditation', 'behavioral activation'
        ]
        
        has_cbt_indicator = any(indicator in query_lower for indicator in cbt_indicators)
        has_condition_keyword = any(keyword in query_lower for keyword in condition_keywords)
        has_direct_request = any(request in query_lower for request in direct_cbt_requests)
        
        # More flexible logic: CBT relevant if any of these conditions are met:
        # 1. Has both CBT indicator and condition keyword (original logic)
        # 2. Has direct CBT technique request
        # 3. Has strong emotional/mental health indicators (even without explicit help request)
        strong_indicators = [
            'feel anxious', 'feeling anxious', 'feel depressed', 'feeling depressed',
            'feel overwhelmed', 'feeling overwhelmed', 'panic attacks', 'anxiety attacks',
            'negative thoughts', 'negative thinking', 'cant sleep', "can't sleep",
            'feel stressed', 'feeling stressed'
        ]
        
        has_strong_indicator = any(indicator in query_lower for indicator in strong_indicators)
        
        return (has_cbt_indicator and has_condition_keyword) or has_direct_request or has_strong_indicator
        
    def get_cbt_status(self) -> Dict:
        """Get CBT integration status"""
        return {
            'available': self.is_available(),
            'total_techniques': len([m for m in self.metadata if m.get('type') == 'cbt_technique']) if self.metadata else 0,
            'total_content': len([m for m in self.metadata if m.get('type') == 'main_content']) if self.metadata else 0,
            'categories': list(set(m.get('category', 'unknown') for m in self.metadata)) if self.metadata else []
        }

    def is_available(self) -> bool:
        """Check if CBT knowledge base is available"""
        return self.index is not None and self.embedding_model is not None

class CBTIntegration:
    def __init__(self, base_dir="cbt_data"):
        self.cbt_kb = CBTKnowledgeBase(base_dir)
        self.logger = logging.getLogger("cbt_integration")
        
    def enhance_response_with_cbt(self, user_query: str, context: str, base_response: str) -> str:
        """Enhance a response with CBT techniques if relevant"""
        
        if not self.cbt_kb.is_available():
            self.logger.debug("CBT knowledge base not available")
            return base_response
        
        # First check for social interactions that should never be enhanced with CBT
        query_lower = user_query.lower().strip()
        
        # Simple social interactions - return base response without CBT
        social_patterns = [
            'thank you', 'thanks', 'hello', 'hi there', 'good morning', 'good afternoon',
            'goodbye', 'bye', 'see you', 'have a good day', 'you too', 'same to you',
            'no problem', 'you\'re welcome', 'okay', 'ok', 'sure', 'yes', 'no'
        ]
        
        if any(pattern in query_lower for pattern in social_patterns) and len(query_lower.split()) <= 5:
            self.logger.debug(f"Social interaction detected, skipping CBT enhancement")
            return base_response
            
        # Check if query would benefit from CBT techniques
        should_enhance = self.should_include_cbt(user_query)
        self.logger.debug(f"CBT relevance for '{user_query}': {should_enhance}")
        
        if should_enhance:
            try:
                recommendations = self.cbt_kb.get_cbt_recommendation(user_query, context)
                self.logger.debug(f"CBT recommendations found: {len(recommendations.get('recommended_techniques', []))} techniques")
                
                if recommendations.get('recommended_techniques'):
                    cbt_response = self.cbt_kb.format_cbt_response(recommendations, user_query, context)
                    
                    # Check if base response is a crisis intervention (keep it)
                    # Otherwise, use the guided CBT response for better user experience
                    if self._is_crisis_response(base_response):
                        # Keep crisis response and add CBT guidance
                        enhanced_response = f"{base_response}\n\n{cbt_response}"
                    else:
                        # Replace with guided CBT response for better conversation flow
                        enhanced_response = cbt_response
                        
                    self.logger.debug(f"CBT enhancement applied, response length: {len(enhanced_response)}")
                    return enhanced_response
                else:
                    self.logger.debug("No CBT techniques found for this query")
                    
            except Exception as e:
                self.logger.error(f"CBT enhancement failed: {e}")
                import traceback
                self.logger.debug(f"CBT enhancement error details: {traceback.format_exc()}")
        else:
            self.logger.debug("Query not suitable for CBT enhancement")
                
        return base_response
        
    def _is_crisis_response(self, response: str) -> bool:
        """Check if response contains crisis intervention content"""
        response_lower = response.lower()
        
        # Strong crisis intervention indicators (should not appear in normal responses)
        strong_crisis_indicators = [
            'immediate safety', 'risk level', 'safety assessment',
            'self-harm', 'suicide', 'emergency services', 'crisis hotline',
            'danger to yourself', 'harm yourself', 'want to die',
            'assess immediate safety', 'safety guidance'
        ]
        
        # Mild crisis indicators (need multiple to trigger)
        mild_crisis_indicators = [
            'crisis', 'emergency', 'immediate help', 'professional help',
            'support systems', 'safety', 'risk'
        ]
        
        # Check for strong indicators (any one triggers)
        has_strong_indicator = any(indicator in response_lower for indicator in strong_crisis_indicators)
        
        # Check for multiple mild indicators  
        mild_count = sum(1 for indicator in mild_crisis_indicators if indicator in response_lower)
        has_multiple_mild = mild_count >= 3
        
        return has_strong_indicator or has_multiple_mild
        
    def should_include_cbt(self, query: str) -> bool:
        """Determine if query would benefit from CBT techniques"""
        return self.cbt_kb.should_include_cbt(query)
        
    def get_cbt_status(self) -> Dict:
        """Get CBT integration status"""
        return self.cbt_kb.get_cbt_status()

def main():
    """Test CBT integration"""
    print("CBT Integration Test")
    print("=" * 40)
    
    integration = CBTIntegration()
    status = integration.get_cbt_status()
    
    print(f"CBT Available: {status['available']}")
    if status['available']:
        print(f"CBT Techniques: {status['total_techniques']}")
        print(f"Content Items: {status['total_content']}")
        print(f"Categories: {status['categories']}")
        
        # Test queries
        test_queries = [
            "I feel anxious all the time, what can I do?",
            "How can I stop negative thinking?",
            "I'm depressed and need coping strategies"
        ]
        
        for query in test_queries:
            print(f"\nTest Query: {query}")
            
            if integration.should_include_cbt(query):
                recommendations = integration.cbt_kb.get_cbt_recommendation(query)
                response = integration.cbt_kb.format_cbt_response(recommendations, query)
                print(f"CBT Response: {response[:200]}...")
            else:
                print("No CBT enhancement needed")
    else:
        print("CBT knowledge base not available")
        print("Run the data collection and vectorization scripts first")

if __name__ == "__main__":
    main() 