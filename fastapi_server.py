#!/usr/bin/env python3
"""
FastAPI Server for ICD-11 RAG System - Enhanced Version
Integrated with Enhanced RAG and Intelligent Fusion Systems
"""
global enhanced_rag_retriever, intelligent_fusion_system
import os
import sys
import json
from pathlib import Path
from typing import List, Optional, Dict, Tuple
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from app.schemas.chat import (
    Message,
    RAGRequest,
    RAGResponse,
    FeedbackRequest,
    HealthResponse,
    CBTRequest,
    CBTResponse,
)
from app.services.memory_service import ConversationStore, summarize_session_transcript
from app.repositories.session_repo import (
    init_session_db,
    save_session_summary,
    load_session_summary,
    append_session_metrics,
)
import logging
import uvicorn
import time
import asyncio
import sqlite3
from functools import lru_cache
from threading import RLock
from datetime import datetime
import re

# HuggingFace and LLM
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.llms.huggingface_pipeline import HuggingFacePipeline

# Enhanced RAG and Fusion Systems
try:
    from enhanced_rag_retriever import EnhancedRAGRetriever, QueryIntent, RetrievalContext, RetrievalResult, RetrievalStrategy
    from intelligent_fusion_system import IntelligentFusionSystem, ProblemType, FusionContext, FusedResponse, FusionStrategy
    ENHANCED_SYSTEMS_AVAILABLE = True
    print("Enhanced RAG and Fusion systems loaded successfully")
except ImportError as e:
    print(f"Enhanced systems not available: {e}")
    ENHANCED_SYSTEMS_AVAILABLE = False
    EnhancedRAGRetriever = None
    IntelligentFusionSystem = None

# Fallback to basic LangChain RAG components
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from transformers import pipeline

# CBT Integration
try:
    from CBT_System.integration import CBTIntegration
    CBT_AVAILABLE = True
    print("CBT integration module loaded successfully")
except ImportError as e:
    print(f"CBT integration not available: {e}")
    CBT_AVAILABLE = False
    CBTIntegration = None

# Set environment variables to avoid tokenizer warnings
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['TRANSFORMERS_OFFLINE'] = '0'

# Check GPU availability
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize FastAPI
app = FastAPI(
    title="ICD-11 Enhanced RAG API - Intelligent Fusion",
    description="Enhanced API with Intelligent RAG and Fusion systems for comprehensive mental health assistance",
    version="4.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enable debug logging for enhanced systems
enhanced_logger = logging.getLogger("enhanced_systems")
enhanced_logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
enhanced_logger.addHandler(console_handler)

# Global variables
psychologist_llm = None
psychologist_tokenizer = None
store = None
embedder = None
emotion_classifier = None
memory = ConversationBufferMemory(return_messages=True)
cbt_integration = None

# Enhanced Systems
enhanced_rag_retriever = None
intelligent_fusion_system = None

# Unified Conversation Store with session persistence

# Initialize unified conversation store
conversation_store = ConversationStore()

    

# OPRO Integration
OPRO_PROMPT_PATH = "OPRO_Streamlined/prompts/optimized_prompt.txt"  
OPRO_FALLBACK_PATH =  "OPRO_Streamlined/prompts/optimized_prompt.txt"  
INTERACTIONS_FILE = "interactions.json"

# Enhanced System Configuration
ENHANCED_CONFIG_PATH = "intelligent_fusion_config.json"

# Debug settings (default to false to reduce overhead)
SHOW_PROMPT_DEBUG = os.getenv("SHOW_PROMPT_DEBUG", "false").lower() == "true"
SHOW_ENHANCED_DEBUG = os.getenv("SHOW_ENHANCED_DEBUG", "false").lower() == "true"

# Fallback prompts (used when OPRO prompt is not available)
FALLBACK_PROMPTS = {
    "professional": """You are a professional mental health advisor. Provide concise, evidence-based responses.

MEDICAL CONTEXT:
{context}

CONVERSATION HISTORY:
{history}

USER QUESTION: {question}

INSTRUCTIONS:
- Keep response to 2-4 sentences maximum
- Reference medical context only when highly relevant
- Ask 1 thoughtful follow-up question
- Maintain professional but warm tone
- Avoid generic lifestyle advice

RESPONSE:""",

    "caring": """You are a compassionate mental health companion. Provide brief emotional support.

MEDICAL CONTEXT:
{context}

CONVERSATION HISTORY:
{history}

USER MESSAGE: {question}

INSTRUCTIONS:
- Start with emotional validation (1 sentence)
- Ask 1 open-ended question to explore feelings
- Keep response to 2-3 sentences maximum
- Focus on emotional support over medical information
- Avoid generic advice

RESPONSE:""",

    "empathetic_professional": """You are a compassionate mental health professional. Provide concise emotional support with gentle guidance.

MEDICAL CONTEXT:
{context}

CONVERSATION HISTORY:
{history}

USER'S CONCERN: {question}

INSTRUCTIONS:
- Start with empathy (1 sentence)
- Cite ICD-11 context if relevant (1 sentence)
- Ask 1 gentle follow-up question
- Keep response to 2-4 sentences maximum
- Avoid generic lifestyle advice unless ICD-11 mentions it

RESPONSE:"""
}

def load_opro_prompt() -> str:
    """Load the latest OPRO optimized prompt"""
    try:
        # Try new optimized prompt first
        if os.path.exists(OPRO_PROMPT_PATH):
            with open(OPRO_PROMPT_PATH, 'r', encoding='utf-8') as f:
                prompt = f.read().strip()
            logger.info(f"Loaded OPRO Streamlined prompt ({len(prompt)} characters)")
            return prompt
        # Fallback to original OPRO path
        elif os.path.exists(OPRO_FALLBACK_PATH):
            with open(OPRO_FALLBACK_PATH, 'r', encoding='utf-8') as f:
                prompt = f.read().strip()
            logger.info(f"Loaded OPRO fallback prompt ({len(prompt)} characters)")
            return prompt
        else:
            logger.warning(f"No OPRO prompt found, using system fallback")
            return FALLBACK_PROMPTS["empathetic_professional"]
    except Exception as e:
        logger.error(f"Error loading OPRO prompt: {e}")
        return FALLBACK_PROMPTS["empathetic_professional"]

def get_dynamic_prompt(tone: str = "empathetic_professional") -> str:
    """Get the appropriate prompt based on tone and OPRO availability"""
    # All tones now use OPRO optimized prompt
    opro_prompt = load_opro_prompt()
    if opro_prompt and opro_prompt != FALLBACK_PROMPTS["empathetic_professional"]:
        return opro_prompt
    else:
        # Fallback to tone-specific prompts if OPRO not available
        return FALLBACK_PROMPTS.get(tone, FALLBACK_PROMPTS["empathetic_professional"])

def save_interaction(question: str, answer: str, tone: str, user_feedback: Optional[int] = None):
    """Save interaction for OPRO optimization"""
    try:
        # Load existing interactions
        interactions = []
        if os.path.exists(INTERACTIONS_FILE):
            with open(INTERACTIONS_FILE, 'r', encoding='utf-8') as f:
                interactions = json.load(f)
        
        # Create new interaction record
        interaction = {
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "answer": answer,
            "tone": tone,
            "user_feedback": user_feedback,
            "context_length": len(answer),
            "response_time": time.time()  # Will be updated with actual time
        }
        
        interactions.append(interaction)
        
        # Save back to file
        with open(INTERACTIONS_FILE, 'w', encoding='utf-8') as f:
            json.dump(interactions, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved interaction (total: {len(interactions)})")
        
    except Exception as e:
        logger.error(f"Error saving interaction: {e}")

# Pydantic request/response schemas
class Message(BaseModel):
    role: str
    content: str

class RAGRequest(BaseModel):
    question: str
    type: str = "empathetic_professional"
    session_id: Optional[str] = None
    history: Optional[List[Message]] = Field(default_factory=list)
    weekly_goal: Optional[str] = None
    feasibility: Optional[float] = None  # 0-10
    anxiety_level: Optional[float] = None  # 0-10

class RAGResponse(BaseModel):
    answer: str
    question: str
    tone: str
    status: str
    context_used: Optional[str] = None
    prompt_source: str = "fallback"  # "opro" or "fallback"
    confidence: Optional[float] = None
    fusion_strategy: Optional[str] = None
    source_breakdown: Optional[Dict[str, float]] = None
    follow_up_suggestions: Optional[List[str]] = None
    safety_notes: Optional[List[str]] = None
    # conversation and intent metadata
    session_id: Optional[str] = None
    intent: Optional[str] = None
    strategy: Optional[str] = None
    tone_suggested: Optional[str] = None
    weekly_goal: Optional[str] = None
    feasibility: Optional[float] = None
    anxiety_level: Optional[float] = None

class FeedbackRequest(BaseModel):
    question: str
    answer: str
    rating: int  # 1-5 scale
    feedback_text: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    psychologist_llm_loaded: bool
    store_loaded: bool
    device: str
    gpu_memory: Optional[str] = None
    opro_prompt_loaded: bool
    interactions_count: int
    cbt_available: bool
    cbt_techniques: Optional[int] = None
    cbt_content: Optional[int] = None
    cbt_smoke_test_passed: Optional[bool] = None
    enhanced_systems_available: bool
    enhanced_rag_loaded: bool
    intelligent_fusion_loaded: bool
    resolved_paths: Optional[dict] = None

# Load components
def load_embeddings():
    try:
        logger.info(f"Loading HuggingFace embeddings on {DEVICE}...")
        embedder = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={"device": DEVICE},
            encode_kwargs={"normalize_embeddings": True}
        )
        logger.info(f"Embeddings loaded successfully on {DEVICE}")
        return embedder
    except Exception as e:
        logger.error(f"Error loading embeddings: {e}")
        return None

def load_faiss_index(embedder):
    try:
        logger.info("Loading FAISS index...")
        store = FAISS.load_local("embeddings", embedder, allow_dangerous_deserialization=True)
        logger.info("FAISS index loaded successfully")
        return store
    except Exception as e:
        logger.exception("Error loading FAISS index")
        return None

def load_emotion_classifier():
    """Load emotion classification model with fallback options"""
    try:
        logger.info("Loading emotion classifier...")
        classifier = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            device=DEVICE,
            torch_dtype=torch.float32
        )
        logger.info("Emotion classifier loaded successfully")
        return classifier
    except Exception as e:
        logger.warning(f"Primary emotion classifier failed: {e}")
        
        # Fallback: Try alternative model
        try:
            logger.info("Trying fallback emotion classifier...")
            classifier = pipeline(
                "text-classification",
                model="cardiffnlp/twitter-roberta-base-emotion",
                device=DEVICE,
                torch_dtype=torch.float32
            )
            logger.info("Fallback emotion classifier loaded successfully")
            return classifier
        except Exception as e2:
            logger.warning(f"Fallback emotion classifier also failed: {e2}")
            
            # Final fallback: Simple keyword-based emotion detection
            logger.info("Using simple keyword-based emotion detection")
            return "keyword_based"
    except Exception as e:
        logger.error(f"Error loading emotion classifier: {e}")
        return None

def load_cbt_integration():
    """Load CBT integration system"""
    try:
        if CBT_AVAILABLE:
            logger.info("Initializing CBT integration...")
            cbt = CBTIntegration(base_dir="CBT_System/cbt_data")
            status = cbt.get_cbt_status()
            
            if status['available']:
                logger.info(f"CBT integration loaded successfully")
                logger.info(f"CBT Techniques: {status['total_techniques']}")
                logger.info(f"CBT Content: {status['total_content']}")
                logger.info(f"CBT Categories: {status['categories']}")
                return cbt
            else:
                logger.warning("CBT knowledge base not available")
                return None
        else:
            logger.warning("CBT integration module not available")
            return None
    except Exception as e:
        logger.error(f"Error loading CBT integration: {e}")
        return None

def load_enhanced_systems():
    """Load enhanced RAG and fusion systems"""
    global enhanced_rag_retriever, intelligent_fusion_system
    
    if not ENHANCED_SYSTEMS_AVAILABLE:
        logger.warning("Enhanced systems not available, using fallback")
        return False
    
    try:
        # Load configuration
        config_path = Path(ENHANCED_CONFIG_PATH)
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
        else:
            logger.warning(f"Enhanced config not found at {config_path}, using defaults")
            config = {}
        
        # Initialize Enhanced RAG Retriever with flattened config
        enhanced_rag_config = config.get("enhanced_rag_config", {})
        # Flatten the config structure to match what EnhancedRAGRetriever expects
        flattened_config = {
            "semantic_weight": enhanced_rag_config.get("retrieval_weights", {}).get("semantic_weight", 0.5),
            "keyword_weight": enhanced_rag_config.get("retrieval_weights", {}).get("keyword_weight", 0.3),
            "context_weight": enhanced_rag_config.get("retrieval_weights", {}).get("context_weight", 0.2),
            "intent_boost": enhanced_rag_config.get("retrieval_weights", {}).get("intent_boost", 0.15),
            "urgency_boost": enhanced_rag_config.get("retrieval_weights", {}).get("urgency_boost", 0.1),
            "min_relevance_threshold": enhanced_rag_config.get("filtering", {}).get("min_relevance_threshold", 0.45),  # Increased from 0.3
            "max_results": enhanced_rag_config.get("filtering", {}).get("max_results", 5),
            "diversity_threshold": enhanced_rag_config.get("filtering", {}).get("diversity_threshold", 0.85),  # Added diversity threshold
            "enable_query_expansion": enhanced_rag_config.get("features", {}).get("enable_query_expansion", True),
            "enable_intent_filtering": enhanced_rag_config.get("features", {}).get("enable_intent_filtering", True),
            "enable_context_awareness": enhanced_rag_config.get("features", {}).get("enable_context_awareness", True),
            "enable_diversity_filtering": True  # Enable diversity filtering
        }
        
        # Debug: print the flattened config
        logger.info(f"Enhanced RAG config: {flattened_config}")
        
        enhanced_rag_retriever = EnhancedRAGRetriever(flattened_config)
        
        # Validate and resolve knowledge base paths from config
        knowledge_bases = config.get('knowledge_bases', {})
        resolved_paths = {}
        
        for kb_name, kb_config in knowledge_bases.items():
            primary_path = kb_config.get('index_path')
            fallback_paths = kb_config.get('fallback_paths', [])
            
            # Try primary path first
            if primary_path and os.path.exists(primary_path):
                resolved_paths[kb_name] = primary_path
                logger.info(f"Using primary path for {kb_name}: {primary_path}")
            else:
                # Try fallback paths
                for fallback_pattern in fallback_paths:
                    if '*' in fallback_pattern:
                        # Handle wildcard patterns
                        import glob
                        matches = glob.glob(fallback_pattern)
                        if matches:
                            # Sort by modification time, use most recent
                            matches.sort(key=lambda x: os.path.getmtime(x), reverse=True)
                            resolved_paths[kb_name] = matches[0]
                            logger.info(f"Using fallback path for {kb_name}: {matches[0]}")
                            break
                    elif os.path.exists(fallback_pattern):
                        resolved_paths[kb_name] = fallback_pattern
                        logger.info(f"Using fallback path for {kb_name}: {fallback_pattern}")
                        break
                else:
                    logger.warning(f"No valid path found for {kb_name}")
        
        # Update config with resolved paths
        config['resolved_paths'] = resolved_paths
        
        # Load knowledge bases with resolved paths
        cbt_index_path = resolved_paths.get('cbt', "CBT_System/cbt_data/embeddings/cbt_index.faiss")
        icd11_index_path = resolved_paths.get('icd11', "embeddings/index.faiss")
        
        enhanced_rag_retriever.load_knowledge_bases(cbt_index_path, icd11_index_path)
        # Store the full config separately without overwriting retriever's internal config
        setattr(enhanced_rag_retriever, "full_config", config)
        logger.info("Enhanced RAG Retriever loaded successfully")
        
        # Initialize Intelligent Fusion System
        intelligent_fusion_system = IntelligentFusionSystem(config.get("fusion_system_config", {}))
        # Use the same enhanced retriever instance
        intelligent_fusion_system.enhanced_retriever = enhanced_rag_retriever
        intelligent_fusion_system.initialize_knowledge_bases(cbt_index_path, icd11_index_path)
        logger.info("Intelligent Fusion System loaded successfully")
        
        # Set global variables explicitly
        globals()['enhanced_rag_retriever'] = enhanced_rag_retriever
        globals()['intelligent_fusion_system'] = intelligent_fusion_system
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to load enhanced systems: {e}")
        import traceback
        logger.error(f"Enhanced systems error details: {traceback.format_exc()}")
        return False

def load_psychologist_llm():
    """Load the psychologist LLM for empathetic professional responses"""
    logger.info(f"Loading Psychologist LLM on {DEVICE}...")
    try:
        model_id = os.getenv("LLM_MODEL_ID", "Qwen/Qwen2.5-3B-Instruct")
        tok = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=True,
            padding_side="left"
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            device_map=DEVICE,
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
            low_cpu_mem_usage=True
        )
        
        # Get decoding parameters from environment
        max_new_tokens = int(os.getenv("LLM_MAX_NEW_TOKENS", "64"))
        temperature = float(os.getenv("LLM_TEMPERATURE", "0.4"))
        do_sample = os.getenv("LLM_DO_SAMPLE", "false").lower() == "true"
        
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tok,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tok.eos_token_id,
            eos_token_id=tok.eos_token_id,
            return_full_text=False
        )
        
        psychologist_llm = HuggingFacePipeline(pipeline=pipe)
        globals()['psychologist_tokenizer'] = tok
        logger.info(f"Psychologist LLM loaded successfully on {DEVICE}")
        return psychologist_llm
    except Exception as e:
        logger.exception("Error loading psychologist LLM")
        return None

def initialize_rag_system():
    global psychologist_llm, store, embedder, emotion_classifier, cbt_integration, enhanced_rag_retriever, intelligent_fusion_system
    logger.info(f"Initializing Enhanced RAG system on {DEVICE}...")
    
    # Load basic components
    embedder = load_embeddings()
    if embedder is None:
        return False
    store = load_faiss_index(embedder)
    if store is None:
        return False
    psychologist_llm = load_psychologist_llm()
    if psychologist_llm is None:
        return False
    emotion_classifier = load_emotion_classifier()
    if emotion_classifier is None:
        logger.warning("Emotion classifier not loaded, continuing without emotion analysis")
    
    # Load CBT integration
    cbt_integration = load_cbt_integration()
    if cbt_integration is None:
        logger.warning("CBT integration not loaded, continuing without CBT features")
    
    # Load enhanced systems
    enhanced_systems_loaded = load_enhanced_systems()
    if enhanced_systems_loaded:
        logger.info("Enhanced RAG and Fusion systems loaded successfully")
    else:
        logger.warning("Enhanced systems not loaded, using fallback RAG system")
    
    logger.info("Enhanced RAG system initialization completed")
    return True

@app.on_event("startup")
async def startup_event():
    if not initialize_rag_system():
        logger.error("Failed to initialize RAG system")
        sys.exit(1)
    init_session_db()

@app.get("/health", response_model=HealthResponse)
async def health_check():
    gpu_memory = None
    if DEVICE == "cuda":
        try:
            gpu_memory = f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB"
        except:
            pass
    
    # Check OPRO prompt availability
    opro_prompt_loaded = os.path.exists(OPRO_PROMPT_PATH)
    
    # Get interactions count
    interactions_count = 0
    if os.path.exists(INTERACTIONS_FILE):
        try:
            with open(INTERACTIONS_FILE, 'r', encoding='utf-8') as f:
                interactions = json.load(f)
                interactions_count = len(interactions)
        except:
            pass
    
    # Get CBT status with smoke test
    cbt_available = False
    cbt_techniques = None
    cbt_content = None
    cbt_smoke_test_passed = False
    
    if cbt_integration is not None:
        try:
            cbt_status = cbt_integration.get_cbt_status()
            cbt_available = cbt_status['available']
            cbt_techniques = cbt_status['total_techniques']
            cbt_content = cbt_status['total_content']
            
            # Perform smoke test if CBT is available
            if cbt_available and cbt_techniques > 0:
                try:
                    # Test query for CBT functionality
                    test_query = "I feel anxious, what can I do?"
                    is_relevant = cbt_integration.should_include_cbt(test_query)
                    if is_relevant:
                        recommendations = cbt_integration.cbt_kb.get_cbt_recommendation(test_query)
                        if recommendations.get('recommended_techniques'):
                            cbt_smoke_test_passed = True
                            logger.info("CBT smoke test passed")
                        else:
                            logger.warning("CBT smoke test failed: no techniques found")
                    else:
                        logger.warning("CBT smoke test failed: query not recognized as CBT relevant")
                except Exception as e:
                    logger.warning(f"CBT smoke test failed: {e}")
        except Exception as e:
            logger.error(f"Error getting CBT status: {e}")
    
        # Check enhanced systems status
    enhanced_systems_available = ENHANCED_SYSTEMS_AVAILABLE
    enhanced_rag_loaded = globals().get('enhanced_rag_retriever') is not None
    intelligent_fusion_loaded = globals().get('intelligent_fusion_system') is not None
    
    # Get resolved paths from enhanced systems if available
    resolved_paths = {}
    if enhanced_rag_retriever and hasattr(enhanced_rag_retriever, 'config'):
        resolved_paths = enhanced_rag_retriever.config.get('resolved_paths', {})
    
    # Calculate CBT statistics if available
    if enhanced_rag_retriever and hasattr(enhanced_rag_retriever, 'cbt_metadata'):
        try:
            # Count techniques and content from CBT metadata
            cbt_techniques = 0
            cbt_content = 0
            if enhanced_rag_retriever.cbt_metadata:
                for item in enhanced_rag_retriever.cbt_metadata:
                    if isinstance(item, dict):
                        # Check if it's a technique or content based on metadata
                        content_type = (item.get('content_type') or '').lower()
                        primary_category = (item.get('primary_category') or '').lower()
                        
                        # Count as technique if it's a technique, protocol, or assessment tool
                        if any(keyword in content_type for keyword in ['technique', 'protocol', 'assessment']):
                            cbt_techniques += 1
                        elif any(keyword in primary_category for keyword in ['technique', 'protocol', 'assessment']):
                            cbt_techniques += 1
                        else:
                            cbt_content += 1
                    else:
                        cbt_content += 1
                        
                logger.info(f"CBT statistics calculated: {cbt_techniques} techniques, {cbt_content} content items")
        except Exception as e:
            logger.warning(f"Error calculating CBT statistics: {e}")
            cbt_techniques = 0
            cbt_content = 0
    else:
        cbt_techniques = 0
        cbt_content = 0
    
    return HealthResponse(
        status="healthy",
        psychologist_llm_loaded=psychologist_llm is not None,
        store_loaded=store is not None,
        device=DEVICE,
        gpu_memory=gpu_memory,
        opro_prompt_loaded=opro_prompt_loaded,
        interactions_count=interactions_count,
        cbt_available=cbt_available,
        cbt_techniques=cbt_techniques,
        cbt_content=cbt_content,
        cbt_smoke_test_passed=cbt_smoke_test_passed,
        enhanced_systems_available=enhanced_systems_available,
        enhanced_rag_loaded=enhanced_rag_loaded,
        intelligent_fusion_loaded=intelligent_fusion_loaded,
        resolved_paths=resolved_paths
    )

def analyze_emotion(text):
    """Analyze emotion from user input with fallback support"""
    if emotion_classifier is None:
        return "neutral"
    
    # Handle keyword-based fallback
    if emotion_classifier == "keyword_based":
        return analyze_emotion_keywords(text)
    
    try:
        result = emotion_classifier(text)
        return result[0]['label'] if result else "neutral"
    except Exception as e:
        logger.error(f"Emotion analysis error: {e}")
        return analyze_emotion_keywords(text)

def analyze_emotion_keywords(text):
    """Simple keyword-based emotion detection as fallback"""
    text_lower = text.lower()
    
    # Emotion keywords mapping
    emotion_keywords = {
        'joy': ['happy', 'joy', 'excited', 'great', 'wonderful', 'amazing', 'fantastic', 'good'],
        'sadness': ['sad', 'depressed', 'down', 'hopeless', 'worthless', 'empty', 'lonely', 'miserable'],
        'anger': ['angry', 'mad', 'furious', 'irritated', 'frustrated', 'annoyed', 'upset'],
        'fear': ['afraid', 'scared', 'fearful', 'anxious', 'worried', 'nervous', 'terrified', 'panic'],
        'surprise': ['surprised', 'shocked', 'amazed', 'astonished', 'unexpected'],
        'disgust': ['disgusted', 'sick', 'nauseated', 'repulsed'],
        'love': ['love', 'loved', 'caring', 'affectionate', 'warm', 'tender']
    }
    
    # Count emotion keywords
    emotion_scores = {}
    for emotion, keywords in emotion_keywords.items():
        score = sum(1 for keyword in keywords if keyword in text_lower)
        if score > 0:
            emotion_scores[emotion] = score
    
    # Return the emotion with highest score, or neutral if none found
    if emotion_scores:
        return max(emotion_scores, key=emotion_scores.get)
    else:
        return "neutral"

def get_conversation_history():
    """Get conversation history using unified conversation store"""
    try:
        return conversation_store.get_conversation_history()
    except Exception as e:
        logger.error(f"Error getting conversation history: {e}")
        return ""

def validate_user_input(question):
    """Validate and clean user input"""
    if not question or len(question.strip()) < 2:
        return False, "Could you please tell me more about what you're experiencing?"
    
    # Check for potential input errors or unclear text
    if len(question) < 5 and not any(word in question.lower() for word in ['sad', 'happy', 'angry', 'anxious', 'help']):
        return False, "I'd like to understand better. Could you share more about what's on your mind?"
    
    return True, question

def detect_crisis_keywords(text: str) -> bool:
    """Detect crisis keywords indicating potential suicide risk"""
    crisis_keywords = [
        'suicide', 'kill myself', 'end my life', 'want to die',
        'no reason to live', 'better off dead', 'hurt myself',
        'want to suicide', 'commit suicide', 'take my life',
        'don\'t want to live', 'can\'t go on', 'give up',
        'no point in living', 'everyone would be better off'
    ]
    return any(keyword in text.lower() for keyword in crisis_keywords)

def generate_crisis_response() -> str:
    """Generate appropriate crisis response with UK emergency resources"""
    return """I'm very concerned about what you're sharing. Your life has value, and there are people who want to help you.

**Immediate Help (UK):**
• Call 999 for emergency services (immediate danger)
• Call NHS 111 for urgent medical advice
• Contact Samaritans (24/7): 116 123 or visit https://www.samaritans.org
• Go to your nearest A&E (Accident & Emergency)

**You're Not Alone:**
Professional help is available and effective. Please reach out to one of these resources right now. You deserve support and care."""

def analyze_conversation_context(question: str, history: List[Message]) -> dict:
    """
    Analyze conversation context to determine appropriate response strategy
    """
    analysis = {
        "information_level": "basic",  # basic, detailed, comprehensive
        "response_strategy": "ask_clarification",  # ask_clarification, provide_advice, give_specific_help
        "already_discussed": [],
        "key_topics": [],
        "emotional_state": "neutral"
    }
    
    # Analyze information level
    question_lower = question.lower()
    
    # Check for detailed information patterns
    detailed_patterns = [
        r"because\s+\w+",  # "because my mind..."
        r"since\s+\w+",    # "since I started..."
        r"for\s+\d+",      # "for 3 months..."
        r"every\s+\w+",    # "every night..."
        r"always\s+\w+",   # "always thinking..."
        r"constantly\s+\w+", # "constantly anxious..."
        r"trouble\s+\w+",  # "trouble sleeping..."
        r"can't\s+\w+",    # "can't stop..."
        r"won't\s+\w+",    # "won't stop..."
        r"almost\s+\w+",   # "almost every night..."
    ]
    
    detailed_count = 0
    for pattern in detailed_patterns:
        if re.search(pattern, question_lower):
            detailed_count += 1
    
    # Check for comprehensive information patterns
    comprehensive_patterns = [
        r"and\s+\w+.*and\s+\w+",  # Multiple "and" clauses
        r"not\s+only.*but\s+also", # "not only... but also"
        r"both\s+\w+.*and\s+\w+", # "both... and..."
        r"either\s+\w+.*or\s+\w+", # "either... or..."
    ]
    
    comprehensive_count = 0
    for pattern in comprehensive_patterns:
        if re.search(pattern, question_lower):
            comprehensive_count += 1
    
    # Determine information level
    if comprehensive_count >= 1 or detailed_count >= 3:
        analysis["information_level"] = "comprehensive"
        analysis["response_strategy"] = "provide_advice"
    elif detailed_count >= 1:
        analysis["information_level"] = "detailed"
        analysis["response_strategy"] = "provide_advice"
    else:
        analysis["information_level"] = "basic"
        analysis["response_strategy"] = "ask_clarification"
    
    # Check for "how to" questions that should get direct advice
    how_to_patterns = [
        r'how to\s+\w+',  # "how to cope", "how to manage"
        r'how do i\s+\w+',  # "how do i deal", "how do i handle"
        r'what should i do\s+\w+',  # "what should i do about"
        r'what can i do\s+\w+',  # "what can i do about"
        r'how can i\s+\w+',  # "how can i help", "how can i manage"
        r'what steps\s+\w+',  # "what steps should i take"
        r'what techniques\s+\w+',  # "what techniques can i use"
        r'can you show me\s+\w+',  # "can you show me how"
        r'can you tell me\s+\w+',  # "can you tell me how"
    ]
    
    # Check for mental health conditions in "how to" questions
    mental_health_conditions = [
        'anxiety', 'anxious', 'depression', 'depressed', 'stress', 'stressed',
        'worry', 'worried', 'panic', 'fear', 'afraid', 'nervous',
        'thoughts', 'thinking', 'behavior', 'mood', 'overwhelmed',
        'sad', 'down', 'upset', 'tense', 'release', 'relieve', 'reduce',
        'cope', 'manage', 'deal', 'handle', 'overcome'
    ]
    
    # If it's a "how to" question with mental health content, provide direct advice
    is_how_to_question = any(re.search(pattern, question_lower) for pattern in how_to_patterns)
    has_mental_health_content = any(condition in question_lower for condition in mental_health_conditions)
    
    if is_how_to_question and has_mental_health_content:
        analysis["response_strategy"] = "provide_advice"
        # If it's a detailed "how to" question, give specific help
        if detailed_count >= 1:
            analysis["response_strategy"] = "give_specific_help"
    
    # Check for simple greetings or social interactions
    simple_greetings = [
        'hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening',
        'how are you', 'how\'s it going', 'nice to meet you', 'pleasure to meet you'
    ]
    
    # If it's a simple greeting without additional context, ask for clarification
    if any(greeting in question_lower for greeting in simple_greetings) and len(question_lower.split()) <= 3:
        analysis["response_strategy"] = "ask_clarification"
        analysis["information_level"] = "basic"
    
    # Analyze emotional state
    emotional_keywords = {
        "anxiety": ["anxious", "anxiety", "worried", "worry", "fear", "afraid", "scared"],
        "depression": ["sad", "depressed", "hopeless", "worthless", "empty", "numb"],
        "stress": ["stressed", "overwhelmed", "pressure", "tension", "burnout"],
        "anger": ["angry", "frustrated", "irritated", "mad", "upset"],
        "fear": ["terrified", "panic", "panic attack", "frightened"]
    }
    
    for emotion, keywords in emotional_keywords.items():
        if any(keyword in question_lower for keyword in keywords):
            analysis["emotional_state"] = emotion
            break
    
    # Extract key topics
    topic_keywords = {
        "work": ["work", "job", "career", "office", "boss", "colleague", "deadline"],
        "sleep": ["sleep", "insomnia", "tired", "exhausted", "rest", "bed"],
        "relationships": ["relationship", "partner", "family", "friend", "marriage"],
        "health": ["health", "medical", "doctor", "symptoms", "pain", "illness"],
        "finances": ["money", "financial", "bills", "debt", "expenses", "salary"]
    }
    
    for topic, keywords in topic_keywords.items():
        if any(keyword in question_lower for keyword in keywords):
            analysis["key_topics"].append(topic)
    
    # Check for already discussed topics in history
    if history:
        recent_history = history[-3:]  # Last 3 messages
        for msg in recent_history:
            if msg.role == "user":
                for topic in analysis["key_topics"]:
                    if topic in msg.content.lower():
                        analysis["already_discussed"].append(topic)
    
    # Adjust response strategy based on already discussed topics
    if analysis["already_discussed"] and analysis["information_level"] in ["detailed", "comprehensive"]:
        analysis["response_strategy"] = "give_specific_help"
    
    return analysis

def post_process_response(answer: str, question: str = "") -> str:
    if not answer:
        return answer
    
    # Remove template formatting and debug information
    template_patterns = [
        r'USER SITUATION:.*?(?=\n|$)',  # Remove USER SITUATION: lines
        r'MEDICAL CONTEXT:.*?(?=\n|$)',  # Remove MEDICAL CONTEXT: lines
        r'CONVERSATION HISTORY:.*?(?=\n|$)',  # Remove CONVERSATION HISTORY: lines
        r'CRISIS RESPONSE PROTOCOL:.*?(?=\n|$)',  # Remove protocol headers
        r'SAFETY NOTICE:.*?(?=\n|$)',  # Remove safety notice headers
        r'CRISIS\s*-\s*.*?(?=I understand|Let\'s|What|How|\.|$)',  # Remove crisis protocol text
        r'Assess immediate safety.*?accessible\.',  # Remove specific crisis protocol
        r'{context}|{history}|{question}',  # Remove unreplaced variables
        r'User:.*?(?=\n|$)',  # Remove debug user messages
        r'Assistant asked:.*?(?=\n|$)',  # Remove debug assistant questions
        r'Assistant:.*?(?=\n|$)',  # Remove debug assistant prefixes
        r'Context:.*?(?=\n|$)',  # Remove context debug info
        r'History:.*?(?=\n|$)',  # Remove history debug info
    ]
    
    for pattern in template_patterns:
        answer = re.sub(pattern, '', answer, flags=re.IGNORECASE | re.MULTILINE)
    
    # Remove labels like Empathy:, Citation:, Follow-up question: etc
    answer = re.sub(r'(Empathy:|Citation:|Follow-up question:|User:|Assistant:|Context:|History:)', '', answer, flags=re.IGNORECASE)
    
    # Remove any remaining format tags that might have leaked
    answer = re.sub(r'<\|assistant\|\|?\s*', '', answer, flags=re.IGNORECASE)
    answer = re.sub(r'<\|.*?\|>', '', answer, flags=re.IGNORECASE)
    
    # Remove AI self-questioning patterns (AI asking itself questions)
    self_question_patterns = [
        r'\s*How do I cope with\s+[^?]*\?',  # "How do I cope with anxiety?"
        r'\s*What should I do about\s+[^?]*\?',  # "What should I do about this?"
        r'\s*How can I help with\s+[^?]*\?',  # "How can I help with this?"
        r'\s*What steps can I take for\s+[^?]*\?',  # "What steps can I take for anxiety?"
        r'\s*How do I manage\s+[^?]*\?',  # "How do I manage anxiety?"
        r'\s*What techniques can I use for\s+[^?]*\?',  # "What techniques can I use for anxiety?"
    ]
    
    for pattern in self_question_patterns:
        answer = re.sub(pattern, '', answer, flags=re.IGNORECASE)
    
    # Remove internal instruction leaks (CRITICAL FIX) - More targeted approach
    internal_instruction_patterns = [
        r'INSTRUCTIONS?:.*?(?=\n|$)',  # "INSTRUCTIONS: - Start with empathy..."
        r'INSTRUCTATIONS?:.*?(?=\n|$)',  # Typo version
        r'RESPONSE TEMPLATE:.*?(?=\n|$)',  # "RESPONSE TEMPLATE:"
        r'GUIDELINES:.*?(?=\n|$)',  # "GUIDELINES:"
        r'Start with empathy.*?(?=\n|$)',  # "Start with empathy (1 sentence)"
        r'Cite ICD-11.*?(?=\n|$)',  # "Cite ICD-11 context if relevant"
        r'Ask 1 gentle.*?(?=\n|$)',  # "Ask 1 gentle follow-up question"
        r'Keep response.*?(?=\n|$)',  # "Keep response to 2-4 sentences"
        r'Avoid generic.*?(?=\n|$)',  # "Avoid generic lifestyle advice"
        r'Response Structure.*?(?=\n|$)',  # "Response Structure Guidelines:"
        r'Formatting Guidelines.*?(?=\n|$)',  # "Formatting Guidelines:"
        r'Content Depth.*?(?=\n|$)',  # "Content Depth Guidelines:"
        r'Professional Resource.*?(?=\n|$)',  # "Professional Resource Guidelines:"
        r'Balance Guidelines.*?(?=\n|$)',  # "Balance Guidelines:"
        r'Question Type.*?(?=\n|$)',  # "Question Type Adaptations:"
        r'Quality Standards.*?(?=\n|$)',  # "Quality Standards:"
        r'Personalization Elements.*?(?=\n|$)',  # "Personalization Elements:"
        r'Response Template.*?(?=\n|$)',  # "Response Template Structure:"
        r'1\. Empathy.*?(?=\n|$)',  # "1. Empathy Opening"
        r'2\. Problem.*?(?=\n|$)',  # "2. Problem Acknowledgment"
        r'3\. Structured.*?(?=\n|$)',  # "3. Structured Advice"
        r'4\. Professional.*?(?=\n|$)',  # "4. Professional Resources"
        r'5\. Encouragement.*?(?=\n|$)',  # "5. Encouragement Closing"
        r'Crisis Questions.*?(?=\n|$)',  # "Crisis Questions:"
        r'How-to Questions.*?(?=\n|$)',  # "How-to Questions:"
        r'Symptom Questions.*?(?=\n|$)',  # "Symptom Questions:"
        r'General Support.*?(?=\n|$)',  # "General Support:"
        r'Each piece of advice.*?(?=\n|$)',  # "Each piece of advice should be"
        r'Include the reasoning.*?(?=\n|$)',  # "Include the reasoning behind"
        r'Provide multiple options.*?(?=\n|$)',  # "Provide multiple options"
        r'Ensure advice is.*?(?=\n|$)',  # "Ensure advice is evidence-based"
        r'Include appropriate.*?(?=\n|$)',  # "Include appropriate disclaimers"
        r'Reference user.*?(?=\n|$)',  # "Reference user's specific situation"
        r'Adapt advice based.*?(?=\n|$)',  # "Adapt advice based on emotional state"
        r'Consider cultural.*?(?=\n|$)',  # "Consider cultural and contextual"
        r'Provide age-appropriate.*?(?=\n|$)',  # "Provide age-appropriate"
        # Additional patterns for template structure leaks
        r'Empathy Opening.*?(?=\n|$)',  # "Empathy Opening (1-2 sentences)"
        r'Problem Acknowledgment.*?(?=\n|$)',  # "Problem Acknowledgment (1 sentence)"
        r'Structured Advice.*?(?=\n|$)',  # "Structured Advice (numbered steps)"
        r'Professional Resources.*?(?=\n|$)',  # "Professional Resources (when relevant)"
        r'Encouragement Closing.*?(?=\n|$)',  # "Encouragement Closing (1-2 sentences)"
        r'Acknowledge the user.*?(?=\n|$)',  # "Acknowledge the user's feelings"
        r'Let the user know.*?(?=\n|$)',  # "Let the user know that they understand"
        r'Express empathy.*?(?=\n|$)',  # "Express empathy and understanding"
        r'Understanding how.*?(?=\n|$)',  # "Understanding how sad they are feeling"
        # Remove specific problematic patterns from the output
        r'EMPATHIC Connection:.*?(?=\n|$)',  # Remove "EMPATHIC Connection:" lines
        r'EMPHATIC Connection:.*?(?=\n|$)',  # Remove "EMPHATIC Connection:" lines  
        r'EVIDENCE-BASED support:.*?(?=\n|$)',  # Remove EVIDENCE-BASED support lines
        r'SAFETY Awareness:.*?(?=\n|$)',  # Remove "SAFETY Awareness:" lines
        r'PROFESSIONAL Boundaries:.*?(?=\n|$)',  # Remove "PROFESSIONAL Boundaries:" lines
        r'Please provide your feedback.*?(?=\n|$)',  # Remove feedback requests
    ]
    
    for pattern in internal_instruction_patterns:
        answer = re.sub(pattern, '', answer, flags=re.IGNORECASE | re.MULTILINE)
    
    # Remove any non-ASCII characters that might have crept in
    answer = re.sub(r'[^\x00-\x7F]+', '', answer)
    
    # Remove extra newlines and spaces but preserve paragraph structure
    answer = re.sub(r'\n+', ' ', answer)
    answer = re.sub(r'\s+', ' ', answer).strip()
    
    # Remove repetitive phrases that appear in the conversation
    repetitive_patterns = [
        r'Please consider taking breaks throughout the day\. It doesn\'t have to be an hour.*?',
        r'It doesn\'t have to be a big deal.*?',
        r'just a little bit of extra time will go a long way.*?',
        r'Remember to take breaks and rest when you need to.*?',
        r'It\'s important to prioritize your safety and well-being.*?',
    ]
    
    for pattern in repetitive_patterns:
        answer = re.sub(pattern, '', answer, flags=re.IGNORECASE | re.DOTALL)
    
    # Enhanced response structuring and length control
    sentences = re.split(r'(?<=[.!?]) +', answer)
    
    # Check if this is a "how to" question or emotional expression
    is_how_to_question = any(phrase in question.lower() for phrase in 
                            ['how to', 'how do', 'what steps', 'what should', 'can you show', 'improve', 'feel better'])
    
    # Check if this is an emotional expression
    is_emotional_expression = any(phrase in question.lower() for phrase in 
                                 ['i feel', 'feeling', 'i am feeling', 'i\'m feeling', 'sad', 'anxious', 'depressed'])
    
    # Check for incomplete sentences (sentences that don't end with punctuation)
    incomplete_sentences = []
    complete_sentences = []
    
    for sentence in sentences:
        if sentence.strip() and sentence.strip()[-1] in '.!?':
            complete_sentences.append(sentence)
        elif sentence.strip():
            incomplete_sentences.append(sentence)
    
    # If we have incomplete sentences, try to complete them or remove them
    if incomplete_sentences:
        # For incomplete sentences, try to complete them with common endings
        for incomplete in incomplete_sentences:
            if 'could involve' in incomplete.lower():
                # Complete common patterns
                if 'relaxation' in incomplete.lower():
                    complete_sentences.append(incomplete + ' deep breathing, progressive muscle relaxation, or guided imagery.')
                elif 'technique' in incomplete.lower():
                    complete_sentences.append(incomplete + ' various relaxation methods.')
                else:
                    complete_sentences.append(incomplete + ' different approaches.')
            elif 'this could' in incomplete.lower():
                complete_sentences.append(incomplete + ' help you feel better.')
            else:
                # For other incomplete sentences, try to complete them
                complete_sentences.append(incomplete + '.')
    
    # Reconstruct answer with complete sentences
    answer = ' '.join(complete_sentences)
    
    # Enhanced length control with better structure preservation
    if is_emotional_expression or is_how_to_question:
        # For emotional expressions and how-to questions, allow structured responses
        if len(complete_sentences) > 12:
            # Keep empathy + 3-4 structured points
            empathy_sentences = []
            technique_sentences = []
            
            # Separate empathy from techniques
            for sentence in complete_sentences:
                if any(word in sentence.lower() for word in ['understand', 'hear', 'sorry', 'feel', 'know']):
                    empathy_sentences.append(sentence)
                else:
                    technique_sentences.append(sentence)
            
            # Keep first 2 empathy sentences and first 3-4 technique sentences
            selected_sentences = empathy_sentences[:2] + technique_sentences[:4]
            answer = ' '.join(selected_sentences)
        elif len(complete_sentences) > 8:
            # For medium length, keep first 6-8 sentences
            answer = ' '.join(complete_sentences[:8])
    else:
        # For regular questions, keep it concise (3-4 sentences max)
        if len(complete_sentences) > 6:
            important_keywords = ['understand', 'feel', 'support', 'help', 'care', 'important', 'okay', 'mindfulness', 'breathing', 'technique']
            important_sentences = complete_sentences[:2]  # Always keep first 2 sentences
            
            for sentence in complete_sentences[2:5]:  # Check next 3 sentences
                if any(keyword in sentence.lower() for keyword in important_keywords):
                    important_sentences.append(sentence)
            
            answer = ' '.join(important_sentences)
        elif len(complete_sentences) > 8:
            # For regular questions with many sentences, limit more strictly
            answer = ' '.join(complete_sentences[:6])
    
    # Remove generic customer service phrases
    generic_phrases = [
        "is there anything else i can help you with",
        "feel free to reach out",
        "don't hesitate to contact",
        "please let me know if you need anything else"
    ]
    for phrase in generic_phrases:
        if phrase in answer.lower():
            answer = re.sub(re.escape(phrase), '', answer, flags=re.IGNORECASE)
            answer = re.sub(r'\s+', ' ', answer).strip()
    
    # Remove conversational flow issues (AI asking questions to itself)
    conversational_issues = [
        r'\s*Would you like to know\s+[^?]*\?',  # "Would you like to know more about..."
        r'\s*Do you want to learn\s+[^?]*\?',  # "Do you want to learn about..."
        r'\s*Should I explain\s+[^?]*\?',  # "Should I explain..."
        r'\s*Can I help you with\s+[^?]*\?',  # "Can I help you with..."
        r'\s*Would you like me to\s+[^?]*\?',  # "Would you like me to..."
    ]
    
    for pattern in conversational_issues:
        answer = re.sub(pattern, '', answer, flags=re.IGNORECASE)
    
    # Ensure answer ends with proper punctuation
    if answer and not answer[-1] in '.!?':
        answer += '.'
    
    # Final optimization: ensure structured format for numbered lists
    answer = optimize_response_structure(answer, question)
    
    return answer.strip()

def optimize_response_structure(answer: str, question: str) -> str:
    """Optimize response structure for better readability"""
    
    # Check if this is an emotional expression or how-to question
    is_emotional_expression = any(phrase in question.lower() for phrase in 
                                 ['i feel', 'feeling', 'i am feeling', 'i\'m feeling', 'sad', 'anxious', 'depressed'])
    is_how_to_question = any(phrase in question.lower() for phrase in 
                            ['how to', 'how do', 'what steps', 'what should', 'can you show', 'improve', 'feel better'])
    
    # If it's an emotional expression or how-to question, ensure proper structure
    if is_emotional_expression or is_how_to_question:
        # Look for numbered patterns and ensure they're properly formatted
        sentences = re.split(r'(?<=[.!?]) +', answer)
        
        # Find empathy sentences and technique sentences
        empathy_sentences = []
        technique_sentences = []
        
        for sentence in sentences:
            # Skip sentences that are just "Here are some suggestions" or similar
            if any(phrase in sentence.lower() for phrase in ['here are some suggestions', 'here are some techniques', 'here are some ways']):
                continue
            elif any(word in sentence.lower() for word in ['understand', 'hear', 'sorry', 'feel', 'know']):
                empathy_sentences.append(sentence)
            else:
                technique_sentences.append(sentence)
        
        # Reconstruct with better structure
        if empathy_sentences and technique_sentences:
            # Keep first empathy sentence and first 3 technique sentences
            selected_empathy = empathy_sentences[:1]
            selected_techniques = technique_sentences[:3]
            
            # Clean up technique sentences - remove any existing numbering and extract the core content
            formatted_techniques = []
            for i, technique in enumerate(selected_techniques, 1):
                # Remove any existing numbering and clean up
                technique = re.sub(r'^\d+[\.\)]\s*', '', technique.strip())
                technique = re.sub(r'^\d+\)\s*', '', technique.strip())
                
                # Extract the main content (before any colons or dashes)
                if ':' in technique:
                    technique = technique.split(':', 1)[1].strip()
                elif ' - ' in technique:
                    technique = technique.split(' - ', 1)[1].strip()
                
                # Ensure it's not empty and add proper numbering
                if technique and len(technique) > 5:
                    formatted_techniques.append(f"{i}) {technique}")
            
            # Combine with proper spacing
            if formatted_techniques:
                result = ' '.join(selected_empathy) + ' Here are some suggestions: ' + ' '.join(formatted_techniques)
                
                # Add a follow-up question if not present
                if not any(word in result.lower() for word in ['what', 'how', 'can you', 'tell me']):
                    result += ' What has been contributing to these feelings?'
                
                return result
    
    return answer

def summarize_session_transcript(transcript: str) -> str:
    """Heuristic short summary for session memory. Keep it model-agnostic."""
    if not transcript:
        return ""
    # Keep first 600 chars, focus on problem/goal signals
    text = transcript[-1200:]
    # Simple compression: remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text[:600]

@app.post("/api/empathetic_professional", response_model=RAGResponse)
async def empathetic_professional_endpoint(request_data: RAGRequest):
    start_time = time.time()
    
    try:
        if store is None or psychologist_llm is None:
            raise HTTPException(status_code=500, detail="RAG system not initialized")

        # Step 1: Crisis detection (HIGHEST PRIORITY)
        if detect_crisis_keywords(request_data.question):
            logger.warning(f"CRISIS DETECTED in user input: {request_data.question}")
            return RAGResponse(
                answer=generate_crisis_response(),
                question=request_data.question,
                tone="empathetic_professional",
                status="crisis_detected",
                context_used="crisis_response",
                prompt_source="crisis",
                confidence=1.0,
                fusion_strategy="crisis_intervention",
                source_breakdown={"crisis": 1.0},
                follow_up_suggestions=["Please call 999 immediately"],
                safety_notes=["User expressed suicidal thoughts - immediate intervention required"]
            )

        # Step 2: Input validation
        is_valid, processed_question = validate_user_input(request_data.question)
        if not is_valid:
            return RAGResponse(
                answer=processed_question,
                question=request_data.question,
                tone="empathetic_professional",
                status="clarification_needed",
                context_used=""
            )

        logger.info(f"Processing request: {processed_question}")
        
        # Display user input for debugging
        if SHOW_PROMPT_DEBUG:
            print("\n" + "="*80)
            print("USER INPUT ANALYSIS:")
            print("="*80)
            print(f"Original question: {request_data.question}")
            print(f"Processed question: {processed_question}")
            print("="*80)
        
        # Step 2: Emotion analysis
        emotion = analyze_emotion(processed_question)
        logger.info(f"Detected emotion: {emotion}")
        
        # Display emotion analysis for debugging
        if SHOW_PROMPT_DEBUG:
            print(f"DETECTED EMOTION: {emotion}")
            print("="*80)
        
        # Step 3: Get conversation history and session summary
        history = get_conversation_history()
        session_id = request_data.session_id or str(int(time.time() * 1000))
        session_summary = load_session_summary(session_id)
        
        # Display conversation history for debugging
        if SHOW_PROMPT_DEBUG:
            print("CONVERSATION HISTORY:")
            print("="*80)
            if history:
                print(history)
            else:
                print("No conversation history")
            print("="*80)
        
        # Step 3.5: Analyze conversation context for response strategy
        conversation_analysis = analyze_conversation_context(processed_question, request_data.history)
        response_strategy = conversation_analysis["response_strategy"]
        
        logger.info(f"Conversation analysis: {conversation_analysis}")
        
        # Display conversation analysis for debugging
        if SHOW_PROMPT_DEBUG:
            print("CONVERSATION ANALYSIS:")
            print("="*80)
            print(f"Information level: {conversation_analysis['information_level']}")
            print(f"Response strategy: {conversation_analysis['response_strategy']}")
            print(f"Emotional state: {conversation_analysis['emotional_state']}")
            print(f"Key topics: {conversation_analysis['key_topics']}")
            print(f"Already discussed: {conversation_analysis['already_discussed']}")
            print("="*80)
        
        # Step 4: Enhanced RAG search with intelligent fusion (CORRECTED LOGIC)
        context = None
        enhanced_metadata = {}
        docs = []  # Initialize docs for later use in debug printing

        current_enhanced_rag_retriever = globals().get('enhanced_rag_retriever')
        current_intelligent_fusion_system = globals().get('intelligent_fusion_system')
        
        if current_enhanced_rag_retriever and current_intelligent_fusion_system:
            try:
                if SHOW_ENHANCED_DEBUG:
                    print("\n" + "="*80)
                    print("ATTEMPTING ENHANCED RAG AND FUSION SYSTEMS")
                    print("="*80)
                
                # Debug: check enhanced_rag_retriever before calling retrieve
                logger.info(f"Enhanced RAG retriever type: {type(current_enhanced_rag_retriever)}")
                logger.info(f"Enhanced RAG retriever config type: {type(current_enhanced_rag_retriever.config)}")
                logger.info(f"Enhanced RAG retriever config: {current_enhanced_rag_retriever.config}")
                
                # Create retrieval context
                retrieval_context = RetrievalContext(
                    conversation_history=request_data.history,
                    emotional_state=emotion,
                    urgency_level=1
                )
                
                # Create fusion context
                fusion_context = FusionContext(
                    urgency_level=1,
                    user_history=request_data.history,
                    session_stage="ongoing"
                )
                
                # Perform intelligent fusion which internally handles retrieval
                fused_response = current_intelligent_fusion_system.fuse_response(
                    query=processed_question,
                    context=fusion_context
                )
                
                # If successful, populate context and metadata
                context = fused_response.primary_content + "\n\n" + fused_response.supporting_content
                enhanced_metadata = {
                    "confidence": fused_response.confidence,
                    "fusion_strategy": fused_response.fusion_strategy.value,
                    "source_breakdown": fused_response.source_breakdown,
                    "follow_up_suggestions": fused_response.follow_up_suggestions,
                    "safety_notes": fused_response.safety_notes
                }
                
                if SHOW_ENHANCED_DEBUG:
                    print(f"Enhanced retrieval successful")
                    print(f"Fusion strategy: {fused_response.fusion_strategy.value}")
                    print(f"Confidence: {fused_response.confidence:.2f}")
                    print(f"Source breakdown: {fused_response.source_breakdown}")
                    print("="*80)

            except Exception as e:
                logger.error(f"Enhanced systems failed: {e}. Falling back to basic RAG.", exc_info=True)
                # Ensure context remains None to trigger the fallback logic
                context = None
        
        # If context is still None, it means enhanced systems were unavailable or failed. Use fallback.
        if context is None:
            if SHOW_ENHANCED_DEBUG:
                print("\n" + "="*80)
                print("USING FALLBACK RAG SYSTEM")
                print("="*80)
            
            retriever = store.as_retriever(search_kwargs={"k": 5})
            docs = retriever.invoke(processed_question)
            
            # Filter out irrelevant medical conditions for general mental health queries
            relevant_docs = []
            irrelevant_keywords = [
                'parkinson', 'tremor', 'dementia', 'alzheimer', 'dystonia', 
                'myoclonus', 'chorea', 'movement disorder', 'neurological'
            ]
            
            general_mh_query = any(term in processed_question.lower() for term in [
                'mental problem', 'mental health', 'feeling', 'sad', 'anxious', 
                'depressed', 'stressed', 'worried'
            ])
            
            for doc in docs:
                # For general mental health queries, filter out neurological conditions
                if general_mh_query:
                    if not any(keyword in doc.page_content.lower() for keyword in irrelevant_keywords):
                        relevant_docs.append(doc)
                else:
                    relevant_docs.append(doc)
            
            # Use filtered docs, fallback to original if none remain
            final_docs = relevant_docs if relevant_docs else docs[:2]  # Limit to 2 if fallback
            context = "\n\n".join([doc.page_content for doc in final_docs])
            
            # Basic metadata for fallback
            enhanced_metadata = {
                "confidence": 0.7,
                "fusion_strategy": "fallback_basic_rag",
                "source_breakdown": {"icd11": 1.0},
                "follow_up_suggestions": [],
                "safety_notes": []
            }
        
        # Display retrieved context for debugging
        if SHOW_PROMPT_DEBUG:
            print("\n" + "="*80)
            print("RETRIEVED CONTEXT FROM RAG:")
            print("="*80)
            print(f"Found {len(docs)} documents:")
            for i, doc in enumerate(docs, 1):
                print(f"\n--- Document {i} ---")
                print(doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content)
            print("="*80)
        
        # Limit context length for faster processing (tighter for latency)
        if len(context) > 800:
            context = context[:800] + "..."
        
        logger.info(f"Retrieved context length: {len(context)} characters")

        # Step 4.5: Intent-aware strategy adjustment (work-stress specialization)
        try:
            work_stress_keywords = [
                'job', 'deadline', 'workload', 'boss', 'manager', 'office', 'overtime', 'project',
                'deliverables', 'sprint', 'jira', 'ticket', 'shift'
            ]
            if any(k in processed_question.lower() for k in work_stress_keywords):
                response_strategy = 'work_stress_support'
        except Exception:
            pass

        # Step 5: Check if this is a definitional question and handle accordingly
        is_definitional = is_definitional_question(processed_question)
        
        if is_definitional:
            logger.info(f"Definitional question detected: {processed_question}")
            # Use definitional prompt for medical information
            prompt = load_definitional_prompt()
            if not prompt:
                # Fallback to dynamic prompt if definitional prompt not available
                prompt = get_dynamic_prompt(request_data.type)
            
            # Get specialized medical context for definitional questions
            medical_context = get_medical_context_for_definition(processed_question, store)
            if medical_context:
                context = medical_context
                logger.info(f"Using specialized medical context for definitional question")
        else:
            # Use regular dynamic prompt for non-definitional questions
            prompt = get_dynamic_prompt(request_data.type)
        
        # Display prompt source for debugging
        if SHOW_PROMPT_DEBUG:
            print("PROMPT SOURCE:")
            print("="*80)
            print(f"Selected tone: {request_data.type}")
            print(f"Is definitional question: {is_definitional}")
            
            if is_definitional:
                print("Using definitional prompt for medical information")
            elif request_data.type == "empathetic_professional":
                opro_exists = os.path.exists(OPRO_PROMPT_PATH)
                print(f"OPRO file exists: {opro_exists}")
                if opro_exists:
                    print(f"Using OPRO optimized prompt from: {OPRO_PROMPT_PATH}")
                else:
                    print("Using fallback prompt (OPRO not available)")
            else:
                print(f"Using {request_data.type} tone prompt")
            print("="*80)
        
        # Step 1: Always replace variables in prompt first
        try:
            # Limit context and history to prevent template overflow (tighter for latency)
            limited_context = context[:800] + "..." if len(context) > 800 else context
            combined_history = (session_summary + "\n\n" + history).strip() if session_summary else history
            limited_history = combined_history[:300] + "..." if len(combined_history) > 300 else combined_history
            
            formatted_prompt = prompt.format(
                context=limited_context,
                question=processed_question,
                history=limited_history,
                response_strategy=response_strategy,
                tone=request_data.type
            )
            logger.info("Variables successfully replaced in prompt")
        except KeyError as e:
            logger.warning(f"Variable replacement failed: {e}, using original prompt")
            formatted_prompt = prompt
        
        # Step 2: Handle different ChatML formats
        if "<|im_start|>" in formatted_prompt:
            # Already in Qwen ChatML format
            logger.info("Using prompt in Qwen ChatML format")
        elif "<|system|>" in formatted_prompt:
            # Already in standard ChatML format
            logger.info("Using prompt in standard ChatML format")
        else:
            # Wrap prompt in ChatML format
            formatted_prompt = f"""
<|system|>
{formatted_prompt}
<|user|>
{processed_question}
<|assistant|>
"""
            logger.info("Prompt wrapped in ChatML format")
        
        logger.info(f"Formatted prompt length: {len(formatted_prompt)} characters")
        
        # Step 6.5: Display the formatted prompt for debugging
        if SHOW_PROMPT_DEBUG:
            print("\n" + "="*80)
            print("PROMPT TO BE SENT TO LLM:")
            print("="*80)
            print(formatted_prompt)
            print("="*80)
            print("Starting LLM inference...\n")

        # Step 7: LLM generation
        try:
            logger.info("Starting LLM inference...")
            # Reset LLM generation parameters - optimize for detailed, helpful responses
            pipe = psychologist_llm.pipeline
            # Reduce generation length to lower latency
            default_max_new_tokens = os.getenv("LLM_MAX_NEW_TOKENS", "64")
            pipe.model.config.max_new_tokens = int(default_max_new_tokens)
            pipe.model.config.temperature = float(os.getenv("LLM_TEMPERATURE", "0.4"))
            pipe.model.config.top_p = 0.9

            # Add timeout to avoid hanging requests returning 500 (tighter default)
            llm_timeout_seconds = float(os.getenv("LLM_TIMEOUT_SECONDS", "22"))

            # Optional hard wall for generation time inside HF generate
            try:
                max_time_sec = float(os.getenv("LLM_MAX_TIME_SECONDS", str(max(1.0, llm_timeout_seconds - 2))))
                if hasattr(pipe.model, "generation_config"):
                    pipe.model.generation_config.max_time = max_time_sec
            except Exception:
                pass

            async def _invoke_llm() -> str:
                loop = asyncio.get_event_loop()
                result_inner = await loop.run_in_executor(None, psychologist_llm.invoke, formatted_prompt)
                return result_inner if isinstance(result_inner, str) else str(result_inner)

            try:
                answer = await asyncio.wait_for(_invoke_llm(), timeout=llm_timeout_seconds)
            except asyncio.TimeoutError:
                logger.warning(f"LLM inference timed out after {llm_timeout_seconds}s; raising 504")
                raise HTTPException(status_code=504, detail=f"LLM inference timed out after {llm_timeout_seconds} seconds")
            
            # Display raw LLM output for debugging
            if SHOW_PROMPT_DEBUG:
                print("RAW LLM OUTPUT:")
                print("-" * 60)
                print(answer)
                print("-" * 60)
            
            # Extract only content after <|assistant|> to avoid instruction leakage
            if "<|assistant|>" in answer:
                answer = answer.split("<|assistant|>")[-1].strip()
            
            # Remove any remaining ChatML tags and pipe characters
            answer = re.sub(r'<\|.*?\|>', '', answer, flags=re.DOTALL)
            answer = re.sub(r'\|+', '', answer)  # Remove multiple pipe characters
            
            # Clean up any other ChatML or template remnants
            chatml_patterns = [
                r'<\|.*?\|>',  # Remove any ChatML tags
                r'<\|im_start\|>.*?<\|im_end\|>',  # Remove Qwen ChatML blocks
                r'<\|system\|>.*?(?=<\||$)',  # Remove system messages
                r'<\|user\|>.*?(?=<\||$)',  # Remove user messages
            ]
            
            for pattern in chatml_patterns:
                answer = re.sub(pattern, '', answer, flags=re.DOTALL | re.IGNORECASE)
            
            # Enhanced filtering for problematic content and debug information
            problematic_patterns = [
                r'I understand you\'re experiencing ".*?" and I want to acknowledge',  # Template repetition
                r'Be mindful of the user\'s emotions.*?support\.',  # Instruction leakage
                r'Example Input \d+:.*?Response:',  # Example content
                r'RESPONSE GUIDELINES:.*?RESPONSE:',  # Template content
                r'USER MESSAGE:.*?RESPONSE:',  # Template variables
                r'emergency involved, and I would appreciate your assistance',  # Specific weird content
                r'User:.*?(?=\n|$)',  # Remove debug user messages
                r'Assistant asked:.*?(?=\n|$)',  # Remove debug assistant questions
                r'Assistant:.*?(?=\n|$)',  # Remove debug assistant prefixes
                r'Context:.*?(?=\n|$)',  # Remove context debug info
                r'History:.*?(?=\n|$)',  # Remove history debug info
                r'<\|.*?\|>.*?(?=\n|$)',  # Remove any remaining ChatML tags with content
                # OPRO specific patterns
                r'Empathetic Response:.*?(?=\n|$)',  # Remove OPRO response prefixes
                r'Tone:.*?validating.*?(?=\n|$)',  # Remove tone validation messages
                r'Tone Adaptation.*?(?=\n|$)',  # Remove tone adaptation messages
                r'Response Strategy.*?(?=\n|$)',  # Remove response strategy messages
                r'Guidelines:.*?(?=\n|$)',  # Remove guidelines messages
                r'Instructions:.*?(?=\n|$)',  # Remove instructions messages
                r'Template.*?(?=\n|$)',  # Remove template messages
                r'Format.*?(?=\n|$)',  # Remove format messages
                r'Structure.*?(?=\n|$)',  # Remove structure messages
                r'Content.*?(?=\n|$)',  # Remove content messages
                r'Balance.*?(?=\n|$)',  # Remove balance messages
                r'Quality.*?(?=\n|$)',  # Remove quality messages
                r'Personalization.*?(?=\n|$)',  # Remove personalization messages
                r'Question Type.*?(?=\n|$)',  # Remove question type messages
                r'Crisis Questions.*?(?=\n|$)',  # Remove crisis questions messages
                r'How-to Questions.*?(?=\n|$)',  # Remove how-to questions messages
                r'Symptom Questions.*?(?=\n|$)',  # Remove symptom questions messages
                r'General Support.*?(?=\n|$)',  # Remove general support messages
                r'Response Template.*?(?=\n|$)',  # Remove response template messages
                r'Empathy Opening.*?(?=\n|$)',  # Remove empathy opening messages
                r'Problem Acknowledgment.*?(?=\n|$)',  # Remove problem acknowledgment messages
                r'Structured Advice.*?(?=\n|$)',  # Remove structured advice messages
                r'Professional Resources.*?(?=\n|$)',  # Remove professional resources messages
                r'Encouragement Closing.*?(?=\n|$)',  # Remove encouragement closing messages
                r'Each piece of advice.*?(?=\n|$)',  # Remove advice messages
                r'Reference user.*?(?=\n|$)',  # Remove reference messages
                r'Adapt advice.*?(?=\n|$)',  # Remove adapt messages
                r'Consider cultural.*?(?=\n|$)',  # Remove cultural messages
                r'Provide age-appropriate.*?(?=\n|$)',  # Remove age-appropriate messages
                r'Response:.*?(?=\n|$)',  # Remove response messages
            ]
            
            for pattern in problematic_patterns:
                answer = re.sub(pattern, '', answer, flags=re.DOTALL | re.IGNORECASE)
            
            # Remove any content that looks like template variables, instructions, or irrelevant medical terms
            problematic_phrases = [
                "be mindful of", "example input", "response guidelines", 
                "user message:", "context:", "conversation history:",
                "parkinson", "tremor", "dementia", "neurological", "rigidity",
                "sudden onset", "early disability", "cognitive function related to mathematics",
                "user:", "assistant:", "assistant asked:", "context:", "history:",
                # OPRO specific problematic phrases
                "empathetic response:", "tone:", "validating", "tone adaptation",
                "response strategy", "guidelines:", "instructions:", "template",
                "format", "structure", "content", "balance", "quality",
                "personalization", "question type", "crisis questions",
                "how-to questions", "symptom questions", "general support",
                "response template", "empathy opening", "problem acknowledgment",
                "structured advice", "professional resources", "encouragement closing",
                "each piece of advice", "reference user", "adapt advice",
                "consider cultural", "provide age-appropriate", "response:"
            ]
            
            # Only use fallback if the response is completely problematic
            problematic_count = sum(1 for phrase in problematic_phrases if phrase in answer.lower())
            if problematic_count > 3:  # Only if more than 3 problematic phrases found
                logger.warning("Detected too many problematic phrases in response, using fallback")
                answer = create_fallback_response(processed_question, context, "empathetic_professional")
            
            # Post-process: automatically trim overly long answers to ensure conciseness
            answer = post_process_response(answer, processed_question)
            
            # Additional cleanup for OPRO-specific issues
            # Remove any lines that start with common OPRO template patterns
            lines = answer.split('\n')
            cleaned_lines = []
            for line in lines:
                line = line.strip()
                if line and not any(line.lower().startswith(prefix) for prefix in [
                    'empathetic response:', 'tone:', 'validating', 'tone adaptation',
                    'response strategy', 'guidelines:', 'instructions:', 'template',
                    'format', 'structure', 'content', 'balance', 'quality',
                    'personalization', 'question type', 'crisis questions',
                    'how-to questions', 'symptom questions', 'general support',
                    'response template', 'empathy opening', 'problem acknowledgment',
                    'structured advice', 'professional resources', 'encouragement closing',
                    'each piece of advice', 'reference user', 'adapt advice',
                    'consider cultural', 'provide age-appropriate', 'response:'
                ]):
                    cleaned_lines.append(line)
            
            answer = '\n'.join(cleaned_lines).strip()
            
            # Final cleanup: remove any remaining template artifacts
            answer = re.sub(r'^.*?(?=I understand|I hear|I can see|It sounds like|I\'m sorry)', '', answer, flags=re.DOTALL | re.IGNORECASE)
            answer = re.sub(r'^.*?(?=Here are|Let me|I would|You might|Consider)', '', answer, flags=re.DOTALL | re.IGNORECASE)
            
            # Enhance response with CBT techniques if available
            if cbt_integration is not None:
                try:
                    # Check if CBT enhancement should be applied
                    should_enhance = cbt_integration.should_include_cbt(processed_question)
                    logger.info(f"CBT relevance check for '{processed_question}': {should_enhance}")
                    
                    if should_enhance:
                        enhanced_answer = cbt_integration.enhance_response_with_cbt(
                            user_query=processed_question,
                            context=context,
                            base_response=answer
                        )
                        if enhanced_answer != answer:
                            answer = enhanced_answer
                            logger.info("Response enhanced with CBT techniques")
                            
                            # Debug info for CBT enhancement
                            if SHOW_PROMPT_DEBUG:
                                print("CBT ENHANCEMENT APPLIED:")
                                print("-" * 60)
                                original_part = answer.split('\n\n')[0]
                                print(f"Original length: {len(original_part)} chars")
                                print(f"Enhanced length: {len(answer)} chars")
                                print("-" * 60)
                        else:
                            logger.info("CBT enhancement returned same response")
                    else:
                        logger.info("CBT enhancement not relevant for this query")
                        
                except Exception as e:
                    logger.warning(f"CBT enhancement failed: {e}")
                    import traceback
                    logger.debug(f"CBT enhancement error details: {traceback.format_exc()}")
            
            # Display processed answer for debugging
            if SHOW_PROMPT_DEBUG:
                print("PROCESSED ANSWER (WITH CBT):")
                print("-" * 60)
                print(answer)
                print("-" * 60)
                print()
            
            logger.info(f"Generated answer length: {len(answer)} characters")
            # Check response quality - more lenient for emotional expressions and how-to questions
            is_emotional_expression = any(phrase in processed_question.lower() for phrase in 
                                         ['i feel', 'feeling', 'i am feeling', 'i\'m feeling', 'sad', 'anxious', 'depressed'])
            is_how_to_question = any(phrase in processed_question.lower() for phrase in 
                                    ['how to', 'how do', 'what steps', 'what should', 'can you show', 'improve', 'feel better'])
            
            min_length = 10 if (is_emotional_expression or is_how_to_question) else 20
            
            if len(answer.strip()) < min_length:
                logger.warning(f"[FALLBACK] Generated answer too short ({len(answer.strip())} chars), using fallback for question: %s", processed_question)
                answer = create_fallback_response(processed_question, context, "empathetic_professional")
            
            # Step 8: Update conversation memory using unified store
            try:
                conversation_store.add_interaction(
                    user_message=processed_question,
                    assistant_message=answer,
                    metadata={
                        'emotion': emotion,
                        'response_strategy': response_strategy,
                        'confidence': enhanced_metadata.get("confidence"),
                        'fusion_strategy': enhanced_metadata.get("fusion_strategy"),
                        'weekly_goal': request_data.weekly_goal,
                        'feasibility': request_data.feasibility,
                        'anxiety_level': request_data.anxiety_level,
                    },
                    session_id=session_id
                )
                # Update emotional state tracking
                if emotion:
                    conversation_store.update_emotional_state(emotion, 0.8, session_id=session_id)  # Default confidence
            except Exception as e:
                logger.error(f"Error updating conversation memory: {e}")

            # Step 8.1: Periodic summary memory update (every N interactions)
            try:
                N = int(os.getenv("SUMMARY_EVERY_N", "3"))
                # Count per-session records
                interactions = conversation_store.session_data.get(session_id, {})
                if isinstance(interactions, dict) and len(interactions) % N == 0:
                    transcript = conversation_store.get_conversation_history()
                    summary = summarize_session_transcript(transcript)
                    if summary:
                        conversation_store.update_session_summary(session_id, summary)
                        save_session_summary(session_id, summary)
                        logger.info("Session summary updated")
            except Exception as e:
                logger.warning(f"Summary update failed: {e}")
            
            # Step 9: Save interaction for OPRO optimization
            save_interaction(processed_question, answer.strip(), "empathetic_professional")
            append_session_metrics(session_id, request_data.weekly_goal, request_data.feasibility, request_data.anxiety_level)
            
            response_time = time.time() - start_time
            logger.info(f"Response generated in {response_time:.2f} seconds")
            
            # Determine prompt source based on question type
            if is_definitional:
                prompt_source = "definitional"
            elif os.path.exists(OPRO_PROMPT_PATH):
                prompt_source = "opro"
            else:
                prompt_source = "fallback"
            
            return RAGResponse(
                answer=answer.strip(),
                question=processed_question,
                tone=request_data.type,
                status="success",
                context_used=context[:500] + "..." if len(context) > 500 else context,
                prompt_source=prompt_source,
                confidence=enhanced_metadata.get("confidence"),
                fusion_strategy=enhanced_metadata.get("fusion_strategy"),
                source_breakdown=enhanced_metadata.get("source_breakdown"),
                follow_up_suggestions=enhanced_metadata.get("follow_up_suggestions"),
                safety_notes=enhanced_metadata.get("safety_notes"),
                session_id=session_id,
                intent=conversation_analysis.get("response_strategy"),
                strategy=response_strategy,
                tone_suggested=request_data.type,
                weekly_goal=request_data.weekly_goal,
                feasibility=request_data.feasibility,
                anxiety_level=request_data.anxiety_level,
            )
            
        except Exception as e:
            logger.exception(f"LLM inference error: {e}")
            fallback = create_fallback_response(processed_question, context, "empathetic_professional")
            response_time = time.time() - start_time
            logger.info(f"Fallback response generated in {response_time:.2f} seconds")
            
            # Determine prompt source based on question type
            if is_definitional:
                prompt_source = "definitional"
            elif os.path.exists(OPRO_PROMPT_PATH):
                prompt_source = "opro"
            else:
                prompt_source = "fallback"
            
            return RAGResponse(
                answer=fallback,
                question=processed_question,
                tone=request_data.type,
                status="fallback_used",
                context_used=context[:500] + "..." if len(context) > 500 else context,
                prompt_source=prompt_source,
                confidence=enhanced_metadata.get("confidence"),
                fusion_strategy=enhanced_metadata.get("fusion_strategy"),
                source_breakdown=enhanced_metadata.get("source_breakdown"),
                follow_up_suggestions=enhanced_metadata.get("follow_up_suggestions"),
                safety_notes=enhanced_metadata.get("safety_notes")
            )
            
    except Exception as e:
        logger.exception(f"Endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def create_fallback_response(question: str, context: str, tone: str) -> str:
    """Create appropriate fallback responses with practical suggestions"""
    
    question_lower = question.lower() if question else ""
    
    # Check if this is a definitional question
    if is_definitional_question(question):
        # Provide medical definitions for common mental health conditions
        if any(word in question_lower for word in ['depression', 'depressed']):
            return "Depression is a mental health disorder characterized by persistent feelings of sadness, hopelessness, and loss of interest in activities. According to medical guidelines, common symptoms include persistent low mood, loss of interest in previously enjoyable activities, changes in sleep patterns, fatigue, difficulty concentrating, feelings of worthlessness, changes in appetite, and thoughts of death or suicide. If you're experiencing these symptoms, it's important to speak with a mental health professional for proper evaluation and treatment."
        elif any(word in question_lower for word in ['anxiety', 'anxious']):
            return "Anxiety is a mental health condition characterized by excessive worry, fear, and nervousness. According to medical guidelines, common symptoms include persistent worry, restlessness, difficulty concentrating, irritability, muscle tension, sleep problems, and physical symptoms like rapid heartbeat or sweating. If you're experiencing these symptoms, it's important to speak with a mental health professional for proper evaluation and treatment."
        elif any(word in question_lower for word in ['ptsd', 'post traumatic', 'trauma']):
            return "Post-traumatic stress disorder (PTSD) is a mental health condition that can develop after experiencing or witnessing a traumatic event. According to medical guidelines, common symptoms include intrusive memories, nightmares, flashbacks, avoidance of reminders, negative changes in thinking and mood, and changes in physical and emotional reactions. If you're experiencing these symptoms, it's important to speak with a mental health professional for proper evaluation and treatment."
        elif any(word in question_lower for word in ['ocd', 'obsessive', 'compulsive']):
            return "Obsessive-compulsive disorder (OCD) is a mental health condition characterized by unwanted, intrusive thoughts (obsessions) and repetitive behaviors (compulsions). According to medical guidelines, common symptoms include persistent unwanted thoughts, repetitive behaviors performed to reduce anxiety, and significant distress or impairment in daily functioning. If you're experiencing these symptoms, it's important to speak with a mental health professional for proper evaluation and treatment."
        else:
            return "I can provide information about mental health conditions. Could you please specify which condition you'd like to learn more about? Common conditions include depression, anxiety, PTSD, OCD, and others."
    
    # Enhanced emotion/topic detection with concise, structured responses
    if any(word in question_lower for word in ['sad', 'sadness', 'down', 'depressed', 'mood', 'feel better', 'i feel sad']):
        return "I understand you're feeling sad. Here are some practical suggestions: 1) Try a 5-minute mindfulness meditation focusing on your breath. 2) Practice the 5-4-3-2-1 grounding technique - name 5 things you can see, 4 you can touch, 3 you can hear, 2 you can smell, and 1 you can taste. 3) Write down three things you're grateful for today. Remember, it's okay to not feel okay. What's been contributing to these feelings?"
    elif any(word in question_lower for word in ['anxious', 'anxiety', 'worried', 'stressed', 'nervous']):
        return "I hear that anxiety is affecting you. Here are some techniques that might help: 1) Try the 4-7-8 breathing technique - inhale for 4 counts, hold for 7, exhale for 8. 2) Progressive muscle relaxation - tense and release each muscle group. 3) Take a 10-minute walk outside to help clear your mind. What's been causing you the most worry lately?"
    elif any(word in question_lower for word in ['angry', 'mad', 'frustrated', 'irritated']):
        return "It sounds like you're feeling frustrated. Here are some ways to help: 1) Take a few deep breaths and count to 10 before responding. 2) Try journaling your feelings to process them. 3) Physical activity like walking can help release tension. What's been bothering you the most?"
    elif any(word in question_lower for word in ['lonely', 'alone', 'isolated', 'disconnected']):
        return "Feeling lonely can be painful. Here are some suggestions: 1) Reach out to a friend or family member for a brief chat. 2) Join an online community with similar interests. 3) Practice self-compassion - treat yourself with kindness. What's making you feel this way?"
    elif any(word in question_lower for word in ['help', 'support', 'advice', 'suggestions']):
        return "I'm here to support you. Here are some wellness techniques: 1) Practice daily gratitude by writing down one good thing each day. 2) Try 5-10 minutes of mindfulness meditation. 3) Set small, achievable goals for yourself. Can you tell me more about what you need help with?"
    elif any(word in question_lower for word in ['thank', 'thanks', 'hello', 'hi']):
        return "You're welcome! I'm here to support you. Is there anything specific you'd like to talk about or any techniques you'd like to learn more about?"
    else:
        return "I'm here to listen and support you. Here are some wellness practices: 1) Take a few deep breaths and notice how you're feeling. 2) Try a short mindfulness exercise - focus on your breath for 2-3 minutes. 3) Practice self-compassion by being kind to yourself. Can you tell me more about what's been on your mind?"

@app.post("/api/reset_conversation")
async def reset_conversation():
    """Reset conversation memory using unified store"""
    try:
        conversation_store.reset_conversation()
        return {"message": "Conversation reset successfully", "status": "success"}
    except Exception as e:
        logger.error(f"Error resetting conversation: {e}")
        raise HTTPException(status_code=500, detail="Failed to reset conversation")

@app.post("/api/feedback")
async def collect_feedback(feedback_data: FeedbackRequest):
    """Collect user feedback for OPRO optimization"""
    try:
        save_interaction(feedback_data.question, feedback_data.answer, feedback_data.rating)
        return {"message": "Feedback received", "status": "success"}
    except Exception as e:
        logger.error(f"Error collecting feedback: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# CBT-specific endpoints
class CBTRequest(BaseModel):
    query: str
    context: Optional[str] = ""

class CBTResponse(BaseModel):
    query: str
    cbt_relevant: bool
    recommended_techniques: List[dict]
    supporting_content: List[dict]
    formatted_response: str
    status: str

@app.get("/api/cbt/status")
async def get_cbt_status():
    """Get CBT integration status"""
    if cbt_integration is not None:
        try:
            status = cbt_integration.get_cbt_status()
            return {
                "cbt_available": True,
                "status": status
            }
        except Exception as e:
            logger.error(f"Error getting CBT status: {e}")
            return {
                "cbt_available": False,
                "error": str(e)
            }
    else:
        return {
            "cbt_available": False,
            "message": "CBT integration not available"
        }

@app.post("/api/cbt/recommend", response_model=CBTResponse)
async def get_cbt_recommendations(request: CBTRequest):
    """Get CBT technique recommendations"""
    if cbt_integration is None:
        raise HTTPException(
            status_code=503, 
            detail="CBT integration not available. Please ensure CBT system is properly set up."
        )
        
    try:
        # Check if query is CBT relevant
        is_relevant = cbt_integration.should_include_cbt(request.query)
        
        if is_relevant:
            # Get CBT recommendations
            recommendations = cbt_integration.cbt_kb.get_cbt_recommendation(
                request.query, 
                request.context
            )
            
            # Format response
            formatted_response = cbt_integration.cbt_kb.format_cbt_response(
                recommendations, 
                request.query,
                request.context
            )
            
            return CBTResponse(
                query=request.query,
                cbt_relevant=True,
                recommended_techniques=recommendations['recommended_techniques'],
                supporting_content=recommendations['supporting_content'],
                formatted_response=formatted_response,
                status="success"
            )
        else:
            return CBTResponse(
                query=request.query,
                cbt_relevant=False,
                recommended_techniques=[],
                supporting_content=[],
                formatted_response="This query does not appear to be related to CBT techniques. Please ask about anxiety, depression, stress management, or specific therapeutic techniques.",
                status="not_relevant"
            )
            
    except Exception as e:
        logger.error(f"CBT recommendation error: {e}")
        raise HTTPException(status_code=500, detail=f"CBT recommendation failed: {e}")

@app.post("/api/cbt/search")
async def search_cbt_techniques(request: CBTRequest):
    """Search for specific CBT techniques"""
    if cbt_integration is None:
        raise HTTPException(
            status_code=503, 
            detail="CBT integration not available"
        )
        
    try:
        # Search CBT techniques
        results = cbt_integration.cbt_kb.search_cbt_techniques(
            request.query, 
            top_k=5
        )
        
        return {
            "query": request.query,
            "results": results,
            "total_found": len(results),
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"CBT search error: {e}")
        raise HTTPException(status_code=500, detail=f"CBT search failed: {e}")

@app.get("/")
async def root():
    cbt_status = "available" if cbt_integration is not None else "not_available"
    
    return {
        "message": "ICD-11 Enhanced RAG API with CBT Integration",
        "version": "3.1.0",
        "description": "Enhanced API with emotion analysis, conversation memory, OPRO optimization, and CBT techniques integration",
        "cbt_integration": cbt_status,
        "endpoints": {
            "health": "/health",
            "empathetic_professional": "/api/empathetic_professional",
            "reset_conversation": "/api/reset_conversation",
            "feedback": "/api/feedback",
            "cbt_status": "/api/cbt/status",
            "cbt_recommend": "/api/cbt/recommend",
            "cbt_search": "/api/cbt/search"
        }
    }

def is_definitional_question(question: str) -> bool:
    """
    Detect if a question is asking for a definition or medical information
    """
    if not question or not question.strip():
        return False
    
    question_lower = question.lower().strip()
    
    # Definitional question patterns
    definitional_patterns = [
        r'what is\s+\w+',  # "what is anxiety", "what is depression"
        r'what are\s+\w+',  # "what are symptoms"
        r'define\s+\w+',  # "define anxiety"
        r'meaning of\s+\w+',  # "meaning of anxiety"
        r'definition of\s+\w+',  # "definition of anxiety"
        r'what does\s+\w+\s+mean',  # "what does depression mean"
        r'explain\s+\w+',  # "explain depression"
        r'tell me about\s+\w+',  # "tell me about depression"
        r'describe\s+\w+',  # "describe anxiety"
        r'what is the\s+\w+',  # "what is the difference"
        r'what are the\s+\w+',  # "what are the symptoms"
        r'how is\s+\w+\s+defined',  # "how is depression defined"
        r'what constitutes\s+\w+',  # "what constitutes anxiety"
    ]
    
    # Check if question matches any definitional pattern
    for pattern in definitional_patterns:
        if re.search(pattern, question_lower):
            return True
    
    # Additional checks for medical terminology questions
    medical_terms = [
        'depression', 'anxiety', 'ptsd', 'ocd', 'bipolar', 'schizophrenia',
        'panic disorder', 'social anxiety', 'generalized anxiety',
        'major depressive disorder', 'dysthymia', 'cyclothymia',
        'borderline personality', 'narcissistic personality',
        'antisocial personality', 'avoidant personality'
    ]
    
    # Check for help-seeking patterns that should NOT be definitional
    help_seeking_patterns = [
        r'what should i do\s+\w+',  # "what should i do about"
        r'what can i do\s+\w+',  # "what can i do about"
        r'how can i\s+\w+',  # "how can i help"
        r'how do i\s+\w+',  # "how do i deal"
        r'i need help\s+\w+',  # "i need help with"
        r'can you help\s+\w+',  # "can you help me"
        r'help me\s+\w+',  # "help me with"
        r'struggling with\s+\w+',  # "struggling with"
        r'dealing with\s+\w+',  # "dealing with"
        r'coping with\s+\w+',  # "coping with"
    ]
    
    # If it's a help-seeking question, it's not definitional
    for pattern in help_seeking_patterns:
        if re.search(pattern, question_lower):
            return False
    
    # If question contains medical terms and asks for information (but not help)
    has_medical_term = any(term in question_lower for term in medical_terms)
    is_information_request = any(word in question_lower for word in ['what', 'define', 'explain', 'tell', 'describe'])
    
    return has_medical_term and is_information_request

def load_definitional_prompt() -> str:
    """
    Load the definitional question prompt template
    """
    definitional_prompt_path = "OPRO_Streamlined/prompts/definitional_prompt.txt"
    
    try:
        if os.path.exists(definitional_prompt_path):
            with open(definitional_prompt_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        else:
            logger.warning(f"Definitional prompt not found at {definitional_prompt_path}")
            return ""
    except Exception as e:
        logger.error(f"Error loading definitional prompt: {e}")
        return ""

def get_medical_context_for_definition(question: str, store) -> str:
    """
    Retrieve relevant medical context for definitional questions
    """
    if not store:
        return ""
    
    try:
        # Extract the medical term from the question
        question_lower = question.lower()
        
        # Common medical terms to search for
        medical_terms = {
            'depression': ['depression', 'depressive', 'major depressive', 'dysthymia'],
            'anxiety': ['anxiety', 'anxious', 'panic', 'generalized anxiety'],
            'ptsd': ['ptsd', 'post traumatic', 'trauma', 'traumatic'],
            'ocd': ['ocd', 'obsessive', 'compulsive'],
            'bipolar': ['bipolar', 'manic', 'mania', 'cyclothymia'],
            'schizophrenia': ['schizophrenia', 'psychotic', 'psychosis'],
            'personality': ['personality disorder', 'borderline', 'narcissistic', 'antisocial']
        }
        
        # Find the most relevant medical term
        search_terms = []
        for category, terms in medical_terms.items():
            if any(term in question_lower for term in terms):
                search_terms.extend(terms)
                break
        
        if not search_terms:
            # If no specific term found, use the question itself
            search_terms = [question]
        
        # Search for relevant medical information
        context_parts = []
        for term in search_terms[:3]:  # Limit to 3 terms to avoid too much context
            try:
                # Use similarity search to find relevant medical information
                results = store.similarity_search(term, k=2)
                for result in results:
                    if result.page_content and len(result.page_content) > 50:
                        context_parts.append(result.page_content[:300])  # Limit each part
            except Exception as e:
                logger.debug(f"Error searching for term '{term}': {e}")
                continue
        
        # Combine context parts
        if context_parts:
            combined_context = " ".join(context_parts)
            # Clean up the context
            combined_context = re.sub(r'\s+', ' ', combined_context).strip()
            return combined_context[:800]  # Limit total context length
        
        return ""
        
    except Exception as e:
        logger.error(f"Error getting medical context: {e}")
        return ""

if __name__ == "__main__":
    logger.info(f"Starting FastAPI server on port 8000 with {DEVICE}...")
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        workers=1  # Single worker for GPU efficiency
    )
