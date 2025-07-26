#!/usr/bin/env python3
"""
FastAPI Server for ICD-11 RAG System - OPRO Integrated Version
Integrated with OPRO optimization system for dynamic prompt management
"""

import os
import sys
import json
from pathlib import Path
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import logging
import uvicorn
import time
from datetime import datetime
import re

# HuggingFace and LLM
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.llms.huggingface_pipeline import HuggingFacePipeline

# LangChain RAG components (vector store + embeddings + chains)
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
    title="ICD-11 Empathetic Professional RAG API - OPRO Integrated",
    description="Unified API with OPRO optimization integration for empathetic and professional mental health assistance",
    version="3.0.0"
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

# Enable debug logging for CBT integration
cbt_logger = logging.getLogger("cbt_integration")
cbt_logger.setLevel(logging.DEBUG)
# Add console handler for CBT debug messages
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
cbt_logger.addHandler(console_handler)

# Global variables
psychologist_llm = None
store = None
embedder = None
emotion_classifier = None
memory = ConversationBufferMemory(return_messages=True)
cbt_integration = None

# OPRO Integration
OPRO_PROMPT_PATH = "OPRO_Streamlined/prompts/optimized_prompt.txt"  
OPRO_FALLBACK_PATH = "ICD11_OPRO/prompts/optimized_prompt.txt"  
INTERACTIONS_FILE = "interactions.json"

# Debug settings
SHOW_PROMPT_DEBUG = os.getenv("SHOW_PROMPT_DEBUG", "true").lower() == "true"

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
    if tone == "empathetic_professional":
        return load_opro_prompt()
    else:
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
    history: Optional[List[Message]] = []

class RAGResponse(BaseModel):
    answer: str
    question: str
    tone: str
    status: str
    context_used: Optional[str] = None
    prompt_source: str = "fallback"  # "opro" or "fallback"

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

# Load components
def load_embeddings():
    try:
        logger.info(f"Loading HuggingFace embeddings on {DEVICE}...")
        embedder = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
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
    """Load emotion classification model"""
    try:
        logger.info("Loading emotion classifier...")
        classifier = pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            device=DEVICE
        )
        logger.info("Emotion classifier loaded successfully")
        return classifier
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

def load_psychologist_llm():
    """Load the psychologist LLM for empathetic professional responses"""
    logger.info(f"Loading Psychologist LLM on {DEVICE}...")
    try:
        tok = AutoTokenizer.from_pretrained(
            "Qwen/Qwen-1_8B-Chat",
            trust_remote_code=True,
            padding_side="left"
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen-1_8B-Chat",
            trust_remote_code=True,
            device_map=DEVICE,
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
            low_cpu_mem_usage=True
        )
        
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tok,
            max_new_tokens=120,  # Further reduced for concise responses
            do_sample=True,
            temperature=0.8, 
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tok.eos_token_id,
            eos_token_id=tok.eos_token_id,
            return_full_text=False
        )
        
        psychologist_llm = HuggingFacePipeline(pipeline=pipe)
        logger.info(f"Psychologist LLM loaded successfully on {DEVICE}")
        return psychologist_llm
    except Exception as e:
        logger.exception("Error loading psychologist LLM")
        return None

def initialize_rag_system():
    global psychologist_llm, store, embedder, emotion_classifier, cbt_integration
    logger.info(f"Initializing RAG system on {DEVICE}...")
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
    
    # Initialize CBT integration
    cbt_integration = load_cbt_integration()
    if cbt_integration is None:
        logger.warning("CBT integration not loaded, continuing without CBT enhancement")
    
    logger.info(f"RAG system initialized successfully on {DEVICE}")
    return True

@app.on_event("startup")
async def startup_event():
    if not initialize_rag_system():
        logger.error("Failed to initialize RAG system")
        sys.exit(1)

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
    
    # Get CBT status
    cbt_available = False
    cbt_techniques = None
    cbt_content = None
    if cbt_integration is not None:
        try:
            cbt_status = cbt_integration.get_cbt_status()
            cbt_available = cbt_status['available']
            cbt_techniques = cbt_status['total_techniques']
            cbt_content = cbt_status['total_content']
        except:
            pass
    
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
        cbt_content=cbt_content
    )

def analyze_emotion(text):
    """Analyze emotion from user input"""
    if emotion_classifier is None:
        return "neutral"
    
    try:
        result = emotion_classifier(text)
        return result[0]['label'] if result else "neutral"
    except Exception as e:
        logger.error(f"Emotion analysis error: {e}")
        return "neutral"

def get_conversation_history():
    """Get conversation history with enhanced context understanding"""
    try:
        messages = memory.chat_memory.messages
        if not messages:
            return ""
            
        if len(messages) > 6:  # Keep last 3 exchanges
            messages = messages[-6:]
        
        # Analyze conversation for key context
        user_emotions = []
        user_goals = []
        topics_discussed = []
        
        history_parts = []
        for i, msg in enumerate(messages):
            content = msg.content
            
            if msg.type == "human":
                # Extract user emotional state and goals
                content_lower = content.lower()
                
                # Detect emotional keywords
                emotion_words = ['anxious', 'stressed', 'worried', 'calm', 'better', 'worse', 'overwhelmed', 'peaceful']
                found_emotions = [word for word in emotion_words if word in content_lower]
                user_emotions.extend(found_emotions)
                
                # Detect goal keywords  
                goal_words = ['want to', 'need to', 'help with', 'stop', 'feel calmer', 'think clearly', 'be better']
                found_goals = [goal for goal in goal_words if goal in content_lower]
                user_goals.extend(found_goals)
                
                history_parts.append(f"User: {content}")
            else:
                # Summarize assistant responses
                if len(content) > 100:
                    # Extract the key question or guidance from assistant response
                    if '?' in content:
                        questions = [q.strip() + '?' for q in content.split('?') if q.strip()]
                        if questions:
                            summary = f"Assistant asked: {questions[-1]}"
                        else:
                            summary = f"Assistant: {content[:80]}..."
                    else:
                        summary = f"Assistant: {content[:80]}..."
                else:
                    summary = f"Assistant: {content}"
                    
                history_parts.append(summary)
        
        # Build enhanced context
        context_parts = []
        
        if history_parts:
            context_parts.append("Recent conversation:")
            context_parts.extend(history_parts)
        
        # Add emotional context if available
        if user_emotions:
            recent_emotions = list(set(user_emotions[-3:]))  # Last 3 unique emotions
            context_parts.append(f"User's recent emotional state: {', '.join(recent_emotions)}")
        
        # Add goal context if available
        if user_goals:
            recent_goals = list(set(user_goals[-2:]))  # Last 2 unique goals
            context_parts.append(f"User's expressed goals: {', '.join(recent_goals)}")
        
        return '\n'.join(context_parts)
        
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

def post_process_response(answer: str) -> str:
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
    ]
    
    for pattern in template_patterns:
        answer = re.sub(pattern, '', answer, flags=re.IGNORECASE | re.MULTILINE)
    
    # Remove labels like Empathy:, Citation:, Follow-up question: etc
    answer = re.sub(r'(Empathy:|Citation:|Follow-up question:)', '', answer, flags=re.IGNORECASE)
    
    # Remove any non-ASCII characters that might have crept in
    answer = re.sub(r'[^\x00-\x7F]+', '', answer)
    
    # Remove extra newlines and spaces but preserve paragraph structure
    answer = re.sub(r'\n+', ' ', answer)
    answer = re.sub(r'\s+', ' ', answer).strip()
    
    # For crisis intervention responses, keep more content (up to 6-7 sentences)
    # Don't truncate professional safety assessments
    sentences = re.split(r'(?<=[.!?]) +', answer)
    if len(sentences) > 7:
        # Keep safety-related content by checking for crisis keywords
        crisis_keywords = ['safety', 'crisis', 'emergency', 'self-harm', 'suicide', 'help', 'support']
        important_sentences = []
        
        for i, sentence in enumerate(sentences[:7]):
            # Always keep first few sentences and those with crisis keywords
            if i < 3 or any(keyword in sentence.lower() for keyword in crisis_keywords):
                important_sentences.append(sentence)
        
        # If we have important crisis content, use that; otherwise use first 6 sentences
        if len(important_sentences) >= 3:
            answer = ' '.join(important_sentences)
        else:
            answer = ' '.join(sentences[:6])
    
    # DO NOT remove professional psychological phrases - they're important for crisis intervention
    # Only remove truly generic customer service phrases
    generic_phrases = [
        "is there anything else i can help you with",
        "feel free to reach out",
        "don't hesitate to contact"
    ]
    for phrase in generic_phrases:
        if phrase in answer.lower():
            answer = re.sub(re.escape(phrase), '', answer, flags=re.IGNORECASE)
            answer = re.sub(r'\s+', ' ', answer).strip()
    
    # Ensure answer ends with proper punctuation
    if answer and not answer[-1] in '.!?':
        answer += '.'
    
    return answer.strip()

@app.post("/api/empathetic_professional", response_model=RAGResponse)
async def empathetic_professional_endpoint(request_data: RAGRequest):
    start_time = time.time()
    
    try:
        if store is None or psychologist_llm is None:
            raise HTTPException(status_code=500, detail="RAG system not initialized")

        # Step 1: Input validation
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
        
        # Step 3: Get conversation history
        history = get_conversation_history()
        
        # Display conversation history for debugging
        if SHOW_PROMPT_DEBUG:
            print("CONVERSATION HISTORY:")
            print("="*80)
            if history:
                print(history)
            else:
                print("No conversation history")
            print("="*80)
        
        # Step 4: RAG search (optimized for mental health relevance)
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
        
        # Limit context length for faster processing
        if len(context) > 2000:
            context = context[:2000] + "..."
        
        logger.info(f"Retrieved context length: {len(context)} characters")

        # Step 5: Dynamically load OPRO optimized prompt, check latest version each time
        opro_prompt = load_opro_prompt()
        
        # Display OPRO prompt source for debugging
        if SHOW_PROMPT_DEBUG:
            print("OPRO PROMPT SOURCE:")
            print("="*80)
            opro_exists = os.path.exists(OPRO_PROMPT_PATH)
            print(f"OPRO file exists: {opro_exists}")
            if opro_exists:
                print(f"Using OPRO optimized prompt from: {OPRO_PROMPT_PATH}")
            else:
                print("Using fallback prompt (OPRO not available)")
            print("="*80)
        
        # Step 1: Always replace variables in OPRO prompt first
        try:
            # Limit context and history to prevent template overflow
            limited_context = context[:1000] + "..." if len(context) > 1000 else context
            limited_history = history[:500] + "..." if len(history) > 500 else history
            
            formatted_opro = opro_prompt.format(
                context=limited_context,
                question=processed_question,
                history=limited_history
            )
            logger.info("Variables successfully replaced in OPRO prompt")
        except KeyError as e:
            logger.warning(f"Variable replacement failed: {e}, using original prompt")
            formatted_opro = opro_prompt
        
        # Step 2: Handle different ChatML formats
        if "<|im_start|>" in formatted_opro:
            # Already in Qwen ChatML format
            formatted_prompt = formatted_opro
            logger.info("Using OPRO prompt in Qwen ChatML format")
        elif "<|system|>" in formatted_opro:
            # Already in standard ChatML format
            formatted_prompt = formatted_opro
            logger.info("Using OPRO prompt in standard ChatML format")
        else:
            # Wrap OPRO prompt in ChatML format
            formatted_prompt = f"""
<|system|>
{formatted_opro}
<|user|>
{processed_question}
<|assistant|>
"""
            logger.info("OPRO prompt wrapped in ChatML format")
        
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
            # Reset LLM generation parameters - optimize for concise, helpful responses
            pipe = psychologist_llm.pipeline
            pipe.model.config.max_new_tokens = 150  # Shorter to prevent template leakage
            pipe.model.config.temperature = 0.6     # More consistent responses
            pipe.model.config.top_p = 0.8
            result = psychologist_llm.invoke(formatted_prompt)
            answer = result if isinstance(result, str) else str(result)
            
            # Display raw LLM output for debugging
            if SHOW_PROMPT_DEBUG:
                print("RAW LLM OUTPUT:")
                print("-" * 60)
                print(answer)
                print("-" * 60)
            
            # Extract only content after <|assistant|> to avoid instruction leakage
            if "<|assistant|>" in answer:
                answer = answer.split("<|assistant|>")[-1].strip()
            
            # Clean up any other ChatML or template remnants
            chatml_patterns = [
                r'<\|.*?\|>',  # Remove any ChatML tags
                r'<\|im_start\|>.*?<\|im_end\|>',  # Remove Qwen ChatML blocks
                r'<\|system\|>.*?(?=<\||$)',  # Remove system messages
                r'<\|user\|>.*?(?=<\||$)',  # Remove user messages
            ]
            
            for pattern in chatml_patterns:
                answer = re.sub(pattern, '', answer, flags=re.DOTALL | re.IGNORECASE)
            
            # Enhanced filtering for problematic content
            problematic_patterns = [
                r'I understand you\'re experiencing ".*?" and I want to acknowledge',  # Template repetition
                r'Be mindful of the user\'s emotions.*?support\.',  # Instruction leakage
                r'Example Input \d+:.*?Response:',  # Example content
                r'RESPONSE GUIDELINES:.*?RESPONSE:',  # Template content
                r'USER MESSAGE:.*?RESPONSE:',  # Template variables
                r'emergency involved, and I would appreciate your assistance',  # Specific weird content
            ]
            
            for pattern in problematic_patterns:
                answer = re.sub(pattern, '', answer, flags=re.DOTALL | re.IGNORECASE)
            
            # Remove any content that looks like template variables, instructions, or irrelevant medical terms
            problematic_phrases = [
                "be mindful of", "example input", "response guidelines", 
                "user message:", "context:", "conversation history:",
                "parkinson", "tremor", "dementia", "neurological", "rigidity",
                "sudden onset", "early disability", "cognitive function related to mathematics"
            ]
            
            if any(phrase in answer.lower() for phrase in problematic_phrases):
                logger.warning("Detected problematic content in response, using fallback")
                answer = create_fallback_response(processed_question, context, "empathetic_professional")
            
            # Post-process: automatically trim overly long answers to ensure conciseness
            answer = post_process_response(answer)
            
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
            # Check response quality
            if len(answer.strip()) < 10:
                logger.warning("[FALLBACK] Generated answer too short, using fallback for question: %s", processed_question)
                answer = create_fallback_response(processed_question, context, "empathetic_professional")
            
            # Step 8: Update conversation memory
            try:
                memory.chat_memory.add_user_message(processed_question)
                memory.chat_memory.add_ai_message(answer)
            except Exception as e:
                logger.error(f"Error updating conversation memory: {e}")
            
            # Step 9: Save interaction for OPRO optimization
            save_interaction(processed_question, answer.strip(), "empathetic_professional")
            
            response_time = time.time() - start_time
            logger.info(f"Response generated in {response_time:.2f} seconds")
            
            # Determine prompt source
            prompt_source = "opro" if os.path.exists(OPRO_PROMPT_PATH) else "fallback"
            
            return RAGResponse(
                answer=answer.strip(),
                question=processed_question,
                tone="empathetic_professional",
                status="success",
                context_used=context[:500] + "..." if len(context) > 500 else context,
                prompt_source=prompt_source
            )
            
        except Exception as e:
            logger.exception(f"LLM inference error: {e}")
            fallback = create_fallback_response(processed_question, context, "empathetic_professional")
            response_time = time.time() - start_time
            logger.info(f"Fallback response generated in {response_time:.2f} seconds")
            
            # Determine prompt source
            prompt_source = "opro" if os.path.exists(OPRO_PROMPT_PATH) else "fallback"
            
            return RAGResponse(
                answer=fallback,
                question=processed_question,
                tone="empathetic_professional",
                status="fallback_used",
                context_used=context[:500] + "..." if len(context) > 500 else context,
                prompt_source=prompt_source
            )
            
    except Exception as e:
        logger.exception(f"Endpoint error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def create_fallback_response(question: str, context: str, tone: str) -> str:
    """Create appropriate fallback responses without repeating user input"""
    
    question_lower = question.lower() if question else ""
    
    # Detect emotion/topic and respond appropriately
    if any(word in question_lower for word in ['sad', 'sadness', 'down', 'depressed']):
        return "I understand you're feeling sad. That's difficult. Would you like to talk about what's weighing on you?"
    elif any(word in question_lower for word in ['anxious', 'anxiety', 'worried', 'stressed']):
        return "I hear that anxiety is affecting you. What's been causing you the most worry?"
    elif any(word in question_lower for word in ['angry', 'mad', 'frustrated']):
        return "It sounds like you're feeling frustrated. What's been bothering you?"
    elif any(word in question_lower for word in ['lonely', 'alone', 'isolated']):
        return "Feeling lonely can be painful. I'm here to listen. What's making you feel this way?"
    elif any(word in question_lower for word in ['help', 'support', 'advice']):
        return "I'm here to support you. Can you tell me more about what you need help with?"
    elif any(word in question_lower for word in ['thank', 'thanks', 'hello', 'hi']):
        return "You're welcome. Is there anything else I can help you with?"
    else:
        return "I'm here to listen and support you. Can you tell me more about what's been on your mind?"

@app.post("/api/reset_conversation")
async def reset_conversation():
    """Reset conversation memory"""
    try:
        global memory
        memory = ConversationBufferMemory(return_messages=True)
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

if __name__ == "__main__":
    logger.info(f"Starting FastAPI server on port 8000 with {DEVICE}...")
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        workers=1  # Single worker for GPU efficiency
    )
