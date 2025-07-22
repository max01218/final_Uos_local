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

# Global variables
psychologist_llm = None
store = None
embedder = None
emotion_classifier = None
memory = ConversationBufferMemory(return_messages=True)

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
    global psychologist_llm, store, embedder, emotion_classifier
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
    
    return HealthResponse(
        status="healthy",
        psychologist_llm_loaded=psychologist_llm is not None,
        store_loaded=store is not None,
        device=DEVICE,
        gpu_memory=gpu_memory,
        opro_prompt_loaded=opro_prompt_loaded,
        interactions_count=interactions_count
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
    """Get conversation history for context"""
    try:
        messages = memory.chat_memory.messages
        if len(messages) > 6:  # Keep last 3 exchanges
            messages = messages[-6:]
        
        history = ""
        for msg in messages:
            role = "User" if msg.type == "human" else "Assistant"
            history += f"{role}: {msg.content}\n"
        
        return history.strip()
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
    
    # Remove labels like Empathy:, Citation:, Follow-up question: etc
    answer = re.sub(r'(Empathy:|Citation:|Follow-up question:)', '', answer, flags=re.IGNORECASE)
    
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
        
        # Step 4: RAG search (optimized for diversity)
        retriever = store.as_retriever(search_kwargs={"k": 5})
        docs = retriever.invoke(processed_question)
        context = "\n\n".join([doc.page_content for doc in docs])
        
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
            formatted_opro = opro_prompt.format(
                context=context,
                question=processed_question,
                history=history
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
            # Reset LLM generation parameters - optimize for crisis intervention responses
            pipe = psychologist_llm.pipeline
            pipe.model.config.max_new_tokens = 250  # Increased for detailed crisis responses
            pipe.model.config.temperature = 0.7     # More consistent for professional responses
            pipe.model.config.top_p = 0.9
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
            # Post-process: automatically trim overly long answers to ensure conciseness
            answer = post_process_response(answer)
            
            # Display processed answer for debugging
            if SHOW_PROMPT_DEBUG:
                print("PROCESSED ANSWER:")
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
    """Creating concise fallback responses with better structure"""
    
    if tone == "caring":
        return f"I hear that you're dealing with '{question}' and your feelings are completely valid. Would you like to tell me more about what you're going through?"

    elif tone == "professional":
        return f"Thank you for sharing your concern about '{question}'. This is a recognized area where professional guidance can be helpful. Would you like to discuss this further?"

    else:  # empathetic_professional
        return f"I understand you're experiencing '{question}' and want to acknowledge how difficult that must be. Would you like to share more about your experience?"

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

@app.get("/")
async def root():
    return {
        "message": "ICD-11 Enhanced RAG API",
        "version": "3.0.0",
        "description": "Enhanced API with emotion analysis, conversation memory, and improved dialogue",
        "endpoints": {
            "health": "/health",
            "empathetic_professional": "/api/empathetic_professional",
            "reset_conversation": "/api/reset_conversation",
            "feedback": "/api/feedback"
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
