#!/usr/bin/env python3
"""
FastAPI Integration Example for Streamlined OPRO System
Shows how to integrate OPRO-optimized prompts with a FastAPI server
"""

import os
import json
from datetime import datetime
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="OPRO-Integrated Mental Health API",
    description="Example integration of streamlined OPRO system with FastAPI",
    version="1.0.0"
)

class ChatRequest(BaseModel):
    question: str
    history: Optional[list] = []
    tone: str = "empathetic_professional"

class ChatResponse(BaseModel):
    answer: str
    question: str
    tone: str
    status: str
    prompt_source: str

class FeedbackRequest(BaseModel):
    question: str
    answer: str
    satisfaction: Optional[int] = None
    empathy: Optional[int] = None
    accuracy: Optional[int] = None
    safety: Optional[int] = None
    comment: Optional[str] = None

def load_opro_prompt() -> str:
    """Load optimized prompt from OPRO system"""
    opro_prompt_path = "prompts/optimized_prompt.txt"
    
    try:
        if os.path.exists(opro_prompt_path):
            with open(opro_prompt_path, 'r', encoding='utf-8') as f:
                prompt = f.read().strip()
                logger.info("Loaded OPRO-optimized prompt")
                return prompt
        else:
            # Fallback to default prompt
            logger.warning("OPRO prompt not found, using fallback")
            return get_fallback_prompt()
    except Exception as e:
        logger.error(f"Error loading OPRO prompt: {e}")
        return get_fallback_prompt()

def get_fallback_prompt() -> str:
    """Fallback prompt when OPRO prompt is unavailable"""
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

def save_interaction(question: str, answer: str, tone: str, feedback: Optional[dict] = None):
    """Save interaction to interactions.json for OPRO processing"""
    interaction = {
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "answer": answer,
        "tone": tone,
        "feedback": feedback
    }
    
    interactions_file = "interactions.json"
    
    try:
        # Load existing interactions
        if os.path.exists(interactions_file):
            with open(interactions_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    interactions = data
                else:
                    interactions = data.get('interactions', [])
        else:
            interactions = []
        
        # Add new interaction
        interactions.append(interaction)
        
        # Save back to file
        with open(interactions_file, 'w', encoding='utf-8') as f:
            json.dump(interactions, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved interaction to {interactions_file}")
        
    except Exception as e:
        logger.error(f"Error saving interaction: {e}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    opro_prompt_exists = os.path.exists("prompts/optimized_prompt.txt")
    
    return {
        "status": "healthy",
        "opro_prompt_loaded": opro_prompt_exists,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Main chat endpoint using OPRO-optimized prompts"""
    try:
        # Load optimized prompt
        prompt_template = load_opro_prompt()
        prompt_source = "opro" if os.path.exists("prompts/optimized_prompt.txt") else "fallback"
        
        # Format prompt (simplified example - you would integrate with your LLM here)
        formatted_prompt = prompt_template.format(
            context="[Medical context would be retrieved here]",
            history=str(request.history),
            question=request.question
        )
        
        # Generate response (this is a placeholder - integrate with your LLM)
        response_text = f"This is a placeholder response to: {request.question}. " \
                       f"In a real implementation, this would be generated by your LLM using the optimized prompt."
        
        # Save interaction for future OPRO optimization
        save_interaction(
            question=request.question,
            answer=response_text,
            tone=request.tone
        )
        
        return ChatResponse(
            answer=response_text,
            question=request.question,
            tone=request.tone,
            status="success",
            prompt_source=prompt_source
        )
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/api/feedback")
async def feedback_endpoint(request: FeedbackRequest):
    """Feedback endpoint for collecting user feedback"""
    try:
        feedback_data = {
            "satisfaction": request.satisfaction,
            "empathy": request.empathy,
            "accuracy": request.accuracy,
            "safety": request.safety,
            "comment": request.comment
        }
        
        # Save interaction with feedback for OPRO processing
        save_interaction(
            question=request.question,
            answer=request.answer,
            tone="empathetic_professional",
            feedback=feedback_data
        )
        
        return {
            "status": "success",
            "message": "Feedback saved successfully",
            "feedback": feedback_data
        }
        
    except Exception as e:
        logger.error(f"Error saving feedback: {e}")
        raise HTTPException(status_code=500, detail="Failed to save feedback")

@app.get("/api/opro/status")
async def opro_status():
    """Get OPRO system status"""
    try:
        # Check if interactions file exists
        interactions_count = 0
        if os.path.exists("interactions.json"):
            with open("interactions.json", 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    interactions_count = len(data)
                else:
                    interactions_count = len(data.get('interactions', []))
        
        # Check if optimized prompt exists
        optimized_prompt_exists = os.path.exists("prompts/optimized_prompt.txt")
        
        # Check scheduler state
        scheduler_state = {}
        if os.path.exists("logs/opro_scheduler_state.json"):
            with open("logs/opro_scheduler_state.json", 'r') as f:
                scheduler_state = json.load(f)
        
        return {
            "interactions_collected": interactions_count,
            "optimized_prompt_available": optimized_prompt_exists,
            "scheduler_state": scheduler_state,
            "system_status": "operational"
        }
        
    except Exception as e:
        logger.error(f"Error getting OPRO status: {e}")
        return {
            "system_status": "error",
            "error": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 