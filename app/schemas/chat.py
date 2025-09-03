from typing import List, Optional, Dict
from pydantic import BaseModel, Field


class Message(BaseModel):
    role: str
    content: str


class UserProfile(BaseModel):
    name: Optional[str] = None
    gender: Optional[str] = None  # 'male' | 'female' | 'prefer_not_to_say'
    age: Optional[int] = None
    occupation: Optional[str] = None


class RAGRequest(BaseModel):
    question: str
    type: Optional[str] = None
    session_id: Optional[str] = None
    history: Optional[List[Message]] = Field(default_factory=list)
    user_profile: Optional[UserProfile] = None
    weekly_goal: Optional[str] = None
    feasibility: Optional[float] = None  # 0-10
    anxiety_level: Optional[float] = None  # 0-10


class RAGResponse(BaseModel):
    answer: str
    question: str
    tone: str
    status: str
    context_used: Optional[str] = None
    prompt_source: str = "fallback"
    confidence: Optional[float] = None
    fusion_strategy: Optional[str] = None
    source_breakdown: Optional[Dict[str, float]] = None
    follow_up_suggestions: Optional[List[str]] = None
    safety_notes: Optional[List[str]] = None
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
    rating: int
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


