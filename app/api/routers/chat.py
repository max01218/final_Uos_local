from fastapi import APIRouter
from app.schemas.chat import RAGRequest, RAGResponse, FeedbackRequest
from fastapi import HTTPException, Depends
from app.core.di import get_chat_service


router = APIRouter()


@router.post("/api/empathetic_professional", response_model=RAGResponse)
async def empathetic_professional_route(request_data: RAGRequest, chat_service = Depends(get_chat_service)):
    try:
        answer, meta = await chat_service.handle_chat(request_data)
        return RAGResponse(
            answer=answer.strip(),
            question=request_data.question,
            tone=request_data.type,
            status="success",
            context_used=None,
            prompt_source="v2",
            confidence=None,
            fusion_strategy=None,
            source_breakdown=None,
            follow_up_suggestions=None,
            safety_notes=None,
            session_id=meta.get("session_id"),
            intent=meta.get("intent"),
            strategy=meta.get("strategy"),
            tone_suggested=meta.get("tone_suggested"),
            weekly_goal=meta.get("weekly_goal"),
            feasibility=meta.get("feasibility"),
            anxiety_level=meta.get("anxiety_level"),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"chat service error: {e}")


@router.post("/api/reset_conversation")
async def reset_conversation(chat_service = Depends(get_chat_service)):
    # soft reset in new service layer
    chat_service.cs.reset_conversation()
    return {"message": "Conversation reset successfully", "status": "success"}


@router.post("/api/feedback")
async def collect_feedback(feedback_data: FeedbackRequest):
    # TODO: move to a dedicated feedback service; keep simple for now
    return {"message": "Feedback received", "status": "success"}


