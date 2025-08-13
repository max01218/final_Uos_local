from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from app.schemas.chat import RAGRequest, RAGResponse
from app.core.di import get_chat_service
from app.utils.prompting import get_dynamic_prompt


router = APIRouter()


@router.post("/api/v2/empathetic_professional", response_model=RAGResponse)
async def empathetic_professional_v2(request_data: RAGRequest, chat_service = Depends(get_chat_service)):
    try:
        # Optional: select prompt dynamically and pass via request extension in future iterations
        answer, meta = await chat_service.handle_chat(request_data)
        resp = RAGResponse(
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
        return JSONResponse(status_code=200, content=resp.model_dump())
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


