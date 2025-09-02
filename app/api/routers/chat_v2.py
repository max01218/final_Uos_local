# app/api/routers/chat_v2.py
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse

from app.schemas.chat import RAGRequest, RAGResponse
from app.core.di import get_chat_service
from app.utils.esq import OUTPUT_CONTRACT, format_esq, fallback_esq

router = APIRouter()

@router.post("/api/v2/empathetic_professional", response_model=RAGResponse)
async def empathetic_professional_v2(
    request_data: RAGRequest,
    request: Request,
    chat_service = Depends(get_chat_service),
):
    try:
        # 如果你的 ChatService 支援附加 system tail，則把合約一併丟進去（不支援則忽略）
        if hasattr(request_data, "system_tail"):
            current_tail = getattr(request_data, "system_tail") or ""
            request_data.system_tail = current_tail + OUTPUT_CONTRACT

        # 取得模型原始輸出
        answer, meta = await chat_service.handle_chat(request_data)

        # 根據路由決定是否套用ESQ格式
        route = meta.get("route", "other") if isinstance(meta, dict) else "other"
        
        if route in ("greeting", "small_talk"):
            # 簡單路由直接使用原始回應，不套用ESQ格式
            final_text = answer
        else:
            # 複雜路由轉為 E/S/Q 三行；若失敗或為空，用保底
            word_limit = getattr(request.app.state, "esq_word_limit", 120)
            safe_answer = answer if isinstance(answer, str) else str(answer)
            final_text = format_esq(safe_answer, word_limit=word_limit)
            if not final_text.strip():
                final_text = fallback_esq(request_data.question)

        resp = RAGResponse(
            answer=final_text,
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
            session_id=meta.get("session_id") if isinstance(meta, dict) else None,
            intent=meta.get("intent") if isinstance(meta, dict) else None,
            strategy=meta.get("strategy") if isinstance(meta, dict) else None,
            tone_suggested=meta.get("tone_suggested") if isinstance(meta, dict) else None,
            weekly_goal=meta.get("weekly_goal") if isinstance(meta, dict) else None,
            feasibility=meta.get("feasibility") if isinstance(meta, dict) else None,
            anxiety_level=meta.get("anxiety_level") if isinstance(meta, dict) else None,
        )
        return JSONResponse(status_code=200, content=resp.model_dump())

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
