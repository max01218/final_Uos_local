# app/api/routers/chat_v2.py
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
import logging

from app.schemas.chat import RAGRequest, RAGResponse
from app.core.di import get_chat_service
from app.utils.esq import OUTPUT_CONTRACT, format_esq, fallback_esq

logger = logging.getLogger(__name__)
router = APIRouter()

def _get_word_limit(request: Request) -> int:
    try:
        # 若 main.py 沒設，回預設
        return int(getattr(request.app.state, "esq_word_limit", 120))
    except Exception:
        return 120

@router.post("/api/v2/empathetic_professional", response_model=RAGResponse)
async def empathetic_professional_v2(
    request_data: RAGRequest,
    request: Request,
    chat_service = Depends(get_chat_service),
):
    try:
        # ---- 安全處理 system_tail：不要直接賦值到 Pydantic 物件 ----
        # 僅當 RAGRequest 定義了 system_tail 欄位時才帶入
        rd = request_data
        if "system_tail" in getattr(RAGRequest, "model_fields", {}):
            current_tail = getattr(request_data, "system_tail", "") or ""
            rd = request_data.model_copy(update={
                "system_tail": current_tail + (OUTPUT_CONTRACT or "")
            })

        logger.info("v2 handler: received payload")

        # ---- 呼叫服務層 ----
        answer, meta = await chat_service.handle_chat(rd)

        # ---- 依 route 決定是否做 ESQ 格式化 ----
        route = meta.get("route", "other") if isinstance(meta, dict) else "other"
        if route in ("greeting", "small_talk"):
            final_text = answer if isinstance(answer, str) else str(answer)
        else:
            word_limit = _get_word_limit(request)
            safe_answer = answer if isinstance(answer, str) else str(answer)
            try:
                final_text = format_esq(
                    raw_text=safe_answer,
                    output_contract=OUTPUT_CONTRACT,
                    word_limit=word_limit,
                )
            except Exception as fe:
                # 不做 fallback（照你的要求）；直接記錄錯誤並丟 500
                logger.exception("format_esq failed")
                raise HTTPException(status_code=500, detail=f"format_esq failed: {fe}")

            if not isinstance(final_text, str) or not final_text.strip():
                # 同樣：不做 fallback，明確告知是空輸出
                raise HTTPException(status_code=500, detail="formatted text is empty")

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
        # 一定要把 stack 打出來，之後你才找得到真正的行號
        logger.exception("v2 handler uncaught exception")
        raise HTTPException(status_code=500, detail=str(e))
