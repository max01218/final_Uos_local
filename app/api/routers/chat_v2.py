from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
import logging

from app.schemas.chat import RAGRequest, RAGResponse
from app.core.di import get_chat_service
from app.utils.esq import OUTPUT_CONTRACT, format_esq, fallback_esq

logger = logging.getLogger(__name__)
router = APIRouter()

TONE_MAP = {
    "balanced": "balanced",
    "warm": "warm",
    "direct": "direct",
}
DEFAULT_TONE = "balanced"

def _get_esq_config(request: Request):
    try:
        cfg = getattr(request.app.state, "config", {}) or {}
        esq = cfg.get("esq", {}) or {}
        return {"word_limit": int(esq.get("word_limit", 28))}
    except Exception:
        return {"word_limit": 28}

def _resolve_tone(rd: RAGRequest, request: Request) -> str:
    # Accept new 'tone', legacy 'type', or header 'x-tone'
    tone = None
    # Safely try attrs (pydantic model)
    tone = getattr(rd, "tone", None) or getattr(rd, "type", None)
    if not tone:
        tone = request.headers.get("x-tone")
    tone = (tone or DEFAULT_TONE).strip().lower()
    return TONE_MAP.get(tone, DEFAULT_TONE)

@router.post("/api/v2/empathetic_professional", response_model=RAGResponse)
async def empathetic_professional_v2(
    request_data: RAGRequest,
    request: Request,
    chat_service = Depends(get_chat_service),
):
    try:
        esq_cfg = _get_esq_config(request)
        rd = request_data
        tone = _resolve_tone(rd, request)

        answer, meta = await chat_service.handle_chat(rd, tone_override=tone)
        if not isinstance(answer, str):
            answer = str(answer)

        route = meta.get("route", "other") if isinstance(meta, dict) else "other"
        # Greeting: 直接給 LLM 的短訊，不做 ESQ
        if route == "greeting":
            final_text = answer.strip()
        # mh_support：維持 ESQ，但轉成單條短訊（移除 E:/S:/Q: 與多餘句）
        elif route == "mh_support":
            try:
                final_text = format_esq(
                    raw_text=answer,
                    output_contract=OUTPUT_CONTRACT,
                    word_limit=esq_cfg["word_limit"],
                )
                if not isinstance(final_text, str) or not final_text.strip():
                    raise ValueError("empty formatted text")
            except Exception as fe:
                logger.warning("format_esq failed for mh_support: %s; using fallback_esq", fe)
                final_text = fallback_esq(request_data.question or "")
        # info_definition / other：LLM 產生後做輕度清理已在 orchestrator 完成
        else:
            final_text = answer.strip()

        resp = RAGResponse(
            answer=final_text,
            question=request_data.question,
            tone=tone,               # 回傳給前端目前 tone
            status="success",
            context_used=None,
            prompt_source="v2",
            confidence=None,
            fusion_strategy=None,
            source_breakdown=None,
            follow_up_suggestions=None,
            safety_notes=None,
            session_id=(meta.get("session_id") if isinstance(meta, dict) else None),
            intent=(meta.get("intent") if isinstance(meta, dict) else None),
            strategy=(meta.get("strategy") if isinstance(meta, dict) else None),
            tone_suggested=tone,
            weekly_goal=(meta.get("weekly_goal") if isinstance(meta, dict) else None),
            feasibility=(meta.get("feasibility") if isinstance(meta, dict) else None),
            anxiety_level=(meta.get("anxiety_level") if isinstance(meta, dict) else None),
        )
        return JSONResponse(status_code=200, content=resp.model_dump())
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("v2 handler uncaught exception")
        raise HTTPException(status_code=500, detail=str(e))
