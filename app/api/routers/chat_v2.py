# app/api/routers/chat_v2.py
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
import logging

from app.schemas.chat import RAGRequest, RAGResponse
from app.core.di import get_chat_service
from app.utils.esq import OUTPUT_CONTRACT, format_esq, fallback_esq

logger = logging.getLogger(__name__)
router = APIRouter()


def _get_esq_config(request: Request):
    """Read ESQ word_limit from app.state.config if present."""
    try:
        cfg = getattr(request.app.state, "config", {}) or {}
        esq = cfg.get("esq", {}) or {}
        return {"word_limit": int(esq.get("word_limit", 45))}
    except Exception:
        return {"word_limit": 45}


@router.post("/api/v2/empathetic_professional", response_model=RAGResponse)
async def empathetic_professional_v2(
    request_data: RAGRequest,
    request: Request,
    chat_service=Depends(get_chat_service),
):
    """
    v2 handler:
    - Always route first (handled in ChatService/Orchestrator).
    - Only when route == 'mh_support' do we enforce E/S/Q formatting.
    - Other routes return the model's natural answer as-is.
    """
    try:
        esq_cfg = _get_esq_config(request)
        rd = request_data

        answer, meta = await chat_service.handle_chat(rd)
        if not isinstance(answer, str):
            answer = str(answer) if answer is not None else ""

        route = "other"
        if isinstance(meta, dict):
            route = meta.get("route", "other") or "other"

        # === Route-specific post-processing ===
        if route == "mh_support":
            # Only mh_support needs E/S/Q
            try:
                formatted = format_esq(
                    raw_text=answer,
                    output_contract=OUTPUT_CONTRACT,
                    word_limit=esq_cfg["word_limit"],
                )
                final_text = formatted if isinstance(formatted, str) and formatted.strip() else fallback_esq(rd.question or "")
            except Exception as fe:
                logger.warning("format_esq failed for mh_support: %s; using fallback_esq", fe)
                final_text = fallback_esq(rd.question or "")
        else:
            # greeting / info_definition / crisis / other → plain text
            final_text = (answer or "").strip()

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
            session_id=(meta.get("session_id") if isinstance(meta, dict) else None),
            intent=(meta.get("intent") if isinstance(meta, dict) else None),
            strategy=(meta.get("strategy") if isinstance(meta, dict) else None),
            tone_suggested=(meta.get("tone_suggested") if isinstance(meta, dict) else None),
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
