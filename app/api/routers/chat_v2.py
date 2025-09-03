from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
import logging

from app.schemas.chat import RAGRequest, RAGResponse
from app.core.di import get_chat_service
from app.utils.esq import OUTPUT_CONTRACT, format_esq, fallback_esq

logger = logging.getLogger(__name__)
router = APIRouter()


def _get_esq_config(request: Request):
    """Read ESQ config from app.state.config if present; otherwise use defaults."""
    try:
        cfg = getattr(request.app.state, "config", {}) or {}
        esq = cfg.get("esq", {}) or {}
        return {"word_limit": int(esq.get("word_limit", 45))}
    except Exception:
        return {"word_limit": 45}


def _normalize_tone(t: str | None) -> str:
    val = (t or "").strip().lower()
    if val in ("balanced", "warm", "direct"):
        return val
    # legacy value from some frontends
    if val == "empathetic_professional":
        return "balanced"
    return "balanced"


@router.post("/api/v2/empathetic_professional", response_model=RAGResponse)
async def empathetic_professional_v2(
    request_data: RAGRequest,
    request: Request,
    chat_service=Depends(get_chat_service),
):
    try:
        esq_cfg = _get_esq_config(request)
        rd = request_data

        tone = _normalize_tone(rd.type)
        # Try to forward tone to the service (backward compatible)
        try:
            answer, meta = await chat_service.handle_chat(rd, tone_override=tone)  # type: ignore[arg-type]
        except TypeError:
            answer, meta = await chat_service.handle_chat(rd)

        if not isinstance(answer, str):
            answer = str(answer)
        if not isinstance(meta, dict):
            meta = {}

        route = (meta.get("route") or "other").strip().lower()

        # Route-specific post-processing
        if route == "mh_support":
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
                final_text = fallback_esq(rd.question or "")
        else:
            # greeting / info_definition / crisis / other are already sanitized upstream
            final_text = (answer or "").strip()

        # Build the normal payload (RAGResponse-compatible)
        resp = RAGResponse(
            answer=final_text,
            question=rd.question,
            tone=tone,
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

        # Attach a lightweight meta for front-end debugging/telemetry
        meta_out = {
            "route": route,
            "route_score": meta.get("route_score"),
            "flow_active": meta.get("flow_active"),
            "repaired": meta.get("repaired"),
            "naturalized": meta.get("naturalized"),
            "session_id": meta.get("session_id"),
            "tone_used": tone,  # <= the field you asked for
        }

        content = resp.model_dump()
        content["meta"] = meta_out

        headers = {
            "X-Route": route,
            "X-Tone": tone,
        }
        return JSONResponse(status_code=200, content=content, headers=headers)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("v2 handler uncaught exception")
        raise HTTPException(status_code=500, detail=str(e))
