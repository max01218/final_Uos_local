# app/orchestration/router_service.py
from __future__ import annotations
import re
import logging
from dataclasses import dataclass
from typing import Optional
from app.clients.model_manager import model_manager

logger = logging.getLogger(__name__)

@dataclass
class RouterDecision:
    route: str
    confidence: float

_LABEL_RE = re.compile(r"\b([GMSIO])\b")
_CONF_RE = re.compile(r"\b(1\.00|0\.\d{2})\b")

# We intentionally do NOT classify "crisis" here.
# Crisis is handled upstream by hard keyword gate in Orchestrator.

PROMPT = (
    "Classify the user's message into ONE label:\n"
    "G=greeting, M=mh_support, I=info_definition, O=other.\n"
    "Rules:\n"
    "- Output exactly two tokens separated by space: <LABEL> <CONFIDENCE>\n"
    "- <LABEL> is one of G/M/I/O.\n"
    "- <CONFIDENCE> is 0.90 or 1.00.\n"
    "- No extra text.\n\n"
    "User: {text}\n"
    "Answer:"
)

class LLMRouter:
    def __init__(self):
        self.client = model_manager.get_router_client()

    async def classify(self, text: str) -> RouterDecision:
        prompt = PROMPT.format(text=text.strip())
        try:
            out = await self.client.complete(prompt, temperature=0.0, top_p=0.0, max_new_tokens=12, max_time=6.0)
            lab_m = _LABEL_RE.search(out or "")
            conf_m = _CONF_RE.search(out or "")
            if not lab_m:
                raise ValueError("no label")
            label = lab_m.group(1)
            conf = float(conf_m.group(1)) if conf_m else 0.90
        except Exception as e:
            logger.warning("Router classify fallback due to parse: %s", e)
            # trivial heuristic fallback
            t = (text or "").lower()
            if any(x in t for x in ("hello", "hi", "hey")) and len(t) <= 40:
                label, conf = "G", 0.90
            elif any(x in t for x in ("what is", "explain", "define", "definition", "meaning of")):
                label, conf = "I", 0.90
            else:
                label, conf = "O", 0.50

        route_map = {"G": "greeting", "M": "mh_support", "I": "info_definition", "O": "other"}
        route = route_map.get(label, "other")
        logger.info("Router classified '%s' -> %s (label=%s, conf=%.2f)", text[:30], route, label, conf)
        return RouterDecision(route=route, confidence=conf)
