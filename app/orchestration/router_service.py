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

PROMPT = (
    "Classify the user's message into ONE label:\n"
    "G=greeting, M=mh_support, I=info_definition, O=other.\n"
    "Heuristics:\n"
    "- First-person feelings/symptoms, seeking coping/next steps -> M\n"
    "- Definitions, \"what is / explain / define\" -> I\n"
    "- Short hello/intro without problem -> G\n"
    "- Everything else -> O\n"
    "Output exactly two tokens: <LABEL> <CONFIDENCE>\n"
    "<LABEL> in {G,M,I,O}; <CONFIDENCE> in {1.00, 0.90}\n"
    "No extra text.\n\n"
    "Examples:\n"
    "User: Hi, I’m Max\n"
    "Answer: G 0.90\n\n"
    "User: At night I can’t fall asleep and keep overthinking\n"
    "Answer: M 0.90\n\n"
    "User: What is depression?\n"
    "Answer: I 0.90\n\n"
    "User: Give me expert advice for this week\n"
    "Answer: O 0.90\n\n"
    "User: {text}\n"
    "Answer:"
)

# Lightweight lexical override for common MH signals that LLM仍偶爾誤分
_MH_HINTS = (
    "can’t sleep", "cant sleep", "can't sleep", "overthinking",
    "panic", "anxious", "anxiety", "worry", "worrying",
    "feel sad", "feeling sad", "low mood", "stressed", "stress",
    "lonely", "hopeless",
)

class LLMRouter:
    def __init__(self):
        self.client = model_manager.get_router_client()

    async def classify(self, text: str) -> RouterDecision:
        t = (text or "").strip()
        prompt = PROMPT.format(text=t)
        try:
            out = await self.client.complete(
                prompt, temperature=0.0, top_p=0.0, max_new_tokens=12, max_time=6.0
            )
            lab_m = _LABEL_RE.search(out or "")
            conf_m = _CONF_RE.search(out or "")
            if not lab_m:
                raise ValueError("no label")
            label = lab_m.group(1)
            conf = float(conf_m.group(1)) if conf_m else 0.90
        except Exception as e:
            logger.warning("Router classify fallback due to parse: %s", e)
            low = t.lower()
            if any(x in low for x in ("hello", "hi ", "hi,", "hey")) and len(low) <= 40:
                label, conf = "G", 0.90
            elif any(x in low for x in ("what is", "explain", "define", "definition", "meaning of")):
                label, conf = "I", 0.90
            else:
                label, conf = "O", 0.50

        # override to M if strong MH hints are present
        low = t.lower()
        if label in ("G", "I", "O") and any(h in low for h in _MH_HINTS):
            label, conf = "M", max(conf, 0.90)

        route_map = {"G": "greeting", "M": "mh_support", "I": "info_definition", "O": "other"}
        route = route_map.get(label, "other")
        logger.info("Router classified '%s' -> %s (label=%s, conf=%.2f)", t[:30], route, label, conf)
        return RouterDecision(route=route, confidence=conf)
