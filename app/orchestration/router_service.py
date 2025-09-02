# app/orchestration/router_service.py
import json
import logging
import re
from dataclasses import dataclass
from typing import Tuple

from app.clients.model_manager import model_manager

logger = logging.getLogger(__name__)

_ALLOWED = [
    "greeting", "small_talk", "mh_support", "info_definition",
    "crisis", "other"
]

_JSON_RE = re.compile(r"\{.*\}", re.S)
CRISIS_RE = re.compile(r"(suicid(e|al)|kill myself|self[- ]?harm|end my life|homicide|kill (him|her|them))", re.I)

PROMPT = """You are a routing classifier for a mental health assistant.
Read the user's message and output a SINGLE LINE of strict JSON with keys:
- "route": one of ["greeting","small_talk","mh_support","info_definition","crisis","other"]
- "confidence": float between 0 and 1

Return ONLY JSON. No prose.

User: {text}
JSON:
"""

@dataclass
class RouteDecision:
    route: str
    score: float

class LLMRouter:
    def __init__(self):
        self.client = model_manager.get_router_client()

    async def classify(self, text: str) -> RouteDecision:
        # hard crisis gate always wins
        if CRISIS_RE.search(text or ""):
            return RouteDecision("crisis", 1.0)

        prompt = PROMPT.format(text=(text or "").strip())
        raw = await self.client.complete(prompt, temperature=0.0, top_p=1.0, max_new_tokens=96, max_time=8.0)

        route, score = self._extract_json(raw)
        if route not in _ALLOWED:
            logger.warning("Router produced invalid route '%s'; using 'other'", route)
            route = "other"
        return RouteDecision(route, score)

    def _extract_json(self, s: str) -> Tuple[str, float]:
        try:
            m = _JSON_RE.search(s or "")
            data = json.loads(m.group(0)) if m else json.loads(s)
            route = str(data.get("route", "other"))
            conf = float(data.get("confidence", 0.0))
            return route, max(0.0, min(1.0, conf))
        except Exception as e:
            logger.warning("Router JSON parse failed: %s; applying simple heuristic", e)
            t = (s or "").lower()
            # heuristic (still considered "classified", no cross-route fallback)
            if any(x in t for x in ["hi", "hello", "hey"]):
                return "greeting", 0.5
            if any(x in t for x in ["how are you", "what's up"]):
                return "small_talk", 0.6
            if any(x in t for x in ["depress", "anxious", "sad", "panic", "lonely"]):
                return "mh_support", 0.7
            if any(x in t for x in ["what is", "define", "meaning of"]):
                return "info_definition", 0.6
            return "other", 0.0
