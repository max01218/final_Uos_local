# app/orchestration/router_service.py
import logging
import re
from dataclasses import dataclass
from typing import Tuple

from app.clients.model_manager import model_manager

logger = logging.getLogger(__name__)

# Crisis hard gate
CRISIS_RE = re.compile(r"(suicid(e|al)|kill myself|self[- ]?harm|end my life|homicide|kill (him|her|them))", re.I)

# Allowed single-char labels -> canonical routes
LABEL2ROUTE = {
    "G": "greeting",        # greeting
    "S": "small_talk",      # small talk
    "M": "mh_support",      # mental-health support
    "I": "info_definition", # information / definition
    "C": "crisis",          # safety-critical
    "O": "other",           # other
}
ALLOWED = set(LABEL2ROUTE.keys())

PROMPT = """You are a strict router for a mental-health assistant.
Classify the user's message into ONE label:
G=greeting, S=small_talk, M=mh_support, I=info_definition, C=crisis, O=other.

Return EXACTLY ONE letter from [G,S,M,I,C,O].
No words, no punctuation, no markdown, no explanation.

User: {text}
Answer:
"""

@dataclass
class RouteDecision:
    route: str
    score: float  # simple confidence; 0..1

class LLMRouter:
    def __init__(self):
        self.client = model_manager.get_router_client()

    async def classify(self, text: str) -> RouteDecision:
        user_text = (text or "").strip()

        # 0) Hard crisis gate wins immediately
        if CRISIS_RE.search(user_text):
            return RouteDecision("crisis", 1.0)

        # 1) Ask LLM for ONE-LETTER label
        prompt = PROMPT.format(text=user_text)
        raw = await self.client.complete(
            prompt,
            temperature=0.0,
            top_p=1.0,
            max_new_tokens=2,
            stop=["\n", " ", "\t", "."],
            max_time=6.0,
        )

        label, route = self._parse_label(raw)

        # 2) If LLM gives a valid label, accept (high confidence)
        if route is not None:
            conf = 1.0 if label == "C" else 0.9
            logger.info("Router classified '%s' -> %s (label=%s, conf=%.2f)",
                        user_text[:40] + ("..." if len(user_text) > 40 else ""),
                        route, label, conf)
            return RouteDecision(route, conf)

        # 3) Fallback heuristic — on USER TEXT (not model raw)
        route, conf = self._heuristic(user_text)
        logger.warning("Router label parse failed (raw='%s'); heuristic -> %s (conf=%.2f)",
                       raw.strip().replace("\n", "\\n")[:80], route, conf)
        return RouteDecision(route, conf)

    # ----- helpers -----

    def _parse_label(self, raw: str) -> Tuple[str, str | None]:
        """
        Extract the first A-Z letter and map it to our routes.
        Returns (label, route-or-None).
        """
        if not raw:
            return "", None
        # take first non-space ASCII letter
        for ch in raw.strip():
            up = ch.upper()
            if up in ALLOWED:
                return up, LABEL2ROUTE.get(up)
            # if model accidentally returned a word, try first letter of it
            if up.isalpha():
                # keep scanning, but only letters in ALLOWED count
                continue
        return "", None

    def _heuristic(self, user_text: str) -> Tuple[str, float]:
        t = (user_text or "").lower()

        # crisis (second guard)
        if CRISIS_RE.search(t):
            return "crisis", 1.0

        # prioritize mental health cues before greetings
        if re.search(r"\b(sad|depress|anxious|anxiety|panic|overthink|can[’']?t sleep|insomnia|lonely|stress|stressed|worry|hopeless)\b", t):
            return "mh_support", 0.8

        if re.search(r"\b(what is|define|meaning of)\b", t):
            return "info_definition", 0.7

        if re.search(r"\b(how are you|what'?s up)\b", t):
            return "small_talk", 0.6

        if re.search(r"\b(hi|hello|hey|good (morning|afternoon|evening))\b", t):
            return "greeting", 0.5

        return "other", 0.4
