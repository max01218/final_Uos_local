# app/orchestration/router_service.py
import re
import logging
from dataclasses import dataclass
from app.clients.model_manager import model_manager

logger = logging.getLogger(__name__)

@dataclass
class RouteDecision:
    route: str
    confidence: float

def _map_label_to_route(label: str) -> str:
    mapping = {"G": "greeting", "M": "mh_support", "I": "info_definition", "O": "other"}
    return mapping.get(label.upper(), "other")

def _parse_label(text: str) -> str | None:
    if not text:
        return None
    m = re.search(r"\b([GMIO])\b", text.strip(), re.I)
    return m.group(1).upper() if m else None

# ---------- Robust MH heuristic (ASCII-only, regex-safe) ----------
# word stems (plain regex, word-bounded)
MH_STEMS_RE = r"\b(?:sad|depress|anxiet|anxious|panic|overthink|insomnia|lonely|worthless|empty|burnout|stressed|therapy|therapist|cbt|cope|coping)\b"

# phrases with spaces/apostrophes — escape each safely
_MH_PHRASES = ["low mood", "can't sleep", "cant sleep"]
_MH_PHRASES_ESC = [re.escape(p) for p in _MH_PHRASES]
MH_HINTS = re.compile(r"(?:%s|%s)" % (MH_STEMS_RE, "|".join(_MH_PHRASES_ESC)), re.I)
# ------------------------------------------------------------------

def _fallback_label_by_words(text: str) -> str:
    t = (text or "").lower()
    if any(w in t for w in ("hi", "hello", "hey", "nice to meet you", "my name is")):
        return "G"
    if re.search(MH_HINTS, t):
        return "M"
    if any(w in t for w in ("what is", "define", "meaning of", "explain", "symptom", "treatment", "cbt", "depression")):
        return "I"
    return "O"

# NOTE: Escape braces for str.format with double braces {{ }}
PROMPT = (
    "You are a router that classifies the user's message into EXACTLY ONE label.\n"
    "Labels:\n"
    "  G = greeting (hello/introductions)\n"
    "  M = mh_support (personal mental-health concerns, feelings, coping help)\n"
    "  I = info_definition (factual/definition/explanation about mental health)\n"
    "  O = other (anything else)\n"
    "Output ONLY one letter from this set: {{G,M,I,O}}.\n\n"
    "Text: {text}\n"
    "Answer:"
)

class LLMRouter:
    def __init__(self):
        self.client = model_manager.get_router_client()
        logger.info("[router] Router using model: %s", getattr(self.client, "model_name", "unknown"))

    async def classify(self, text: str) -> RouteDecision:
        t = (text or "").strip()
        try:
            prompt = PROMPT.format(text=t)
        except KeyError as e:
            logger.error("Router prompt format error: %s", e)
            prompt = PROMPT.replace("{text}", t).replace("{", "{{").replace("}", "}}")
            prompt = prompt.replace("{{text}}", t)

        try:
            resp = await self.client.complete(
                prompt,
                temperature=0.0,
                top_p=1.0,
                max_new_tokens=8,
                max_time=4.0,
            )
        except Exception as e:
            logger.warning("[router] LLM call failed: %s; using heuristic", e)
            label = _fallback_label_by_words(t)
            # force M when strong MH cues are present
            if label != "M" and re.search(MH_HINTS, t):
                label = "M"
            return RouteDecision(route=_map_label_to_route(label), confidence=0.6)

        label = _parse_label(resp)
        if not label:
            # try to infer from response content; fallback to original text too
            label = _fallback_label_by_words((resp or "") + " " + t)

        # final override to M if text clearly contains MH cues
        if label != "M" and re.search(MH_HINTS, t):
            label = "M"

        route = _map_label_to_route(label)
        conf = 0.9 if label in ("G", "M", "I", "O") else 0.6
        logger.info(
            "Router classified %r -> %s (label=%s, conf=%.2f)",
            t[:40] + ("..." if len(t) > 40 else ""),
            route, label, conf
        )
        return RouteDecision(route=route, confidence=conf)
