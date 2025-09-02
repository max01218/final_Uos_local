# app/orchestration/router_service.py
import re
import logging
from dataclasses import dataclass
from app.clients.model_manager import model_manager

logger = logging.getLogger(__name__)

# Heuristic keywords to force mh_support when LLM is unsure/mislabels
MH_HINTS = re.compile(
    r"(sad|depress|anxiet|anxious|panic|overthink|can[’']?t sleep|insomnia|lonely|"
    r"worthless|low mood|empty|burnout|stressed|therapy|therapist|cbt|cope|coping|"
    , re.I
)

@dataclass
class RouteDecision:
    route: str
    confidence: float

def _map_label_to_route(label: str) -> str:
    # G: greeting, M: mh_support, I: info_definition, O: other
    m = {"G": "greeting", "M": "mh_support", "I": "info_definition", "O": "other"}
    return m.get(label.upper(), "other")

def _parse_label(text: str) -> str | None:
    """Extract first label letter from model output."""
    if not text:
        return None
    # common patterns: "G", "Label: G", "Answer: M", "G - greeting"
    m = re.search(r"\b([GMIO])\b", text.strip(), re.I)
    return m.group(1).upper() if m else None

def _fallback_label_by_words(text: str) -> str:
    """Very light heuristic fallback if LLM output is noisy."""
    t = (text or "").lower()
    if any(w in t for w in ("hi", "hello", "hey", "nice to meet you", "my name is")):
        return "G"
    if re.search(MH_HINTS, t):
        return "M"
    if any(w in t for w in ("what is", "define", "meaning of", "explain", "symptom", "treatment", "cbt", "depression")):
        return "I"
    return "O"

# NOTE: escape braces for str.format — use double braces {{ }} around the label set.
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
            # Safety: if future edits introduce braces again
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
            # mh override
            if label != "M" and re.search(MH_HINTS, t):
                label = "M"
            return RouteDecision(route=_map_label_to_route(label), confidence=0.6)

        label = _parse_label(resp)
        if not label:
            label = _fallback_label_by_words(resp + " " + t)

        # Heuristic override to mh_support if strong signals present
        if label != "M" and re.search(MH_HINTS, t):
            label = "M"

        route = _map_label_to_route(label)
        conf = 0.9 if label in ("G", "M", "I", "O") else 0.6
        logger.info("Router classified %r -> %s (label=%s, conf=%.2f)", t[:40] + ("..." if len(t) > 40 else ""), route, label, conf)
        return RouteDecision(route=route, confidence=conf)
