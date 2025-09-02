import logging
import re
from dataclasses import dataclass
from typing import Optional

from app.clients.model_manager import model_manager

logger = logging.getLogger(__name__)


# ----- Helpers -----

def _normalize_text(t: str) -> str:
    """Lowercase and keep only a-z letters as space-separated tokens."""
    t = (t or "").lower()
    t = re.sub(r"[^a-z]+", " ", t)
    return t.strip()


def _tokenize(t: str) -> list[str]:
    return [w for w in (t or "").split() if w]


def _contains_any(haystack: str, phrases: set[str]) -> int:
    """Count how many phrases appear in normalized text (space-separated)."""
    count = 0
    for p in phrases:
        if " " in p:
            # multi-word phrase
            if p in haystack:
                count += 1
        else:
            # single token
            if f" {p} " in f" {haystack} ":
                count += 1
    return count


def _is_greeting(tokens: list[str]) -> bool:
    """A minimal heuristic greeting: very short and only greeting-like tokens."""
    if not tokens:
        return False
    if len(tokens) > 6:
        return False
    allow = {
        "hi", "hello", "hey", "yo",
        "i", "im", "i’m", "am", "name", "is",
        "my", "max", "there"
    }
    return all(t in allow for t in tokens)


# ----- Router via LLM (fallback only) -----

PROMPT = (
    "You are a message router. Classify the USER message into exactly one label:\n"
    "- G = greeting or small talk (simple hello/intro)\n"
    "- M = mental-health support request (feelings, sleep trouble, anxiety, overthinking, coping help)\n"
    "- I = information/definition/explanation request (what is, explain, definition, symptoms)\n"
    "- O = other\n\n"
    "Return ONLY ONE LETTER (G/M/I/O).\n\n"
    "USER: {text}\n"
    "LABEL:"
)

_LABEL_MAP = {
    "G": "greeting",
    "M": "mh_support",
    "I": "info_definition",
    "O": "other",
}


@dataclass
class Decision:
    route: str
    confidence: float
    label: str


class LLMRouter:
    def __init__(self, client: Optional[object] = None) -> None:
        self.client = client or model_manager.get_router_client()
        try:
            name = getattr(self.client, "model_name", "unknown")
        except Exception:
            name = "unknown"
        logger.info("[router] Router using model: %s", name)

        # keyword sets (normalized, lowercased)
        self.mh_terms = {
            "sad", "sadness", "down", "low", "empty", "numb",
            "overthink", "ruminate", "anxiety", "anxious", "panic",
            "insomnia", "cant", "cannot", "sleep", "burnout",
            "overwhelmed", "stressed", "worry", "worried", "fear",
            "hopeless", "no", "motivation"
        }
        self.defn_terms = {
            "what", "is", "explain", "definition", "define",
            "difference", "symptoms", "signs", "how", "does", "work"
        }

    async def classify(self, text: str) -> Decision:
        raw = text or ""
        norm = _normalize_text(raw)
        tokens = _tokenize(norm)

        # Heuristic: greeting
        if _is_greeting(tokens):
            logger.info("Router classified '%s' -> greeting (heuristic)", raw[:30])
            return Decision(route="greeting", confidence=0.90, label="G")

        # Heuristic: MH vs Info by keyword counts
        mh_hits = _contains_any(f" {norm} ", self.mh_terms)
        defn_hits = _contains_any(f" {norm} ", self.defn_terms)

        if mh_hits >= 1 and mh_hits > defn_hits:
            logger.info("Router classified '%s' -> mh_support (heuristic)", raw[:30])
            return Decision(route="mh_support", confidence=0.90, label="M")

        if defn_hits >= 1:
            logger.info("Router classified '%s' -> info_definition (heuristic)", raw[:30])
            return Decision(route="info_definition", confidence=0.90, label="I")

        # Fallback to LLM single-letter response
        try:
            prompt = PROMPT.format(text=raw)
            resp = await self.client.complete(prompt, temperature=0.0, top_p=1.0, max_new_tokens=4, max_time=6.0)
            letter = None
            for ch in (resp or "").strip():
                up = ch.upper()
                if up in _LABEL_MAP:
                    letter = up
                    break
            if not letter:
                letter = "O"
            route = _LABEL_MAP[letter]
            logger.info("Router classified '%s' -> %s (label=%s, conf=0.50)", raw[:30], route, letter)
            return Decision(route=route, confidence=0.50, label=letter)
        except Exception as e:
            logger.warning("Router LLM call failed: %s; defaulting to 'other'", e)
            return Decision(route="other", confidence=0.50, label="O")
