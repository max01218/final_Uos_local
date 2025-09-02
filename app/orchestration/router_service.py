# app/orchestration/router_service.py
import json
import re
import time
import logging
from typing import Optional, Any, Dict, Tuple

from app.clients.model_manager import model_manager
from app.schemas.route import RouteDecision

logger = logging.getLogger(__name__)
VERSION = "router-r3"  # bump this if you need to confirm reloads

_JSON_OBJ = re.compile(r"\{.*?\}", re.S)
_GREETING = re.compile(r"^\s*(hi|hello|hey|yo|good\s+(morning|afternoon|evening))\W*$", re.I)
ROUTES = {"greeting", "small_talk", "crisis", "mh_support", "info_definition", "other"}


def _extract_json_obj(text: str) -> Optional[str]:
    if not text:
        return None
    cleaned = str(text).replace("```json", "").replace("```", "").replace("JSON:", "").strip()
    m = _JSON_OBJ.search(cleaned)
    return m.group(0) if m else None


class _Cache:
    def __init__(self, ttl: int = 30):
        self.ttl = ttl
        self._m: Dict[str, Tuple[float, RouteDecision]] = {}

    def get(self, k: str) -> Optional[RouteDecision]:
        now = time.time()
        item = self._m.get(k)
        if not item:
            return None
        exp, val = item
        if now > exp:
            self._m.pop(k, None)
            return None
        return val

    def set(self, k: str, v: RouteDecision) -> None:
        self._m[k] = (time.time() + self.ttl, v)


CLASSIFIER_PROMPT = """You are a router for a mental-health assistant.
Return JSON only as {"route":"...", "confidence":0.0-1.0, "triggers":["..."]}.
- route in: greeting, small_talk, crisis, mh_support, info_definition, other
- confidence: float in [0,1]
- triggers: JSON array (can be [])

Examples:
INPUT: "hi"
OUTPUT: {"route":"greeting","confidence":0.95,"triggers":[]}

INPUT: "what is depression?"
OUTPUT: {"route":"info_definition","confidence":0.90,"triggers":["definition"]}

INPUT: "I feel really sad recently, what can I do?"
OUTPUT: {"route":"mh_support","confidence":0.86,"triggers":["low_mood"]}

If you are unsure, output exactly: {"route":"other","confidence":0.0,"triggers":[]}

USER: {user_text}
JSON:"""

MINI_PROMPT = """Choose one label for the user message:
[greeting, small_talk, crisis, mh_support, info_definition, other]
Return ONLY the label word.

USER: {user_text}
LABEL:"""


class LLMRouter:
    def __init__(self, cache_ttl: int = 30):
        self.client = model_manager.get_router_client()
        logger.info(f"[{VERSION}] Router using model: {getattr(self.client, 'model_id', 'unknown')}")
        self.cache = _Cache(cache_ttl) if cache_ttl > 0 else None

    async def classify(self, user_text: str) -> RouteDecision:
        text = (user_text or "").strip()
        default_guess = "greeting" if _GREETING.match(text) else "other"

        key = text.lower()
        if self.cache and len(text) <= 15:
            hit = self.cache.get(key)
            if hit:
                logger.info(f"[{VERSION}] Router cache hit for: {text[:20]}...")
                return hit

        logger.info(f"Stage-1 Router LLM classifying: {text[:30]}...")

        # ---- 1) call only (no parsing in this try) ----
        try:
            raw = await self.client.complete(
                CLASSIFIER_PROMPT.format(user_text=text),
                temperature=0.1, top_p=0.9, max_new_tokens=80, max_time=8.0,
            )
            try:
                logger.debug(f"[{VERSION}] Router raw: {(raw or '')[:400].replace(chr(10), ' ')}")
            except Exception:
                pass
        except Exception as e:
            logger.warning(f"[{VERSION}] Router call failed: {e!r}; fallback='{default_guess}'")
            dec = RouteDecision(
                route=default_guess,
                confidence=0.5 if default_guess == "greeting" else 0.0,
                triggers=[],
            )
            if self.cache and len(text) <= 15:
                self.cache.set(key, dec)
            return dec

        # ---- 2) parse & normalize (separate try) ----
        try:
            obj: Dict[str, Any]
            if isinstance(raw, dict):
                obj = raw
            else:
                payload = _extract_json_obj(raw)
                obj = json.loads(payload) if payload else {}

            label = (obj.get("route") or obj.get("label") or obj.get("class") or "").strip().lower()
            conf_raw = obj.get("confidence", obj.get("score", obj.get("prob", 0.0)))
            try:
                conf = float(conf_raw)
            except Exception:
                conf = 0.0
            triggers = obj.get("triggers") or []
            if not isinstance(triggers, list):
                triggers = [str(triggers)]

            if label not in ROUTES:
                raise KeyError("route missing/invalid")

            dec = RouteDecision(route=label, confidence=conf, triggers=triggers)
            try:
                setattr(dec, "score", conf)
            except Exception:
                pass
            logger.info(f"[{VERSION}] Stage-1 classified '{text[:30]}...' as '{dec.route}' (conf: {dec.confidence:.2f})")
        except Exception as e:
            logger.warning(f"[{VERSION}] Router parsing failed: {e!r}; trying mini router...")
            # ---- 3) mini backup classifier ----
            try:
                mini = await self.client.complete(
                    MINI_PROMPT.format(user_text=text),
                    temperature=0.0, max_new_tokens=6, max_time=4.0,
                )
                lbl = (mini or "").strip().split()[0].strip(",. ").lower()
                if lbl not in ROUTES:
                    lbl = default_guess
                dec = RouteDecision(
                    route=lbl,
                    confidence=0.6 if lbl == "greeting" else 0.5 if lbl == "mh_support" else 0.0,
                    triggers=[],
                )
                try:
                    setattr(dec, "score", dec.confidence)
                except Exception:
                    pass
            except Exception as e2:
                logger.warning(f"[{VERSION}] Mini router failed: {e2!r}; using default '{default_guess}'")
                dec = RouteDecision(
                    route=default_guess,
                    confidence=0.5 if default_guess == "greeting" else 0.0,
                    triggers=[],
                )

        if self.cache and len(text) <= 15:
            self.cache.set(key, dec)
        return dec
