# app/orchestration/router_service.py
import json, re, time, logging
from typing import Optional, Any, Dict, Tuple
from app.clients.model_manager import model_manager
from app.schemas.route import RouteDecision

logger = logging.getLogger(__name__)
_JSON_RE = re.compile(r"\{.*?\}", re.S)
_GREETING_RE = re.compile(r"^\s*(hi|hello|hey|yo|good\s+(morning|afternoon|evening))\W*$", re.I)

class RouterCache:
    def __init__(self, ttl_seconds: int = 30):
        self.ttl = ttl_seconds
        self._store: Dict[str, Tuple[float, RouteDecision]] = {}

    def get(self, key: str) -> Optional[RouteDecision]:
        now = time.time()
        item = self._store.get(key)
        if not item: return None
        exp, val = item
        if now > exp:
            self._store.pop(key, None)
            return None
        return val

    def set(self, key: str, value: RouteDecision) -> None:
        self._store[key] = (time.time() + self.ttl, value)

CLASSIFIER_PROMPT = """You are a router for a mental-health assistant.
Return JSON only as {"route":"...", "confidence":0.0-1.0, "triggers":["..."]}.
- route ∈ {greeting, small_talk, crisis, mh_support, info_definition, other}
- confidence ∈ [0,1] (float)
- triggers is a JSON array (can be [])

Examples:
INPUT: "hi"
OUTPUT: {"route":"greeting","confidence":0.95,"triggers":[]}

INPUT: "what is depression?"
OUTPUT: {"route":"info_definition","confidence":0.90,"triggers":["definition"]}

INPUT: "I feel really sad recently, what can I do?"
OUTPUT: {"route":"mh_support","confidence":0.86,"triggers":["low_mood"]}

USER: {user_text}
JSON:"""

def _extract_json(text: str) -> Optional[str]:
    if not text: return None
    cleaned = text.replace("```json","").replace("```","").replace("JSON:","").strip()
    m = _JSON_RE.search(cleaned)
    return m.group(0) if m else None

class LLMRouter:
    def __init__(self, cache_ttl: int = 30):
        self.client = model_manager.get_router_client()
        self.cache = RouterCache(cache_ttl) if cache_ttl > 0 else None

    async def classify(self, user_text: str) -> RouteDecision:
        text = (user_text or "").strip()
        # 只作為回退的猜測（仍然會呼叫 LLM）
        default_guess = "greeting" if _GREETING_RE.match(text) else "other"

        key = text.lower()
        if self.cache and len(text) <= 15:
            hit = self.cache.get(key)
            if hit:
                logger.info(f"Router cache hit for: {text[:20]}...")
                return hit

        logger.info(f"Stage-1 Router LLM classifying: {text[:30]}...")
        try:
            raw_out = await self.client.complete(CLASSIFIER_PROMPT.format(user_text=text),
                                                temperature=0.1, max_new_tokens=80)
        except Exception as e:
            logger.warning(f"Router LLM call failed: {e}; falling back to '{default_guess}'")
            dec = RouteDecision(route=default_guess,
                                confidence=0.5 if default_guess=="greeting" else 0.0,
                                triggers=[])
            if self.cache and len(text) <= 15: self.cache.set(key, dec)
            return dec

        payload = _extract_json(raw_out)
        try:
            obj: Dict[str, Any] = json.loads(payload) if payload else {}
            label = obj.get("route") or obj.get("label") or obj.get("class") or default_guess
            conf = obj.get("confidence")
            if conf is None: conf = obj.get("score", obj.get("prob", 0.0))
            try: conf = float(conf)
            except Exception: conf = 0.0
            triggers = obj.get("triggers") or []
            if not isinstance(triggers, list): triggers = [str(triggers)]
            dec = RouteDecision(route=label, confidence=conf, triggers=triggers)
            try: setattr(dec, "score", conf)
            except Exception: pass
            logger.info(f"Stage-1 classified '{text[:30]}...' as '{dec.route}' (conf: {dec.confidence:.2f})")
        except Exception as e:
            logger.warning(f"Router parsing failed: {e}; raw_out={raw_out!r}")
            dec = RouteDecision(route=default_guess,
                                confidence=0.5 if default_guess=="greeting" else 0.0,
                                triggers=[])
            try: setattr(dec, "score", dec.confidence)
            except Exception: pass

        if self.cache and len(text) <= 15: self.cache.set(key, dec)
        return dec
