# app/orchestration/router_service.py
import json
import re
import time
import logging
from typing import Optional, Any, Dict, Tuple

from app.clients.model_manager import model_manager
from app.schemas.route import RouteDecision

logger = logging.getLogger(__name__)

# 抓出第一個 JSON 物件
_JSON_RE = re.compile(r"\{.*?\}", re.S)

# 用於回退預測（不是短路）：若像招呼語，解析失敗時偏向 greeting
_GREETING_RE = re.compile(
    r"^\s*(hi|hello|hey|yo|good\s+(morning|afternoon|evening))\W*$", re.I
)


class RouterCache:
    """極簡 TTL cache：只用在短字串輸入上避免重算。"""
    def __init__(self, ttl_seconds: int = 30):
        self.ttl = ttl_seconds
        self._store: Dict[str, Tuple[float, RouteDecision]] = {}

    def get(self, key: str) -> Optional[RouteDecision]:
        now = time.time()
        item = self._store.get(key)
        if not item:
            return None
        exp, val = item
        if now > exp:
            self._store.pop(key, None)
            return None
        return val

    def set(self, key: str, value: RouteDecision) -> None:
        self._store[key] = (time.time() + self.ttl, value)


CLASSIFIER_PROMPT = """You are a router for a mental-health assistant.
Return STRICT JSON only as {"route":"...", "confidence":0.0-1.0, "triggers":["..."]}.
- The "route" must be one of: greeting, small_talk, crisis, mh_support, info_definition, other.
- "confidence" is a float between 0 and 1.
- "triggers" must be a JSON array (can be empty).
- Output JSON only, no extra text.

Routes:
- greeting: brief salutations or introductions, no emotional content.
- small_talk: short acknowledgements or farewells.
- crisis: self-harm, other-harm, imminent danger, plan or means.
- mh_support: emotional/psychological concerns, coping/therapy requests.
- info_definition: direct requests for definitions or factual information.
- other: everything else.

USER: {user_text}
JSON:"""


def _extract_json(text: str) -> Optional[str]:
    """Extract the first JSON object from LLM raw text."""
    if not text:
        return None
    # 移除常見前綴再抓 JSON
    cleaned = text.replace("```json", "").replace("```", "")
    cleaned = cleaned.replace("JSON:", "").replace("Human:", "").strip()
    m = _JSON_RE.search(cleaned)
    return m.group(0) if m else None


class LLMRouter:
    """負責呼叫分類 LLM，並把輸出標準化為 RouteDecision。"""

    def __init__(self, cache_ttl: int = 30):
        self.client = model_manager.get_router_client()
        self.cache = RouterCache(cache_ttl) if cache_ttl > 0 else None

    async def classify(self, user_text: str) -> RouteDecision:
        text = (user_text or "").strip()

        # 回退用的「最合理預測」：不是短路，僅在解析失敗時使用
        default_guess = "greeting" if _GREETING_RE.match(text) else "other"

        # 短輸入可以快取（<= 15 字元）
        key = text.lower()
        if self.cache and len(text) <= 15:
            cached = self.cache.get(key)
            if cached:
                logger.info(f"Router cache hit for: {text[:20]}...")
                return cached

        logger.info(f"Stage-1 Router LLM classifying: {text[:30]}...")
        try:
            raw_out = await self.client.complete(CLASSIFIER_PROMPT.format(user_text=text))
        except Exception as e:
            logger.warning(f"Router LLM call failed: {e}; falling back to '{default_guess}'")
            decision = RouteDecision(route=default_guess,
                                     confidence=0.5 if default_guess == "greeting" else 0.0,
                                     triggers=[])
            if self.cache and len(text) <= 15:
                self.cache.set(key, decision)
            return decision

        payload = _extract_json(raw_out)

        try:
            raw: Dict[str, Any] = json.loads(payload) if payload else {}
            # ---- 標準化鍵名：確保有 route / confidence / triggers ----
            label = raw.get("route") or raw.get("label") or raw.get("class") or default_guess

            conf = raw.get("confidence")
            if conf is None:
                conf = raw.get("score", raw.get("prob", 0.0))
            try:
                conf = float(conf)
            except Exception:
                conf = 0.0

            triggers = raw.get("triggers") or []
            if not isinstance(triggers, list):
                triggers = [str(triggers)]

            decision = RouteDecision(route=label, confidence=conf, triggers=triggers)

            # 兼容：提供 .score 屬性（部分舊程式會讀 decision.score）
            try:
                setattr(decision, "score", conf)
            except Exception:
                pass

            logger.info(
                f"Stage-1 classified '{text[:30]}...' as '{decision.route}' (conf: {decision.confidence:.2f})"
            )
        except Exception as e:
            logger.warning(f"Router parsing failed: {e}; raw_out={raw_out!r}")
            decision = RouteDecision(route=default_guess,
                                     confidence=0.5 if default_guess == "greeting" else 0.0,
                                     triggers=[])
            try:
                setattr(decision, "score", decision.confidence)
            except Exception:
                pass

        if self.cache and len(text) <= 15:
            self.cache.set(key, decision)
        return decision
