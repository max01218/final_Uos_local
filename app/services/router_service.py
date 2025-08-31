# app/services/router_service.py
import json
import re
import time
import logging
from typing import Optional
from app.schemas.route import RouteDecision
from app.core.settings import settings
from app.clients.llm_client import LLMClient
from app.utils.router_prompts import CLASSIFIER_PROMPT

logger = logging.getLogger(__name__)

_JSON_RE = re.compile(r"\{.*\}", re.S)

class RouterCache:
    def __init__(self, ttl_seconds: int):
        self.ttl = ttl_seconds
        self.store = {}

    def get(self, key: str) -> Optional[RouteDecision]:
        item = self.store.get(key)
        if not item:
            return None
        value, ts = item
        if time.time() - ts > self.ttl:
            self.store.pop(key, None)
            return None
        return value

    def set(self, key: str, value: RouteDecision):
        self.store[key] = (value, time.time())

class LLMRouter:
    def __init__(self):
        self.client = LLMClient(
            model_id=settings.router_model_id,
            temperature=settings.router_temperature,
            top_p=settings.router_top_p,
            max_new_tokens=settings.router_max_new_tokens,
        )
        self.cache = RouterCache(settings.router_cache_ttl_seconds) if settings.enable_router_cache else None

    async def classify(self, user_text: str) -> RouteDecision:
        key = None
        if self.cache and len(user_text) <= 15:
            key = f"router::{user_text.strip().lower()}"
            cached = self.cache.get(key)
            if cached:
                logger.info(f"Router cache hit for: {user_text[:20]}...")
                return cached

        prompt = CLASSIFIER_PROMPT.format(user_text=user_text.strip())
        try:
            out = await self.client.complete(prompt)
            
            # Extract JSON from response
            m = _JSON_RE.search(out)
            payload = m.group(0) if m else out
            
            try:
                data = json.loads(payload)
                decision = RouteDecision(**data)
                logger.info(f"Router classified '{user_text[:30]}...' as '{decision.route}' (conf: {decision.confidence:.2f})")
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Failed to parse router JSON response: {e}. Raw output: {out}")
                decision = RouteDecision(route="other", confidence=0.0, triggers=[])
                
        except Exception as e:
            logger.error(f"Router classification failed: {e}")
            decision = RouteDecision(route="other", confidence=0.0, triggers=[])

        if key and self.cache:
            self.cache.set(key, decision)
            
        return decision
