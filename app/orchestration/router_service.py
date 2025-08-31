# app/orchestration/router_service.py
import json
import re
import time
import logging
from typing import Optional
from app.core.settings import settings
from app.clients.model_manager import model_manager
from app.schemas.route import RouteDecision

_JSON_RE = re.compile(r"\{.*?\}", re.S)
logger = logging.getLogger(__name__)

CLASSIFIER_PROMPT = """You are a router for a mental-health assistant.
Return STRICT JSON only as {{"route":"...", "confidence":0.0-1.0, "triggers":["..."]}}.

Routes:
- "greeting": Simple greetings like "hi", "hello", "hey" with no emotional content
- "small_talk": Thanks, bye, simple acknowledgments  
- "crisis": Self-harm, suicide, danger words
- "mh_support": Emotional distress, anxiety, depression, coping needs, therapy requests
- "info_definition": Questions about definitions or factual information
- "other": Everything else

Rules:
1. Simple greetings ("hi", "hello", "hey") are ALWAYS "greeting"
2. Crisis keywords override everything else
3. Anxiety, stress, emotional distress, coping strategies = "mh_support"
4. Only choose "mh_support" if there's clear emotional content or therapy request

USER: {user_text}
JSON:"""

def _extract_json(text: str) -> Optional[str]:
    m = _JSON_RE.search(text.replace("JSON:", "").replace("Human:", ""))
    return m.group(0) if m else None

class RouterCache:
    def __init__(self, ttl: int): 
        self.ttl, self.store = ttl, {}
    
    def get(self, key: str): 
        v = self.store.get(key)
        return v[0] if v and (time.time()-v[1] < self.ttl) else None
    
    def set(self, key, value): 
        self.store[key] = (value, time.time())

class LLMRouter:
    def __init__(self):
        # Use shared model manager instead of creating new client
        self.client = model_manager.get_router_client()
        self.cache = RouterCache(settings.router_cache_ttl_seconds) if settings.enable_router_cache else None

    async def classify(self, user_text: str) -> RouteDecision:
        # Check cache first
        key = user_text.strip().lower() if (self.cache and len(user_text) <= 15) else None
        if key:
            cached = self.cache.get(key)
            if cached: 
                logger.info(f"Router cache hit for: {user_text[:20]}...")
                return cached
        
        # LLM classification (first stage)
        logger.info(f"Stage-1 Router LLM classifying: {user_text[:30]}...")
        out = await self.client.complete(CLASSIFIER_PROMPT.format(user_text=user_text.strip()))
        payload = _extract_json(out)
        
        try:
            data = json.loads(payload) if payload else {}
            decision = RouteDecision(**data) if data else RouteDecision(route="other", confidence=0.0, triggers=[])
            logger.info(f"Stage-1 classified '{user_text[:30]}...' as '{decision.route}' (conf: {decision.confidence:.2f})")
        except Exception as e:
            logger.warning(f"Router parsing failed: {e}")
            decision = RouteDecision(route="other", confidence=0.0, triggers=[])
            
        if key and self.cache: 
            self.cache.set(key, decision)
            
        return decision
