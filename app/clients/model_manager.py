# app/clients/model_manager.py
"""
Centralized model management to avoid duplicate loading
"""
import logging
from typing import Dict, Optional
from app.clients.llm_client import LLMClient
from app.core.settings import settings

logger = logging.getLogger(__name__)

class ModelManager:
    """Singleton model manager to share models across services"""
    _instance = None
    _models: Dict[str, LLMClient] = {}
    _prewarmed = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
        return cls._instance
    
    def prewarm_models(self):
        """Pre-initialize models at startup to reduce first-request latency"""
        if self._prewarmed:
            return
            
        logger.info("Pre-warming models for faster response times...")
        
        # Pre-initialize router model
        router_client = self.get_router_client()
        logger.info("Router model ready")
        
        # Pre-initialize main model
        main_client = self.get_main_client()
        logger.info("Main model ready")
        
        self._prewarmed = True
        logger.info("All models pre-warmed successfully")
    
    def get_router_client(self) -> LLMClient:
        """Get router model client (shared)"""
        key = f"router_{settings.router_model_id}"
        if key not in self._models:
            logger.info(f"Creating router client: {settings.router_model_id}")
            client = LLMClient(
                model_id=settings.router_model_id,
                temperature=settings.router_temperature,
                top_p=settings.router_top_p,
                max_new_tokens=settings.router_max_new_tokens,
            )
            # Force initialization during creation to avoid delays
            client._initialize()
            self._models[key] = client
        return self._models[key]
    
    def get_main_client(self) -> LLMClient:
        """Get main model client (shared)"""
        key = f"main_{settings.llm_model_id}"
        if key not in self._models:
            logger.info(f"Creating main client: {settings.llm_model_id}")
            client = LLMClient(
                model_id=settings.llm_model_id,
                temperature=settings.llm_temperature,
                top_p=settings.llm_top_p,
                max_new_tokens=settings.llm_max_new_tokens,
                repetition_penalty=settings.llm_repetition_penalty,
            )
            # Force initialization during creation to avoid delays
            client._initialize()
            self._models[key] = client
        return self._models[key]

# Global instance
model_manager = ModelManager()
