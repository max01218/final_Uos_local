import logging
from functools import lru_cache
from typing import Optional

from app.services.chat_service import ChatService
from app.services.memory_service import ConversationStore
from app.clients.llm_adapter import LLMAdapter
from app.clients.vectorstore import load_embeddings, load_faiss_index

logger = logging.getLogger(__name__)


_chat_service: Optional[ChatService] = None


def set_chat_service(svc: ChatService) -> None:
    global _chat_service
    _chat_service = svc


@lru_cache(maxsize=1)
def get_chat_service() -> ChatService:
    if _chat_service is not None:
        return _chat_service
    # Fallback path for legacy startup; try building lazily
    try:
        conv_store = ConversationStore()
        embedder = load_embeddings("cpu")
        store = load_faiss_index(embedder) if embedder else None
        # LLM must be provided by bootstrap; raise if not available
        raise RuntimeError("ChatService not initialized. Call set_chat_service at startup.")
    except Exception:
        logger.exception("ChatService DI fallback failed")
        raise


