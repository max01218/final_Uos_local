# app/core/di.py
# Minimal DI container with safe getters/setters and lazy bootstrap fallback.

from typing import Optional

_llm = None
_conversation_store = None
_store = None
_chat_service = None

def set_llm(llm):  # noqa
    global _llm; _llm = llm

def set_conversation_store(cs):  # noqa
    global _conversation_store; _conversation_store = cs

def set_store(store):  # noqa
    global _store; _store = store

def set_chat_service(cs):  # noqa
    global _chat_service; _chat_service = cs

def get_llm():
    if _llm is None:
        _try_bootstrap()
    if _llm is None:
        raise RuntimeError("LLM not initialized. Call bootstrap_services at startup.")
    return _llm

def get_conversation_store():
    if _conversation_store is None:
        _try_bootstrap()
    if _conversation_store is None:
        raise RuntimeError("ConversationStore not initialized.")
    return _conversation_store

def get_store():
    if _store is None:
        _try_bootstrap()
    return _store

def get_chat_service():
    if _chat_service is None:
        _try_bootstrap()
    if _chat_service is None:
        raise RuntimeError("ChatService not initialized. Call set_chat_service at startup.")
    return _chat_service

def _try_bootstrap():
    # Lazy attempt to bootstrap if not yet initialized
    try:
        import logging
        logger = logging.getLogger(__name__)
        logger.info("Attempting lazy bootstrap...")
        from app.bootstrap import bootstrap_services
        success = bootstrap_services()
        if success:
            logger.info("Lazy bootstrap successful")
        else:
            logger.error("Lazy bootstrap failed")
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Lazy bootstrap exception: {e}")
