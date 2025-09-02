# app/bootstrap.py
import os
import inspect
import logging
import torch

from app.core.settings import settings
from app.clients.model_manager import model_manager

logger = logging.getLogger(__name__)


def _dtype_from_env() -> torch.dtype | None:
    name = os.getenv("TORCH_DTYPE", "").lower()
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    if name in ("fp16", "float16", "half"):
        return torch.float16
    if name in ("fp32", "float32"):
        return torch.float32
    return None


def _safe_kwargs_for_ctor(cls, raw: dict) -> dict:
    """Keep only kwargs accepted by cls.__init__ and drop the rest."""
    sig = inspect.signature(cls.__init__)
    allowed = set(sig.parameters.keys())
    clean = {k: v for k, v in (raw or {}).items() if k in allowed and v is not None}
    dropped = {k: v for k, v in (raw or {}).items() if k not in allowed and v is not None}
    if dropped:
        logger.info(f"{cls.__name__}: dropping unsupported init kwargs: {sorted(dropped.keys())}")
    return clean


def bootstrap_services():
    """
    Build and register core singletons without duplicating model loads.
    This version avoids loading the same 7B model multiple times.
    """
    try:
        logger.info("Starting bootstrap_services...")

        # Optional: make CUDA allocator less fragment-prone
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

        # 1) Build / prewarm LLM clients via model_manager ONLY (single source of truth)
        #    - model_manager should instantiate router/main internally from settings.
        prewarm_flag = getattr(settings, "prewarm_models", True)
        env_prewarm = os.getenv("PREWARM_MODELS", "true").lower() == "true"
        do_prewarm = prewarm_flag and env_prewarm

        if do_prewarm:
            logger.info("Pre-warming models via model_manager...")
            model_manager.prewarm_models()
            logger.info("All models pre-warmed successfully")

        # Ensure main client exists (lazy build if not prewarmed)
        main_client = model_manager.get_main_client()
        router_client = model_manager.get_router_client()  # also ensure router exists

        # 2) Conversation store
        from app.services.memory_service import ConversationStore
        cs = ConversationStore()
        logger.info("ConversationStore initialized successfully")

        # 3) Optional vector store (RAG)
        kb_store = None
        try:
            from app.core import di  # DI container
            if hasattr(di, "get_vector_store"):
                logger.info("Attempting to initialize vector store...")
                kb_store = di.get_vector_store()  # type: ignore
                logger.info("Vector store initialized successfully")
        except Exception as e:
            logger.warning(f"Vector store init failed, continue without RAG: {e}")

        # 4) Chat service uses the SINGLE main client (no legacy pipeline)
        from app.services.chat_service import ChatService
        chat_service = ChatService(
            store=kb_store,
            llm_client=main_client,
            conversation_store=cs,
            embedder=None,
        )
        logger.info("ChatService initialized successfully")

        # 5) Register into DI
        from app.core import di  # re-import to set attributes
        # Prefer setters when available
        if hasattr(di, "set_conversation_store"):
            di.set_conversation_store(cs)  # type: ignore
        else:
            di.conversation_store = cs  # type: ignore

        if hasattr(di, "set_store"):
            di.set_store(kb_store)  # type: ignore
        else:
            di.store = kb_store  # type: ignore

        if hasattr(di, "set_chat_service"):
            di.set_chat_service(chat_service)  # type: ignore
        else:
            di.chat_service = chat_service  # type: ignore

        # (Optional) For backwards-compat code that expects di.llm,
        # provide a thin shim that delegates to main_client.complete(...)
        class _LLMShim:
            async def complete(self, prompt: str, **kwargs):
                return await main_client.complete(prompt, **kwargs)

        if hasattr(di, "set_llm"):
            di.set_llm(_LLMShim())  # type: ignore
        else:
            di.llm = _LLMShim()  # type: ignore

        logger.info("bootstrap_services completed (ChatService registered).")
        return True

    except Exception as e:
        logger.exception(f"bootstrap_services failed: {e}")
        return False
