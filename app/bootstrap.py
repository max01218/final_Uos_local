# app/bootstrap.py
import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from app.core.settings import settings

logger = logging.getLogger(__name__)

def build_llm(device: str = None):
    model_id = getattr(settings, "llm_model_id", "Qwen/Qwen2.5-7B-Instruct")
    device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token_id = tok.eos_token_id

    torch.backends.cuda.matmul.allow_tf32 = True

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=(torch.bfloat16 if device.startswith("cuda") else torch.float32),
        attn_implementation=getattr(settings, "llm_attn_impl", "sdpa"),
    )
    model.to(device)

    gen_pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tok,
        do_sample=True,
        max_new_tokens=getattr(settings, "llm_max_new_tokens", 90),
        temperature=getattr(settings, "llm_temperature", 0.35),
        top_p=getattr(settings, "llm_top_p", 0.85),
        repetition_penalty=getattr(settings, "llm_repetition_penalty", 1.05),
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
        return_full_text=False,
        device=0 if device.startswith("cuda") else -1,
    )
    return gen_pipe, tok

def warmup_pipeline(gen_pipe):
    try:
        _ = gen_pipe("E: ok\nS: ok\nQ: ok?<END>", max_new_tokens=1)
        logger.info("LLM warmup complete.")
    except Exception as e:
        logger.warning(f"LLM warmup skipped: {e}")

def bootstrap_services():
    """
    Build core singletons and register them in app.core.di, including ChatService.
    """
    try:
        logger.info("Starting bootstrap_services...")
        from app.clients.llm_adapter import LLMAdapter
        from app.clients.llm_client import LLMClient
        from app.services.memory_service import ConversationStore
        from app.services.chat_service import ChatService
        from app.core import di  # DI container

        # Main LLM (legacy adapter for compatibility)
        logger.info("Initializing legacy LLM adapter...")
        pipe, tok = build_llm()
        warmup_pipeline(pipe)
        legacy_llm = LLMAdapter(pipe, tokenizer=tok)
        logger.info("Legacy LLM adapter initialized successfully")

        # Main LLM Client (new architecture)
        logger.info("Initializing main LLM client...")
        main_llm_client = LLMClient(
            model_id=settings.llm_model_id,
            temperature=settings.llm_temperature,
            top_p=settings.llm_top_p,
            repetition_penalty=settings.llm_repetition_penalty,
            max_new_tokens=settings.llm_max_new_tokens,
        )
        logger.info("Main LLM client initialized successfully")

        # Conversation store
        logger.info("Initializing ConversationStore...")
        cs = ConversationStore()
        logger.info("ConversationStore initialized successfully")

        # Optional knowledge store (RAG); tolerate absence
        kb_store = None
        if hasattr(di, "get_vector_store"):
            try:
                logger.info("Attempting to initialize vector store...")
                kb_store = di.get_vector_store()  # type: ignore
                logger.info("Vector store initialized successfully")
            except Exception as e:
                logger.warning(f"Vector store init failed, continue without RAG: {e}")

        # Pre-warm models for faster response times
        logger.info("Pre-warming models...")
        from app.clients.model_manager import model_manager
        model_manager.prewarm_models()
        
        # Build ChatService with new architecture
        logger.info("Initializing ChatService...")
        chat_service = ChatService(
            store=kb_store, 
            llm_client=main_llm_client, 
            conversation_store=cs, 
            embedder=None
        )
        logger.info("ChatService initialized successfully")

        # Register into DI (prefer setters)
        logger.info("Registering services in DI container...")
        if hasattr(di, "set_llm"):
            di.set_llm(legacy_llm)  # type: ignore
            logger.info("LLM registered in DI")
        else:
            di.llm = legacy_llm  # type: ignore
            logger.info("LLM set as DI attribute")

        if hasattr(di, "set_conversation_store"):
            di.set_conversation_store(cs)  # type: ignore
            logger.info("ConversationStore registered in DI")
        else:
            di.conversation_store = cs  # type: ignore
            logger.info("ConversationStore set as DI attribute")

        if hasattr(di, "set_store"):
            di.set_store(kb_store)  # type: ignore
            logger.info("Store registered in DI")
        else:
            di.store = kb_store  # type: ignore
            logger.info("Store set as DI attribute")

        if hasattr(di, "set_chat_service"):
            di.set_chat_service(chat_service)  # type: ignore
            logger.info("ChatService registered in DI successfully")
        else:
            di.chat_service = chat_service  # type: ignore
            logger.info("ChatService set as DI attribute")

        logger.info("bootstrap_services completed (ChatService registered).")
        return True
    except Exception as e:
        logger.exception(f"bootstrap_services failed: {e}")
        return False
