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
        from app.clients.llm_adapter import LLMAdapter
        from app.services.memory_service import ConversationStore
        from app.services.chat_service import ChatService
        from app.core import di  # DI container

        # LLM
        pipe, tok = build_llm()
        warmup_pipeline(pipe)
        llm = LLMAdapter(pipe, tokenizer=tok)

        # Conversation store
        cs = ConversationStore()

        # Optional knowledge store (RAG); tolerate absence
        kb_store = None
        if hasattr(di, "get_vector_store"):
            try:
                kb_store = di.get_vector_store()  # type: ignore
            except Exception as e:
                logger.warning(f"Vector store init failed, continue without RAG: {e}")

        # Build ChatService
        chat_service = ChatService(store=kb_store, llm_client=llm, conversation_store=cs, embedder=None)

        # Register into DI (prefer setters)
        if hasattr(di, "set_llm"):
            di.set_llm(llm)  # type: ignore
        else:
            di.llm = llm  # type: ignore

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

        logger.info("bootstrap_services completed (ChatService registered).")
        return True
    except Exception as e:
        logger.exception(f"bootstrap_services failed: {e}")
        return False
