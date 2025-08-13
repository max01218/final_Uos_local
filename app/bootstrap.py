import logging
import os
import torch
from app.clients.vectorstore import load_embeddings, load_faiss_index
from app.clients.llm_adapter import LLMAdapter
from app.services.chat_service import ChatService
from app.services.memory_service import ConversationStore
from app.core.di import set_chat_service
from app.repositories.session_repo import init_session_db
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from app.core.settings import settings

logger = logging.getLogger(__name__)


def build_llm(device: str):
    model_id = os.getenv("LLM_MODEL_ID", "Qwen/Qwen2.5-3B-Instruct")
    tok = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        padding_side="left",
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        device_map=device,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        low_cpu_mem_usage=True,
    )
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tok,
        max_new_tokens=settings.llm_max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
        pad_token_id=tok.eos_token_id,
        eos_token_id=tok.eos_token_id,
        return_full_text=False,
    )
    return HuggingFacePipeline(pipeline=pipe), tok


def bootstrap_services():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Ensure session DB schema exists before any read/write
    try:
        init_session_db()
    except Exception:
        logger.exception("Failed to initialize session DB; proceeding may cause errors")
    embedder = load_embeddings(device)
    store = load_faiss_index(embedder) if embedder else None
    llm, tok = build_llm(device)
    conv_store = ConversationStore()
    svc = ChatService(store=store, llm_client=LLMAdapter(llm, tokenizer=tok), conversation_store=conv_store, embedder=embedder)
    set_chat_service(svc)
    logger.info("ChatService initialized and registered")


