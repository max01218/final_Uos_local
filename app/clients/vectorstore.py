import logging
from typing import Any
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import FAISS

logger = logging.getLogger(__name__)


def load_embeddings(device: str = "cpu") -> HuggingFaceEmbeddings | None:
    try:
        logger.info(f"Loading HuggingFace embeddings on {device}...")
        embedder = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True},
        )
        logger.info(f"Embeddings loaded successfully on {device}")
        return embedder
    except Exception:
        logger.exception("Error loading embeddings")
        return None


def load_faiss_index(embedder: Any, index_dir: str = "embeddings") -> FAISS | None:
    try:
        logger.info("Loading FAISS index...")
        store = FAISS.load_local(index_dir, embedder, allow_dangerous_deserialization=True)
        logger.info("FAISS index loaded successfully")
        return store
    except Exception:
        logger.exception("Error loading FAISS index")
        return None





