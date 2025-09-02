# app/services/chat_service.py
from typing import Tuple, Optional
import logging

from app.schemas.chat import RAGRequest
from app.orchestration.orchestrator import Orchestrator
from app.services.memory_service import ConversationStore

logger = logging.getLogger(__name__)

class ChatService:
    def __init__(self, store=None, llm_client=None, conversation_store: ConversationStore = None, embedder=None):
        rag = None
        if store:
            try:
                from app.services.rag_service import RAGService
                rag = RAGService(store, embedder=embedder)
            except Exception as e:
                logger.warning("RAGService init failed: %s", e)
                rag = None

        self.cs = conversation_store or ConversationStore()
        self.orch = Orchestrator(rag, self.cs)

    async def handle_chat(self, req: RAGRequest) -> Tuple[str, dict]:
        question: str = (getattr(req, "question", "") or "").strip()
        tone: str = (getattr(req, "type", None) or "balanced").strip() or "balanced"
        history_len: int = int(getattr(req, "historyLength", 50) or 50)

        session_id: Optional[str] = getattr(req, "session_id", None) or getattr(req, "sessionId", None)
        if not session_id or not str(session_id).strip():
            import time
            session_id = str(int(time.time() * 1000))

        history_text: str = self.cs.get_conversation_history(session_id=session_id, limit=history_len, as_text=True)

        try:
            self.cs.append_user_message(question, session_id=session_id)
        except Exception as e:
            logger.warning("append_user_message failed: %s", e)

        try:
            if self.orch.flow and self.orch.flow.is_flow_active(session_id):
                self.orch.flow.advance_or_adjust(question, session_id)
        except Exception as e:
            logger.warning("advance_or_adjust failed: %s", e)

        answer, meta = await self.orch.generate(question=question, history=history_text, tone=tone, session_id=session_id)

        try:
            self.cs.append_assistant_message(answer, session_id=session_id)
        except Exception as e:
            logger.warning("append_assistant_message failed: %s", e)

        if not isinstance(meta, dict):
            meta = {}
        meta.setdefault("session_id", session_id)
        return (answer or "").strip(), meta
