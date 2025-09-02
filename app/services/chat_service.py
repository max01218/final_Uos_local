from typing import Tuple, Optional
from app.schemas.chat import RAGRequest
from app.orchestration.orchestrator import Orchestrator
from app.services.memory_service import ConversationStore

class ChatService:
    def __init__(self, store=None, llm_client=None, conversation_store: ConversationStore = None, embedder=None):
        # Optional RAG
        rag = None
        if store:
            try:
                from app.services.rag_service import RAGService
                rag = RAGService(store, embedder=embedder)
            except Exception:
                rag = None

        self.cs = conversation_store or ConversationStore()
        self.orch = Orchestrator(rag, self.cs)

    async def handle_chat(self, req: RAGRequest, tone_override: Optional[str] = None) -> Tuple[str, dict]:
        question = (req.question or "").strip()
        # If you keep a long history elsewhere, plug it here.
        history = getattr(self.cs, "get_conversation_history", lambda: "")()
        # Accept tone from override, request.tone, or legacy request.type
        tone = (tone_override
                or getattr(req, "tone", None)
                or getattr(req, "type", None)
                or "balanced")

        # Generate a session_id if not provided
        session_id = getattr(req, "session_id", None) or str(int(__import__('time').time() * 1000))

        # Update flow state if active (optional)
        if self.orch.flow and self.orch.flow.is_flow_active(session_id):
            if history:
                self.orch.flow.advance_or_adjust(question, session_id)

        answer, meta = await self.orch.generate(
            question=question,
            history=history,
            tone=tone,
            session_id=session_id,
        )

        # Write back to store if such APIs exist
        if hasattr(self.cs, "add_interaction"):
            self.cs.add_interaction(
                user_message=question,
                assistant_message=answer,
                metadata=meta,
                session_id=session_id,
            )

        meta = meta or {}
        meta["session_id"] = session_id
        meta["tone"] = tone
        return answer.strip(), meta
