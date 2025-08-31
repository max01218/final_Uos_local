# app/services/chat_service.py
from typing import Tuple
from app.schemas.chat import RAGRequest
from app.orchestration.orchestrator import Orchestrator
from app.services.memory_service import ConversationStore

class ChatService:
    def __init__(self, store=None, llm_client=None, conversation_store: ConversationStore = None, embedder=None):
        # Initialize RAG service if available
        rag = None
        if store:
            try:
                from app.services.rag_service import RAGService
                rag = RAGService(store, embedder=embedder)
            except Exception:
                rag = None
        
        self.cs = conversation_store or ConversationStore()
        self.orch = Orchestrator(rag, self.cs)

    async def handle_chat(self, req: RAGRequest) -> Tuple[str, dict]:
        question = req.question.strip() if req.question else ""
        history = self.cs.get_conversation_history()
        tone = req.type or "balanced"
        
        # Generate session_id if not provided
        session_id = req.session_id or str(int(__import__('time').time() * 1000))
        
        # Handle guided flow state progression if active
        if self.orch.flow and self.orch.flow.is_flow_active(session_id):
            # Check if user provided feedback for step progression
            if history:  # If there's conversation history, process user feedback
                self.orch.flow.advance_or_adjust(question, session_id)
        
        answer, meta = await self.orch.generate(
            question=question, 
            history=history, 
            tone=tone,
            session_id=session_id
        )
        
        # Add session_id to metadata
        meta["session_id"] = session_id
        
        self.cs.add_interaction(
            user_message=question, 
            assistant_message=answer, 
            metadata=meta,
            session_id=session_id
        )
        
        return answer.strip(), meta