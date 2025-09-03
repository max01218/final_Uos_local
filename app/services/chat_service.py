from typing import Tuple, Optional, List, Dict, Any
import time

from app.schemas.chat import RAGRequest
from app.orchestration.orchestrator import Orchestrator
from app.services.memory_service import ConversationStore


def _flatten_user_history(history: Optional[List[str]]) -> str:
    if not history:
        return ""
    return "\n".join(str(h) for h in history if h is not None).strip()


def _flatten_store_messages(msgs: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for m in msgs or []:
        role = (m.get("role") or "").strip().lower()
        content = (m.get("content") or "").strip()
        if not content:
            continue
        if role == "assistant":
            lines.append(f"Assistant: {content}")
        else:
            lines.append(f"User: {content}")
    return "\n".join(lines).strip()


def _normalize_tone(t: Optional[str]) -> str:
    tone = (t or "").strip().lower()
    if not tone:
        return "balanced"
    aliases = {
        "caring": "warm",
        "friendly": "warm",
        "empathetic": "warm",
        "supportive": "warm",
        "gentle": "warm",
        "professional": "direct",
        "concise": "direct",
        "formal": "direct",
        "neutral": "balanced",
    }
    return aliases.get(tone, tone)


class ChatService:
    def __init__(self, store=None, llm_client=None, conversation_store: ConversationStore = None, embedder=None):
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
        session_id = getattr(req, "session_id", None) or str(int(time.time() * 1000))
        question = (req.question or "").strip()

        recent_msgs = []
        if hasattr(self.cs, "get_recent_messages"):
            try:
                recent_msgs = self.cs.get_recent_messages(n=10, session_id=session_id)
            except Exception:
                recent_msgs = []
        store_history = _flatten_store_messages(recent_msgs)
        user_hist = _flatten_user_history(getattr(req, "history", None))
        history = "\n".join([h for h in [store_history, user_hist] if h])

        tone_in = _normalize_tone(
            tone_override or getattr(req, "tone", None) or getattr(req, "type", None) or "balanced"
        )

        if self.orch.flow and self.orch.flow.is_flow_active(session_id):
            if store_history:
                self.orch.flow.advance_or_adjust(question, session_id)

        answer, meta = await self.orch.generate(
            question=question,
            history=history,
            tone=tone_in,
            session_id=session_id,
        )

        if hasattr(self.cs, "add_message"):
            try:
                self.cs.add_message(session_id=session_id, role="user", content=question)
            except Exception:
                pass
            try:
                self.cs.add_message(session_id=session_id, role="assistant", content=answer)
            except Exception:
                pass

        meta = meta or {}
        meta["session_id"] = session_id
        meta["tone_used"] = tone_in  # always stamp

        return (answer or "").strip(), meta
