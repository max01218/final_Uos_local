# app/services/chat_service.py
from typing import Tuple, Optional, List, Dict, Any
import time

from app.schemas.chat import RAGRequest
from app.orchestration.orchestrator import Orchestrator
from app.services.memory_service import ConversationStore


def _flatten_user_history(history: Optional[List[str]]) -> str:
    """
    Flattens a client-supplied history array (if any) into a single string.
    """
    if not history:
        return ""
    return "\n".join(str(h) for h in history if h is not None).strip()


def _flatten_store_messages(msgs: List[Dict[str, Any]]) -> str:
    """
    Flattens ConversationStore message dicts into a readable turn-by-turn string.
    Expected message shape: {"role": "user"|"assistant", "content": "..."}.
    """
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
    """
    Normalizes tone aliases; orchestrator also normalizes, but we keep this robust.
    """
    tone = (t or "").strip().lower()
    if not tone:
        return "balanced"
    aliases = {
        "professional": "direct",
        "neutral": "balanced",
        "friendly": "warm",
    }
    return aliases.get(tone, tone)


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
        # Session id first (so we can read the right recent messages)
        session_id = getattr(req, "session_id", None) or str(int(time.time() * 1000))

        # Prepare question and history
        question = (req.question or "").strip()

        # Read recent messages from store (per session) and flatten
        recent_msgs = []
        if hasattr(self.cs, "get_recent_messages"):
            try:
                recent_msgs = self.cs.get_recent_messages(n=10, session_id=session_id)  # last ~10 turns
            except Exception:
                recent_msgs = []
        store_history = _flatten_store_messages(recent_msgs)

        # Also allow client-provided history array
        user_hist = _flatten_user_history(getattr(req, "history", None))

        # Combine histories: store first (older), then client-sent (if any)
        history_parts = [h for h in [store_history, user_hist] if h]
        history = "\n".join(history_parts)

        # Tone preference
        tone_in = _normalize_tone(
            tone_override or getattr(req, "tone", None) or getattr(req, "type", None) or "balanced"
        )

        # Advance guided flow state if active and we have prior turns
        if self.orch.flow and self.orch.flow.is_flow_active(session_id):
            if store_history:
                self.orch.flow.advance_or_adjust(question, session_id)

        # Generate answer
        answer, meta = await self.orch.generate(
            question=question,
            history=history,
            tone=tone_in,
            session_id=session_id,
        )

        # Persist interaction back to ConversationStore
        if hasattr(self.cs, "add_message"):
            try:
                self.cs.add_message(session_id=session_id, role="user", assistant_message=None)  # backward-proofing
            except TypeError:
                # Fallback to the expected signature
                try:
                    self.cs.add_message(session_id=session_id, role="user", content=question)
                except Exception:
                    pass

            try:
                self.cs.add_message(session_id=session_id, role="assistant", content=answer)
            except Exception:
                pass

        # Ensure metadata is present and consistent
        meta = meta or {}
        meta["session_id"] = session_id
        # Guarantee tone_used even if the orchestrator didn't set it
        meta["tone_used"] = meta.get("tone_used") or tone_in

        return (answer or "").strip(), meta
