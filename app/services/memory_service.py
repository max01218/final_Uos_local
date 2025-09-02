# app/services/memory_service.py
from __future__ import annotations
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field

# If you already have ConversationBufferMemory etc. keep them; this is only the store.

def _norm_session_id(session_id: Optional[str]) -> str:
    s = (session_id or "").strip()
    return s if s else "default"

@dataclass
class _Session:
    flags: Dict[str, Any] = field(default_factory=dict)
    messages: List[Dict[str, Any]] = field(default_factory=list)

class ConversationStore:
    """
    Simple in-memory conversation store that tolerates missing session_id.
    Methods used by GuidedFlowService:
      - get_flag(session_id, key)
      - set_flag(session_id, key, value)
      - get_recent_messages(n)
    """
    def __init__(self) -> None:
        self._sessions: Dict[str, _Session] = {}

    # internal
    def _get_sess(self, session_id: Optional[str]) -> _Session:
        sid = _norm_session_id(session_id)
        if sid not in self._sessions:
            self._sessions[sid] = _Session()
        return self._sessions[sid]

    # flags
    def get_flag(self, session_id: Optional[str], key: str, default: Any = None) -> Any:
        sess = self._get_sess(session_id)
        return sess.flags.get(key, default)

    def set_flag(self, session_id: Optional[str], key: str, value: Any) -> None:
        sess = self._get_sess(session_id)
        sess.flags[key] = value

    # messages (optional; only used for quick_check history)
    def add_message(self, session_id: Optional[str], role: str, content: str) -> None:
        sess = self._get_sess(session_id)
        sess.messages.append({"role": role, "content": content})

    def get_recent_messages(self, n: int, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        sess = self._get_sess(session_id)
        if n <= 0:
            return []
        return sess.messages[-n:]