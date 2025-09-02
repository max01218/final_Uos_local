# app/services/memory_service.py
from __future__ import annotations
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass, field

def _norm_session_id(session_id: Optional[str]) -> str:
    s = (session_id or "").strip()
    return s if s else "default"

@dataclass
class _Session:
    flags: Dict[str, Any] = field(default_factory=dict)
    messages: List[Dict[str, Any]] = field(default_factory=list)

class ConversationStore:
    def __init__(self) -> None:
        self._sessions: Dict[str, _Session] = {}

    def _get_sess(self, session_id: Optional[str]) -> _Session:
        sid = _norm_session_id(session_id)
        if sid not in self._sessions:
            self._sessions[sid] = _Session()
        return self._sessions[sid]

    # flags
    def get_flag(self, session_id: Optional[str], key: str, default: Any = None) -> Any:
        return self._get_sess(session_id).flags.get(key, default)

    def set_flag(self, session_id: Optional[str], key: str, value: Any) -> None:
        self._get_sess(session_id).flags[key] = value

    # messages
    def add_message(self, session_id: Optional[str], role: str, content: str) -> None:
        self._get_sess(session_id).messages.append({"role": (role or "").lower(), "content": content or ""})

    def append_user_message(self, content: str, session_id: Optional[str] = None) -> None:
        self.add_message(session_id, "user", content)

    def append_assistant_message(self, content: str, session_id: Optional[str] = None) -> None:
        self.add_message(session_id, "assistant", content)

    def get_recent_messages(self, n: int, session_id: Optional[str] = None, role: Optional[str] = None) -> List[Dict[str, Any]]:
        msgs = self._get_sess(session_id).messages
        if n <= 0:
            return []
        chunk = msgs[-n:]
        if role:
            r = role.lower()
            chunk = [m for m in chunk if (m.get("role") or "").lower() == r]
        return chunk

    def get_conversation_history(self, session_id: Optional[str] = None, limit: int = 50, as_text: bool = False) -> Union[List[Dict[str, Any]], str]:
        msgs = self._get_sess(session_id).messages
        out = msgs[-limit:] if (limit and limit > 0) else list(msgs)
        if not as_text:
            return out
        lines: List[str] = []
        for m in out:
            lines.append(f"{(m.get('role') or 'user').upper()}: {m.get('content') or ''}")
        return "\n".join(lines).strip()

    def clear_session(self, session_id: Optional[str] = None) -> None:
        sid = _norm_session_id(session_id)
        self._sessions.pop(sid, None)
