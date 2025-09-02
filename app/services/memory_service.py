# app/services/memory_service.py
from __future__ import annotations
from typing import Optional, Dict, Any, List, Iterable, Union, Tuple
from dataclasses import dataclass, field

def _norm_session_id(session_id: Optional[str]) -> str:
    s = (session_id or "").strip()
    return s if s else "default"

@dataclass
class _Session:
    flags: Dict[str, Any] = field(default_factory=dict)
    messages: List[Dict[str, Any]] = field(default_factory=list)  # [{"role": "user|assistant", "content": "..."}]

class ConversationStore:
    """
    In-memory conversation store that tolerates missing session_id.
    Public API (used across the project):
      - get_flag(session_id, key, default=None) -> Any
      - set_flag(session_id, key, value) -> None
      - add_message(session_id, role, content) -> None
      - append_user_message(content, session_id=None) -> None
      - append_assistant_message(content, session_id=None) -> None
      - get_conversation_history(session_id=None, limit=50, as_text=False) -> list|str
      - get_recent_messages(n, session_id=None, role=None) -> list
      - get_recent_messages_text(n, session_id=None, role=None, sep="\\n") -> str
      - get_last_user_message(session_id=None) -> Optional[str]
      - get_last_assistant_message(session_id=None) -> Optional[str]
      - clear_session(session_id=None) -> None
    """

    def __init__(self) -> None:
        self._sessions: Dict[str, _Session] = {}

    # ---- internal helpers ----
    def _get_sess(self, session_id: Optional[str]) -> _Session:
        sid = _norm_session_id(session_id)
        if sid not in self._sessions:
            self._sessions[sid] = _Session()
        return self._sessions[sid]

    # ---- flags ----
    def get_flag(self, session_id: Optional[str], key: str, default: Any = None) -> Any:
        sess = self._get_sess(session_id)
        return sess.flags.get(key, default)

    def set_flag(self, session_id: Optional[str], key: str, value: Any) -> None:
        sess = self._get_sess(session_id)
        sess.flags[key] = value

    # ---- messages write ----
    def add_message(self, session_id: Optional[str], role: str, content: str) -> None:
        sess = self._get_sess(session_id)
        sess.messages.append({"role": (role or "").lower(), "content": content or ""})

    def append_user_message(self, content: str, session_id: Optional[str] = None) -> None:
        self.add_message(session_id, "user", content)

    def append_assistant_message(self, content: str, session_id: Optional[str] = None) -> None:
        self.add_message(session_id, "assistant", content)

    # ---- messages read (history) ----
    def get_conversation_history(
        self,
        session_id: Optional[str] = None,
        limit: int = 50,
        as_text: bool = False,
    ) -> Union[List[Dict[str, Any]], str]:
        sess = self._get_sess(session_id)
        msgs = sess.messages[-limit:] if (limit and limit > 0) else list(sess.messages)
        if not as_text:
            return msgs
        lines: List[str] = []
        for m in msgs:
            role = (m.get("role") or "").upper() or "USER"
            content = m.get("content") or ""
            lines.append(f"{role}: {content}")
        return "\n".join(lines).strip()

    def get_recent_messages(
        self,
        n: int,
        session_id: Optional[str] = None,
        role: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Return last n messages. If role is provided ("user" or "assistant"),
        the result is filtered to that role.
        """
        sess = self._get_sess(session_id)
        if n <= 0:
            return []
        chunk = sess.messages[-n:]
        if role:
            r = role.lower()
            chunk = [m for m in chunk if (m.get("role") or "").lower() == r]
        return chunk

    def get_recent_messages_text(
        self,
        n: int,
        session_id: Optional[str] = None,
        role: Optional[str] = None,
        sep: str = "\n",
    ) -> str:
        """
        Return last n messages as a single string with role headers.
        """
        msgs = self.get_recent_messages(n, session_id=session_id, role=role)
        lines: List[str] = []
        for m in msgs:
            role_up = (m.get("role") or "").upper() or "USER"
            content = m.get("content") or ""
            lines.append(f"{role_up}: {content}")
        return sep.join(lines).strip()

    def get_last_user_message(self, session_id: Optional[str] = None) -> Optional[str]:
        sess = self._get_sess(session_id)
        for m in reversed(sess.messages):
            if (m.get("role") or "").lower() == "user":
                return m.get("content") or ""
        return None

    def get_last_assistant_message(self, session_id: Optional[str] = None) -> Optional[str]:
        sess = self._get_sess(session_id)
        for m in reversed(sess.messages):
            if (m.get("role") or "").lower() == "assistant":
                return m.get("content") or ""
        return None

    # ---- utils ----
    def clear_session(self, session_id: Optional[str] = None) -> None:
        sid = _norm_session_id(session_id)
        self._sessions.pop(sid, None)
