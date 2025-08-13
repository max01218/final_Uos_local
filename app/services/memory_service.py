import time
from threading import RLock
from typing import Dict, List, Optional
from langchain.memory import ConversationBufferMemory
from app.repositories.session_repo import (
    save_session_summary,
    load_session_summary,
)
from app.core.settings import settings


class ConversationStore:
    def __init__(self):
        self.memory = ConversationBufferMemory(return_messages=True)
        self.session_data = {}
        self.user_preferences = {}
        self.emotional_trajectory = []
        self.summary_memory: Dict[str, str] = {}
        self.lock = RLock()

    def add_interaction(self, user_message: str, assistant_message: str, metadata: dict = None, session_id: Optional[str] = None):
        with self.lock:
            self.memory.chat_memory.add_user_message(user_message)
            self.memory.chat_memory.add_ai_message(assistant_message)
            if metadata:
                if session_id:
                    self.session_data.setdefault(session_id, {})[time.time()] = metadata
                else:
                    self.session_data[time.time()] = metadata

    def get_conversation_history(self) -> str:
        messages = self.memory.chat_memory.messages
        if not messages:
            return ""
        history: List[str] = []
        for i in range(0, len(messages), 2):
            if i + 1 < len(messages):
                history.append(f"User: {messages[i].content}")
                history.append(f"Assistant: {messages[i+1].content}")
        return "\n".join(history[-10:])

    def get_recent_messages(self, count: int = 5) -> List[dict]:
        messages = self.memory.chat_memory.messages
        recent = []
        for i in range(max(0, len(messages) - count * 2), len(messages), 2):
            if i + 1 < len(messages):
                recent.append({"role": "user", "content": messages[i].content})
                recent.append({"role": "assistant", "content": messages[i+1].content})
        return recent

    def reset_conversation(self, session_id: Optional[str] = None):
        with self.lock:
            self.memory = ConversationBufferMemory(return_messages=True)
            if session_id and session_id in self.session_data:
                del self.session_data[session_id]
            else:
                self.session_data = {}
            self.emotional_trajectory = []

    def update_emotional_state(self, emotion: str, confidence: float, session_id: Optional[str] = None):
        with self.lock:
            record = {
                'emotion': emotion,
                'confidence': confidence,
                'timestamp': time.time(),
                'session_id': session_id
            }
            self.emotional_trajectory.append(record)

    def get_session_summary(self, session_id: str) -> str:
        return self.summary_memory.get(session_id, "")

    def update_session_summary(self, session_id: str, new_summary: str):
        with self.lock:
            self.summary_memory[session_id] = new_summary
            save_session_summary(session_id, new_summary)


def summarize_session_transcript(transcript: str) -> str:
    if not transcript:
        return ""
    text = transcript[-1200:]
    return " ".join(text.split())[:600]


