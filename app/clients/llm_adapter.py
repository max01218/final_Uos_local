import asyncio
from typing import List, Dict, Optional
from app.core.settings import settings


class LLMAdapter:
    def __init__(self, llm, tokenizer: Optional[object] = None):
        self.llm = llm
        self.tokenizer = tokenizer

    async def generate(self, prompt: str) -> str:
        async def _invoke():
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, self.llm.invoke, prompt)
            return result if isinstance(result, str) else str(result)

        return await asyncio.wait_for(_invoke(), timeout=settings.llm_timeout_seconds)

    def build_chat_prompt(self, system_content: str, user_content: str, history_messages: Optional[List[Dict[str, str]]] = None) -> str:
        if not self.tokenizer or not hasattr(self.tokenizer, 'apply_chat_template'):
            # Fallback to simple ChatML style
            history_blob = ""
            if history_messages:
                for m in history_messages:
                    role = m.get('role', 'user')
                    content = m.get('content', '')
                    history_blob += f"<|{role}|>\n{content}\n"
            return (
                f"<|system|>\n{system_content}\n{history_blob}<|user|>\n{user_content}\n<|assistant|>\n"
            )

        messages = []
        if system_content:
            messages.append({"role": "system", "content": system_content})
        if history_messages:
            # Keep a small slice of history
            for m in history_messages[-6:]:
                messages.append({"role": m.get('role', 'user'), "content": m.get('content', '')})
        messages.append({"role": "user", "content": user_content})

        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )


