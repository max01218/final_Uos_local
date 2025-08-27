# app/clients/llm_adapter.py
import asyncio
from typing import Optional, List, Dict
from app.core.settings import settings

class LLMAdapter:
    def __init__(self, pipeline, tokenizer=None):
        self.pipeline = pipeline          # HF text-generation pipeline
        self.tokenizer = tokenizer        # Qwen tokenizer (for chat template)

    def build_chat_prompt(self, system_content: str, user_content: str,
                          history_messages: Optional[List[Dict[str, str]]] = None) -> str:
        """Prefer Qwen chat template; fallback to ChatML."""
        if self.tokenizer and hasattr(self.tokenizer, "apply_chat_template"):
            messages = [{"role": "system", "content": system_content}]
            if history_messages:
                messages.extend(history_messages)
            messages.append({"role": "user", "content": user_content})
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        # ChatML fallback
        history_blob = ""
        if history_messages:
            for m in history_messages:
                history_blob += f"<|{m.get('role','user')}|>\n{m.get('content','')}\n"
        return f"<|system|>\n{system_content}\n{history_blob}<|user|>\n{user_content}\n<|assistant|>\n"

    async def generate(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """Run generation with decoding params and manual stop cut."""
        stop = stop or []
        stop_set = [s for s in stop if s]
        # include common Qwen stops
        if "<|im_end|>" not in stop_set:
            stop_set.append("<|im_end|>")
        if self.tokenizer and self.tokenizer.eos_token and self.tokenizer.eos_token not in stop_set:
            stop_set.append(self.tokenizer.eos_token)

        loop = asyncio.get_event_loop()
        def _run():
            out = self.pipeline(
                prompt,
                do_sample=True,
                max_new_tokens=settings.llm_max_new_tokens,
                temperature=settings.llm_temperature,
                top_p=settings.llm_top_p,
                repetition_penalty=settings.llm_repetition_penalty,
            )[0]["generated_text"]
            # manual stop
            for s in stop_set:
                idx = out.find(s)
                if idx != -1:
                    out = out[:idx]
                    break
            return out
        return await asyncio.wait_for(loop.run_in_executor(None, _run),
                                      timeout=getattr(settings, "llm_timeout_seconds", 25))
