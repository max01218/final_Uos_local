# app/clients/llm_client.py
import asyncio
from typing import List, Optional

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    StoppingCriteria,
    StoppingCriteriaList,
)
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from app.core.settings import settings


class _StopOnText(StoppingCriteria):
    """Stop when any of the stop strings appears at the end of the generated ids."""
    def __init__(self, stop_strings: List[str], tokenizer):
        super().__init__()
        self.stop_ids = [tokenizer(s, add_special_tokens=False).input_ids for s in stop_strings]
        self.max_len = max((len(s) for s in self.stop_ids), default=0)

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        if self.max_len == 0:
            return False
        seq = input_ids[0].tolist()  # assumes batch size 1
        for sids in self.stop_ids:
            L = len(sids)
            if L and len(seq) >= L and seq[-L:] == sids:
                return True
        return False


class LLMClient:
    def __init__(
        self,
        device: str,
        *,
        stop: Optional[List[str]] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        repetition_penalty: float = 1.05,
    ):
        self.device = device
        # defaults (overridable via settings or init kwargs)
        self.model_id = getattr(settings, "llm_model_id", "meta-llama/Meta-Llama-3-8B-Instruct")
        self.stop = stop or ["<END>"]
        self.temperature = temperature if temperature is not None else getattr(settings, "llm_temperature", 0.3)
        self.top_p = top_p if top_p is not None else getattr(settings, "llm_top_p", 0.8)
        self.max_new_tokens = (
            max_new_tokens
            if max_new_tokens is not None
            else min(90, getattr(settings, "llm_max_new_tokens", 90))
        )
        self.repetition_penalty = repetition_penalty

        self.tokenizer = None
        self.pipeline = None
        self.llm = None
        self._stopper = None

    def load(self):
        tok = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            padding_side="left",
        )
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            device_map=self.device,
            torch_dtype=None,
            low_cpu_mem_usage=True,
        )

        # ensure pad/eos ids set
        if tok.pad_token_id is None and tok.eos_token_id is not None:
            tok.pad_token_id = tok.eos_token_id

        self._stopper = _StopOnText(self.stop, tok)

        pipe = pipeline(
            task="text-generation",
            model=model,
            tokenizer=tok,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=self.temperature,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty,
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
            eos_token_id=tok.eos_token_id,
            return_full_text=False,
        )
        # Keep both for backward compatibility
        self.llm = HuggingFacePipeline(pipeline=pipe)
        self.pipeline = pipe
        self.tokenizer = tok

    async def generate(
        self,
        prompt: str,
        *,
        stop: Optional[List[str]] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
    ) -> str:
        """Generate text with unified decoding params and stop strings."""
        if self.pipeline is None:
            raise RuntimeError("LLMClient not loaded. Call load() first.")

        stop_strings = stop or self.stop
        stopper = self._stopper if stop_strings == self.stop else _StopOnText(stop_strings, self.tokenizer)

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens or self.max_new_tokens,
            temperature=self._coalesce(temperature, self.temperature),
            top_p=self._coalesce(top_p, self.top_p),
            repetition_penalty=self._coalesce(repetition_penalty, self.repetition_penalty),
            stopping_criteria=StoppingCriteriaList([stopper]),
        )

        def _run_sync() -> str:
            out = self.pipeline(prompt, **gen_kwargs)
            text = out[0]["generated_text"] if isinstance(out, list) else str(out)
            # hard-trim at first occurrence of any stop string
            for s in stop_strings:
                idx = text.find(s)
                if idx != -1:
                    text = text[:idx]
                    break
            return text

        loop = asyncio.get_event_loop()
        result = await asyncio.wait_for(_to_thread(_run_sync), timeout=getattr(settings, "llm_timeout_seconds", 25))
        return result if isinstance(result, str) else str(result)

    @staticmethod
    def _coalesce(x, default):
        return default if x is None else x


# small helper to keep asyncio code tidy across Python versions
async def _to_thread(func, /, *args, **kwargs):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: func(*args, **kwargs))
