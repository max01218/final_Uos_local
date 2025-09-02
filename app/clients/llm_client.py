# app/clients/llm_client.py
import logging
from dataclasses import dataclass
from typing import Optional, List, Union

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

logger = logging.getLogger(__name__)

@dataclass
class LLMConfig:
    model_name: str
    dtype: str = "bfloat16"  # "float16" or "bfloat16"
    device: Optional[str] = None  # "cuda" or "cpu"
    load_in_8bit: bool = False
    trust_remote_code: bool = True
    use_cache: bool = True

class LLMClient:
    def __init__(self, model_name: str, **kwargs):
        self.cfg = LLMConfig(model_name=model_name, **{k: v for k, v in kwargs.items() if k in {
            "dtype", "device", "load_in_8bit", "trust_remote_code", "use_cache"
        }})
        self.tokenizer = None
        self.model = None
        self.device = self.cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._initialized = False

    def _torch_dtype(self):
        m = self.cfg.dtype.lower()
        if m in ("float16", "fp16"):
            return torch.float16
        if m in ("bfloat16", "bf16"):
            return torch.bfloat16
        return torch.float32

    def _initialize(self):
        if self._initialized:
            return
        logger.info("LLMClient initializing model: %s", self.cfg.model_name)
        dtype = self._torch_dtype()
        kwargs = {
            "trust_remote_code": self.cfg.trust_remote_code,
        }
        if self.device == "cuda":
            kwargs["torch_dtype"] = dtype

        if self.cfg.load_in_8bit:
            try:
                import bitsandbytes as bnb  # noqa: F401
                kwargs["load_in_8bit"] = True
                kwargs.pop("torch_dtype", None)
                kwargs["device_map"] = "auto"
            except Exception:
                logger.warning("bitsandbytes not available; falling back to full precision.")
        else:
            kwargs["device_map"] = "auto"

        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_name, trust_remote_code=self.cfg.trust_remote_code)
        self.model = AutoModelForCausalLM.from_pretrained(self.cfg.model_name, **kwargs)
        if hasattr(self.model.config, "use_cache"):
            self.model.config.use_cache = self.cfg.use_cache
        self._initialized = True
        logger.info("LLMClient initialized successfully")

    async def complete(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        stop: Optional[Union[str, List[str]]] = None,
        max_time: Optional[float] = None,
    ) -> str:
        if not self._initialized:
            self._initialize()

        do_sample = False
        gen_cfg = GenerationConfig()
        if temperature is not None and temperature > 0:
            gen_cfg.temperature = float(temperature)
            do_sample = True
        if top_p is not None and top_p > 0:
            gen_cfg.top_p = float(top_p)
            do_sample = True
        if max_new_tokens is not None:
            gen_cfg.max_new_tokens = int(max_new_tokens)

        eos_id = self.tokenizer.eos_token_id
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else eos_id

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                do_sample=do_sample,
                generation_config=gen_cfg,
                eos_token_id=eos_id,
                pad_token_id=pad_id,
                max_time=max_time,
            )[0]
        text = self.tokenizer.decode(output_ids[len(inputs["input_ids"][0]):], skip_special_tokens=True)

        if stop:
            if isinstance(stop, str):
                stop = [stop]
            cut = len(text)
            for s in stop:
                idx = text.find(s)
                if idx != -1:
                    cut = min(cut, idx)
            text = text[:cut]

        return text
