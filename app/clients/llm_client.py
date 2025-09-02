# app/clients/llm_client.py
import os
import time
import asyncio
import logging
from typing import Optional, Dict, Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    from transformers import BitsAndBytesConfig  # optional, for 4/8bit
except Exception:  # pragma: no cover
    BitsAndBytesConfig = None  # type: ignore

logger = logging.getLogger(__name__)


def _parse_dtype(name: Optional[str]) -> Optional[torch.dtype]:
    if not name:
        return None
    name = str(name).lower()
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    if name in ("fp16", "float16", "half"):
        return torch.float16
    if name in ("fp32", "float32"):
        return torch.float32
    return None


class LLMClient:
    """
    Lightweight HF Transformers client.

    Public API expected by the app:
      - _initialize()
      - complete(prompt, ...)

    Constructor now accepts default generation params like temperature/top_p/etc.
    """

    def __init__(
        self,
        model_id: str,
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        # default generation params (optional)
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        do_sample: Optional[bool] = None,
        max_new_tokens: Optional[int] = None,
        max_time: Optional[float] = None,
        **extra_defaults: Any,  # swallow unknown keys from bootstrap/settings
    ):
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        env_dtype = _parse_dtype(os.getenv("TORCH_DTYPE"))
        self.torch_dtype = env_dtype or torch_dtype
        self.load_in_4bit = load_in_4bit or (os.getenv("LOAD_IN_4BIT", "false").lower() == "true")
        self.load_in_8bit = load_in_8bit or (os.getenv("LOAD_IN_8BIT", "false").lower() == "true")

        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[AutoModelForCausalLM] = None
        self._initialized = False

        # store default generation params
        self._gen_defaults: Dict[str, Any] = {}
        if temperature is not None:
            self._gen_defaults["temperature"] = float(temperature)
        if top_p is not None:
            self._gen_defaults["top_p"] = float(top_p)
        if top_k is not None:
            self._gen_defaults["top_k"] = int(top_k)
        if repetition_penalty is not None:
            self._gen_defaults["repetition_penalty"] = float(repetition_penalty)
        if do_sample is not None:
            self._gen_defaults["do_sample"] = bool(do_sample)
        if max_new_tokens is not None:
            self._gen_defaults["max_new_tokens"] = int(max_new_tokens)
        if max_time is not None:
            self._gen_defaults["max_time"] = float(max_time)

        # accept a few common aliases from settings/bootstrap
        for alias in ("temp", "temperature_default"):
            if alias in extra_defaults and "temperature" not in self._gen_defaults:
                try:
                    self._gen_defaults["temperature"] = float(extra_defaults[alias])
                except Exception:
                    pass
        for k in ("top_p_default", "topk", "top_k_default"):
            if k in extra_defaults and "top_k" not in self._gen_defaults:
                try:
                    self._gen_defaults["top_k"] = int(extra_defaults[k])
                except Exception:
                    pass
        if "max_tokens" in extra_defaults and "max_new_tokens" not in self._gen_defaults:
            try:
                self._gen_defaults["max_new_tokens"] = int(extra_defaults["max_tokens"])
            except Exception:
                pass

    # ---------- lifecycle ----------

    def _initialize(self) -> None:
        if self._initialized:
            return

        logger.info(f"LLMClient initializing model: {self.model_id}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, use_fast=True)
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        kwargs: Dict[str, Any] = {}
        if self.load_in_4bit or self.load_in_8bit:
            if BitsAndBytesConfig is None:
                raise RuntimeError(
                    "Quantization requested but BitsAndBytesConfig is unavailable. "
                    "Install bitsandbytes or disable 4/8-bit."
                )
            if self.load_in_4bit:
                kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
            elif self.load_in_8bit:
                kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            kwargs["device_map"] = "auto"
            self.model = AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs)
        else:
            if self.torch_dtype is not None:
                kwargs["torch_dtype"] = self.torch_dtype
            self.model = AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs)
            self.model.to(self.device)

        self.model.eval()
        self._initialized = True
        logger.info("LLMClient initialized successfully")

    # ---------- generation ----------

    async def complete(
        self,
        prompt: str,
        max_time: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> str:
        """
        Generate text. Per-call kwargs override constructor defaults.

        Supported kwargs:
          temperature, top_p, top_k, repetition_penalty, do_sample,
          max_new_tokens, max_time
        Unknown kwargs are ignored for compatibility.
        """
        if not self._initialized:
            self._initialize()

        assert self.model is not None and self.tokenizer is not None

        # start with defaults from ctor
        gen: Dict[str, Any] = dict(self._gen_defaults)

        # per-call overrides
        if max_new_tokens is not None:
            gen["max_new_tokens"] = int(max_new_tokens)
        if max_time is not None:
            gen["max_time"] = float(max_time)

        for key in (
            "temperature",
            "top_p",
            "top_k",
            "repetition_penalty",
            "do_sample",
            "max_new_tokens",
            "max_time",
        ):
            if key in kwargs and kwargs[key] is not None:
                gen[key] = kwargs[key]

        # sane fallbacks
        gen.setdefault("max_new_tokens", 256)
        gen.setdefault("eos_token_id", self.tokenizer.eos_token_id)
        gen.setdefault("pad_token_id", self.tokenizer.pad_token_id or self.tokenizer.eos_token_id)

        # auto decide do_sample if not specified
        if "do_sample" not in gen:
            gen["do_sample"] = (
                (float(gen.get("temperature", 0)) > 0)
                or ("top_p" in gen)
                or ("top_k" in gen)
            )

        device = getattr(self.model, "device", torch.device(self.device))
        enc = self.tokenizer(prompt, return_tensors="pt")
        enc = {k: v.to(device) for k, v in enc.items()}

        def _generate_sync() -> str:
            with torch.no_grad():
                output = self.model.generate(**enc, **{k: v for k, v in gen.items() if v is not None})
            new_tokens = output[0, enc["input_ids"].shape[1]:]
            return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        t0 = time.time()
        loop = asyncio.get_running_loop()
        text = await loop.run_in_executor(None, _generate_sync)
        logger.info(f"LLM complete in {time.time()-t0:.2f}s / max_time={gen.get('max_time')}")
        return text.strip()
