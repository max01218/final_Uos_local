# app/clients/llm_client.py
import os
import time
import asyncio
import logging
from typing import Optional, Dict, Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    # Optional: only needed if you plan to use 4bit/8bit quantization
    from transformers import BitsAndBytesConfig  # type: ignore
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
    Minimal text-generation client around Hugging Face Transformers.

    Public API used by the rest of your app:
      - _initialize(): loads tokenizer/model lazily
      - complete(prompt, ...): returns generated text (supports sampling args)
    """

    def __init__(
        self,
        model_id: str,
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
    ):
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # allow override via env TORCH_DTYPE
        env_dtype = _parse_dtype(os.getenv("TORCH_DTYPE"))
        self.torch_dtype = env_dtype or torch_dtype
        self.load_in_4bit = load_in_4bit or (os.getenv("LOAD_IN_4BIT", "false").lower() == "true")
        self.load_in_8bit = load_in_8bit or (os.getenv("LOAD_IN_8BIT", "false").lower() == "true")

        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[AutoModelForCausalLM] = None
        self._initialized = False

    # ---------- lifecycle ----------

    def _initialize(self) -> None:
        """Load tokenizer/model once. Safe to call multiple times."""
        if self._initialized:
            return

        logger.info(f"LLMClient initializing model: {self.model_id}")

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, use_fast=True)
        # Ensure pad token exists
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Model
        quantized = False
        kwargs: Dict[str, Any] = {}

        if self.load_in_4bit or self.load_in_8bit:
            if BitsAndBytesConfig is None:
                raise RuntimeError(
                    "bitsandbytes/transformers quantization not available. "
                    "Install `bitsandbytes` and ensure compatible CUDA."
                )
            quantized = True
            if self.load_in_4bit:
                kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True)
            elif self.load_in_8bit:
                kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            kwargs["device_map"] = "auto"
            # dtype is ignored in quantized path
            self.model = AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs)
        else:
            if self.torch_dtype is not None:
                kwargs["torch_dtype"] = self.torch_dtype
            self.model = AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs)
            # Move to target device
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
        **kwargs,
    ) -> str:
        """
        Generate text from a prompt.

        Supported kwargs (all optional):
          - temperature: float
          - top_p: float
          - top_k: int
          - repetition_penalty: float
          - do_sample: bool
          - max_new_tokens: int (also available as positional arg)
          - max_time: float (also available as positional arg)
        Unknown kwargs are ignored for compatibility.
        """
        if not self._initialized:
            self._initialize()

        assert self.model is not None and self.tokenizer is not None

        gen: Dict[str, Any] = {
            "max_new_tokens": max_new_tokens if max_new_tokens is not None else 256,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        }

        # Extract supported sampling args from kwargs
        temperature = kwargs.pop("temperature", None)
        top_p = kwargs.pop("top_p", None)
        top_k = kwargs.pop("top_k", None)
        repetition_penalty = kwargs.pop("repetition_penalty", None)
        do_sample = kwargs.pop("do_sample", None)

        if temperature is not None:
            gen["temperature"] = float(temperature)
        if top_p is not None:
            gen["top_p"] = float(top_p)
        if top_k is not None:
            gen["top_k"] = int(top_k)
        if repetition_penalty is not None:
            gen["repetition_penalty"] = float(repetition_penalty)

        # Auto-decide do_sample if not explicitly set
        if do_sample is not None:
            gen["do_sample"] = bool(do_sample)
        else:
            gen["do_sample"] = (
                ("temperature" in gen and gen["temperature"] and gen["temperature"] > 0)
                or ("top_p" in gen)
                or ("top_k" in gen)
            )

        if max_time is not None:
            gen["max_time"] = float(max_time)

        device = getattr(self.model, "device", torch.device(self.device))
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        def _generate_sync() -> str:
            with torch.no_grad():
                output = self.model.generate(**inputs, **{k: v for k, v in gen.items() if v is not None})
            new_tokens = output[0, inputs["input_ids"].shape[1]:]
            text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            return text

        t0 = time.time()
        loop = asyncio.get_running_loop()
        text = await loop.run_in_executor(None, _generate_sync)
        dt = time.time() - t0
        logger.info(f"LLM complete in {dt:.2f}s / max_time={max_time}")
        return text.strip()
