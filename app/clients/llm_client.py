# app/clients/llm_client.py
import logging
from dataclasses import dataclass
from typing import Optional, List, Union, Dict, Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

logger = logging.getLogger(__name__)

@dataclass
class LLMConfig:
    model_name: str
    dtype: str = "bfloat16"            # "float16" | "bfloat16" | "float32"
    device: Optional[str] = None       # "cuda" | "cpu" | None(auto)
    load_in_8bit: bool = False
    trust_remote_code: bool = True
    use_cache: bool = True

class LLMClient:
    def __init__(self, model_name: Optional[str] = None, **kwargs):
        """
        Backward-compatible constructor.

        Accepts legacy aliases:
          - model (str), name (str), model_id (str)

        Ignores unknown kwargs safely. If callers pass generation defaults
        (temperature/top_p/max_new_tokens/stop/max_time), they are stored
        and merged into complete().
        """
        alias = kwargs.pop("model", None) or kwargs.pop("name", None) or kwargs.pop("model_id", None)
        self.model_name: Optional[str] = model_name or alias
        if not self.model_name:
            raise ValueError("LLMClient: 'model_name' is required (or pass legacy 'model').")

        cfg_keys = {"dtype", "device", "load_in_8bit", "trust_remote_code", "use_cache"}
        cfg_kwargs = {k: v for k, v in kwargs.items() if k in cfg_keys}
        self.cfg = LLMConfig(model_name=self.model_name, **cfg_kwargs)

        # Optional default generation params (used if not overridden in complete()).
        self.default_gen: Dict[str, Any] = {}
        for k in ("temperature", "top_p", "max_new_tokens", "stop", "max_time"):
            if k in kwargs and kwargs[k] is not None:
                self.default_gen[k] = kwargs[k]

        self.tokenizer = None
        self.model = None
        self.device = self.cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._initialized = False

    def _torch_dtype(self):
        m = (self.cfg.dtype or "").lower()
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

        load_kwargs: Dict[str, Any] = {"trust_remote_code": self.cfg.trust_remote_code}
        # Device mapping & dtype
        if self.cfg.load_in_8bit:
            # Try 8-bit; fall back gracefully if bitsandbytes not available.
            try:
                import bitsandbytes as _bnb  # noqa: F401
                load_kwargs["load_in_8bit"] = True
                load_kwargs["device_map"] = "auto"
                # do not pass torch_dtype with 8bit
            except Exception:
                logger.warning("bitsandbytes not available; falling back to full precision.")
                load_kwargs["device_map"] = "auto"
                if self.device == "cuda":
                    load_kwargs["torch_dtype"] = dtype
        else:
            load_kwargs["device_map"] = "auto"
            if self.device == "cuda":
                load_kwargs["torch_dtype"] = dtype

        # Tokenizer & model
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_name, trust_remote_code=self.cfg.trust_remote_code)
        self.model = AutoModelForCausalLM.from_pretrained(self.cfg.model_name, **load_kwargs)

        # cache behavior
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
        """
        Generate a completion. Sampling controls here truly affect generation.
        If a parameter is None, uses the default set in __init__ (if any).
        """
        if not self._initialized:
            self._initialize()

        # Merge defaults
        if temperature is None:
            temperature = self.default_gen.get("temperature")
        if top_p is None:
            top_p = self.default_gen.get("top_p")
        if max_new_tokens is None:
            max_new_tokens = self.default_gen.get("max_new_tokens")
        if stop is None:
            stop = self.default_gen.get("stop")
        if max_time is None:
            max_time = self.default_gen.get("max_time")

        do_sample = False
        gen_cfg = GenerationConfig()
        if temperature is not None and float(temperature) > 0:
            gen_cfg.temperature = float(temperature)
            do_sample = True
        if top_p is not None and float(top_p) > 0:
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
                max_time=max_time,  # supported in recent transformers
            )[0]

        # Slice off the prompt
        gen_part = output_ids[len(inputs["input_ids"][0]):]
        text = self.tokenizer.decode(gen_part, skip_special_tokens=True)

        # Honor stop sequence(s)
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
