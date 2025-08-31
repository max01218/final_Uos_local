# app/clients/llm_client.py
import asyncio
import logging
import time
import threading
from typing import Optional, List
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, TextIteratorStreamer
import torch
from app.core.settings import settings

logger = logging.getLogger(__name__)

class LLMClient:
    def __init__(self, model_id: str, temperature: float, top_p: float, 
                 max_new_tokens: int, repetition_penalty: Optional[float] = None):
        self.model_id = model_id
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens
        self.repetition_penalty = repetition_penalty or 1.0
        self.pipeline = None
        self.tokenizer = None
        self.model = None
        self._initialized = False

    def _initialize(self):
        """Lazy initialization of the model pipeline"""
        if self._initialized:
            return
            
        try:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_id, 
                trust_remote_code=True
            )
            if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

            # Load model
            model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                trust_remote_code=True,
                torch_dtype=(torch.bfloat16 if device.startswith("cuda") else torch.float32),
                attn_implementation=getattr(settings, "llm_attn_impl", "sdpa"),
            )
            model.to(device)
            self.model = model

            # Create pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=self.tokenizer,
                do_sample=True,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                repetition_penalty=self.repetition_penalty,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                return_full_text=False,
                device=0 if device.startswith("cuda") else -1,
            )
            
            self._initialized = True
            logger.info(f"LLMClient initialized successfully with model: {self.model_id}")
            
        except Exception as e:
            logger.error(f"Failed to initialize LLMClient with model {self.model_id}: {e}")
            raise

    async def complete(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        *,
        max_time: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        stream: bool = False,
    ) -> str:
        """Complete the given prompt and return the generated text.
        Supports optional model-level timeboxing and streaming.
        """
        if not self._initialized:
            self._initialize()
            
        stop = stop or []
        stop_set = [s for s in stop if s]
        
        # Add common stop tokens
        common_stops = ["<|im_end|>", "\n\n", "User:", "Human:", "\nUser", "\nHuman"]
        for stop in common_stops:
            if stop not in stop_set:
                stop_set.append(stop)
        if self.tokenizer and self.tokenizer.eos_token and self.tokenizer.eos_token not in stop_set:
            stop_set.append(self.tokenizer.eos_token)

        loop = asyncio.get_event_loop()

        # Shared generation kwargs
        gen_kwargs = {
            "max_new_tokens": max_new_tokens or self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "repetition_penalty": self.repetition_penalty,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if max_time is not None:
            gen_kwargs["max_time"] = max_time
        
        def _run():
            try:
                t0 = time.perf_counter()
                out = self.pipeline(prompt, **gen_kwargs)[0]["generated_text"]
                dt = time.perf_counter() - t0
                logger.info(f"LLM complete in {dt:.2f}s / max_time={max_time}")
                
                # Apply manual stop token cutting and cleaning
                for s in stop_set:
                    idx = out.find(s)
                    if idx != -1:
                        out = out[:idx]
                        break
                
                # Simple cleaning for now
                if "Human:" in out:
                    out = out.split("Human:")[0]
                if "User:" in out:
                    out = out.split("User:")[0]
                        
                return out.strip()
            except Exception as e:
                logger.error(f"LLM generation failed: {e}")
                return ""

        if not stream:
            try:
                return await asyncio.wait_for(
                    loop.run_in_executor(None, _run),
                    timeout=getattr(settings, "llm_timeout_seconds", 60)
                )
            except asyncio.TimeoutError:
                logger.error("LLM generation timed out")
                return ""

        # Streaming path using TextIteratorStreamer
        def _run_streaming(chunks: List[str]):
            try:
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
                streamer = TextIteratorStreamer(self.tokenizer, skip_special_tokens=True)
                kwargs = dict(gen_kwargs)
                kwargs["streamer"] = streamer

                def _gen():
                    self.model.generate(**inputs, **kwargs)

                th = threading.Thread(target=_gen)
                th.start()
                t0 = time.perf_counter()
                first = True
                for piece in streamer:
                    if first:
                        first = False
                        ttft = time.perf_counter() - t0
                        logger.info(f"TTFT={ttft:.2f}s")
                    chunks.append(piece)
                th.join()
            except Exception as e:
                logger.error(f"LLM streaming failed: {e}")

        chunks: List[str] = []
        await loop.run_in_executor(None, _run_streaming, chunks)
        out = "".join(chunks)
        # Apply stop trimming on the combined output
        for s in stop_set:
            idx = out.find(s)
            if idx != -1:
                out = out[:idx]
                break
        return out.strip()