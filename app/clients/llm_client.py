import asyncio
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from app.core.settings import settings


class LLMClient:
    def __init__(self, device: str):
        self.device = device
        self.tokenizer = None
        self.pipeline = None
        self.llm = None

    def load(self):
        tok = AutoTokenizer.from_pretrained(
            "meta-llama/Meta-Llama-3-8B-Instruct",
            trust_remote_code=True,
            padding_side="left"
        )
        model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Meta-Llama-3-8B-Instruct",
            trust_remote_code=True,
            device_map=self.device,
            torch_dtype=None,
            low_cpu_mem_usage=True
        )
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tok,
            max_new_tokens=settings.llm_max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            pad_token_id=tok.eos_token_id,
            eos_token_id=tok.eos_token_id,
            return_full_text=False
        )
        self.llm = HuggingFacePipeline(pipeline=pipe)
        self.pipeline = pipe
        self.tokenizer = tok

    async def generate(self, prompt: str) -> str:
        pipe = self.llm.pipeline
        pipe.model.config.max_new_tokens = settings.llm_max_new_tokens
        pipe.model.config.temperature = 0.7
        pipe.model.config.top_p = 0.9

        async def _invoke():
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, self.llm.invoke, prompt)
            return result if isinstance(result, str) else str(result)

        return await asyncio.wait_for(_invoke(), timeout=settings.llm_timeout_seconds)


