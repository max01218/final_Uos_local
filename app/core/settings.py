from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    show_prompt_debug: bool = False
    show_enhanced_debug: bool = False
    llm_timeout_seconds: int = 1200  # Increased for 20B model
    llm_max_new_tokens: int = 100
    llm_max_time_seconds: float = 45.0  # Increased for 20B model
    llm_model_id: str = "Qwen/Qwen2.5-7B-Instruct"  # Default model for iridis5
    llm_temperature: float = 0.35
    use_quantization: bool = True  # Enable 4-bit quantization for efficiency
    llm_repetition_penalty: float = 1.05
    llm_top_p: float = 0.85 
    max_memory_gb: int = 20  # Reduced memory limit for 7B model
    summary_every_n: int = 3
    session_db_path: str = "sessions.db"
    opro_prompt_path: str = "OPRO_Streamlined/prompts/optimized_prompt.txt"
    opro_fallback_path: str = "ICD11_OPRO/prompts/optimized_prompt.txt"
    # Quantization / dtype / attention accel
    llm_load_in_4bit: bool = False         # set True for 4-bit with bitsandbytes
    llm_dtype: str = "auto"                # "auto" | "bfloat16" | "float16"
    llm_attn_impl: str = "sdpa"            # "flash_attention_2" if FA2 is installed
    llm_device_map: str = "auto"           # "auto" or specific gpu ids
    class Config:
        env_prefix = ""
        case_sensitive = False


settings = Settings()

