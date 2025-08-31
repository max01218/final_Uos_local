from pydantic import BaseModel


class Settings(BaseModel):
    # Main LLM settings
    llm_model_id: str = "Qwen/Qwen2.5-3B-Instruct"
    llm_temperature: float = 0.55
    llm_top_p: float = 0.9
    llm_repetition_penalty: float = 1.12
    llm_max_new_tokens: int = 120
    llm_timeout_seconds: int = 1200
    llm_max_time_seconds: float = 45.0
    
    # Router LLM settings (lightweight)
    router_model_id: str = "Qwen/Qwen2.5-1.5B-Instruct"
    router_temperature: float = 0.1
    router_top_p: float = 0.9
    router_max_new_tokens: int = 120
    
    # Router cache settings
    enable_router_cache: bool = True
    router_cache_ttl_seconds: int = 300
    
    # Legacy settings (keeping for compatibility)
    show_prompt_debug: bool = False
    show_enhanced_debug: bool = False
    use_quantization: bool = True
    max_memory_gb: int = 20
    summary_every_n: int = 3
    session_db_path: str = "sessions.db"
    opro_prompt_path: str = "OPRO_Streamlined/prompts/optimized_prompt.txt"
    opro_fallback_path: str = "ICD11_OPRO/prompts/optimized_prompt.txt"
    
    # Quantization / dtype / attention accel
    llm_load_in_4bit: bool = False
    llm_dtype: str = "auto"
    llm_attn_impl: str = "sdpa"
    llm_device_map: str = "auto"
    class Config:
        env_prefix = ""
        case_sensitive = False


settings = Settings()

