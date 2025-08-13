from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    show_prompt_debug: bool = False
    show_enhanced_debug: bool = False
    llm_timeout_seconds: float = 25.0
    llm_max_new_tokens: int = 160
    llm_max_time_seconds: float = 20.0
    llm_model_id: str = "Qwen/Qwen2.5-3B-Instruct"
    summary_every_n: int = 3
    session_db_path: str = "sessions.db"
    opro_prompt_path: str = "OPRO_Streamlined/prompts/optimized_prompt.txt"
    opro_fallback_path: str = "ICD11_OPRO/prompts/optimized_prompt.txt"

    class Config:
        env_prefix = ""
        case_sensitive = False


settings = Settings()


