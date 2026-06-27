"""Application configuration loaded from environment / .env file."""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings. Values come from env vars or a local .env file."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # Provider selection
    llm_provider: str = "mock"  # mock | groq | openrouter | ollama
    llm_model: str = "mock-llama-3.1-8b"

    # Credentials
    groq_api_key: str = ""
    openrouter_api_key: str = ""

    # Base URLs (all OpenAI-compatible chat/completions endpoints)
    groq_base_url: str = "https://api.groq.com/openai/v1"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    ollama_base_url: str = "http://localhost:11434/v1"

    # Storage
    db_path: str = "data/observability.db"

    # Reliability
    llm_timeout_seconds: float = 30.0
    llm_max_retries: int = 2

    # Data governance
    store_prompts: bool = True

    def base_url_for(self, provider: str) -> str:
        return {
            "groq": self.groq_base_url,
            "openrouter": self.openrouter_base_url,
            "ollama": self.ollama_base_url,
        }.get(provider, "")

    def api_key_for(self, provider: str) -> str:
        return {
            "groq": self.groq_api_key,
            "openrouter": self.openrouter_api_key,
            "ollama": "ollama",  # Ollama ignores the key but the client wants one
        }.get(provider, "")


@lru_cache
def get_settings() -> Settings:
    return Settings()
