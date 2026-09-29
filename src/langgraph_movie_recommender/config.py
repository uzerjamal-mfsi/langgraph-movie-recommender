from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openrouter_api_key: str 
    openrouter_model: str
    tmdb_api_key: str
    log_level: str = "INFO"

@lru_cache
def get_settings():
    return Settings()
