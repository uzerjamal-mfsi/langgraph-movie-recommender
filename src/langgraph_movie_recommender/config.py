import os
from functools import lru_cache
from pathlib import Path

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_DATA_DIR = Path("data")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openrouter_api_key: str
    openrouter_model: str
    tmdb_access_token: str | None = None
    tmdb_api_key: str | None = None
    log_level: str = "INFO"

    data_dir: Path = DEFAULT_DATA_DIR
    library_db_path: Path | None = None
    checkpoint_db_path: Path | None = None

    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "taste_memories"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimensions: int = 384

    langsmith_tracing: bool = False
    langsmith_api_key: str | None = None

    @model_validator(mode="after")
    def apply_default_paths(self):
        if self.library_db_path is None:
            self.library_db_path = self.data_dir / "library.db"
        if self.checkpoint_db_path is None:
            self.checkpoint_db_path = self.data_dir / "checkpoints.db"
        return self

    @model_validator(mode="after")
    def require_tmdb_credentials(self):
        if not (self.tmdb_access_token or self.tmdb_api_key):
            raise ValueError("TMDB_ACCESS_TOKEN or TMDB_API_KEY must be set.")
        return self

    @model_validator(mode="after")
    def apply_langsmith_config(self):
        if self.langsmith_tracing and self.langsmith_api_key:
            os.environ["LANGSMITH_TRACING"] = "true"
            os.environ["LANGSMITH_API_KEY"] = self.langsmith_api_key
        return self


@lru_cache
def get_settings():
    return Settings()
