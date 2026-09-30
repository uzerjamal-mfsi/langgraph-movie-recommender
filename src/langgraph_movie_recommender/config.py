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
    tmdb_api_key: str
    log_level: str = "INFO"

    data_dir: Path = DEFAULT_DATA_DIR
    library_db_path: Path | None = None
    checkpoint_db_path: Path | None = None

    @model_validator(mode="after")
    def apply_default_paths(self):
        if self.library_db_path is None:
            self.library_db_path = self.data_dir / "library.db"
        if self.checkpoint_db_path is None:
            self.checkpoint_db_path = self.data_dir / "checkpoints.db"
        return self


@lru_cache
def get_settings():
    return Settings()
