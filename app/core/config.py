"""Application configuration using Pydantic Settings v2."""

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """DevGuide application configuration loaded from environment and optional .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    APP_HOST: str = "127.0.0.1"
    APP_PORT: int = Field(default=8000, ge=1, le=65535)
    DATA_DIR: Path = Path("./data")
    SQLITE_PATH: Path = Path("./data/devguide.db")
    CHROMA_PATH: Path = Path("./data/chroma")
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"
    OPENAI_CHAT_MODEL: str = "gpt-6-luna"
    CHUNK_SIZE: int = Field(default=1000, gt=0)
    CHUNK_OVERLAP: int = Field(default=150, ge=0)
    EMBEDDING_BATCH_SIZE: int = Field(default=64, gt=0)
    DEFAULT_TOP_K: int = Field(default=5, ge=1)
    MAX_TOP_K: int = Field(default=10, ge=1, le=10)
    MAX_COSINE_DISTANCE: float = Field(default=0.70, ge=0.0, le=2.0)
    MAX_UPLOAD_MB: int = Field(default=10, gt=0)
    MAX_CONTEXT_CHARS: int = Field(default=12000, gt=0)

    @field_validator("APP_HOST")
    @classmethod
    def validate_app_host(cls, v: str) -> str:
        """Ensure APP_HOST is not empty or whitespace-only."""
        if not v or not v.strip():
            raise ValueError("APP_HOST must not be empty")
        return v

    @model_validator(mode="after")
    def validate_relationships(self) -> "Settings":
        """Validate interdependent field relationships."""
        if self.CHUNK_OVERLAP >= self.CHUNK_SIZE:
            raise ValueError(
                f"CHUNK_OVERLAP ({self.CHUNK_OVERLAP}) must be strictly smaller than CHUNK_SIZE ({self.CHUNK_SIZE})"
            )
        if self.DEFAULT_TOP_K > self.MAX_TOP_K:
            raise ValueError(
                f"DEFAULT_TOP_K ({self.DEFAULT_TOP_K}) must be between 1 and MAX_TOP_K ({self.MAX_TOP_K})"
            )
        return self


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()
