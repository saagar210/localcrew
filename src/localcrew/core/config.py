"""Application configuration using pydantic-settings."""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Application
    app_name: str = "LocalCrew"
    app_version: str = "0.1.0"
    debug: bool = False

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Database
    database_url: str = "postgresql+asyncpg://localcrew:localcrew@localhost:5432/localcrew"

    # MLX Configuration
    mlx_model_id: str = "mlx-community/Qwen2.5-14B-Instruct-4bit"
    mlx_fallback_model_id: str = "mlx-community/Qwen2.5-7B-Instruct-4bit"
    mlx_max_tokens: int = 4096
    mlx_temperature: float = 0.7

    # Human Review
    confidence_threshold: int = 70

    # MLflow
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "localcrew"

    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


settings = get_settings()
