from __future__ import annotations

from functools import lru_cache
import torch

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MULTIMODEL_", extra="ignore")

    app_name: str = "Text-to-Image Generation with Multi-Model Analysis"
    api_timeout_seconds: float = Field(default=90.0, ge=1.0)
    default_image_size: int = Field(default=512, ge=64, le=1024)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
