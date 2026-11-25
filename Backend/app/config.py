"""
app/config.py
─────────────
Centralised application configuration loaded from environment variables.
All modules import settings from here; nothing reads os.environ directly.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import List

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings resolved from environment variables / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Frame analysis (Groq) ────────────────────────────────────────────────
    groq_api_key: str = ""
    groq_model: str = "qwen/qwen3.8-27b"

    # ── Database ──────────────────────────────────────────────────────────────
    database_url: str = "sqlite:///./drone_security.db"

    # ── CORS ──────────────────────────────────────────────────────────────────
    frontend_origin: str = "http://localhost:5173"

    # ── Alert rules ───────────────────────────────────────────────────────────
    restricted_hour_start: int = 22   # 10 PM
    restricted_hour_end: int = 6      # 6 AM
    loitering_min_frames: int = 2     # consecutive detections needed

    # Stored as a comma-separated string in .env, parsed into a list here.
    authorized_vehicles: str = "blue Ford F150,white Toyota Camry"

    # ── Logging ───────────────────────────────────────────────────────────────
    log_level: str = "INFO"

    # ── Derived helpers ───────────────────────────────────────────────────────
    @property
    def authorized_vehicle_list(self) -> List[str]:
        """Return normalized authorized vehicle strings."""
        return [v.strip().lower() for v in self.authorized_vehicles.split(",") if v.strip()]

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in valid:
            raise ValueError(f"log_level must be one of {valid}")
        return upper


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings instance (singleton pattern)."""
    return Settings()


def configure_logging() -> None:
    """Apply log level from settings to the root logger."""
    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
