"""
config.py
---------
Centralised configuration management for the Voice & Chat Assistant.
Loads all settings from environment variables via pydantic-settings.
Provides a single `settings` instance imported across the entire app.

Services used:
  - Sarvam AI  : STT (Saaras:v3) + TTS (Bulbul:v3) — all 3 languages
  - Groq       : LLM inference (intent, sentiment, response)
"""

import logging
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator


# ── Module Logger ─────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)


# ── Settings Model ────────────────────────────────────────────────────────────
class Settings(BaseSettings):
    """
    Application-wide settings loaded from environment variables / .env file.

    All fields map 1-to-1 with keys defined in .env.example.
    Pydantic validates types and raises clear errors on misconfiguration.
    """

    model_config = SettingsConfigDict(
        env_file          = ".env",
        env_file_encoding = "utf-8",
        case_sensitive    = False,
        extra             = "ignore",   # Silently ignore unknown env vars
    )

    # ── Sarvam AI (STT + TTS) ─────────────────────────────────────────────────
    sarvam_api_key:   str = Field(..., description="Sarvam AI API key for STT and TTS")
    sarvam_stt_model: str = Field(default="saaras:v3",  description="Sarvam STT model")
    sarvam_tts_model: str = Field(default="bulbul:v3",  description="Sarvam TTS model")
    sarvam_tts_speaker: str = Field(default="rahul",   description="Sarvam TTS speaker voice")

    # ── Groq (LLM) ────────────────────────────────────────────────────────────
    groq_api_key:     str   = Field(...,    description="Groq API key for LLM inference")
    groq_model:       str   = Field(default="llama-3.3-70b-versatile", description="Groq model identifier")
    groq_max_tokens:  int   = Field(default=1024, description="Max tokens for LLM response")
    groq_temperature: float = Field(default=0.2,  description="LLM temperature — lower = more deterministic")

    # ── App ───────────────────────────────────────────────────────────────────
    app_env:   str = Field(default="development", description="Runtime environment")
    log_level: str = Field(default="INFO",        description="Logging level")
    app_host:  str = Field(default="0.0.0.0",     description="FastAPI host")
    app_port:  int = Field(default=8000,           description="FastAPI port")

    # ── Validators ────────────────────────────────────────────────────────────
    @field_validator("groq_temperature")
    @classmethod
    def validate_temperature(cls, value: float) -> float:
        """Ensure temperature stays within valid LLM range [0.0, 1.0]."""
        if not (0.0 <= value <= 1.0):
            raise ValueError(
                f"groq_temperature must be between 0.0 and 1.0, got {value}"
            )
        return value

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, value: str) -> str:
        """Ensure log level is a valid Python logging level."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = value.upper()
        if upper not in valid_levels:
            raise ValueError(
                f"log_level must be one of {valid_levels}, got '{value}'"
            )
        return upper

    @field_validator("app_env")
    @classmethod
    def validate_app_env(cls, value: str) -> str:
        """Ensure environment is either development or production."""
        valid_envs = {"development", "production"}
        lower = value.lower()
        if lower not in valid_envs:
            raise ValueError(
                f"app_env must be one of {valid_envs}, got '{value}'"
            )
        return lower


# ── Singleton Factory ─────────────────────────────────────────────────────────
@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Return a cached singleton instance of Settings.

    Using @lru_cache ensures the .env file is read only once at startup,
    avoiding repeated I/O on every request.

    Returns:
        Settings: Validated application settings instance.

    Raises:
        ValidationError: If required env vars are missing or invalid.
        FileNotFoundError: If .env file is missing in production.
    """
    try:
        logger.info("Loading application settings from environment...")
        config = Settings()
        logger.info(
            f"Settings loaded | env={config.app_env} | "
            f"log_level={config.log_level} | "
            f"groq_model={config.groq_model} | "
            f"sarvam_stt={config.sarvam_stt_model} | "
            f"sarvam_tts={config.sarvam_tts_model} | "
            f"sarvam_speaker={config.sarvam_tts_speaker}"
        )
        return config
    except Exception as e:
        logger.critical(
            f"Failed to load application settings. "
            f"Please verify your .env file. Error: {str(e)}"
        )
        raise


# ── Module-level settings instance ───────────────────────────────────────────
# Import this directly across all modules: `from config import settings`
settings: Settings = get_settings()