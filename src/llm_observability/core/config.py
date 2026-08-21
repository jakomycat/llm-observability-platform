"""Application configuration, loaded from the environment and validated at startup."""

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from llm_observability.core.exceptions import ConfigurationError

ENV_PREFIX = "LLMOBS_"

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
Device = Literal["cpu", "cuda", "auto"]


class Settings(BaseSettings):
    """Runtime configuration for the platform.

    Values are read from environment variables and from a local `.env` file,
    validated once at startup. Every field is immutable after construction.

    Environment variables take precedence over the `.env` file, which in turn
    takes precedence over the defaults declared here.
    """

    model_config = SettingsConfigDict(
        env_prefix=ENV_PREFIX,
        env_file=".env",
        env_file_encoding="utf-8",
        frozen=True,
        extra="forbid",
    )

    environment: Literal["local", "ci", "production"] = "local"
    log_level: LogLevel = "INFO"
    log_json: bool = False

    device: Device = "auto"
    default_model: str = "gpt2"
    seed: int = Field(default=42, ge=0)

    max_new_tokens: int = Field(default=64, gt=0, le=4096)
    batch_size: int = Field(default=8, gt=0, le=512)

    artifacts_dir: Path = Path("artifacts")

    @field_validator("default_model")
    @classmethod
    def _model_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            msg = "default_model cannot be blank"
            raise ValueError(msg)
        return value.strip()

    @model_validator(mode="after")
    def _reject_unknown_prefixed_vars(self) -> Self:
        """Fail if the environment defines a prefixed variable with no matching field.

        Guards against silent typos such as `LLMOBS_SEDD`, which the settings
        sources would otherwise ignore, leaving the default value in place.
        """
        known = {f"{ENV_PREFIX}{name}".upper() for name in type(self).model_fields}
        unknown = sorted(
            key
            for key in os.environ
            if key.upper().startswith(ENV_PREFIX) and key.upper() not in known
        )
        if unknown:
            msg = f"Unknown environment variables: {', '.join(unknown)}"
            raise ValueError(msg)
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the application settings, building them only once.

    Raises:
        ConfigurationError: If any value is missing or invalid.
    """
    try:
        return Settings()
    except Exception as exc:
        msg = "Invalid application configuration"
        raise ConfigurationError(msg, reason=str(exc)) from exc
