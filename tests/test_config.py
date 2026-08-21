import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from llm_observability.core.config import Settings, get_settings
from llm_observability.core.exceptions import ConfigurationError


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Give each test a clean environment and an empty working directory."""
    for key in list(os.environ):
        if key.startswith("LLMOBS_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(tmp_path)
    get_settings.cache_clear()


def test_defaults_are_sane() -> None:
    settings = Settings()
    assert settings.seed == 42
    assert settings.device == "auto"
    assert settings.batch_size == 8


def test_env_vars_override_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLMOBS_SEED", "123")
    monkeypatch.setenv("LLMOBS_DEVICE", "cpu")
    settings = Settings()
    assert settings.seed == 123
    assert settings.device == "cpu"


def test_env_file_is_read(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("LLMOBS_SEED=7\n", encoding="utf-8")
    assert Settings().seed == 7


def test_env_var_wins_over_env_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("LLMOBS_SEED=7\n", encoding="utf-8")
    monkeypatch.setenv("LLMOBS_SEED", "99")
    assert Settings().seed == 99


def test_settings_are_immutable() -> None:
    settings = Settings()
    with pytest.raises(ValidationError):
        settings.seed = 7


def test_unknown_variable_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLMOBS_SEDD", "42")
    with pytest.raises(ValidationError):
        Settings()


@pytest.mark.parametrize(
    ("variable", "value"),
    [
        ("LLMOBS_BATCH_SIZE", "0"),
        ("LLMOBS_BATCH_SIZE", "-1"),
        ("LLMOBS_MAX_NEW_TOKENS", "99999"),
        ("LLMOBS_SEED", "-1"),
        ("LLMOBS_DEVICE", "gpu"),
        ("LLMOBS_LOG_LEVEL", "TRACE"),
    ],
)
def test_invalid_values_are_rejected(
    monkeypatch: pytest.MonkeyPatch, variable: str, value: str
) -> None:
    monkeypatch.setenv(variable, value)
    with pytest.raises(ValidationError):
        Settings()


def test_blank_model_name_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLMOBS_DEFAULT_MODEL", "   ")
    with pytest.raises(ValidationError):
        Settings()


def test_model_name_is_stripped(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLMOBS_DEFAULT_MODEL", "  gpt2  ")
    assert Settings().default_model == "gpt2"


def test_get_settings_wraps_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLMOBS_BATCH_SIZE", "0")
    with pytest.raises(ConfigurationError):
        get_settings()


def test_get_settings_is_cached() -> None:
    assert get_settings() is get_settings()
