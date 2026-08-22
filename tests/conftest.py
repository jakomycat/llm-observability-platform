"""Fixtures shared by the whole test suite."""

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from llm_observability.core.config import get_settings
from llm_observability.core.logging import reset_logging


@pytest.fixture(autouse=True)
def isolated_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    """Run every test with a clean environment, an empty CWD and fresh globals.

    Prevents the developer's own `LLMOBS_*` variables, the repository `.env`
    file, and module-level caches from leaking between tests.
    """
    for key in list(os.environ):
        if key.upper().startswith("LLMOBS_"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(tmp_path)
    get_settings.cache_clear()
    reset_logging()
    yield
    get_settings.cache_clear()
    reset_logging()
