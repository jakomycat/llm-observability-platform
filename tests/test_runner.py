from typing import Any

import pytest
import torch
from huggingface_hub.errors import HFValidationError, RepositoryNotFoundError
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_observability.core.config import Settings
from llm_observability.core.exceptions import (
    ConfigurationError,
    ModelLoadError,
    ModelNotFoundError,
)
from llm_observability.inference.runner import ModelRunner

TINY_MODEL = "hf-internal-testing/tiny-random-gpt2"


class _FakeParameter:
    def numel(self) -> int:
        return 100


class _FakeModel:
    """Stands in for a PreTrainedModel without touching the network."""

    def __init__(self) -> None:
        self.moved_to: torch.device | None = None
        self.eval_called = False

    def to(self, device: torch.device) -> "_FakeModel":
        self.moved_to = device
        return self

    def eval(self) -> "_FakeModel":
        self.eval_called = True
        return self

    def parameters(self) -> list[_FakeParameter]:
        return [_FakeParameter(), _FakeParameter()]


class _FakeTokenizer:
    pass


def _hub_error(cls: type[Exception]) -> Exception:
    """Build a Hub exception without running its `__init__`.

    `RepositoryNotFoundError.__init__` reaches into `response.headers`, so it
    can't be constructed with `response=None`. These tests only need an object
    of the right type for the `isinstance` check inside `load()`, so bypass the
    constructor entirely — that also keeps the tests immune to signature
    changes in `huggingface_hub`.
    """
    return cls.__new__(cls)


def _wrapped_by_transformers(cause: Exception) -> OSError:
    """Rebuild what `transformers` actually hands back.

    `from_pretrained` catches the `huggingface_hub` error and re-raises
    `OSError(...) from e`. `raise X from e` is nothing more than
    `X.__cause__ = e`, so the same object can be built without raising.
    """
    err = OSError(
        "does/not-exist is not a valid model identifier listed on 'https://huggingface.co/models'"
    )
    err.__cause__ = cause
    return err


@pytest.fixture
def stub_hub(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Replace both `from_pretrained` calls with in-memory fakes.

    Returns the list of names they were called with, in order.
    """
    calls: list[str] = []

    def fake_model(name: str, **kwargs: Any) -> _FakeModel:
        calls.append(f"model:{name}")
        return _FakeModel()

    def fake_tokenizer(name: str, **kwargs: Any) -> _FakeTokenizer:
        calls.append(f"tokenizer:{name}")
        return _FakeTokenizer()

    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", fake_model)
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", fake_tokenizer)
    return calls


def _raiser(exc: Exception) -> Any:
    def _fail(name: str, **kwargs: Any) -> Any:
        raise exc

    return _fail


def test_construction_does_not_load_anything(stub_hub: list[str]) -> None:
    ModelRunner("gpt2")
    assert stub_hub == []


def test_model_name_is_exposed() -> None:
    assert ModelRunner("gpt2").model_name == "gpt2"


def test_device_from_settings_is_resolved() -> None:
    runner = ModelRunner("gpt2", settings=Settings(device="cpu"))
    assert runner.device.type == "cpu"


def test_explicit_device_overrides_settings() -> None:
    runner = ModelRunner("gpt2", device="cpu", settings=Settings(device="auto"))
    assert runner.device.type == "cpu"


def test_invalid_device_is_rejected() -> None:
    with pytest.raises(ConfigurationError):
        ModelRunner("gpt2", device="gpu")


def test_public_attributes_are_read_only() -> None:
    runner = ModelRunner("gpt2")
    with pytest.raises(AttributeError):
        runner.model_name = "bert"  # type: ignore[misc]
    with pytest.raises(AttributeError):
        runner.device = torch.device("cpu")  # type: ignore[misc]


def test_repr_shows_load_state(stub_hub: list[str]) -> None:
    runner = ModelRunner("gpt2", device="cpu")
    assert repr(runner) == "ModelRunner(model_name='gpt2', device='cpu', loaded=False)"

    runner.load()

    assert repr(runner) == "ModelRunner(model_name='gpt2', device='cpu', loaded=True)"


def test_load_fetches_tokenizer_before_model(stub_hub: list[str]) -> None:
    ModelRunner("gpt2").load()
    assert stub_hub == ["tokenizer:gpt2", "model:gpt2"]


def test_load_is_idempotent(stub_hub: list[str]) -> None:
    runner = ModelRunner("gpt2")
    runner.load()
    runner.load()
    runner.load()
    assert stub_hub == ["tokenizer:gpt2", "model:gpt2"]


def test_load_moves_the_model_and_switches_to_eval(stub_hub: list[str]) -> None:
    runner = ModelRunner("gpt2", device="cpu")
    runner.load()

    model: object = runner.model
    assert isinstance(model, _FakeModel)
    assert model.moved_to == runner.device
    assert model.eval_called


def test_accessing_model_triggers_loading(stub_hub: list[str]) -> None:
    runner = ModelRunner("gpt2")
    assert stub_hub == []

    assert runner.model is not None

    assert stub_hub == ["tokenizer:gpt2", "model:gpt2"]


def test_accessing_tokenizer_triggers_loading(stub_hub: list[str]) -> None:
    runner = ModelRunner("gpt2")
    assert runner.tokenizer is not None
    assert stub_hub == ["tokenizer:gpt2", "model:gpt2"]


def test_missing_repository_becomes_model_not_found(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hub_error = _hub_error(RepositoryNotFoundError)
    raised = _wrapped_by_transformers(hub_error)
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", _raiser(raised))

    with pytest.raises(ModelNotFoundError) as exc_info:
        ModelRunner("does/not-exist").load()

    assert exc_info.value.context["model"] == "does/not-exist"
    assert exc_info.value.__cause__ is raised
    assert exc_info.value.__cause__.__cause__ is hub_error


def test_a_deeply_chained_repository_error_is_still_recognised(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hub_error = _hub_error(RepositoryNotFoundError)
    inner = _wrapped_by_transformers(hub_error)
    outer = OSError("could not load the tokenizer")
    outer.__cause__ = inner
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", _raiser(outer))

    with pytest.raises(ModelNotFoundError):
        ModelRunner("does/not-exist").load()


def test_an_unrelated_cause_still_becomes_model_load_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cause = OSError("disk full")
    cause.__cause__ = ValueError("something unrelated")
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", _raiser(cause))

    with pytest.raises(ModelLoadError) as exc_info:
        ModelRunner("gpt2").load()

    assert exc_info.value.context["model"] == "gpt2"


def test_malformed_identifier_becomes_model_not_found(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cause = HFValidationError("bad id")
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", _raiser(cause))

    with pytest.raises(ModelNotFoundError) as exc_info:
        ModelRunner("bad id").load()

    assert exc_info.value.context["model"] == "bad id"
    assert exc_info.value.__cause__ is cause


def test_unreadable_files_become_model_load_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cause = OSError("disk full")
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", _raiser(cause))

    with pytest.raises(ModelLoadError) as exc_info:
        ModelRunner("gpt2").load()

    assert exc_info.value.context["model"] == "gpt2"
    assert exc_info.value.__cause__ is cause


def test_failed_model_load_leaves_the_runner_unloaded(
    monkeypatch: pytest.MonkeyPatch, stub_hub: list[str]
) -> None:
    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", _raiser(OSError("corrupt")))
    runner = ModelRunner("gpt2")

    with pytest.raises(ModelLoadError):
        runner.load()

    assert "loaded=False" in repr(runner)


def test_unknown_errors_are_not_swallowed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", _raiser(ValueError("weird")))

    with pytest.raises(ValueError, match="weird"):
        ModelRunner("gpt2").load()


@pytest.mark.slow
def test_a_real_tiny_model_loads() -> None:
    runner = ModelRunner(TINY_MODEL, device="cpu")
    runner.load()

    assert runner.model.training is False
    assert next(runner.model.parameters()).device.type == "cpu"
    assert runner.tokenizer.encode("hello")


@pytest.mark.slow
def test_a_missing_repository_raises_model_not_found() -> None:
    with pytest.raises(ModelNotFoundError):
        ModelRunner("this-org/definitely-does-not-exist-42", device="cpu").load()
