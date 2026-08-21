import pytest

from llm_observability.core.exceptions import (
    ConfigurationError,
    InferenceError,
    LLMObservabilityError,
    ModelError,
    ModelLoadError,
    ModelNotFoundError,
    OutOfMemoryError,
)


def test_base_error_without_context() -> None:
    error = LLMObservabilityError("something failed")
    assert str(error) == "something failed"
    assert error.message == "something failed"
    assert error.context == {}


def test_base_error_includes_context_in_message() -> None:
    error = LLMObservabilityError("invalid model", model="gpt2", layer=3)
    assert str(error) == "invalid model (layer=3, model='gpt2')"
    assert error.context == {"model": "gpt2", "layer": 3}


def test_context_order_is_deterministic() -> None:
    a = LLMObservabilityError("x", b=2, a=1)
    b = LLMObservabilityError("x", a=1, b=2)
    assert str(a) == str(b)


@pytest.mark.parametrize(
    ("child", "parent"),
    [
        (ConfigurationError, LLMObservabilityError),
        (ModelError, LLMObservabilityError),
        (ModelNotFoundError, ModelError),
        (ModelLoadError, ModelError),
        (InferenceError, LLMObservabilityError),
        (OutOfMemoryError, InferenceError),
    ],
)
def test_hierarchy(child: type[Exception], parent: type[Exception]) -> None:
    assert issubclass(child, parent)


def test_specific_error_is_caught_by_base() -> None:
    with pytest.raises(LLMObservabilityError):
        raise ModelNotFoundError("doesn't exist", model="gpt-9")


def test_sibling_branches_are_independent() -> None:
    with pytest.raises(ModelNotFoundError):
        try:
            raise ModelNotFoundError("doesn't exist")
        except InferenceError:  # pragma: no cover
            pytest.fail("ModelNotFoundError shouldn't be caught as InferenceError")
