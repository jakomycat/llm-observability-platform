import json
from dataclasses import FrozenInstanceError

import pytest

from llm_observability.inference.results import GenerationResult, InferenceMetrics


def _metrics(**overrides: float | int | None) -> InferenceMetrics:
    defaults: dict[str, float | int | None] = {
        "latency_ms": 1000.0,
        "prompt_tokens": 5,
        "completion_tokens": 3,
        "peak_memory_mb": 1320.5,
    }
    return InferenceMetrics(**{**defaults, **overrides})  # type: ignore[arg-type]


def _result(**overrides: object) -> GenerationResult:
    defaults: dict[str, object] = {
        "model": "gpt2",
        "prompt": "The capital of France is",
        "completion": " Paris",
        "token_ids": (40, 1917, 318),
        "device": "cpu",
        "seed": 42,
        "metrics": _metrics(),
    }
    return GenerationResult(**{**defaults, **overrides})  # type: ignore[arg-type]


def test_metrics_expose_every_field() -> None:
    metrics = _metrics()
    assert metrics.latency_ms == 1000.0
    assert metrics.prompt_tokens == 5
    assert metrics.completion_tokens == 3
    assert metrics.peak_memory_mb == 1320.5


def test_peak_memory_defaults_to_none() -> None:
    metrics = InferenceMetrics(latency_ms=10.0, prompt_tokens=1, completion_tokens=1)
    assert metrics.peak_memory_mb is None


def test_total_tokens_sums_prompt_and_completion() -> None:
    assert _metrics(prompt_tokens=5, completion_tokens=3).total_tokens == 8


def test_tokens_per_second_uses_completion_tokens_over_seconds() -> None:
    metrics = _metrics(latency_ms=1000.0, completion_tokens=10)
    assert metrics.tokens_per_second == 10.0


def test_tokens_per_second_converts_milliseconds() -> None:
    metrics = _metrics(latency_ms=500.0, completion_tokens=10)
    assert metrics.tokens_per_second == 20.0


@pytest.mark.parametrize("latency", [0.0, -1.0])
def test_tokens_per_second_is_zero_for_non_positive_latency(latency: float) -> None:
    assert _metrics(latency_ms=latency).tokens_per_second == 0.0


def test_metrics_are_immutable() -> None:
    with pytest.raises(FrozenInstanceError):
        _metrics().latency_ms = 5.0  # type: ignore[misc]


def test_result_is_immutable() -> None:
    with pytest.raises(FrozenInstanceError):
        _result().model = "bert"  # type: ignore[misc]


def test_equal_values_compare_equal() -> None:
    assert _result() == _result()


def test_different_values_compare_unequal() -> None:
    assert _result() != _result(seed=7)


def test_result_is_hashable() -> None:
    assert len({_result(), _result(), _result(seed=7)}) == 2


def test_to_dict_contains_every_top_level_field() -> None:
    payload = _result().to_dict()
    assert payload.keys() == {
        "model",
        "prompt",
        "completion",
        "token_ids",
        "device",
        "seed",
        "metrics",
    }


def test_to_dict_nests_metrics_with_derived_values() -> None:
    payload = _result().to_dict()
    assert payload["metrics"].keys() == {
        "latency_ms",
        "prompt_tokens",
        "completion_tokens",
        "peak_memory_mb",
        "total_tokens",
        "tokens_per_second",
    }


def test_to_dict_derived_values_match_the_properties() -> None:
    result = _result()
    payload = result.to_dict()
    assert payload["metrics"]["total_tokens"] == result.metrics.total_tokens
    assert payload["metrics"]["tokens_per_second"] == result.metrics.tokens_per_second


def test_to_dict_is_json_serializable() -> None:
    payload = json.loads(json.dumps(_result().to_dict()))
    assert payload["model"] == "gpt2"
    assert payload["token_ids"] == [40, 1917, 318]


def test_to_dict_does_not_mutate_the_result() -> None:
    result = _result()
    result.to_dict()
    assert result.to_dict()["metrics"]["total_tokens"] == 8
