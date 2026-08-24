import random
from collections.abc import Iterator

import numpy as np
import pytest
import torch

from llm_observability.core.exceptions import ConfigurationError
from llm_observability.inference.runtime import (
    BYTES_PER_MB,
    MemoryStats,
    resolve_device,
    set_seed,
    track_peak_memory,
)


@pytest.fixture(autouse=True)
def _restore_torch_globals() -> Iterator[None]:
    """Restore PyTorch's process-wide determinism flags after every test."""
    deterministic = torch.are_deterministic_algorithms_enabled()
    cudnn_deterministic = torch.backends.cudnn.deterministic
    cudnn_benchmark = torch.backends.cudnn.benchmark
    yield
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.deterministic = cudnn_deterministic
    torch.backends.cudnn.benchmark = cudnn_benchmark


def _pretend_cuda(monkeypatch: pytest.MonkeyPatch, *, available: bool) -> None:
    """Make `torch.cuda.is_available()` report the given answer."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: available)


@pytest.mark.parametrize("cuda", [True, False])
def test_cpu_is_honoured_regardless_of_hardware(
    monkeypatch: pytest.MonkeyPatch, cuda: bool
) -> None:
    _pretend_cuda(monkeypatch, available=cuda)
    assert resolve_device("cpu").type == "cpu"


def test_cuda_is_returned_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    _pretend_cuda(monkeypatch, available=True)
    assert resolve_device("cuda").type == "cuda"


def test_cuda_fails_when_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    _pretend_cuda(monkeypatch, available=False)
    with pytest.raises(ConfigurationError) as exc_info:
        resolve_device("cuda")
    assert exc_info.value.context == {"requested": "cuda"}


def test_auto_prefers_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    _pretend_cuda(monkeypatch, available=True)
    assert resolve_device("auto").type == "cuda"


def test_auto_falls_back_to_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    _pretend_cuda(monkeypatch, available=False)
    assert resolve_device("auto").type == "cpu"


@pytest.mark.parametrize("requested", ["gpu", "CPU", "", "cuda:0"])
def test_unknown_device_is_rejected(monkeypatch: pytest.MonkeyPatch, requested: str) -> None:
    _pretend_cuda(monkeypatch, available=True)
    with pytest.raises(ConfigurationError) as exc_info:
        resolve_device(requested)
    assert exc_info.value.context == {"requested": requested}


def test_torch_values_repeat_for_the_same_seed() -> None:
    set_seed(42)
    first = torch.rand(5)
    set_seed(42)
    second = torch.rand(5)
    assert torch.equal(first, second)


def test_torch_values_differ_for_different_seeds() -> None:
    set_seed(42)
    first = torch.rand(5)
    set_seed(43)
    second = torch.rand(5)
    assert not torch.equal(first, second)


def test_python_and_numpy_generators_are_seeded() -> None:
    set_seed(7)
    first = (random.random(), float(np.random.rand()))
    set_seed(7)
    second = (random.random(), float(np.random.rand()))
    assert first == second


def test_cuda_generator_is_seeded_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seeds: list[int] = []
    _pretend_cuda(monkeypatch, available=True)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda s: seeds.append(s))

    set_seed(99)

    assert seeds == [99, 99]


def test_cuda_seeding_is_skipped_by_our_guard_when_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seeds: list[int] = []
    _pretend_cuda(monkeypatch, available=False)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda s: seeds.append(s))

    set_seed(99)

    assert seeds == [99]


def test_deterministic_mode_enables_every_flag() -> None:
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    set_seed(42, deterministic=True)

    assert torch.are_deterministic_algorithms_enabled()
    assert torch.backends.cudnn.deterministic
    assert not torch.backends.cudnn.benchmark


def test_flags_are_left_alone_by_default() -> None:
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    set_seed(42)

    assert not torch.are_deterministic_algorithms_enabled()
    assert not torch.backends.cudnn.deterministic
    assert torch.backends.cudnn.benchmark


def _pretend_cuda_memory(
    monkeypatch: pytest.MonkeyPatch, *, peak_bytes: int, calls: list[str]
) -> None:
    """Replace the CUDA memory API with recording stubs."""
    _pretend_cuda(monkeypatch, available=True)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: calls.append("reset"))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: calls.append("sync"))
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: peak_bytes)


def test_memory_stats_default_to_none() -> None:
    assert MemoryStats().peak_mb is None


def test_cpu_leaves_the_measurement_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    _pretend_cuda_memory(monkeypatch, peak_bytes=0, calls=calls)

    with track_peak_memory(torch.device("cpu")) as stats:
        pass

    assert stats.peak_mb is None
    assert calls == []


def test_cuda_resets_on_entry_and_reads_on_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    _pretend_cuda_memory(monkeypatch, peak_bytes=4 * BYTES_PER_MB, calls=calls)

    with track_peak_memory(torch.device("cuda")) as stats:
        calls.append("work")

    assert calls == ["reset", "work", "sync"]
    assert stats.peak_mb == 4.0


def test_bytes_are_converted_to_mebibytes(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    _pretend_cuda_memory(monkeypatch, peak_bytes=1_572_864, calls=calls)

    with track_peak_memory(torch.device("cuda")) as stats:
        pass

    assert stats.peak_mb == 1.5


def test_measurement_survives_a_failing_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    _pretend_cuda_memory(monkeypatch, peak_bytes=2 * BYTES_PER_MB, calls=calls)

    with pytest.raises(RuntimeError), track_peak_memory(torch.device("cuda")) as stats:
        msg = "out of memory"
        raise RuntimeError(msg)

    assert stats.peak_mb == 2.0
