"""Runtime concerns of inference: device resolution, seeding and memory tracking.

These helpers isolate the side effects that inference depends on, querying the
hardware, mutating global RNG state, reading CUDA counters, so the rest of the
engine stays pure and testable without a GPU.
"""

import random
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import torch

from llm_observability.core.exceptions import ConfigurationError
from llm_observability.core.logging import get_logger

logger = get_logger(__name__)


BYTES_PER_MB = 1024 * 1024


def resolve_device(requested: str) -> torch.device:
    """Resolve a requested device string into a concrete `torch.device`.

    Args:
        requested: One of "cpu", "cuda" or "auto".

    Returns:
        The device inference should run on. "auto" picks CUDA when available
        and falls back to CPU.

    Raises:
        ConfigurationError: If "cuda" is requested on a machine without CUDA,
            or if `requested` is not a recognised value.

    Note:
        "cuda" fails rather than silently falling back: an experiment launched
        expecting a GPU and quietly running on CPU wastes hours and produces
        latency figures that mean nothing. "auto" falls back because that is
        what asking for "auto" means.
    """
    match requested:
        case "cpu":
            logger.info("device_resolved", requested="cpu", device="cpu")
            return torch.device("cpu")

        case "cuda":
            if torch.cuda.is_available():
                logger.info("device_resolved", requested="cuda", device="cuda")
                return torch.device("cuda")

            msg = "CUDA is not available. Select another option."
            raise ConfigurationError(msg, requested=requested)

        case "auto":
            if torch.cuda.is_available():
                logger.info("device_resolved", requested="auto", device="cuda")
                return torch.device("cuda")

            logger.warning("device_resolved", requested="auto", device="cpu")
            return torch.device("cpu")

        case _:
            msg = "Your option is not available. Select another option: cpu, cuda or auto"
            raise ConfigurationError(msg, requested=requested)


def set_seed(seed: int, *, deterministic: bool = False) -> None:
    """Seed every random number generator used by the inference pipeline.

    Args:
        seed: Value applied to Python's `random`, NumPy, and PyTorch (CPU and,
            when present, CUDA).
        deterministic: When True, also force deterministic CUDA kernels. This
            is slower, sometimes substantially, so it stays off by default,
            since the platform measures latency and determinism would distort
            those figures. When False, the flags are left untouched rather than
            actively disabled.

    Note:
        Seeding does not cover `PYTHONHASHSEED`, which affects string hash
        ordering and can only be set before the interpreter starts. Full
        determinism also requires `CUBLAS_WORKSPACE_CONFIG` for some CUDA
        operations.
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    logger.info("seed_set", seed=seed, deterministic=deterministic)


@dataclass(slots=True)
class MemoryStats:
    """Peak memory of a tracked block.

    Attributes:
        peak_mb: Peak allocated memory in MiB, or None on devices that
            do not expose the measurement.
    """

    peak_mb: float | None = None


@contextmanager
def track_peak_memory(device: torch.device) -> Iterator[MemoryStats]:
    """Track peak memory allocated while the block runs.

    On CUDA the peak counter is reset on entry and read on exit. On any other
    device the measurement is unavailable and `peak_mb` stays None.

    Args:
        device: Device the tracked work runs on.

    Yields:
        A `MemoryStats` populated when the block exits.
    """
    stats = MemoryStats()
    tracked = device.type == "cuda" and torch.cuda.is_available()

    if tracked:
        torch.cuda.reset_peak_memory_stats()

    try:
        yield stats
    finally:
        if tracked:
            torch.cuda.synchronize()
            stats.peak_mb = torch.cuda.max_memory_allocated() / BYTES_PER_MB
