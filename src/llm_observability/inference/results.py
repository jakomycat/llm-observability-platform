"""Everything that can be measured in execution and everything needed to reproduce execution."""

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True, kw_only=True)
class InferenceMetrics:
    """Store each metric and calculate total tokens and tokens per second.

    Attributes:
        latency_ms: Total time of generation.
        prompt_tokens: Entry tokens.
        completion_tokens: Generated tokens.
        peak_memory_mb: Memory peak.
    """

    latency_ms: float
    prompt_tokens: int
    completion_tokens: int
    peak_memory_mb: float | None = None

    @property
    def total_tokens(self) -> int:
        """Calculate total tokens."""
        return self.prompt_tokens + self.completion_tokens

    @property
    def tokens_per_second(self) -> float:
        """Calculate tokens per second."""
        if self.latency_ms > 0.0:
            return self.completion_tokens / (self.latency_ms / 1000)

        return 0.0


@dataclass(frozen=True, slots=True, kw_only=True)
class GenerationResult:
    """Store each important value for reproducibility, including metrics.

    Attributes:
        model: Model identifier.
        prompt: Exact entry.
        completion: Only generated text.
        token_ids: Generated tokens.
        device: Where it was executed.
        seed: Used seed.
        metrics: The metrics.
    """

    model: str
    prompt: str
    completion: str
    token_ids: tuple[int, ...]
    device: str
    seed: int
    metrics: InferenceMetrics

    def to_dict(self) -> dict[str, Any]:
        """Convert the results into a dictionary.

        The dictionary returned has this shape:
            {
                model: ...,
                prompt: ...,
                completion: ...,
                ...,
                seed: ...,
                metrics: {
                    latency_ms: ...,
                    ...,
                    tokens_per_second: ...
                }
            }
        This contain every attribute from GenerationResult along with information
        about InferenceMetrics stored in metrics.
        """
        dictionary = asdict(self)
        dictionary["metrics"].update(
            {
                "total_tokens": self.metrics.total_tokens,
                "tokens_per_second": self.metrics.tokens_per_second,
            }
        )
        return dictionary
