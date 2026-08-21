"""Project exception hierarchy.

All exceptions in the domain inherit from `LLMObservabilityError`, which
allows them to be selectively caught without catching programming errors.
"""

from typing import Any


class LLMObservabilityError(Exception):
    """Exception base of the platform.

    Attributes:
        message: Legible description of the error.
        context: Additional data useful for diagnostics and logging.
    """

    def __init__(self, message: str, **context: Any) -> None:
        super().__init__(message)

        self.message = message
        self.context = context

    def __str__(self) -> str:
        """Return the message with its context appended, if any."""
        if not self.context:
            return self.message
        details = ", ".join(f"{k}={v!r}" for k, v in sorted(self.context.items()))

        return f"{self.message} ({details})"


class ConfigurationError(LLMObservabilityError):
    """The configuration is invalid, incomplete, or inconsistent."""


class ModelError(LLMObservabilityError):
    """Generic error related to a model."""


class ModelNotFoundError(ModelError):
    """The requested model doesn't exist or is not available."""


class ModelLoadError(ModelError):
    """The model exists but couldn't be loaded."""


class InferenceError(LLMObservabilityError):
    """The inference failed during execution."""


class OutOfMemoryError(InferenceError):
    """There isn't enough memory (GPU or CPU) to complete the operation."""
