"""Loading and running Hugging Face causal models on a resolved device."""

import time

import torch
import torch.nn as nn
from huggingface_hub.errors import HFValidationError, RepositoryNotFoundError
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from llm_observability.core.config import Settings, get_settings
from llm_observability.core.exceptions import ModelLoadError, ModelNotFoundError
from llm_observability.core.logging import get_logger
from llm_observability.inference.runtime import resolve_device, set_seed

logger = get_logger(__name__)


def _caused_by(exc: BaseException, types: tuple[type[BaseException], ...]) -> bool:
    """Walk the exception chain looking for one of `types`.

    ``transformers`` catches Hub errors and re-raises them as :class:`OSError`,
    preserving the original through ``raise ... from``. Inspecting the chain is
    the only reliable way to tell a missing repository from a corrupt download.

    Args:
        exc: Exception to inspect, including its ``__cause__`` ancestors.
        types: Exception types to look for.

    Returns:
        ``True`` if `exc` or any of its causes is an instance of `types`.
    """
    current: BaseException | None = exc
    while current is not None:
        if isinstance(current, types):
            return True
        current = current.__cause__
    return False


class ModelRunner:
    """It wraps a model and its tokenizer along with the device on which they run.

    Configuration is resolved once during construction: if neither
    `device` nor `settings` are passed, they are taken from the global configuration.
    The model and the tokenizer are loaded lazily, so instantiating the class is
    inexpensive. The state is read-only once the object is constructed.

    Args:
        model_name: Hub repository ID of the causal LM to load.
            Also accepts a path to a local checkpoint directory.
        device: Requested device, such as ``"cpu"``, ``"cuda"`` or ``"auto"``.
            Defaults to ``settings.device``. The string is normalised by
            :func:`resolve_device`, which may fall back to CPU if the
            requested accelerator is unavailable.
        settings: Configuration object. Defaults to the process-wide
            settings returned by :func:`get_settings`.

    Raises:
        ConfigurationError: If `device` is not a device string that
            :func:`resolve_device` recognises.
    """

    def __init__(
        self, model_name: str, *, device: str | None = None, settings: Settings | None = None
    ) -> None:
        if settings is None:
            settings = get_settings()

        if device is None:
            device = settings.device

        set_seed(settings.seed)

        self._settings: Settings = settings
        self._model_name: str = model_name
        self._device: torch.device = resolve_device(device)

        self._model: PreTrainedModel | None = None
        self._tokenizer: PreTrainedTokenizerBase | None = None

        logger.info("runner_created", model=model_name, requested=device)

    def __repr__(self) -> str:
        return (
            f"ModelRunner(model_name={self._model_name!r}, "
            f"device={str(self._device)!r}, "
            f"loaded={self._model is not None})"
        )

    @property
    def model_name(self) -> str:
        """Model ID for the hub."""
        return self._model_name

    @property
    def device(self) -> torch.device:
        """Device resolved where the model runs."""
        return self._device

    @property
    def model(self) -> PreTrainedModel:
        """The loaded model, loading it on first access."""
        self.load()
        if self._model is None:
            msg = "Model is not loaded"
            raise ModelLoadError(msg, model=self._model_name)
        return self._model

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        """The loaded tokenizer, loading it on first access."""
        self.load()
        if self._tokenizer is None:
            msg = "Tokenizer is not loaded"
            raise ModelLoadError(msg, model=self._model_name)
        return self._tokenizer

    def load(self) -> None:
        """Load the tokenizer and the model onto the resolved device.

        The model is moved to ``self.device`` and switched to evaluation mode.
        The call is idempotent: if the model is already loaded it returns
        immediately without touching the network.

        Raises:
            ModelNotFoundError: If the repository does not exist on the Hub or
                the identifier is malformed.
            ModelLoadError: If the weights or the tokenizer files exist but
                cannot be read, downloaded or deserialised.
        """
        if self._model is not None:
            return

        start = time.perf_counter()

        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        except (RepositoryNotFoundError, HFValidationError) as exc:
            msg = "Model not found"
            raise ModelNotFoundError(msg, model=self.model_name) from exc
        except OSError as exc:
            if _caused_by(exc, (RepositoryNotFoundError,)):
                msg = "Model not found"
                raise ModelNotFoundError(msg, model=self.model_name) from exc
            msg = "Could not load tokenizer"
            raise ModelLoadError(msg, model=self.model_name, reason=str(exc)) from exc

        try:
            model = AutoModelForCausalLM.from_pretrained(self.model_name)
        except (RepositoryNotFoundError, HFValidationError) as exc:
            msg = "Model not found"
            raise ModelNotFoundError(msg, model=self.model_name) from exc
        except OSError as exc:
            if _caused_by(exc, (RepositoryNotFoundError,)):
                msg = "Model not found"
                raise ModelNotFoundError(msg, model=self.model_name) from exc
            msg = "Could not load model"
            raise ModelLoadError(msg, model=self.model_name, reason=str(exc)) from exc

        module: nn.Module = model
        module.to(self.device)
        module.eval()

        self._model = model
        self._tokenizer = tokenizer

        end = time.perf_counter()

        load_time_ms = (end - start) * 1000
        parameters = sum(p.numel() for p in model.parameters())

        logger.info(
            "model_loaded",
            model=self.model_name,
            device=str(self._device),
            load_time_ms=load_time_ms,
            parameters=parameters,
        )
