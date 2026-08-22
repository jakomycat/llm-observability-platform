"""Structured logging setup.

Every log record is an event with typed fields rather than a formatted
sentence, so records can be filtered and aggregated instead of grepped.

Two renderers share the same call sites: a human-readable one for local
development and a JSON one for any environment where logs are collected.

Uses structlog's native mode (no standard-library integration). Third-party
libraries logging through `logging` are not reformatted; that will be
revisited once the API server is in place.
"""

import logging
import sys
from typing import Any

import structlog
from structlog.typing import FilteringBoundLogger, Processor

from llm_observability.core.config import Settings, get_settings

_configured = False


def configure_logging(settings: Settings | None = None) -> None:
    """Configure structlog and the standard library logging module.

    Idempotent: repeated calls are ignored, so importing a module twice or
    starting several workers cannot duplicate handlers.

    Args:
        settings: Configuration to use. Defaults to the application settings.
    """
    global _configured
    if _configured:
        return

    settings = settings or get_settings()

    processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    renderer: Processor
    if settings.log_json:
        processors.append(structlog.processors.format_exc_info)
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=[*processors, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelNamesMapping()[settings.log_level]
        ),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stderr,
        level=settings.log_level,
        force=True,
    )

    _configured = True


def reset_logging() -> None:
    """Undo the configuration so it can be applied again. Intended for tests."""
    global _configured
    _configured = False
    structlog.reset_defaults()
    structlog.contextvars.clear_contextvars()


def get_logger(name: str, **initial_values: Any) -> FilteringBoundLogger:
    """Return a logger bound to `name` and to any given permanent fields.

    Args:
        name: Logger name, conventionally the module's `__name__`.
        **initial_values: Fields attached to every record from this logger.

    Returns:
        A logger whose methods accept arbitrary keyword fields.
    """
    configure_logging()
    logger: FilteringBoundLogger = structlog.get_logger().bind(logger=name, **initial_values)

    return logger
