import json
import logging

import pytest
import structlog

from llm_observability.core.config import Settings
from llm_observability.core.logging import configure_logging, get_logger


def _capture() -> structlog.testing.CapturingLoggerFactory:
    """Swap the logger factory while keeping the processor chain intact."""
    factory = structlog.testing.CapturingLoggerFactory()
    structlog.configure(logger_factory=factory)
    return factory


def test_configure_is_idempotent() -> None:
    configure_logging(Settings())
    first = structlog.get_config()["processors"]
    configure_logging(Settings())
    assert structlog.get_config()["processors"] is first


def test_logger_emits_event_and_fields() -> None:
    configure_logging(Settings(log_json=True))
    factory = _capture()

    get_logger(__name__).info("inference_completed", model="gpt2", latency_ms=82)

    payload = json.loads(factory.logger.calls[0].args[0])
    assert payload["event"] == "inference_completed"
    assert payload["model"] == "gpt2"
    assert payload["latency_ms"] == 82
    assert payload["level"] == "info"


def test_logger_name_is_bound_as_a_field() -> None:
    configure_logging(Settings(log_json=True))
    factory = _capture()

    get_logger("my.module").info("model_loaded")

    assert json.loads(factory.logger.calls[0].args[0])["logger"] == "my.module"


def test_initial_values_are_bound_to_every_record() -> None:
    configure_logging(Settings(log_json=True))
    factory = _capture()

    logger = get_logger(__name__, component="inference")
    logger.info("model_loaded")
    logger.info("inference_started")

    components = [json.loads(call.args[0])["component"] for call in factory.logger.calls]
    assert components == ["inference", "inference"]


def test_contextvars_are_merged() -> None:
    configure_logging(Settings(log_json=True))
    factory = _capture()
    structlog.contextvars.bind_contextvars(request_id="abc123")
    try:
        structlog.get_logger().info("request_handled")
    finally:
        structlog.contextvars.clear_contextvars()

    payload = json.loads(factory.logger.calls[0].args[0])
    assert payload["request_id"] == "abc123"


def test_level_below_threshold_is_dropped() -> None:
    configure_logging(Settings(log_level="WARNING"))
    with structlog.testing.capture_logs() as logs:
        logger = get_logger(__name__)
        logger.debug("noisy_detail")
        logger.warning("something_odd")

    assert [entry["event"] for entry in logs] == ["something_odd"]


def test_json_output_is_valid_json() -> None:
    configure_logging(Settings(log_json=True))
    factory = _capture()
    get_logger(__name__).info("inference_completed", model="gpt2")

    payload = json.loads(factory.logger.calls[0].args[0])
    assert payload["event"] == "inference_completed"
    assert payload["model"] == "gpt2"
    assert payload["logger"] == __name__
    assert "timestamp" in payload


def test_console_output_is_not_json() -> None:
    configure_logging(Settings(log_json=False))
    factory = _capture()
    structlog.get_logger().info("inference_completed", model="gpt2")

    with pytest.raises(json.JSONDecodeError):
        json.loads(factory.logger.calls[0].args[0])


def test_stdlib_logging_level_follows_settings() -> None:
    configure_logging(Settings(log_level="ERROR"))
    assert logging.getLogger().level == logging.ERROR
