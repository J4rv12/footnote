"""The log contract: one JSON object per line, with the same keys every time.

A log line is the only record a question leaves on the server. So the shape
is a contract, asserted with full equality, like `/healthz`'s body.
"""

import io
import json
import logging
from collections.abc import Iterator

import pytest

from app.logs import JsonFormatter, configure_logging, log_event


@pytest.fixture
def captured() -> Iterator[tuple[logging.Logger, io.StringIO]]:
    """A logger whose JSON lines land in a buffer instead of stderr."""
    buffer = io.StringIO()
    handler = logging.StreamHandler(buffer)
    handler.setFormatter(JsonFormatter())
    logger = logging.getLogger("tests.logs")
    logger.addHandler(handler)
    logger.propagate = False
    yield logger, buffer
    logger.removeHandler(handler)


def test_event_is_one_json_line_carrying_the_required_fields(
    captured: tuple[logging.Logger, io.StringIO],
) -> None:
    logger, buffer = captured

    log_event(
        logger,
        request_id="req-1",
        stage="retrieve",
        outcome="degraded",
        latency_ms=12.34,
        reason="voyage_timeout",
    )

    (line,) = buffer.getvalue().splitlines()
    assert json.loads(line) == {
        "level": "WARNING",
        "msg": "retrieve degraded",
        "request_id": "req-1",
        "stage": "retrieve",
        "outcome": "degraded",
        "latency_ms": 12.3,
        "reason": "voyage_timeout",
    }


def test_unexpected_failure_logs_the_traceback_at_error(
    captured: tuple[logging.Logger, io.StringIO],
) -> None:
    """The last-resort net's line must say what it caught, or it hides bugs."""
    logger, buffer = captured

    try:
        raise ZeroDivisionError("simulated")
    except ZeroDivisionError:
        log_event(
            logger,
            request_id="req-1",
            stage="ask",
            outcome="degraded",
            latency_ms=1.0,
            reason="internal_error",
            exc_info=True,
        )

    (line,) = buffer.getvalue().splitlines()
    entry = json.loads(line)
    assert entry["level"] == "ERROR"
    assert entry["reason"] == "internal_error"
    assert "ZeroDivisionError: simulated" in entry["exc"]
    assert "Traceback" in entry["exc"]


def test_http_client_loggers_are_quiet_at_info() -> None:
    """httpx and httpx2 log one line per request with none of the event fields.

    Drop either line from `configure_logging` and this fails, because the root
    logger is at INFO and both inherit it.
    """
    root = logging.getLogger()
    saved_handlers, saved_level = root.handlers[:], root.level
    try:
        configure_logging("INFO")
        assert not logging.getLogger("httpx").isEnabledFor(logging.INFO)
        assert not logging.getLogger("httpx2").isEnabledFor(logging.INFO)
    finally:
        root.handlers = saved_handlers
        root.setLevel(saved_level)
