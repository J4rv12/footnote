"""Embeddings: the request Voyage receives, the vectors we keep, every refusal.

The success body is real: one voyage-4 call, recorded in U08, embedding the
first sentences of matchday-ops-2026-27 sections 1.1 and 1.2 as documents.
Bodies altered for a test (reordered, truncated, cut short) say so where they
appear. Network failures are simulated at the transport, as in SitRep.
"""

import json
import logging
from pathlib import Path

import httpx
import numpy as np
import pytest

from app.config import get_settings
from app.embed import embed
from app.logs import JsonFormatter

RECORDED = json.loads(
    (Path(__file__).parent / "fixtures" / "voyage_embeddings_two_documents.json").read_bytes()
)
TEXTS = [
    "For home matches, players report to the home dressing room at the Middling Ground "
    "2 h 15 min before kick-off.",
    "The player car park at the Middling Ground opens 4 hours before kick-off.",
]


@pytest.fixture(autouse=True)
def capture_info_logs(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="app.embed")


def embed_records(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [record for record in caplog.records if record.name == "app.embed"]


def assert_no_key_in(records: list[logging.LogRecord]) -> None:
    key = get_settings().voyage_api_key
    for record in records:
        assert key not in JsonFormatter().format(record)


def serve(
    status: int, body: object, seen: list[httpx.Request] | None = None
) -> httpx.MockTransport:
    """A transport that answers every request with `status` and `body` as JSON."""

    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        return httpx.Response(status, json=body)

    return httpx.MockTransport(handler)


def fail_with(error: type[httpx.TransportError]) -> httpx.MockTransport:
    """A transport that raises the way httpx does when the network fails."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise error("simulated", request=request)

    return httpx.MockTransport(handler)


@pytest.mark.parametrize("input_type", ["document", "query"])
async def test_returns_one_row_per_text_and_sends_settings_from_config(
    input_type: str, caplog: pytest.LogCaptureFixture
) -> None:
    seen: list[httpx.Request] = []

    vectors = await embed(TEXTS, input_type, "req-1", transport=serve(200, RECORDED, seen))

    assert vectors is not None
    assert vectors.dtype == np.float32
    assert vectors.shape == (2, 1024)
    expected = np.array([row["embedding"] for row in RECORDED["data"]], dtype=np.float32)
    assert np.array_equal(vectors, expected)

    settings = get_settings()
    (request,) = seen
    assert request.method == "POST"
    assert str(request.url) == "https://api.voyageai.com/v1/embeddings"
    assert request.headers["authorization"] == f"Bearer {settings.voyage_api_key}"
    assert json.loads(request.content) == {
        "input": TEXTS,
        "model": "voyage-4",
        "input_type": input_type,
        "output_dimension": 1024,
    }

    records = embed_records(caplog)
    (record,) = records
    assert (record.levelname, record.request_id, record.stage, record.outcome) == (
        "INFO",
        "req-1",
        "embed",
        "ok",
    )
    assert_no_key_in(records)


async def test_rows_follow_voyage_index_not_body_order() -> None:
    """The recorded body with its two vectors swapped. Written for this test."""
    swapped = {**RECORDED, "data": RECORDED["data"][::-1]}

    vectors = await embed(TEXTS, "document", "req-1", transport=serve(200, swapped))

    assert vectors is not None
    assert np.array_equal(vectors[0], np.array(RECORDED["data"][0]["embedding"], np.float32))


# Malformed bodies are the recorded one, altered. Written for these tests.
ONE_VECTOR_FOR_TWO_TEXTS = {**RECORDED, "data": RECORDED["data"][:1]}
WRONG_DIMENSION = {
    **RECORDED,
    "data": [{**row, "embedding": row["embedding"][:512]} for row in RECORDED["data"]],
}

# Not one of these may raise: on None, /ask falls back to keyword search.
UPSTREAM_FAILURES = [
    pytest.param(serve(500, {"detail": "x"}), "voyage_status_500", id="upstream-500"),
    pytest.param(serve(429, {"detail": "x"}), "voyage_status_429", id="rate-limited"),
    pytest.param(serve(401, {"detail": "x"}), "voyage_status_401", id="bad-key"),
    pytest.param(serve(400, {"detail": "x"}), "voyage_status_400", id="bad-request"),
    pytest.param(fail_with(httpx.ReadTimeout), "voyage_timeout", id="read-timeout"),
    pytest.param(fail_with(httpx.ConnectTimeout), "voyage_timeout", id="connect-timeout"),
    pytest.param(fail_with(httpx.ConnectError), "voyage_unreachable", id="unreachable"),
    pytest.param(fail_with(httpx.LocalProtocolError), "voyage_unreachable", id="unsendable-key"),
    pytest.param(serve(200, ONE_VECTOR_FOR_TWO_TEXTS), "voyage_malformed", id="missing-vector"),
    pytest.param(serve(200, WRONG_DIMENSION), "voyage_malformed", id="wrong-dimension"),
    pytest.param(serve(200, {"object": "list"}), "voyage_malformed", id="no-data"),
    pytest.param(serve(200, "not an object"), "voyage_malformed", id="not-an-object"),
]


@pytest.mark.parametrize(("transport", "reason"), UPSTREAM_FAILURES)
async def test_upstream_failure_degrades_to_none_with_a_reason_code(
    transport: httpx.MockTransport, reason: str, caplog: pytest.LogCaptureFixture
) -> None:
    assert await embed(TEXTS, "document", "req-1", transport=transport) is None

    records = embed_records(caplog)
    (record,) = records
    assert record.levelname == "WARNING"
    assert (record.request_id, record.stage, record.outcome, record.reason) == (
        "req-1",
        "embed",
        "degraded",
        reason,
    )
    assert_no_key_in(records)


async def test_a_body_that_is_not_json_is_malformed(caplog: pytest.LogCaptureFixture) -> None:
    """A proxy's HTML error page with a 200, say. Written for this test."""
    transport = httpx.MockTransport(lambda request: httpx.Response(200, text="<html>"))

    assert await embed(TEXTS, "document", "req-1", transport=transport) is None

    (record,) = embed_records(caplog)
    assert record.reason == "voyage_malformed"
