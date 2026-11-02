"""Text in, vectors out: the Voyage embeddings client.

Raw httpx, no SDK (CLAUDE.md section 4): one dependency fewer, and tests mock
it with `httpx.MockTransport` like every other outside call.

Two callers, one contract. `/ask` embeds the question and, on None, retrieves
by BM25 alone and tags the response `keyword_only` (invariant 5).
`scripts/build_index.py` embeds the chunks and, on None, stops: an index with
missing vectors is worse than no index. Neither caller needs to know why, only
the log does, so `embed` returns None rather than raising.

No retry. When Voyage is slow, falling back to keyword search answers sooner
than a second attempt would, and invariant 6 allows one embedding call.

Vectors come back exactly as Voyage sent them. Normalising them is part of
cosine similarity, which belongs to dense search (U11).
"""

import logging
import time
from typing import Literal

import httpx
import numpy as np

from app.config import get_settings
from app.logs import log_event

logger = logging.getLogger(__name__)


def _parse(response: httpx.Response, rows: int, dimension: int) -> np.ndarray | None:
    # Voyage tags each vector with its input position; sorting by it means the
    # order holds even if a response ever arrives out of order. Any body that
    # is not exactly `rows` vectors of `dimension` numbers is malformed.
    try:
        data = sorted(response.json()["data"], key=lambda row: row["index"])
        vectors = np.array([row["embedding"] for row in data], dtype=np.float32)
    # ValueError covers invalid JSON, ragged rows and non-numbers; KeyError and
    # TypeError a body of the wrong shape.
    except (ValueError, KeyError, TypeError):
        return None
    if vectors.shape != (rows, dimension):
        return None
    return vectors


async def embed(
    texts: list[str],
    input_type: Literal["document", "query"],
    request_id: str,
    transport: httpx.AsyncBaseTransport | None = None,
) -> np.ndarray | None:
    """Embed `texts`; return a float32 array of shape (len(texts), dimension).

    Row i is the vector for texts[i]. `input_type` is "document" for chunks
    and "query" for questions: Voyage embeds the two differently so that a
    question lands near the passage that answers it, and mixing them up
    degrades recall without raising any error.

    Never raises. Every failure becomes None plus one WARNING line carrying
    one of these reason codes:

        voyage_timeout        no response within `voyage_timeout_seconds`
        voyage_unreachable    DNS, TLS or connection failure; also a key
                              that cannot go in a header, such as one with a
                              newline pasted into Render's env settings
        voyage_status_<code>  any non-200: 400 a bad request (an empty
                              `texts`, an unsupported dimension, over 1,000
                              inputs), 401 a bad key, 429 rate limited, 5xx an
                              outage
        voyage_malformed      a 200 whose body is not one vector of the
                              configured dimension per input

    The API key travels in a header, and nothing here logs the request, the
    response or an exception's text.

    Logs exactly one line per call, stage "embed", outcome "ok" or
    "degraded". `transport` exists for tests, which pass an
    `httpx.MockTransport`. Production leaves it None.
    """
    settings = get_settings()
    started = time.perf_counter()
    reason: str | None = None
    vectors: np.ndarray | None = None

    try:
        async with httpx.AsyncClient(
            base_url=settings.voyage_base_url,
            timeout=settings.voyage_timeout_seconds,
            transport=transport,
        ) as client:
            response = await client.post(
                "/embeddings",
                headers={"Authorization": f"Bearer {settings.voyage_api_key}"},
                json={
                    "input": texts,
                    "model": settings.voyage_model,
                    "input_type": input_type,
                    "output_dimension": settings.voyage_output_dimension,
                },
            )
        if response.status_code != 200:
            reason = f"voyage_status_{response.status_code}"
        else:
            vectors = _parse(response, len(texts), settings.voyage_output_dimension)
            if vectors is None:
                reason = "voyage_malformed"
    # TimeoutException subclasses TransportError, so it must be caught first.
    except httpx.TimeoutException:
        reason = "voyage_timeout"
    except httpx.TransportError:
        reason = "voyage_unreachable"

    log_event(
        logger,
        request_id=request_id,
        stage="embed",
        outcome="ok" if reason is None else "degraded",
        latency_ms=(time.perf_counter() - started) * 1000,
        reason=reason,
    )
    return vectors
