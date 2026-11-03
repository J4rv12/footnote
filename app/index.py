"""The committed index: what it was built from, and loading it at startup.

`build_manifest` describes every input that decides the index's contents.
`scripts/build_index.py` writes it to `index/manifest.json`, together with a
receipt: the sha256 of the two files that build wrote, `chunks.jsonl` and
`embeddings.npy`. Two places compare that copy with a fresh one:

    tests/test_index.py  in CI, so a corpus edit without `make index` fails
                         before it merges
    load_index           at startup, so a setting on Render that differs from
                         the build (VOYAGE_MODEL, VOYAGE_OUTPUT_DIMENSION) stops
                         the deploy. CI never sees Render's environment.

Without the second, a different embedding model ranks chunks close to random
while every log line says `ok` (U08).

The receipt ties the three files to one build. `git add` takes an explicit
list, so committing two of them and forgetting the third is easy; without the
receipt, a forgotten `embeddings.npy` after a one-sentence edit boots green
with one chunk's text paired to its old text's vector. Render deploys on push
whether CI passed or not, so this is checked at startup, not only in CI.

Not covered: the code. A change to `chunker.py` is caught by a test that
re-chunks the corpus and compares the result with `chunks.jsonl`; a change to
the text the build script embeds per chunk is caught by nothing.

`load_index` reads files synchronously, against the "all I/O async" rule. It
runs once, at import, before the server accepts a request.
"""

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.config import Settings
from app.logs import log_event
from app.schemas import Chunk

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = ROOT / "corpus" / "inter-mediate"
INDEX_DIR = ROOT / "index"
# The files a build writes before its manifest, in the order it writes them.
OUTPUTS = ("chunks.jsonl", "embeddings.npy")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def receipt(index_dir: Path) -> dict[str, str]:
    """Return the sha256 of each file in `OUTPUTS`, as found in `index_dir`.

    Raises:
        OSError: if a file is missing or cannot be read.
    """
    return {name: _sha256(index_dir / name) for name in OUTPUTS}


class IndexMismatchError(Exception):
    """The committed index doesn't match this deploy's corpus or settings."""


@dataclass(frozen=True)
class LoadedIndex:
    """The index in memory. Row i of `embeddings` is the vector for `chunks[i]`.

    `embeddings` is read-only: a search that normalised it in place would
    change every later search's results.
    """

    chunks: tuple[Chunk, ...]
    embeddings: np.ndarray
    manifest: dict[str, object]


def build_manifest(corpus_dir: Path, settings: Settings) -> dict[str, object]:
    """Return the manifest for an index built from `corpus_dir` with `settings`.

    Hashes every top-level `*.md` file, the files `chunk_corpus` reads. The
    planning file, `corpus/README.md`, sits one level up and is never hashed.
    Every value is JSON-ready and every key order is fixed, so two calls on
    unchanged inputs return equal dicts.

    Raises:
        OSError: if `corpus_dir` or a file in it cannot be read.
    """
    paths = sorted(corpus_dir.glob("*.md"))
    return {
        "corpus": {p.name: _sha256(p) for p in paths},
        "chunker": {
            "max_chunk_words": settings.max_chunk_words,
            "current_edition": settings.current_edition,
        },
        "embedding": {
            "model": settings.voyage_model,
            "dimension": settings.voyage_output_dimension,
        },
    }


def load_index(index_dir: Path, corpus_dir: Path, settings: Settings) -> LoadedIndex:
    """Load the index from `index_dir` and check it matches this deploy.

    Logs one line, stage "index_load", with the load time.

    Raises:
        IndexMismatchError: the manifest differs from `build_manifest` plus
            the receipt of the files on disk. The message names which parts:
            corpus, chunker, embedding, or outputs (the files did not all
            come from one build).
        OSError, ValueError: a file is missing or unreadable. pydantic's
            ValidationError, a ValueError, for a malformed chunk line.

    Every one of these should stop the process. `main.py` doesn't catch them.

    No shape check on the vectors: once the receipt matches, they are the
    build's own output, one row per chunk at the dimension the manifest
    records, and the manifest's dimension has just been checked against
    settings. tests/test_index.py checks the build's output shape instead.
    """
    started = time.perf_counter()
    manifest = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
    expected = build_manifest(corpus_dir, settings) | {"outputs": receipt(index_dir)}
    stale = sorted(
        k for k in manifest.keys() | expected.keys() if manifest.get(k) != expected.get(k)
    )
    if stale:
        raise IndexMismatchError(
            f"index/manifest.json differs from this deploy in: {', '.join(stale)}. "
            "Rebuild with `make index` and commit all three files, or fix the setting."
        )

    lines = (index_dir / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
    chunks = tuple(Chunk.model_validate_json(line) for line in lines)
    embeddings = np.load(index_dir / "embeddings.npy")
    embeddings.flags.writeable = False

    log_event(
        logger,
        request_id="startup",
        stage="index_load",
        outcome="ok",
        latency_ms=(time.perf_counter() - started) * 1000,
    )
    return LoadedIndex(chunks=chunks, embeddings=embeddings, manifest=manifest)
