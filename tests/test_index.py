"""The committed index matches what `make index` would build today.

These tests are CI's guard on `index/`. Edit the corpus, the chunker or an
index setting without rebuilding and one of them fails, before a stale index
can reach Render. The fix in every case is `make index` and committing what
it writes.

Offline: nothing here calls Voyage. The vectors themselves can't be checked
without a call, so only their shape is.
"""

import json
import shutil
from pathlib import Path

import numpy as np

from app.chunker import chunk_corpus
from app.config import get_settings
from app.index import CORPUS_DIR, INDEX_DIR, build_manifest
from app.schemas import Chunk

REBUILD = "index/ is stale: run `make index` and commit what it writes"


def committed_chunks() -> list[Chunk]:
    lines = (INDEX_DIR / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
    return [Chunk.model_validate_json(line) for line in lines]


def test_committed_manifest_matches_the_corpus_and_settings() -> None:
    committed = json.loads((INDEX_DIR / "manifest.json").read_text(encoding="utf-8"))

    assert committed == build_manifest(CORPUS_DIR, get_settings()), REBUILD


def test_committed_chunks_match_a_fresh_chunking() -> None:
    """Catches a chunker code change, which the manifest's hashes cannot see."""
    settings = get_settings()
    fresh = chunk_corpus(
        CORPUS_DIR,
        max_chunk_words=settings.max_chunk_words,
        current_edition=settings.current_edition,
    )

    assert committed_chunks() == fresh, REBUILD


def test_embeddings_have_one_row_per_chunk_at_the_configured_dimension() -> None:
    embeddings = np.load(INDEX_DIR / "embeddings.npy")

    assert embeddings.dtype == np.float32
    assert embeddings.shape == (len(committed_chunks()), get_settings().voyage_output_dimension)


def test_every_index_file_is_under_20_mb() -> None:
    """CLAUDE.md section 4: past this, reconsider the dimension or use float16."""
    for path in INDEX_DIR.iterdir():
        assert path.stat().st_size < 20 * 1024 * 1024, path.name


def test_manifest_changes_when_a_corpus_file_changes(tmp_path: Path) -> None:
    """A copy of the real corpus, one file edited. Written for this test."""
    corpus = tmp_path / "inter-mediate"
    shutil.copytree(CORPUS_DIR, corpus)
    before = build_manifest(corpus, get_settings())

    edited = corpus / "matchday-ops-2026-27.md"
    edited.write_text(edited.read_text(encoding="utf-8") + "\nOne more rule.\n", encoding="utf-8")
    after = build_manifest(corpus, get_settings())

    changed = {name for name in before["corpus"] if before["corpus"][name] != after["corpus"][name]}
    assert changed == {"matchday-ops-2026-27.md"}
