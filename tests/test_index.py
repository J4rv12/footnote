"""The committed index matches what `make index` would build today.

The first group is CI's guard on `index/`. Edit the corpus, the chunker or an
index setting without rebuilding and one of them fails, before a stale index
can reach Render. The fix in every case is `make index` and committing what
it writes. The second group is the same check at startup, `load_index`, which
also sees settings CI never does.

Offline: nothing here calls Voyage. The vectors themselves can't be checked
without a call, so only their shape is.
"""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from app.chunker import chunk_corpus
from app.config import get_settings
from app.index import (
    CORPUS_DIR,
    INDEX_DIR,
    IndexMismatchError,
    build_manifest,
    load_index,
    receipt,
)
from app.schemas import Chunk

REBUILD = "index/ is stale: run `make index` and commit what it writes"


def committed_chunks() -> list[Chunk]:
    lines = (INDEX_DIR / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
    return [Chunk.model_validate_json(line) for line in lines]


def test_committed_manifest_matches_the_corpus_and_settings() -> None:
    committed = json.loads((INDEX_DIR / "manifest.json").read_text(encoding="utf-8"))

    expected = build_manifest(CORPUS_DIR, get_settings()) | {"outputs": receipt(INDEX_DIR)}

    assert committed == expected, REBUILD


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
    """The build's own output shape. `load_index` trusts it once the receipt matches."""
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


# --- load_index --------------------------------------------------------------


def test_committed_index_loads_with_one_read_only_row_per_chunk() -> None:
    index = load_index(INDEX_DIR, CORPUS_DIR, get_settings())

    assert index.chunks == tuple(committed_chunks())
    assert index.embeddings.shape == (len(index.chunks), get_settings().voyage_output_dimension)
    with pytest.raises(ValueError, match="read-only"):
        index.embeddings[0, 0] = 0.0


@pytest.mark.parametrize(
    ("setting", "value"),
    [("voyage_model", "voyage-4-lite"), ("voyage_output_dimension", 512)],
)
def test_an_embedding_setting_unlike_the_build_refuses_to_load(setting: str, value: object) -> None:
    """A VOYAGE_* variable set on Render. CI can't see it; startup must."""
    drifted = get_settings().model_copy(update={setting: value})

    with pytest.raises(IndexMismatchError, match="embedding"):
        load_index(INDEX_DIR, CORPUS_DIR, drifted)


def test_an_edited_corpus_refuses_to_load(tmp_path: Path) -> None:
    """A copy of the real corpus, one file edited. Written for this test."""
    corpus = tmp_path / "inter-mediate"
    shutil.copytree(CORPUS_DIR, corpus)
    edited = corpus / "code-of-conduct-2026-27.md"
    edited.write_text(edited.read_text(encoding="utf-8") + "\nOne more fine.\n", encoding="utf-8")

    with pytest.raises(IndexMismatchError, match="corpus"):
        load_index(INDEX_DIR, corpus, get_settings())


def _older_vectors(index_dir: Path) -> None:
    # Same shape, one value different: what a forgotten embeddings.npy looks
    # like after a one-sentence edit. No count or shape check can see it.
    vectors = np.load(index_dir / "embeddings.npy")
    vectors[0, 0] += 0.01
    np.save(index_dir / "embeddings.npy", vectors)


def _older_chunks(index_dir: Path) -> None:
    # One chunk's text from before an edit: a forgotten chunks.jsonl.
    path = index_dir / "chunks.jsonl"
    text = path.read_text(encoding="utf-8").replace("players", "squad", 1)
    path.write_text(text, encoding="utf-8")


@pytest.mark.parametrize("swap", [_older_vectors, _older_chunks], ids=["vectors", "chunks"])
def test_a_file_from_another_build_refuses_to_load(tmp_path: Path, swap: object) -> None:
    """The committed index with one output altered. Written for this test.

    The manifest and the other files are the committed ones, so every input
    check passes; only the receipt sees that the files come from two builds.
    """
    index_dir = tmp_path / "index"
    shutil.copytree(INDEX_DIR, index_dir)
    swap(index_dir)

    with pytest.raises(IndexMismatchError, match="outputs"):
        load_index(index_dir, CORPUS_DIR, get_settings())
