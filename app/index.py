"""The committed index: what it was built from, and (U10) loading it.

`build_manifest` describes every input that decides the index's contents.
`scripts/build_index.py` writes it to `index/manifest.json`, and
tests/test_index.py compares that copy with a fresh one. A corpus edit or a
settings change without `make index` therefore fails CI. U10 runs the same
comparison at startup.

Not covered: the code. A change to `chunker.py` is caught by a test that
re-chunks the corpus and compares the result with `chunks.jsonl`; a change to
the text the build script embeds per chunk is caught by nothing.
"""

import hashlib
from pathlib import Path

from app.config import Settings

ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = ROOT / "corpus" / "inter-mediate"
INDEX_DIR = ROOT / "index"


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
        "corpus": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        "chunker": {
            "max_chunk_words": settings.max_chunk_words,
            "current_edition": settings.current_edition,
        },
        "embedding": {
            "model": settings.voyage_model,
            "dimension": settings.voyage_output_dimension,
        },
    }
