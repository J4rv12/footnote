"""Build the retrieval index. Run as `make index`: offline, billed, never at startup.

Writes three files to `index/`, all committed:

    chunks.jsonl     one Chunk per line, in `chunk_corpus` order
    embeddings.npy   float32, row i is the vector for line i of chunks.jsonl
    manifest.json    what the index was built from (`app.index.build_manifest`)

BM25 statistics join these in U12.

Exits 1 and writes nothing if Voyage fails; `embed` has already logged the
reason code. An index whose vectors don't match its chunks is worse than the
old one, which stays in place.
"""

import asyncio
import json
import sys

import numpy as np

from app.chunker import chunk_corpus
from app.config import get_settings
from app.embed import embed
from app.index import CORPUS_DIR, INDEX_DIR, build_manifest
from app.logs import configure_logging

# Voyage's limit on inputs per request. A chunk is at most max_chunk_words
# (200 words, about 260 tokens), so a full batch stays under voyage-4's
# 320K-token request limit too.
BATCH_SIZE = 1000


async def main() -> int:
    settings = get_settings()
    configure_logging(settings.log_level)
    chunks = chunk_corpus(
        CORPUS_DIR,
        max_chunk_words=settings.max_chunk_words,
        current_edition=settings.current_edition,
    )
    # Chunk.text has no heading, and the title carries much of what a section
    # is about ("Home reporting time"). The edition stays out: an unchanged
    # rule embeds identically in both, and retrieval breaks the tie.
    texts = [f"{chunk.title}\n\n{chunk.text}" for chunk in chunks]

    batches: list[np.ndarray] = []
    for start in range(0, len(texts), BATCH_SIZE):
        vectors = await embed(texts[start : start + BATCH_SIZE], "document", "build-index")
        if vectors is None:
            return 1
        batches.append(vectors)
    embeddings = np.concatenate(batches)

    INDEX_DIR.mkdir(exist_ok=True)
    lines = "".join(chunk.model_dump_json() + "\n" for chunk in chunks)
    (INDEX_DIR / "chunks.jsonl").write_text(lines, encoding="utf-8")
    np.save(INDEX_DIR / "embeddings.npy", embeddings)
    manifest = json.dumps(build_manifest(CORPUS_DIR, settings), indent=2) + "\n"
    (INDEX_DIR / "manifest.json").write_text(manifest, encoding="utf-8")

    print(f"wrote {len(chunks)} chunks, embeddings {embeddings.shape} {embeddings.dtype}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
