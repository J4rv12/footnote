"""Markdown handbook files → `Chunk`s with section paths and edition metadata.

A chunk is one numbered section. Retrieval returns five of them, so each must
be small enough to be precise and whole enough to be true: a section split
mid-list could be quoted verbatim ("Late to training:") and still be wrong.
That is why oversize sections split only at paragraph boundaries.

Both functions are pure apart from file reads, and take their settings as
arguments rather than importing `config`, so tests need no environment.

`chunk_corpus` reads files synchronously, against the "all I/O async" rule.
It runs only offline, in `scripts/build_index.py` and tests, never on the
event loop.

Not handled here: identical sections in two editions tie exactly in
retrieval, and `chunk_id` order puts 2025-26 first. Breaking that tie in
favour of `is_current` belongs to retrieval (U11/U13).
"""

import re
from pathlib import Path

from app.schemas import Chunk

_FILENAME_PATTERN = r"^[a-z0-9]+(?:-[a-z0-9]+)*-\d{4}-\d{2}\.md$"
_EDITION_PATTERN = r"\b\d{4}-\d{2}\b"
_HEADING_PATTERN = r"^## (.*)"
_TITLE_LINE_PATTERN = r"^# .*"
_SECTION_NUMBER_PATTERN = r"\d+(\.\d+)*"


def _pack_paragraphs(text: str, max_words: int) -> list[str]:
    """Group whole paragraphs into parts of at most `max_words`, in order.

    A single paragraph over the limit stays whole, as a part of its own.
    """
    if len(text.split()) <= max_words:
        return [text]

    parts: list[str] = []
    current: list[str] = []
    count = 0
    for paragraph in text.split("\n\n"):
        words = len(paragraph.split())
        if current and count + words > max_words:
            parts.append("\n\n".join(current))
            current, count = [], 0
        current.append(paragraph)
        count += words
    parts.append("\n\n".join(current))
    return parts


def chunk_document(
    text: str,
    filename: str,
    club: str,
    *,
    max_chunk_words: int,
    current_edition: str,
) -> list[Chunk]:
    """Split one handbook file into chunks, in document order.

    Filename: `<slug>-<yyyy>-<yy>.md`, e.g. `matchday-ops-2026-27.md`.
    `doc_id` is the filename without `.md`; `edition` is the `yyyy-yy` part;
    `is_current` is `edition == current_edition`.

    Sections: every line of the form `## <number> <title>`, where number is
    digits and dots (`1.1`, `12.3`), starts a section. `section_path` is
    `f"{doc_id}#{number}"`. `title` is the heading text after the number.
    `text` is the section body, without the heading line, stripped. The
    `# ` title line at the top of a file is never part of any chunk.

    Splitting: words are counted with `str.split()`. A section whose text is
    over `max_chunk_words` is split into parts by packing whole paragraphs
    (blocks separated by blank lines) in order, starting a new part when the
    next paragraph would take the part over the limit. A single paragraph
    over the limit becomes one part on its own; paragraphs are never cut.
    Parts keep the section's `section_path` and `title`, and get
    `chunk_id = f"{section_path}:p{n}"` with n counting from 1. An unsplit
    section has `chunk_id == section_path`.

    No headings: a file with no `## ` lines is treated as one section,
    `section_path` `f"{doc_id}#0"`, `title` equal to `doc_id`, `text` the whole
    file stripped (minus any `# ` title line).

    Raises:
        ValueError: the filename does not match the pattern; a `## ` heading
            has no number or no title; or a section number appears twice in
            one file. These mean a broken corpus, so `make index` must fail
            loudly rather than build chunks with missing or clashing addresses.
    """
    if not re.search(_FILENAME_PATTERN, filename):
        raise ValueError(f"Not a valid filename: {filename}")

    doc_id = filename.split(".")[0]
    edition = re.search(_EDITION_PATTERN, doc_id).group()
    is_current = edition == current_edition

    parts = re.split(_HEADING_PATTERN, text, flags=re.MULTILINE)

    sections: list[tuple[str, str, str]] = []
    if len(parts) == 1:
        sections.append(("0", doc_id, re.sub(_TITLE_LINE_PATTERN, "", text, flags=re.MULTILINE)))
    else:
        for heading, body in zip(parts[1::2], parts[2::2], strict=True):
            number, _, title = heading.partition(" ")
            sections.append((number, title.strip(), body))

    chunks: list[Chunk] = []
    seen: set[str] = set()
    for number, title, body in sections:
        if not re.fullmatch(_SECTION_NUMBER_PATTERN, number) or not title:
            raise ValueError(f"{filename}: bad heading: {number} {title}")
        if number in seen:
            raise ValueError(f"{filename}: repeated section number: {number}")
        seen.add(number)

        section_path = f"{doc_id}#{number}"
        pieces = _pack_paragraphs(body.strip(), max_chunk_words)
        for n, piece in enumerate(pieces, start=1):
            chunks.append(
                Chunk(
                    chunk_id=section_path if len(pieces) == 1 else f"{section_path}:p{n}",
                    doc_id=doc_id,
                    club=club,
                    edition=edition,
                    is_current=is_current,
                    section_path=section_path,
                    title=title,
                    text=piece,
                )
            )

    return chunks


def chunk_corpus(
    corpus_dir: Path,
    *,
    max_chunk_words: int,
    current_edition: str,
) -> list[Chunk]:
    """Chunk every handbook file directly inside `corpus_dir`.

    Reads only `*.md` files at the top level of `corpus_dir`, never
    subfolders and never `README.md`: the planning file holds every trap's
    answer and the canary, and indexing it would contaminate the eval. Files
    are processed in sorted filename order, so the output is deterministic.
    `club` is `corpus_dir.name` (`inter-mediate`).

    Raises:
        ValueError: from `chunk_document`, for any malformed file.
        OSError: if `corpus_dir` or a file in it cannot be read.
    """
    chunks: list[Chunk] = []
    for path in sorted(corpus_dir.glob("*.md")):
        if path.name == "README.md":
            continue
        chunks.extend(
            chunk_document(
                path.read_text(encoding="utf-8"),
                path.name,
                corpus_dir.name,
                max_chunk_words=max_chunk_words,
                current_edition=current_edition,
            )
        )
    return chunks
