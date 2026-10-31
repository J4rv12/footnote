"""The chunker's contract: addresses, editions, and never cutting a rule.

Small inline documents test each rule in isolation. The last tests run on
the real handbook, so a corpus edit that breaks an address fails here
before it reaches the index or the eval labels.
"""

from pathlib import Path

import pytest

from app.chunker import chunk_corpus, chunk_document
from app.config import get_settings
from app.schemas import Chunk

HANDBOOK = Path(__file__).resolve().parent.parent / "corpus" / "inter-mediate"

TWO_SECTIONS = """\
# Matchday Operations — 2026/27 edition

## 1.1 Home reporting time

Players report early.

## 1.2 Arrival and parking

The car park opens.
"""


def chunk(
    text: str, filename: str = "matchday-ops-2026-27.md", max_chunk_words: int = 200
) -> list[Chunk]:
    return chunk_document(
        text,
        filename,
        "inter-mediate",
        max_chunk_words=max_chunk_words,
        current_edition="2026-27",
    )


def test_splits_at_headings() -> None:
    chunks = chunk(TWO_SECTIONS)

    assert [c.section_path for c in chunks] == [
        "matchday-ops-2026-27#1.1",
        "matchday-ops-2026-27#1.2",
    ]
    assert [c.chunk_id for c in chunks] == [c.section_path for c in chunks]
    assert [c.title for c in chunks] == ["Home reporting time", "Arrival and parking"]
    assert [c.text for c in chunks] == ["Players report early.", "The car park opens."]


def test_title_line_is_never_chunk_text() -> None:
    assert all("Matchday Operations" not in c.text for c in chunk(TWO_SECTIONS))


@pytest.mark.parametrize(
    ("filename", "edition", "is_current"),
    [
        ("matchday-ops-2026-27.md", "2026-27", True),
        ("matchday-ops-2025-26.md", "2025-26", False),
    ],
)
def test_edition_comes_from_the_filename(filename: str, edition: str, is_current: bool) -> None:
    first = chunk(TWO_SECTIONS, filename)[0]

    assert first.doc_id == filename.removesuffix(".md")
    assert first.edition == edition
    assert first.is_current is is_current
    assert first.club == "inter-mediate"


@pytest.mark.parametrize("filename", ["matchday-ops.md", "matchday-ops-2026.md", "notes.txt"])
def test_filename_without_edition_fails(filename: str) -> None:
    with pytest.raises(ValueError):
        chunk(TWO_SECTIONS, filename)


def test_oversize_section_splits_at_paragraph_boundaries() -> None:
    # Three paragraphs of 4 words; a limit of 9 fits two, not three.
    doc = (
        "## 5.1 Fines schedule\n\n"
        "one two three four\n\nfive six seven eight\n\nnine ten eleven twelve\n"
    )

    chunks = chunk(doc, max_chunk_words=9)

    assert [c.chunk_id for c in chunks] == [
        "matchday-ops-2026-27#5.1:p1",
        "matchday-ops-2026-27#5.1:p2",
    ]
    assert {c.section_path for c in chunks} == {"matchday-ops-2026-27#5.1"}
    assert {c.title for c in chunks} == {"Fines schedule"}
    assert chunks[0].text == "one two three four\n\nfive six seven eight"
    assert chunks[1].text == "nine ten eleven twelve"


def test_a_single_paragraph_over_the_limit_is_never_cut() -> None:
    paragraph = "- Late to training: £125\n- Late to a team meeting: £60\n- Wrong kit: £40"
    doc = f"## 5.1 Fines schedule\n\n{paragraph}\n"

    chunks = chunk(doc, max_chunk_words=3)

    assert len(chunks) == 1
    assert chunks[0].text == paragraph


def test_no_headings_is_one_chunk_not_a_crash() -> None:
    chunks = chunk("A note with no sections.\n\nA second paragraph.\n", "notice-2026-27.md")

    assert len(chunks) == 1
    assert chunks[0].section_path == "notice-2026-27#0"
    assert chunks[0].chunk_id == "notice-2026-27#0"
    assert chunks[0].title == "notice-2026-27"
    assert chunks[0].text == "A note with no sections.\n\nA second paragraph."


def test_unnumbered_heading_fails() -> None:
    with pytest.raises(ValueError):
        chunk("## Reporting times\n\nPlayers report early.\n")


@pytest.mark.parametrize("heading", ["## Reporting", "## 1.1"])
def test_heading_without_number_or_title_fails(heading: str) -> None:
    with pytest.raises(ValueError):
        chunk(f"## 1.0 Home\n\nEarly.\n\n{heading}\n\nLate.\n")


def test_repeated_section_number_fails() -> None:
    with pytest.raises(ValueError):
        chunk("## 1.1 Home\n\nEarly.\n\n## 1.1 Away\n\nEarlier.\n")


def test_corpus_reads_only_top_level_handbook_files(tmp_path: Path) -> None:
    club_dir = tmp_path / "inter-mediate"
    club_dir.mkdir()
    (club_dir / "media-policy-2026-27.md").write_text(TWO_SECTIONS)
    (club_dir / "README.md").write_text("## 1.1 Traps\n\nEvery answer, and MARMALADE.\n")
    (club_dir / "drafts").mkdir()
    (club_dir / "drafts" / "media-policy-2026-27.md").write_text("## 9.9 Draft\n\nUnused.\n")

    chunks = chunk_corpus(club_dir, max_chunk_words=200, current_edition="2026-27")

    assert {c.doc_id for c in chunks} == {"media-policy-2026-27"}
    assert len(chunks) == 2


# --- The real handbook ---


def handbook() -> list[Chunk]:
    settings = get_settings()
    return chunk_corpus(
        HANDBOOK,
        max_chunk_words=settings.max_chunk_words,
        current_edition=settings.current_edition,
    )


def test_handbook_chunk_ids_are_unique() -> None:
    ids = [c.chunk_id for c in handbook()]

    assert len(ids) == len(set(ids))


def test_handbook_trap_sections_exist() -> None:
    """Every address corpus/README.md plants a trap at. U07's labels use these."""
    paths = {c.section_path for c in handbook()}

    assert {
        "matchday-ops-2025-26#1.1",
        "matchday-ops-2026-27#1.1",
        "code-of-conduct-2025-26#5.1",
        "code-of-conduct-2026-27#5.1",
        "away-travel-2026-27#4.2",
        "away-travel-2026-27#5.1",
        "training-ground-rules-2026-27#5.2",
        "set-piece-playbook-2026-27#6.1",
        "code-of-conduct-2026-27#5.4",
        "code-of-conduct-2026-27#3.2",
        "media-policy-2026-27#5.1",
    } <= paths


def test_every_handbook_section_is_one_chunk() -> None:
    """max_chunk_words sits above the longest section, so nothing splits.

    If this fails, a section grew past the limit: either shorten it or
    accept that its parts now share one section_path, and check the U07
    labels still point at the part holding the fact.
    """
    assert not [c.chunk_id for c in handbook() if c.chunk_id != c.section_path]
