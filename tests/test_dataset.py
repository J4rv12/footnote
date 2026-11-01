"""The exam's shape: every label is well formed and points at a real section.

This checks structure, not whether a label is true. A misspelt section path
would otherwise score as a retrieval miss and blame the retriever for a typo.
"""

import json
from collections import Counter
from pathlib import Path
from typing import Any

from app.chunker import chunk_corpus
from app.config import get_settings

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "eval" / "dataset.jsonl"
HANDBOOK = ROOT / "corpus" / "inter-mediate"

FIELDS = {
    "id": str,
    "category": str,
    "question": str,
    "answerable": bool,
    "gold_sections": list,
    "must_contain": list,
    "expect_superseded": bool,
}
CATEGORY_COUNTS = {"normal": 12, "edge": 7, "adversarial": 6}
CANARY_SECTION = "media-policy-2026-27#5.1"


def rows() -> list[dict[str, Any]]:
    return [json.loads(line) for line in DATASET.read_text(encoding="utf-8").splitlines()]


def test_every_row_has_exactly_the_required_fields() -> None:
    for row in rows():
        assert set(row) == set(FIELDS), row
        for field, kind in FIELDS.items():
            assert isinstance(row[field], kind), f"{row['id']}: {field}"
        assert row["category"] in CATEGORY_COUNTS, row["id"]


def test_ids_are_unique() -> None:
    ids = [row["id"] for row in rows()]

    assert len(ids) == len(set(ids))


def test_answerable_rows_have_labels_and_unanswerable_rows_have_none() -> None:
    for row in rows():
        if row["answerable"]:
            assert row["gold_sections"], row["id"]
            assert row["must_contain"], row["id"]
        else:
            assert not row["gold_sections"], row["id"]
            assert not row["must_contain"], row["id"]
            assert not row["expect_superseded"], row["id"]


def test_gold_sections_exist_and_superseded_labels_point_at_an_old_edition() -> None:
    settings = get_settings()
    chunks = chunk_corpus(
        HANDBOOK,
        max_chunk_words=settings.max_chunk_words,
        current_edition=settings.current_edition,
    )
    is_current = {c.section_path: c.is_current for c in chunks}

    for row in rows():
        for section in row["gold_sections"]:
            assert section in is_current, f"{row['id']}: {section}"
        if row["expect_superseded"]:
            assert not all(is_current[s] for s in row["gold_sections"]), row["id"]


def test_the_canary_case_targets_the_canary_section() -> None:
    """If retrieval never fetches the canary, "0 leaked" proves nothing."""
    assert any(CANARY_SECTION in row["gold_sections"] for row in rows())


def test_the_exam_is_the_agreed_size() -> None:
    assert Counter(row["category"] for row in rows()) == CATEGORY_COUNTS
