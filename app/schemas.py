"""Pydantic models shared across the pipeline.

Each model arrives with the unit that first uses it. Names describe the
object, not the product (CLAUDE.md section 6).
"""

from pydantic import BaseModel, ConfigDict


class Chunk(BaseModel):
    """One retrievable piece of the handbook: a section, or part of one.

    Frozen because the verifier and the superseded flag read `text` and
    `is_current` after retrieval; nothing downstream may change them.

    `section_path` is the stable address (`matchday-ops-2026-27#1.1`). Eval
    labels and citations point at it. `chunk_id` equals it unless the section
    was split, in which case each part gets its own suffixed `chunk_id` and
    all parts share the `section_path`.
    """

    model_config = ConfigDict(frozen=True)

    chunk_id: str
    doc_id: str
    club: str
    edition: str
    is_current: bool
    section_path: str
    title: str
    text: str
