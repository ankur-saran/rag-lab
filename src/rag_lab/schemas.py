"""Frozen artifact schemas — the interface contract between phases.

Phases communicate *only* through JSONL files whose records validate against the
models defined here. No phase imports another phase's runtime state. That is what
makes each phase independently runnable, and it is why changing a model in this
module is a breaking change for everything downstream. Bump ``SCHEMA_VERSION``
when you do.

The single most important design decision lives on ``Chunk``: the split between
``text`` (what gets returned to the consumer) and ``embed_text`` (what gets
embedded). Heading-path prefixing, asymmetric query/document prefixes, table
summary-indexing and parent-document retrieval are all expressible through that
one split without special-casing any of them downstream. Do not collapse it.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = "1.0"

ContentType = Literal["markdown", "text", "html"]
Difficulty = Literal["lookup", "synthesis", "cross_reference"]
ChunkRole = Literal["standalone", "parent", "child"]


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------- #
# Phase 1
# --------------------------------------------------------------------------- #


class Document(_Base):
    """A normalized source document. Output of Phase 1.

    ``text`` is authoritative: every chunk offset in the system indexes into this
    exact string. All normalization (NFC, CRLF, blank-line collapsing) must happen
    before this object is constructed, never after.
    """

    doc_id: str
    corpus: str
    source_path: str  # relative to repo root
    title: str
    text: str
    content_type: ContentType = "text"
    meta: dict[str, Any] = Field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Phases 2 and 7
# --------------------------------------------------------------------------- #


class Chunk(_Base):
    """A retrievable unit. Output of Phase 2 (baseline) and Phase 7 (advanced)."""

    chunk_id: str
    doc_id: str
    corpus: str
    chunk_set_id: str

    text: str  # returned to the consumer
    embed_text: str  # sent to the embedder — may differ

    char_start: int  # offset into Document.text
    char_end: int
    token_count: int
    ordinal: int  # position within document, 0-based

    parent_id: str | None = None
    role: ChunkRole = "standalone"
    heading_path: list[str] = Field(default_factory=list)

    chunker: str
    chunker_params: dict[str, Any] = Field(default_factory=dict)
    meta: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_span(self) -> Chunk:
        if self.char_start < 0:
            raise ValueError(f"char_start must be >= 0, got {self.char_start}")
        if self.char_end < self.char_start:
            raise ValueError(
                f"char_end ({self.char_end}) must be >= char_start ({self.char_start})"
            )
        return self

    @property
    def span(self) -> tuple[int, int]:
        return (self.char_start, self.char_end)


# --------------------------------------------------------------------------- #
# Phase 3
# --------------------------------------------------------------------------- #


class IndexManifest(_Base):
    """Metadata written alongside every persisted index. Output of Phase 3.

    One JSON object per index directory (``artifacts/indexes/<index_id>/manifest.json``),
    not a JSONL collection — same shape as ``RunResult`` in that respect, so it is
    deliberately not registered in ``ARTIFACT_MODELS``. Phase 6 reads this for
    reproducibility metadata; ``index build``'s caching also reads it to decide
    whether a rebuild is needed.
    """

    index_id: str
    chunk_set_id: str
    embedder: str
    embedder_params: dict[str, Any] = Field(default_factory=dict)
    vector_count: int
    dim: int  # actual stored dim, post-truncation
    build_duration_s: float
    library_versions: dict[str, str] = Field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION
    created_at: datetime = Field(default_factory=datetime.utcnow)


# --------------------------------------------------------------------------- #
# Phase 5
# --------------------------------------------------------------------------- #


class EvalPair(_Base):
    """A gold-labeled query. Output of Phase 5.

    ``gold_char_spans`` is the *authoritative* label and is expressed in
    document coordinates, deliberately, as a **list**: most pairs (``lookup``,
    ``synthesis``) carry exactly one span, but ``cross_reference`` pairs need
    two — often distant — spans in the same document, and gold status must be
    evaluated against each span independently (see ``resolve_gold`` in Phase
    6). Collapsing a pair's spans into one enclosing range would silently mark
    every chunk *between* the two real reference points as gold too.

    Labels expressed as chunk IDs would only be valid for the chunk set that
    produced them, which would make chunker comparison circular — the exact
    thing this project exists to measure. ``sampling_chunk_ids`` is a derived
    convenience field (which chunks the generation-time *sampling* chunk set
    touched), re-resolved per chunk set at evaluation time; it is provenance,
    never authoritative, and downstream code must always re-derive gold via
    ``resolve_gold`` rather than reading it directly.
    """

    query_id: str
    corpus: str
    query: str
    gold_doc_id: str
    gold_char_spans: list[tuple[int, int]] = Field(min_length=1)
    sampling_chunk_ids: list[str] = Field(default_factory=list)  # derived, not authoritative
    answer: str | None = None
    supporting_quotes: list[str] = Field(default_factory=list)  # index-aligned with gold_char_spans
    difficulty: Difficulty = "lookup"
    split: Literal["train", "dev", "test"] = "train"
    generator_model: str = "unknown"
    validated: bool = False

    @model_validator(mode="after")
    def _validate_span(self) -> EvalPair:
        for span in self.gold_char_spans:
            start, end = span
            if start < 0 or end < start:
                raise ValueError(f"invalid gold_char_span: {span}")
        return self


# --------------------------------------------------------------------------- #
# Phases 4 and 6
# --------------------------------------------------------------------------- #


class ScoredChunk(_Base):
    """A retrieval result. ``debug`` carries component ranks and scores so hybrid
    fusion stays explicable rather than magical."""

    chunk: Chunk
    score: float
    rank: int
    retriever: str
    debug: dict[str, Any] = Field(default_factory=dict)


class QueryTrace(_Base):
    """Per-query record within a RunResult. Input to failure analysis (Phase 6)
    and to the optimizer agent (Phase 8)."""

    query_id: str
    query: str
    retrieved_chunk_ids: list[str]
    retrieved_scores: list[float] = Field(default_factory=list)
    gold_chunk_ids: list[str]
    first_hit_rank: int | None = None  # 1-based; None == miss
    metrics: dict[str, float] = Field(default_factory=dict)
    latency_ms: float = 0.0
    excluded: bool = False  # True when gold resolved to zero chunks for this chunk set
    exclusion_reason: str | None = None


class RunResult(_Base):
    """One cell of the experiment matrix. Output of Phase 6."""

    run_id: str
    config: dict[str, Any]  # full resolved config, for reproducibility
    corpus: str
    metrics: dict[str, float] = Field(default_factory=dict)
    per_query: list[QueryTrace] = Field(default_factory=list)
    chunk_stats: dict[str, float] = Field(default_factory=dict)
    timings: dict[str, float] = Field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION
    created_at: datetime = Field(default_factory=datetime.utcnow)


# --------------------------------------------------------------------------- #
# Registry — used by paths.resolve_artifact and by the fixture generator
# --------------------------------------------------------------------------- #

ARTIFACT_MODELS: dict[str, type[BaseModel]] = {
    "documents": Document,
    "chunks": Chunk,
    "evalset": EvalPair,
}

__all__ = [
    "SCHEMA_VERSION",
    "ARTIFACT_MODELS",
    "Chunk",
    "ChunkRole",
    "ContentType",
    "Difficulty",
    "Document",
    "EvalPair",
    "IndexManifest",
    "QueryTrace",
    "RunResult",
    "ScoredChunk",
]
