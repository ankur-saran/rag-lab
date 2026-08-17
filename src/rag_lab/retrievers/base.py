"""Shared retriever primitives: the ``Retriever`` protocol, ``RetrieverSpec``,
and the re-rank helper every concrete retriever ends with.

``ScoredChunk`` is *not* redefined here — it already lives in ``schemas.py``
(Phases 4 and 6) and every retriever below imports it from there. Redeclaring
it as a fresh model would silently produce an incompatible type the moment
anything compared instances across module boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from rag_lab.schemas import ScoredChunk


class Retriever(Protocol):
    name: str

    def retrieve(self, query: str, k: int) -> list[ScoredChunk]: ...


@dataclass(frozen=True)
class RetrieverSpec:
    """A named retriever's fixed identity: its default params, whether building
    it needs ``rank_bm25`` (so the registry can lazy-import around it, same
    convention as ``embedders.build_embedder`` around ``sentence_transformers``),
    and whether it wraps another retriever rather than searching directly."""

    default_params: dict[str, Any] = field(default_factory=dict)
    needs_rank_bm25: bool = False
    is_wrapper: bool = False


def truncate_and_rank(results: list[ScoredChunk], k: int, retriever: str) -> list[ScoredChunk]:
    """Re-derive 1-based rank and relabel ``.retriever`` after any reorder,
    fusion, or dedup, then truncate to ``k``.

    Every concrete retriever below ends with this: a wrapper's own reordering
    (RRF fusion, parent dedup, window dedup) invalidates whatever rank/retriever
    label its base component(s) computed, so those fields must be recomputed
    once, in one place, rather than by each wrapper individually.
    """
    truncated = results[:k]
    return [
        r.model_copy(update={"rank": i, "retriever": retriever})
        for i, r in enumerate(truncated, start=1)
    ]


__all__ = ["Retriever", "RetrieverSpec", "truncate_and_rank"]
