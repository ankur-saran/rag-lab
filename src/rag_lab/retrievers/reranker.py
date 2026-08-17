"""Reranking seam (plan §Phase 4, Step 4.6, deliberately deferred).

Defining this protocol now costs a few minutes; retrofitting it into Phase 6's
experiment runner later costs a day. ``NoOpReranker`` is the only
implementation -- a cross-encoder reranker is left as a stub for Phase 6+ to
fill in. Neither is registered in ``retrievers/registry.py`` or wired into the
CLI; nothing in Phase 4 calls this module.
"""

from __future__ import annotations

from typing import Protocol

from rag_lab.schemas import ScoredChunk


class Reranker(Protocol):
    name: str

    def rerank(self, query: str, results: list[ScoredChunk]) -> list[ScoredChunk]: ...


class NoOpReranker:
    name = "noop"

    def rerank(self, query: str, results: list[ScoredChunk]) -> list[ScoredChunk]:
        return results


__all__ = ["NoOpReranker", "Reranker"]
