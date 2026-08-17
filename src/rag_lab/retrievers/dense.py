"""``dense`` — thin wrapper over ``VectorStore.search`` using ``embed_query``
(plan §Phase 4, Step 4.2). The baseline every other retriever is measured
against.

``ChromaStore.search`` already returns fully-scored ``ScoredChunk``s (score =
cosine similarity, 1-based rank, ``debug={"chunk_id", "distance"}``) labeled
``retriever="vector_store"``; this class only needs to relabel that to
``"dense"``, never recompute score or rank.
"""

from __future__ import annotations

from rag_lab.embedders.base import Embedder
from rag_lab.retrievers.base import truncate_and_rank
from rag_lab.schemas import ScoredChunk
from rag_lab.stores.base import VectorStore

NAME = "dense"


class DenseRetriever:
    name = NAME

    def __init__(self, store: VectorStore, embedder: Embedder, *, name: str = NAME) -> None:
        self.store = store
        self.embedder = embedder
        self.name = name

    def retrieve(self, query: str, k: int) -> list[ScoredChunk]:
        vector = self.embedder.embed_query(query)
        results = self.store.search(vector, k)
        return truncate_and_rank(results, k, self.name)


__all__ = ["NAME", "DenseRetriever"]
