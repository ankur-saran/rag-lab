"""``parent_doc`` -- retrieve children, map to their parent via ``parent_id``,
dedupe parents preserving the best (lowest-rank) child, return parents (plan
§Phase 4, Step 4.5).

Parents are never embedded/indexed -- only children are searched -- so parent
``Chunk`` objects can't come from a ``VectorStore``. The registry (see
``retrievers/registry.py``) loads the parent chunk set from disk via
``chunks.load_chunk_set`` and passes the resulting list in here.

Fails loud rather than degrading silently, matching this repo's existing style
(``chunks.py``/``ids.py``): every retrieved chunk lacking a ``parent_id`` means
the base index wasn't built from a ``role="child"`` chunk set, and a
``parent_id`` that doesn't resolve in the loaded parent set means the wrong
``--parent-chunk-set`` was given. Both are configuration errors, not data
oddities, so both raise instead of silently passing results through.
"""

from __future__ import annotations

from rag_lab.retrievers.base import Retriever, truncate_and_rank
from rag_lab.schemas import Chunk, ScoredChunk

NAME = "parent_doc"


class ParentDocumentRetriever:
    name = NAME

    def __init__(
        self,
        base: Retriever,
        parent_chunks: list[Chunk],
        *,
        fanout: int = 5,
        name: str = NAME,
    ) -> None:
        if not parent_chunks:
            raise ValueError("parent_doc retriever: parent_chunks must be non-empty")
        self.base = base
        self.fanout = fanout
        self.name = name
        self._parents_by_id = {p.chunk_id: p for p in parent_chunks}

    def retrieve(self, query: str, k: int) -> list[ScoredChunk]:
        candidates = self.base.retrieve(query, k * self.fanout)

        if candidates and all(c.chunk.parent_id is None for c in candidates):
            raise ValueError(
                "parent_doc retriever: none of the retrieved chunks have a parent_id set "
                "-- was the index built from a role='child' chunk set "
                "(chunk run --role child --parent-chunk-set ...)?"
            )

        best_by_parent: dict[str, ScoredChunk] = {}
        child_counts: dict[str, int] = {}
        for candidate in candidates:
            parent_id = candidate.chunk.parent_id
            if parent_id is None:
                continue
            if parent_id not in self._parents_by_id:
                raise ValueError(
                    f"parent_doc retriever: chunk {candidate.chunk.chunk_id!r} references "
                    f"parent_id {parent_id!r}, which is not in the loaded parent chunk set "
                    "-- wrong --parent-chunk-set?"
                )
            child_counts[parent_id] = child_counts.get(parent_id, 0) + 1
            current_best = best_by_parent.get(parent_id)
            if current_best is None or candidate.rank < current_best.rank:
                best_by_parent[parent_id] = candidate

        results = [
            ScoredChunk(
                chunk=self._parents_by_id[parent_id],
                score=best_child.score,
                rank=best_child.rank,
                retriever=self.name,
                debug={
                    "child_chunk_id": best_child.chunk.chunk_id,
                    "child_rank": best_child.rank,
                    "n_children_matched": child_counts[parent_id],
                },
            )
            for parent_id, best_child in best_by_parent.items()
        ]
        results.sort(key=lambda r: r.rank)
        return truncate_and_rank(results, k, self.name)


__all__ = ["NAME", "ParentDocumentRetriever"]
