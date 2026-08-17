"""``sentence_window`` -- dedupe overlapping windows returned by a base
retriever (plan §Phase 4, Step 4.5).

The widening itself already lives in ``Chunk.text`` from
``chunkers/sentence_window.py`` (a distinct module -- the chunker widens at
*chunk* time; this retriever only dedupes at *retrieval* time). Two results
are considered the same window once they overlap by more than 50% of the
shorter span:

    overlap_ratio = intersection_length / min(len(a), len(b))

computed only between candidates sharing a ``doc_id``. This is the
denominator the plan doc leaves unstated for "overlap by more than 50%".
"""

from __future__ import annotations

from rag_lab.retrievers.base import Retriever, truncate_and_rank
from rag_lab.schemas import Chunk, ScoredChunk

NAME = "sentence_window"


def overlap_ratio(a: Chunk, b: Chunk) -> float:
    if a.doc_id != b.doc_id:
        return 0.0
    start = max(a.char_start, b.char_start)
    end = min(a.char_end, b.char_end)
    intersection = end - start
    if intersection <= 0:
        return 0.0
    shorter = min(a.char_end - a.char_start, b.char_end - b.char_start)
    if shorter <= 0:
        return 0.0
    return intersection / shorter


class SentenceWindowRetriever:
    name = NAME

    def __init__(
        self,
        base: Retriever,
        *,
        overlap_threshold: float = 0.5,
        fanout: int = 3,
        name: str = NAME,
    ) -> None:
        self.base = base
        self.overlap_threshold = overlap_threshold
        self.fanout = fanout
        self.name = name

    def retrieve(self, query: str, k: int) -> list[ScoredChunk]:
        candidates = self.base.retrieve(query, k * self.fanout)

        kept: list[ScoredChunk] = []
        for candidate in candidates:  # already best-rank-first
            if any(
                overlap_ratio(candidate.chunk, existing.chunk) > self.overlap_threshold
                for existing in kept
            ):
                continue
            kept.append(candidate)

        return truncate_and_rank(kept, k, self.name)


__all__ = ["NAME", "SentenceWindowRetriever", "overlap_ratio"]
