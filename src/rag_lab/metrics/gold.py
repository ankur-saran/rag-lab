"""Gold resolution (plan Phase 6, Step 6.2) -- the critical step.

An ``EvalPair.gold_char_spans`` is authoritative and expressed in *document*
coordinates precisely so it can be re-resolved against any chunk set, which
is what makes cross-chunker comparison non-circular (schemas.py's ``EvalPair``
docstring, README "Two things worth knowing before extending this"). This
module is that re-resolution, plus the machinery ``cross_reference`` pairs'
multi-span gold needs on top of it.

Pure functions only -- no I/O, no knowledge of corpora or experiment cells.
"""

from __future__ import annotations

from rag_lab.schemas import Chunk


def _overlaps(span: tuple[int, int], chunk_span: tuple[int, int], min_overlap: float) -> bool:
    """Two-sided overlap test: a chunk is gold against ``span`` if it covers
    ``>= min_overlap`` of the span (the "tile" case -- several small chunks
    tiling one gold span), OR the span covers ``>= min_overlap`` of the chunk
    (the "swallow" case -- one large chunk containing a small gold span). A
    one-sided rule would systematically penalize whichever chunk size sits on
    the wrong side of the gold span, turning the whole chunker comparison into
    an artifact of the metric rather than a property of the chunkers.
    """
    s_start, s_end = span
    c_start, c_end = chunk_span
    inter = max(0, min(s_end, c_end) - max(s_start, c_start))
    if inter == 0:
        return False
    span_len = s_end - s_start
    chunk_len = c_end - c_start
    if span_len > 0 and inter / span_len >= min_overlap:
        return True
    if chunk_len > 0 and inter / chunk_len >= min_overlap:
        return True
    return False


def resolve_gold_per_span(
    spans: list[tuple[int, int]],
    doc_id: str,
    chunks: list[Chunk],
    min_overlap: float = 0.5,
) -> list[set[str]]:
    """One gold chunk-id set per span, evaluated *independently* -- never
    against the spans' union. Unioning first would mark every chunk sitting
    between two distant ``cross_reference`` spans as gold too, which would
    make that tier trivially easy to score correct (plan Step 6.2).

    ``chunks`` may be an entire chunk set; candidates are filtered to
    ``doc_id`` internally so char-offset coincidences with an unrelated
    document's chunks can never leak in.
    """
    candidates = [c for c in chunks if c.doc_id == doc_id]
    return [
        {c.chunk_id for c in candidates if _overlaps(span, c.span, min_overlap)}
        for span in spans
    ]


def flatten_gold(per_span: list[set[str]]) -> set[str]:
    """The any-span union -- what ``QueryTrace.gold_chunk_ids`` stores, and
    what the headline ``recall@k``/``mrr``/``ndcg@k`` are computed against."""
    result: set[str] = set()
    for span_gold in per_span:
        result |= span_gold
    return result


def all_spans_hit(retrieved_ids: list[str], per_span: list[set[str]], k: int) -> bool:
    """True iff *every* span has a retrieved gold chunk in the top-k -- the
    stricter "true multi-hop credit" reading of ``cross_reference`` recall
    (vs. the any-span headline in ``recall_at_k``). Meaningful only when
    ``len(per_span) > 1``; for single-span pairs this is identical to the
    headline hit and callers should not report it separately.

    A span whose own gold set is empty (this chunk set fragmented that region
    so badly no chunk aligns with it) can never be hit -- that's correct
    signal about the chunk set, not a bug in this function.
    """
    if not per_span:
        return False
    top_k = set(retrieved_ids[:k])
    return all(bool(top_k & span_gold) for span_gold in per_span)


__all__ = ["all_spans_hit", "flatten_gold", "resolve_gold_per_span"]
