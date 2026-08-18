"""Retrieval-quality metrics (plan Phase 6, Step 6.1) -- pure functions over
``(retrieved_ids, gold_ids)``. No I/O, no knowledge of chunk sets, corpora or
experiment cells; ``experiment/runner.py`` resolves gold ids via
``metrics.gold`` and assembles ``QueryTrace``/``RunResult`` around these.
"""

from __future__ import annotations

import math

import numpy as np

from rag_lab.schemas import Chunk


def recall_at_k(retrieved_ids: list[str], gold_ids: set[str], k: int) -> float:
    """Binary hit indicator: 1.0 if any of the top-k retrieved ids is gold,
    else 0.0. For single-gold ``lookup``/``synthesis`` pairs this is exactly
    hit-rate. For multi-span ``cross_reference`` pairs, ``gold_ids`` is
    expected to already be the any-span union (``metrics.gold.flatten_gold``)
    -- this is the headline "any span" reading; see
    ``metrics.gold.all_spans_hit`` for the stricter multi-hop diagnostic.
    """
    if not gold_ids:
        return 0.0
    return 1.0 if set(retrieved_ids[:k]) & gold_ids else 0.0


def mrr(retrieved_ids: list[str], gold_ids: set[str]) -> float:
    """Reciprocal rank of the first gold hit in ``retrieved_ids`` (whatever
    length the caller passes -- not clipped to a k), 0.0 if none hit."""
    if not gold_ids:
        return 0.0
    for rank, cid in enumerate(retrieved_ids, start=1):
        if cid in gold_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: list[str], gold_ids: set[str], k: int) -> float:
    """Binary-relevance nDCG, standard log2(rank + 1) discount, rank 1-based:

        DCG  = sum(1 / log2(rank + 1)  for each top-k hit)
        IDCG = the same sum over the best-case ranking (min(k, |gold|) hits
               placed first)
        nDCG = DCG / IDCG  (0.0 when IDCG == 0, i.e. no gold ids)

    Pinned exactly because discount conventions differ between sources (plan
    Phase 6 AC-2) -- tests/test_phase_6.py hand-computes a toy case against
    this formula.
    """
    if not gold_ids:
        return 0.0
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, cid in enumerate(retrieved_ids[:k], start=1)
        if cid in gold_ids
    )
    ideal_hits = min(k, len(gold_ids))
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def chunk_efficiency(retrieved_chunks: list[Chunk], k: int, hit: bool) -> float | None:
    """Total ``token_count`` across the top-k retrieved chunks -- the context
    cost paid for this query's answer. ``None`` when ``hit`` is False: a
    strategy must not be able to improve its average efficiency by simply
    failing more often, so callers aggregate this by skipping ``None``, never
    by treating it as 0.
    """
    if not hit:
        return None
    return float(sum(c.token_count for c in retrieved_chunks[:k]))


def latency_percentiles(latencies_ms: list[float]) -> dict[str, float]:
    """p50/p95 wall-clock retrieval latency, same ``np.percentile`` convention
    ``corpus.py``/``chunks.py`` already use elsewhere in this repo."""
    if not latencies_ms:
        return {"latency_p50": 0.0, "latency_p95": 0.0}
    return {
        "latency_p50": float(np.percentile(latencies_ms, 50)),
        "latency_p95": float(np.percentile(latencies_ms, 95)),
    }


__all__ = [
    "chunk_efficiency",
    "latency_percentiles",
    "mrr",
    "ndcg_at_k",
    "recall_at_k",
]
