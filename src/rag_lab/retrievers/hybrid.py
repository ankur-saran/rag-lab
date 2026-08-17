"""``hybrid`` — Reciprocal Rank Fusion of a dense and a sparse retriever (plan
§Phase 4, Step 4.4).

    score(d) = Σ_r  weight_r / (k_rrf + rank_r(d))

summed **only over components whose candidate_k list actually contains d** —
a document dense retrieved but BM25 never surfaced gets no sparse term, not a
zero-filled one. This is the detail the plan doc leaves unstated, and it is
what gives a hand-computed toy case a single well-defined answer: a document
present in every component's list sums every component's term; a document
present in only one sums only that one term.

Fusion happens on **rank**, never score — dense cosine similarities and BM25
scores live on incomparable scales, and RRF's whole appeal is needing no
per-corpus tuning to reconcile them.
"""

from __future__ import annotations

from typing import Any

from rag_lab.retrievers.base import Retriever, truncate_and_rank
from rag_lab.schemas import ScoredChunk

NAME = "hybrid"


def reciprocal_rank_fusion(
    ranked_lists: dict[str, list[ScoredChunk]],
    *,
    k_rrf: int = 60,
    weights: dict[str, float] | None = None,
) -> list[ScoredChunk]:
    """Fuse named ranked lists into one, sorted by descending fused score.

    Returns every chunk that appears in at least one input list (not just the
    top ``k``) — callers truncate afterward via ``truncate_and_rank``. Each
    result's ``debug`` carries every component's own rank/score, plus the
    fused score, so a hybrid result stays explicable rather than magical.
    """
    weights = weights or {name: 1.0 for name in ranked_lists}

    fused_scores: dict[str, float] = {}
    representative: dict[str, ScoredChunk] = {}
    components: dict[str, dict[str, dict[str, Any]]] = {}

    for component, results in ranked_lists.items():
        weight = weights.get(component, 1.0)
        for r in results:
            chunk_id = r.chunk.chunk_id
            representative.setdefault(chunk_id, r)
            fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + weight / (k_rrf + r.rank)
            components.setdefault(chunk_id, {})[component] = {"rank": r.rank, "score": r.score}

    ordered_ids = sorted(fused_scores, key=lambda cid: (-fused_scores[cid], cid))
    return [
        ScoredChunk(
            chunk=representative[chunk_id].chunk,
            score=fused_scores[chunk_id],
            rank=i,
            retriever=NAME,
            debug={"rrf_score": fused_scores[chunk_id], "components": components[chunk_id]},
        )
        for i, chunk_id in enumerate(ordered_ids, start=1)
    ]


class HybridRetriever:
    name = NAME

    def __init__(
        self,
        components: dict[str, Retriever],
        *,
        k_rrf: int = 60,
        weights: dict[str, float] | None = None,
        candidate_k: int = 50,
        name: str = NAME,
    ) -> None:
        if not components:
            raise ValueError("HybridRetriever needs at least one component retriever")
        self.components = components
        self.k_rrf = k_rrf
        self.weights = weights
        self.candidate_k = candidate_k
        self.name = name

    def retrieve(self, query: str, k: int) -> list[ScoredChunk]:
        candidate_k = max(self.candidate_k, k)
        ranked_lists = {
            component: retriever.retrieve(query, candidate_k)
            for component, retriever in self.components.items()
        }
        fused = reciprocal_rank_fusion(ranked_lists, k_rrf=self.k_rrf, weights=self.weights)
        return truncate_and_rank(fused, k, self.name)


__all__ = ["NAME", "HybridRetriever", "reciprocal_rank_fusion"]
