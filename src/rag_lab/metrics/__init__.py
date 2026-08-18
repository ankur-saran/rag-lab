"""Retrieval-quality metrics, gold resolution, and bootstrap CIs. Phase 6.

Re-export surface only, mirroring ``retrievers/__init__.py``'s shape --
callers (``experiment/runner.py``, ``experiment/report.py``, tests) import
from here rather than reaching into the individual modules.
"""

from __future__ import annotations

from rag_lab.metrics.bootstrap import bootstrap_ci
from rag_lab.metrics.core import chunk_efficiency, latency_percentiles, mrr, ndcg_at_k, recall_at_k
from rag_lab.metrics.gold import all_spans_hit, flatten_gold, resolve_gold_per_span

__all__ = [
    "all_spans_hit",
    "bootstrap_ci",
    "chunk_efficiency",
    "flatten_gold",
    "latency_percentiles",
    "mrr",
    "ndcg_at_k",
    "recall_at_k",
    "resolve_gold_per_span",
]
