"""Retrieval strategies (Phase 4).

``registry.py`` maps a friendly name (``dense``, ``bm25``, ``hybrid``,
``parent_doc``, ``sentence_window``) to a ready-to-use ``Retriever``, mirroring
``chunkers/__init__.py``'s ``REGISTRY``/``run_chunker`` and
``embedders/registry.py``'s ``NAMED_EMBEDDERS``/``build_embedder`` shape.
Nothing imported here touches ``rank_bm25``/``chromadb`` until
``build_retriever`` is actually called for a retriever that needs them, so
importing this package stays safe under a ``core``-only install.
"""

from __future__ import annotations

from rag_lab.retrievers.base import Retriever, RetrieverSpec, truncate_and_rank
from rag_lab.retrievers.registry import (
    NAMED_RETRIEVERS,
    available_retrievers,
    build_retriever,
    resolve_params,
)

__all__ = [
    "NAMED_RETRIEVERS",
    "Retriever",
    "RetrieverSpec",
    "available_retrievers",
    "build_retriever",
    "resolve_params",
    "truncate_and_rank",
]
