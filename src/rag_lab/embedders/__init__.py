"""Embedding models behind a common protocol (Phase 3).

``registry.py`` maps a friendly name (``bge-small``, ``minilm``, ...) to a
``ModelSpec`` and builds a ``SentenceTransformerEmbedder`` from it, mirroring
``chunkers/__init__.py``'s ``REGISTRY``/``run_chunker`` shape. Nothing
imported here touches ``sentence_transformers``/``torch`` until
``build_embedder`` is actually called, so importing this package stays safe
under a ``core``-only install.
"""

from __future__ import annotations

from rag_lab.embedders.base import Embedder, ModelSpec, l2_normalize, truncate_and_renormalize
from rag_lab.embedders.registry import (
    NAMED_EMBEDDERS,
    available_embedders,
    build_embedder,
    default_params_for,
    resolve_params,
)

__all__ = [
    "NAMED_EMBEDDERS",
    "Embedder",
    "ModelSpec",
    "available_embedders",
    "build_embedder",
    "default_params_for",
    "l2_normalize",
    "resolve_params",
    "truncate_and_renormalize",
]
