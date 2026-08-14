"""``HashEmbedder`` — a deterministic, dependency-free stand-in for a real
embedder.

It carries no semantic meaning whatsoever and must never be used to draw a
conclusion about retrieval quality. Its only job is to satisfy the
``Embedder`` protocol (including query/document prefix mechanics) with no
``torch``/``sentence_transformers`` import anywhere in this module, so it can
build the committed ``fixtures/indexes/sample/`` fixture and drive fast unit
tests without the ``embed`` extra's heavy dependencies or a model download.

This mirrors ``paragraph_fixture`` in ``scripts/build_fixtures.py``: a
fixture-only component, deliberately never registered in
``embedders.registry.NAMED_EMBEDDERS`` and never reachable from the CLI's
``--embedder`` flag.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

DEFAULT_DIM = 32


def _token_vector(token: str, dim: int) -> np.ndarray:
    """A deterministic pseudo-random unit-scale vector for one token.

    Seeded from a SHA1 digest of the token rather than numpy's global RNG, so
    the result is identical across processes and platforms without any shared
    mutable state — the same determinism guarantee ``ids.py``'s hash-based IDs
    rely on.
    """
    digest = hashlib.sha1(token.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "big")
    rng = np.random.default_rng(seed)
    return rng.normal(size=dim)


def hash_embed(text: str, dim: int = DEFAULT_DIM) -> np.ndarray:
    """Sum of per-token hashed vectors, L2-normalized. Empty text still yields
    a well-defined unit vector rather than a zero vector or an error."""
    tokens = text.lower().split() or [""]
    vector = np.zeros(dim, dtype=np.float32)
    for token in tokens:
        vector += _token_vector(token, dim)
    norm = np.linalg.norm(vector)
    if norm == 0:
        vector[0] = 1.0
        norm = 1.0
    return (vector / norm).astype(np.float32)


class HashEmbedder:
    name = "hash-fixture"
    dim = DEFAULT_DIM
    default_params: dict[str, Any] = {"query_prefix": "query: ", "doc_prefix": "passage: "}

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        resolved = {**self.default_params, **(params or {})}
        self.query_prefix: str = resolved["query_prefix"]
        self.doc_prefix: str = resolved["doc_prefix"]

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.stack([hash_embed(self.doc_prefix + t, self.dim) for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return hash_embed(self.query_prefix + text, self.dim)


__all__ = ["DEFAULT_DIM", "HashEmbedder", "hash_embed"]
