"""Shared embedder primitives: the ``Embedder`` protocol, ``ModelSpec``, and the
normalize/truncate math every embedder implementation needs.

Two methods, not one (plan §Phase 3, Step 3.1): ``embed_documents`` and
``embed_query`` are kept separate on purpose. BGE and E5 are trained with
distinct query/passage prefixes, and collapsing this into a single ``embed()``
method makes it structurally impossible to honour that asymmetry correctly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


class Embedder(Protocol):
    name: str
    dim: int  # native output dim, pre-truncation
    default_params: dict[str, Any]

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...  # (n, d)
    def embed_query(self, text: str) -> np.ndarray: ...  # (d,)


@dataclass(frozen=True)
class ModelSpec:
    """A named embedder's fixed identity: which model, which prefixes, whether
    getting the prefixes wrong is a silent correctness bug (``asymmetric``)."""

    model_name: str
    query_prefix: str
    doc_prefix: str
    dim: int
    asymmetric: bool


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalization, guarding against all-zero rows."""
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return vectors / norms


def truncate_and_renormalize(vectors: np.ndarray, dim: int) -> np.ndarray:
    """Matryoshka truncation: slice to the first ``dim`` dims, then re-normalize.

    Re-normalizing after truncation is mandatory (plan §3.2's pitfall note) —
    skipping it makes cosine similarity subtly wrong rather than obviously
    broken, since a truncated-but-unnormalized vector no longer has unit norm.
    """
    vectors = np.asarray(vectors, dtype=np.float32)
    if dim <= 0:
        raise ValueError(f"truncate_dim must be > 0, got {dim}")
    if dim > vectors.shape[-1]:
        raise ValueError(
            f"truncate_dim ({dim}) exceeds the embedder's native dim ({vectors.shape[-1]})"
        )
    return l2_normalize(vectors[..., :dim])


__all__ = [
    "Embedder",
    "ModelSpec",
    "l2_normalize",
    "truncate_and_renormalize",
]
