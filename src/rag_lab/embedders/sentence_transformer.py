"""``SentenceTransformerEmbedder`` — the local ``Embedder`` implementation
(plan §Phase 3, Step 3.2). Requires the ``embed`` extra.

Nothing in this module runs at import time beyond defining the class: the
actual ``sentence_transformers`` import and model download/load happen only
inside ``_model()``, the first time an instance actually encodes something.
That keeps ``rag_lab.embedders`` safe to import under a ``core``-only
install — a caller only pays the ``embed``-extra cost when it calls
``embed_documents``/``embed_query``.
"""

from __future__ import annotations

import functools
from typing import Any

import numpy as np

from rag_lab.embedders.base import ModelSpec, truncate_and_renormalize


@functools.lru_cache(maxsize=4)
def _model(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


class SentenceTransformerEmbedder:
    """Wraps a local sentence-transformers model with asymmetric query/document
    prefixes and optional Matryoshka truncation.

    The model is lazily loaded and cached by ``model_name`` (see ``_model``
    above), so constructing this class never touches disk or network, and two
    instances that share a ``model_name`` share the loaded weights rather than
    reloading them.
    """

    def __init__(self, name: str, spec: ModelSpec, params: dict[str, Any]) -> None:
        self.name = name
        self.spec = spec
        self.params = dict(params)

        self.model_name: str = params.get("model_name", spec.model_name)
        self.query_prefix: str = params.get("query_prefix", spec.query_prefix)
        self.doc_prefix: str = params.get("doc_prefix", spec.doc_prefix)
        self.normalize: bool = bool(params.get("normalize", True))
        self.batch_size: int = int(params.get("batch_size", 32))
        self.truncate_dim: int | None = params.get("truncate_dim")
        self.strict_prefixes: bool = bool(params.get("strict_prefixes", True))

        if (
            self.strict_prefixes
            and spec.asymmetric
            and not self.query_prefix
            and not self.doc_prefix
        ):
            raise ValueError(
                f"embedder {name!r} wraps a known-asymmetric model ({self.model_name}) "
                "but was configured with empty query_prefix and doc_prefix. Getting this "
                "wrong is silent and costs several points of recall — pass explicit "
                "prefixes, or set strict_prefixes=False to opt out deliberately."
            )

        # Advertised dim, for callers that need it before encoding anything.
        # The manifest's authoritative dim comes from the real output shape
        # (see indexing.py), which stays correct even if `model_name` was
        # overridden away from this embedder's named preset.
        self.dim: int = self.truncate_dim or spec.dim

    def _encode(self, texts: list[str]) -> np.ndarray:
        model = _model(self.model_name)
        vectors = model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        vectors = np.asarray(vectors, dtype=np.float32)
        if self.truncate_dim:
            vectors = truncate_and_renormalize(vectors, self.truncate_dim)
        return vectors

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._encode([self.doc_prefix + t for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._encode([self.query_prefix + text])[0]


__all__ = ["SentenceTransformerEmbedder"]
