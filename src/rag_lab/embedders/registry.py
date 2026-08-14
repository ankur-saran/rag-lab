"""Named embedders — the ``--embedder`` CLI values and ``index_id``'s embedder
component. Mirrors ``chunkers/__init__.py``'s registry shape:
``resolve_params``/``build_embedder``/``available_embedders`` play the same
role as ``resolve_params``/``run_chunker``/``available_chunkers`` there.

The four models are plan §3.2's table verbatim. ``HashEmbedder``
(``embedders/fixture.py``) is deliberately absent from this registry — it is a
fixture/test-only component, the same way ``paragraph_fixture`` in
``scripts/build_fixtures.py`` is never a real Phase 2 chunker.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rag_lab.embedders.base import ModelSpec

if TYPE_CHECKING:
    from rag_lab.embedders.sentence_transformer import SentenceTransformerEmbedder

NAMED_EMBEDDERS: dict[str, ModelSpec] = {
    "bge-small": ModelSpec(
        model_name="BAAI/bge-small-en-v1.5",
        query_prefix="Represent this sentence for searching relevant passages: ",
        doc_prefix="",
        dim=384,
        asymmetric=True,
    ),
    "e5-base": ModelSpec(
        model_name="intfloat/e5-base-v2",
        query_prefix="query: ",
        doc_prefix="passage: ",
        dim=768,
        asymmetric=True,
    ),
    "e5-multilingual": ModelSpec(
        model_name="intfloat/multilingual-e5-base",
        query_prefix="query: ",
        doc_prefix="passage: ",
        dim=768,
        asymmetric=True,
    ),
    "minilm": ModelSpec(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        query_prefix="",
        doc_prefix="",
        dim=384,
        asymmetric=False,
    ),
}


def available_embedders() -> list[str]:
    return sorted(NAMED_EMBEDDERS)


def default_params_for(name: str) -> dict[str, Any]:
    """The exact dict that gets hashed into ``index_id`` (via ``resolve_params``
    below), so it must be built the same way every time a given (name,
    overrides) pair is resolved. ``batch_size`` is excluded from hashing via
    the local ``_nohash`` list (see ``config.py``) — it's a performance knob,
    not a parameter that changes the resulting vectors, so changing it should
    not invalidate the cache.
    """
    if name not in NAMED_EMBEDDERS:
        raise ValueError(f"unknown embedder {name!r}; available: {available_embedders()}")
    spec = NAMED_EMBEDDERS[name]
    return {
        "model_name": spec.model_name,
        "query_prefix": spec.query_prefix,
        "doc_prefix": spec.doc_prefix,
        "normalize": True,
        "batch_size": 32,
        "truncate_dim": None,
        "strict_prefixes": True,
        "_nohash": ["batch_size"],
    }


def resolve_params(name: str, overrides: dict[str, Any]) -> dict[str, Any]:
    """Embedder defaults with ``overrides`` layered on top."""
    return {**default_params_for(name), **overrides}


def build_embedder(
    name: str, overrides: dict[str, Any] | None = None
) -> SentenceTransformerEmbedder:
    """Resolve a named embedder into a ready-to-use ``SentenceTransformerEmbedder``.

    Imports the implementation lazily so ``rag_lab.embedders`` stays
    import-safe (no ``sentence_transformers``/``torch`` import) under a
    ``core``-only install — only calling this function requires the ``embed``
    extra.
    """
    from rag_lab.embedders.sentence_transformer import SentenceTransformerEmbedder

    if name not in NAMED_EMBEDDERS:
        raise ValueError(f"unknown embedder {name!r}; available: {available_embedders()}")
    params = resolve_params(name, overrides or {})
    return SentenceTransformerEmbedder(name=name, spec=NAMED_EMBEDDERS[name], params=params)


__all__ = [
    "NAMED_EMBEDDERS",
    "available_embedders",
    "build_embedder",
    "default_params_for",
    "resolve_params",
]
