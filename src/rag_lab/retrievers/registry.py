"""Named retrievers -- the ``--retriever``/``--retrievers`` CLI values. Mirrors
``embedders/registry.py``'s shape: ``resolve_params``/``build_retriever``/
``available_retrievers`` play the same role as ``resolve_params``/
``build_embedder``/``available_embedders`` there.

Wrapper retrievers (``hybrid``, ``parent_doc``, ``sentence_window``)
recursively call ``build_retriever`` for their own base component(s), so a
caller only ever constructs the one retriever it asked for.

``resolve_params`` uses ``config.deep_merge`` rather than a shallow
``{**defaults, **overrides}`` spread (unlike ``embedders/registry.py``,
whose params have no nested dicts): ``hybrid``'s ``weights`` default is
itself a dict, and a shallow merge would drop the untouched component's
weight the moment a caller overrides just one (``--params
hybrid.weights.dense=2.0`` should not silently erase ``weights.sparse``).
"""

from __future__ import annotations

from typing import Any

from rag_lab.config import deep_merge
from rag_lab.retrievers.base import Retriever, RetrieverSpec
from rag_lab.schemas import IndexManifest
from rag_lab.stores.base import VectorStore

NAMED_RETRIEVERS: dict[str, RetrieverSpec] = {
    "dense": RetrieverSpec(default_params={}),
    "bm25": RetrieverSpec(default_params={"k1": 1.5, "b": 0.75}, needs_rank_bm25=True),
    "hybrid": RetrieverSpec(
        default_params={
            "k_rrf": 60,
            "weights": {"dense": 1.0, "sparse": 1.0},
            "candidate_k": 50,
        },
        needs_rank_bm25=True,
        is_wrapper=True,
    ),
    "parent_doc": RetrieverSpec(
        # parent_chunk_set_id has NO default -- required override, validated in build_retriever.
        default_params={"base": "dense", "fanout": 5},
        is_wrapper=True,
    ),
    "sentence_window": RetrieverSpec(
        default_params={"base": "dense", "overlap_threshold": 0.5, "fanout": 3},
        is_wrapper=True,
    ),
}


def available_retrievers() -> list[str]:
    return sorted(NAMED_RETRIEVERS)


def resolve_params(name: str, overrides: dict[str, Any]) -> dict[str, Any]:
    """Retriever defaults with ``overrides`` deep-merged on top. This is the
    exact dict ``build_retriever`` inspects below."""
    if name not in NAMED_RETRIEVERS:
        raise ValueError(f"unknown retriever {name!r}; available: {available_retrievers()}")
    return deep_merge(NAMED_RETRIEVERS[name].default_params, overrides)


def _dense_embedder(manifest: IndexManifest):
    from rag_lab.indexing import embedder_for

    return embedder_for(manifest.embedder, manifest.embedder_params)


def build_retriever(
    name: str,
    overrides: dict[str, Any] | None,
    *,
    manifest: IndexManifest,
    store: VectorStore,
) -> Retriever:
    """Resolve a named retriever into a ready-to-use ``Retriever``.

    Concrete implementations are imported lazily so ``rag_lab.retrievers``
    stays import-safe under a ``core``-only install -- constructing ``bm25``
    or ``hybrid`` is the only thing that requires ``rank_bm25``.
    """
    if name not in NAMED_RETRIEVERS:
        raise ValueError(f"unknown retriever {name!r}; available: {available_retrievers()}")
    params = resolve_params(name, overrides or {})

    if name == "dense":
        from rag_lab.retrievers.dense import DenseRetriever

        return DenseRetriever(store, _dense_embedder(manifest))

    if name == "bm25":
        from rag_lab.retrievers.bm25 import BM25Retriever

        return BM25Retriever(
            manifest.chunk_set_id,
            store.persist_dir,
            k1=float(params["k1"]),
            b=float(params["b"]),
        )

    if name == "hybrid":
        from rag_lab.retrievers.hybrid import HybridRetriever

        dense = build_retriever("dense", {}, manifest=manifest, store=store)
        sparse = build_retriever("bm25", {}, manifest=manifest, store=store)
        return HybridRetriever(
            {"dense": dense, "sparse": sparse},
            k_rrf=int(params["k_rrf"]),
            weights=params.get("weights"),
            candidate_k=int(params["candidate_k"]),
        )

    if name == "parent_doc":
        from rag_lab.chunks import load_chunk_set
        from rag_lab.retrievers.parent_doc import ParentDocumentRetriever

        parent_chunk_set_id = params.get("parent_chunk_set_id")
        if not parent_chunk_set_id:
            raise ValueError(
                "parent_doc retriever requires a 'parent_chunk_set_id' param "
                "(CLI: --parent-chunk-set <chunk_set_id>)"
            )
        base = build_retriever(str(params["base"]), {}, manifest=manifest, store=store)
        parent_chunks = load_chunk_set(str(parent_chunk_set_id))
        return ParentDocumentRetriever(base, parent_chunks, fanout=int(params["fanout"]))

    if name == "sentence_window":
        from rag_lab.retrievers.sentence_window import SentenceWindowRetriever

        base = build_retriever(str(params["base"]), {}, manifest=manifest, store=store)
        return SentenceWindowRetriever(
            base,
            overlap_threshold=float(params["overlap_threshold"]),
            fanout=int(params["fanout"]),
        )

    raise AssertionError(f"unreachable: retriever {name!r} declared but not implemented")


__all__ = [
    "NAMED_RETRIEVERS",
    "available_retrievers",
    "build_retriever",
    "resolve_params",
]
