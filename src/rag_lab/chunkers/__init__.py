"""Chunking strategies (Phase 2 baselines; Phase 7 adds `semantic` and
`table_summary` to the same registry).

``REGISTRY`` maps a chunker name to an instance implementing the ``Chunker``
protocol, mirroring ``loaders.REGISTRY``'s extension-keyed dispatch. ``run_chunker``
is the one entry point the CLI needs: resolve params (chunker defaults, then
caller overrides), then chunk.
"""

from __future__ import annotations

from typing import Any

from rag_lab.chunkers.base import Chunker, ChunkSpec
from rag_lab.chunkers.fixed import FixedChunker
from rag_lab.chunkers.markdown_chunker import MarkdownChunker
from rag_lab.chunkers.recursive import RecursiveChunker
from rag_lab.chunkers.sentence_window import SentenceWindowChunker
from rag_lab.schemas import Chunk, Document

REGISTRY: dict[str, Chunker] = {
    "fixed": FixedChunker(),
    "recursive": RecursiveChunker(),
    "markdown": MarkdownChunker(),
    "sentence_window": SentenceWindowChunker(),
}


def available_chunkers() -> list[str]:
    return sorted(REGISTRY)


def resolve_params(chunker: str, overrides: dict[str, Any]) -> dict[str, Any]:
    """Chunker defaults with ``overrides`` layered on top. This is the exact
    dict that gets hashed into ``chunk_set_id`` and ``chunk_id``, so it must
    be built the same way every time a given (chunker, overrides) pair is run."""
    if chunker not in REGISTRY:
        raise ValueError(f"unknown chunker {chunker!r}; available: {available_chunkers()}")
    return {**REGISTRY[chunker].default_params, **overrides}


def run_chunker(
    chunker: str, doc: Document, overrides: dict[str, Any] | None = None
) -> list[Chunk]:
    """Chunk one document with a named strategy. ``overrides`` are layered
    over that chunker's defaults before chunking."""
    if chunker not in REGISTRY:
        raise ValueError(f"unknown chunker {chunker!r}; available: {available_chunkers()}")
    params = resolve_params(chunker, overrides or {})
    return REGISTRY[chunker].chunk(doc, params)


__all__ = [
    "REGISTRY",
    "Chunker",
    "ChunkSpec",
    "available_chunkers",
    "resolve_params",
    "run_chunker",
]
