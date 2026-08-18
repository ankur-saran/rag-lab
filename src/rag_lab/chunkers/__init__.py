"""Chunking strategies (Phase 2 baselines; Phase 7 adds `semantic` and
`table_summary` to the same registry).

``REGISTRY`` maps a chunker name to an instance implementing the ``Chunker``
protocol, mirroring ``loaders.REGISTRY``'s extension-keyed dispatch. ``run_chunker``
is the one entry point the CLI needs to chunk a single document; ``build_chunk_set``
is the one entry point that produces (and caches) a whole corpus's chunk set --
what both the ``chunk run`` CLI command and Phase 6's experiment runner need,
factored out once so neither reimplements the role/parent handling and caching
by hand.
"""

from __future__ import annotations

from typing import Any

from rag_lab.chunkers.base import Chunker, ChunkSpec
from rag_lab.chunkers.fixed import FixedChunker
from rag_lab.chunkers.hierarchy import OrphanChildError, assign_parents
from rag_lab.chunkers.markdown_chunker import MarkdownChunker
from rag_lab.chunkers.recursive import RecursiveChunker
from rag_lab.chunkers.sentence_window import SentenceWindowChunker
from rag_lab.corpus import list_documents_by_corpus
from rag_lab.ids import make_chunk_set_id
from rag_lab.jsonl import read_jsonl, write_jsonl
from rag_lab.paths import artifact_path
from rag_lab.schemas import Chunk, ChunkRole, Document

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


def build_chunk_set(
    corpus: str,
    chunker: str,
    overrides: dict[str, Any] | None = None,
    *,
    role: ChunkRole = "standalone",
    parent_chunk_set_id: str | None = None,
    force: bool = False,
) -> tuple[str, list[Chunk], bool]:
    """Resolve (or build and cache) the chunk set for
    ``(corpus, chunker, overrides, role[, parent_chunk_set_id])``.

    Mirrors ``indexing.build_index``'s shape: the id is computed first, and a
    cache hit returns without ever touching the chunker or the corpus's
    documents. ``role``/``parent_chunk_set_id`` are folded into the hashed
    params exactly as the ``chunk run`` CLI command does (plan §4.7's
    correction -- a role-tagged chunk set must never collide with its
    standalone twin), so this reproduces byte-identical output to what
    ``chunk run`` would have written for the same inputs.

    Returns ``(chunk_set_id, chunks, cache_hit)``. Raises ``ValueError`` for a
    bad role/chunker, ``LookupError`` if the corpus or (for ``role="child"``)
    the parent chunk set can't be resolved, and ``OrphanChildError`` if a
    child chunk matches zero parents.
    """
    if role not in {"standalone", "parent", "child"}:
        raise ValueError(f"role must be one of standalone|parent|child, got {role!r}")
    if role == "child" and not parent_chunk_set_id:
        raise ValueError("role='child' requires parent_chunk_set_id")
    if role != "child" and parent_chunk_set_id:
        raise ValueError("parent_chunk_set_id only applies to role='child'")

    params = dict(overrides or {})
    if role != "standalone":
        params["role"] = role
    if role == "child":
        params["parent_chunk_set_id"] = parent_chunk_set_id

    resolved_params = resolve_params(chunker, params)
    chunk_set_id = make_chunk_set_id(corpus, chunker, resolved_params)
    out_path = artifact_path("chunks", f"{chunk_set_id}.jsonl")

    if not force and out_path.exists():
        cached = read_jsonl(out_path, Chunk)
        if cached:
            return chunk_set_id, cached, True

    docs = list_documents_by_corpus(corpus)[corpus]
    all_chunks: list[Chunk] = []
    for doc in docs:
        all_chunks.extend(run_chunker(chunker, doc, params))

    if not all_chunks:
        raise ValueError(f"chunker {chunker!r} produced zero chunks for corpus {corpus!r}")

    if role == "parent":
        all_chunks = [c.model_copy(update={"role": "parent"}) for c in all_chunks]
    elif role == "child":
        from rag_lab.chunks import load_chunk_set  # lazy: avoids a circular import

        parent_chunks = load_chunk_set(parent_chunk_set_id)
        all_chunks = assign_parents(all_chunks, parent_chunks)

    write_jsonl(out_path, all_chunks)
    return chunk_set_id, all_chunks, False


__all__ = [
    "REGISTRY",
    "Chunker",
    "ChunkSpec",
    "OrphanChildError",
    "available_chunkers",
    "build_chunk_set",
    "resolve_params",
    "run_chunker",
]
