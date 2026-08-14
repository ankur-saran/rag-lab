"""``ChromaStore`` — the ``VectorStore`` implementation (plan §Phase 3, Step 3.4).

Everything that touches ``chromadb`` is lazily imported inside functions, the
same convention ``embedders.sentence_transformer`` uses for
``sentence_transformers``, so ``rag_lab.stores`` stays import-safe under a
``core``-only install.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from rag_lab.schemas import Chunk, ScoredChunk

COLLECTION_NAME = "chunks"

# Chunk fields duplicated into Chroma's flat metadata so `filters` can query
# them directly, on top of the full chunk stored as "_chunk_json" -- which is
# what satisfies "retrieval never needs to re-read the chunk JSONL" (plan §3.4).
_FILTERABLE_FIELDS = ("doc_id", "corpus", "chunk_set_id", "chunker")


def _client(persist_dir: Path):
    import chromadb
    from chromadb.config import Settings

    persist_dir.mkdir(parents=True, exist_ok=True)
    # anonymized_telemetry=False: Chroma's default telemetry can attempt a
    # network call, which is exactly the offline surprise this project
    # otherwise avoids (--mock-llm, no-secrets CI).
    return chromadb.PersistentClient(
        path=str(persist_dir),
        settings=Settings(anonymized_telemetry=False),
    )


def _open_collection(client: Any, name: str = COLLECTION_NAME) -> Any:
    """Open (or create) the collection with cosine distance explicit.

    Chroma's default distance is L2, not cosine (plan's pitfall note) — with
    normalized vectors the two produce identical *rankings* but unintuitive
    *scores*, and this can only be set at collection-creation time.

    ``embedding_function=None`` is load-bearing: without it, Chroma falls
    back to its own default ONNX MiniLM embedding function the moment some
    call path is missing explicit ``embeddings=``, silently triggering an
    unwanted second model download. Every method below passes embeddings
    explicitly, so that default is never exercised anyway — ``None`` just
    makes that a guarantee instead of an accident.

    Tries the modern (chromadb >= 1.0) ``configuration=`` kwarg first and
    falls back to the legacy ``metadata={"hnsw:space": ...}`` form, so this
    keeps working across the ``chromadb>=0.5`` range pyproject.toml allows.
    """
    try:
        configuration = {"hnsw": {"space": "cosine"}}
        return client.get_or_create_collection(
            name=name, configuration=configuration, embedding_function=None
        )
    except TypeError:
        return client.get_or_create_collection(
            name=name, metadata={"hnsw:space": "cosine"}, embedding_function=None
        )


def _flat_metadata(chunk: Chunk) -> dict[str, Any]:
    meta = {field: getattr(chunk, field) for field in _FILTERABLE_FIELDS}
    meta["_chunk_json"] = chunk.model_dump_json()
    return meta


def _chunk_from_metadata(meta: dict[str, Any]) -> Chunk:
    return Chunk.model_validate_json(meta["_chunk_json"])


class ChromaStore:
    """A ``VectorStore`` backed by a Chroma persistent client at ``persist_dir``.

    One collection ("chunks") per ``persist_dir``, and one ``persist_dir`` per
    ``index_id`` (``artifacts/indexes/<index_id>/`` — see ``indexing.py``), so
    collection names never collide across indexes.
    """

    def __init__(self, persist_dir: Path | str) -> None:
        self.persist_dir = Path(persist_dir)
        self._client = _client(self.persist_dir)
        self._collection = _open_collection(self._client)

    def upsert(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) != len(vectors):
            raise ValueError(
                f"chunks ({len(chunks)}) and vectors ({len(vectors)}) must be the same length"
            )
        if not chunks:
            return
        self._collection.upsert(
            ids=[c.chunk_id for c in chunks],
            embeddings=np.asarray(vectors, dtype=np.float32),
            documents=[c.text for c in chunks],
            metadatas=[_flat_metadata(c) for c in chunks],
        )

    def search(
        self, vector: np.ndarray, k: int, filters: dict | None = None
    ) -> list[ScoredChunk]:
        result = self._collection.query(
            query_embeddings=[np.asarray(vector, dtype=np.float32).tolist()],
            n_results=k,
            where=filters,
            include=["metadatas", "distances"],
        )
        ids, distances, metadatas = result["ids"][0], result["distances"][0], result["metadatas"][0]
        scored = []
        for rank, (chunk_id, distance, meta) in enumerate(
            zip(ids, distances, metadatas, strict=True), start=1
        ):
            scored.append(
                ScoredChunk(
                    chunk=_chunk_from_metadata(meta),
                    score=1.0 - float(distance),  # cosine distance -> similarity
                    rank=rank,
                    retriever="vector_store",
                    debug={"chunk_id": chunk_id, "distance": float(distance)},
                )
            )
        return scored

    def get(self, chunk_ids: list[str]) -> list[Chunk]:
        if not chunk_ids:
            return []
        result = self._collection.get(ids=chunk_ids, include=["metadatas"])
        return [_chunk_from_metadata(meta) for meta in result["metadatas"]]

    def count(self) -> int:
        return self._collection.count()

    def close(self) -> None:
        """Release the underlying Chroma client's file handles.

        Needed before deleting or moving a ``persist_dir`` on Windows: Chroma
        caches clients by path in a process-wide ``SharedSystemClient``
        registry, and the sqlite/HNSW files it opens stay open+locked for the
        life of that cache entry. Three things have to happen together for
        the OS to actually release the lock: drop this instance's own
        references to the client/collection, drop Chroma's process-wide
        cache entry (``clear_system_cache``, which has no narrower,
        per-instance form), and force a collection so CPython's finalizers
        run before the caller proceeds — skipping any one of the three
        leaves the files locked even though it looks like everything was
        cleaned up. Production CLI usage is one-shot per process and never
        needs this; it matters for the fixture-build script and for tests
        that build several stores in one process and then need to delete
        their directories.
        """
        import gc

        import chromadb

        self._collection = None
        self._client = None
        chromadb.api.client.SharedSystemClient.clear_system_cache()
        gc.collect()


__all__ = ["COLLECTION_NAME", "ChromaStore"]
