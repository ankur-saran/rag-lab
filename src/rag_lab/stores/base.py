"""The ``VectorStore`` protocol (plan §Phase 3, Step 3.4)."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from rag_lab.schemas import Chunk, ScoredChunk


class VectorStore(Protocol):
    def upsert(self, chunks: list[Chunk], vectors: np.ndarray) -> None: ...
    def search(
        self, vector: np.ndarray, k: int, filters: dict | None = None
    ) -> list[ScoredChunk]: ...
    def get(self, chunk_ids: list[str]) -> list[Chunk]: ...


__all__ = ["VectorStore"]
