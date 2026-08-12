"""`fixed` — plain token-window chunker: slide a fixed-size token window over
the whole document with no regard for any structural signal. This is the
baseline every other chunker is measured against.
"""

from __future__ import annotations

from typing import Any

from rag_lab.chunkers.base import ChunkSpec, finalize_chunks, token_windows
from rag_lab.schemas import Chunk, Document

NAME = "fixed"
DEFAULT_PARAMS: dict[str, Any] = {
    "chunk_tokens": 512,
    "overlap_tokens": 64,
    "encoding": "cl100k_base",
}


class FixedChunker:
    name = NAME
    default_params = DEFAULT_PARAMS

    def chunk(self, doc: Document, params: dict[str, Any]) -> list[Chunk]:
        resolved = {**DEFAULT_PARAMS, **params}
        encoding = str(resolved["encoding"])

        spans = token_windows(
            doc.text,
            base_offset=0,
            chunk_tokens=int(resolved["chunk_tokens"]),
            overlap_tokens=int(resolved["overlap_tokens"]),
            encoding=encoding,
        )
        specs = [
            ChunkSpec(char_start=start, char_end=end, text=doc.text[start:end])
            for start, end in spans
        ]
        return finalize_chunks(specs, doc, NAME, resolved, encoding=encoding)


__all__ = ["DEFAULT_PARAMS", "NAME", "FixedChunker"]
