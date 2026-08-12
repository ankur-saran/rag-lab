"""Shared chunker primitives: the ``Chunker`` protocol, the boundary-only
``ChunkSpec``, token counting, and ``finalize_chunks()``.

``finalize_chunks()`` is what makes a new chunker "genuinely just the boundary
logic" (plan §Phase 2, Step 2.1): it assigns ``chunk_id``, ``ordinal`` and
``token_count``, defaults ``embed_text`` to ``text``, and validates the offset
invariant. A chunker only ever has to produce a list of ``ChunkSpec`` —
character spans plus whatever ``text``/``embed_text``/``heading_path`` it
wants — and this module turns that into valid ``Chunk`` records.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from typing import Any, Protocol

from rag_lab.ids import chunker_signature, make_chunk_id, make_chunk_set_id
from rag_lab.schemas import Chunk, ChunkRole, Document

DEFAULT_ENCODING = "cl100k_base"


class Chunker(Protocol):
    name: str
    default_params: dict[str, Any]

    def chunk(self, doc: Document, params: dict[str, Any]) -> list[Chunk]: ...


@dataclass
class ChunkSpec:
    """A chunker's boundary decision, before identity and token count exist.

    ``text`` must equal ``doc.text[char_start:char_end]`` — every chunker in
    this module produces genuine slices, widened or not, so the offset
    invariant holds exactly rather than by the weaker "contains" fallback the
    plan allows for. ``embed_text`` left as ``None`` defaults to ``text`` in
    ``finalize_chunks``.
    """

    char_start: int
    char_end: int
    text: str
    embed_text: str | None = None
    heading_path: list[str] = field(default_factory=list)
    parent_id: str | None = None
    role: ChunkRole = "standalone"
    meta: dict[str, Any] = field(default_factory=dict)


@functools.lru_cache(maxsize=4)
def _encoding(name: str):
    import tiktoken

    return tiktoken.get_encoding(name)


def count_tokens(text: str, encoding: str = DEFAULT_ENCODING) -> int:
    if not text:
        return 0
    return len(_encoding(encoding).encode(text))


def token_char_offsets(text: str, encoding: str = DEFAULT_ENCODING) -> list[int]:
    """Per-token start-character-offsets of ``text`` under ``encoding``, plus
    a final sentinel of ``len(text)`` — so a token window ``[i, j)`` maps to
    the character span ``(offsets[i], offsets[j])`` directly.

    Uses ``Encoding.decode_with_offsets``, which tracks the character offset
    of each token as it re-decodes the *cumulative* token sequence — not each
    token in isolation — so it can never mis-map, unlike decoding a window
    and searching for it back in the source text, which silently picks the
    wrong occurrence whenever the decoded text repeats elsewhere in the
    document. It is also immune to the case a naive per-token byte-accumulate
    would have to guard by hand: a BPE token boundary landing mid-character
    (real for CJK/emoji spans) never produces an invalid offset here.
    """
    enc = _encoding(encoding)
    token_ids = enc.encode_ordinary(text)
    if not token_ids:
        return [0]
    decoded, starts = enc.decode_with_offsets(token_ids)
    if decoded != text:
        raise ValueError(
            f"tiktoken round-trip mismatch under {encoding!r}; "
            "encode/decode did not reconstruct the source text"
        )
    return [*starts, len(text)]


def token_windows(
    text: str,
    base_offset: int,
    chunk_tokens: int,
    overlap_tokens: int = 0,
    encoding: str = DEFAULT_ENCODING,
) -> list[tuple[int, int]]:
    """Absolute ``(char_start, char_end)`` spans sliding a ``chunk_tokens``
    token window over ``text`` (a slice of some larger document starting at
    ``base_offset``), stepping by ``chunk_tokens - overlap_tokens`` tokens.

    The last window may be shorter than ``chunk_tokens`` when the text runs
    out; every other window is exactly full. Windows always advance (stride
    is at least 1 token) provided ``overlap_tokens < chunk_tokens``.
    """
    if chunk_tokens <= 0:
        raise ValueError(f"chunk_tokens must be > 0, got {chunk_tokens}")
    if overlap_tokens >= chunk_tokens:
        raise ValueError(
            f"overlap_tokens ({overlap_tokens}) must be < chunk_tokens ({chunk_tokens})"
        )

    offsets = token_char_offsets(text, encoding)
    n_tokens = len(offsets) - 1
    if n_tokens == 0:
        return []

    stride = chunk_tokens - overlap_tokens
    spans: list[tuple[int, int]] = []
    start_tok = 0
    while start_tok < n_tokens:
        end_tok = min(start_tok + chunk_tokens, n_tokens)
        spans.append((base_offset + offsets[start_tok], base_offset + offsets[end_tok]))
        if end_tok >= n_tokens:
            break
        start_tok += stride
    return spans


def finalize_chunks(
    specs: list[ChunkSpec],
    doc: Document,
    chunker: str,
    params: dict[str, Any],
    *,
    encoding: str = DEFAULT_ENCODING,
) -> list[Chunk]:
    """Turn boundary decisions into validated, identity-bearing ``Chunk``
    records, in the order given (``ordinal`` is assigned by position).

    Every ``chunk_set_id`` and ``chunk_id`` in the system is a pure function
    of (corpus, chunker, params) and (doc_id, span, signature) respectively —
    see ``ids.py`` — so calling this twice with identical inputs reproduces
    byte-identical output. That determinism is the whole caching strategy.
    """
    chunk_set_id = make_chunk_set_id(doc.corpus, chunker, params)
    signature = chunker_signature(chunker, params)

    chunks: list[Chunk] = []
    for ordinal, spec in enumerate(specs):
        expected = doc.text[spec.char_start : spec.char_end]
        if expected != spec.text:
            raise ValueError(
                f"offset invariant violated by {chunker!r} on doc {doc.doc_id} "
                f"[{spec.char_start}:{spec.char_end}]: text does not match the document slice"
            )
        chunks.append(
            Chunk(
                chunk_id=make_chunk_id(doc.doc_id, spec.char_start, spec.char_end, signature),
                doc_id=doc.doc_id,
                corpus=doc.corpus,
                chunk_set_id=chunk_set_id,
                text=spec.text,
                embed_text=spec.embed_text if spec.embed_text is not None else spec.text,
                char_start=spec.char_start,
                char_end=spec.char_end,
                token_count=count_tokens(spec.text, encoding),
                ordinal=ordinal,
                parent_id=spec.parent_id,
                role=spec.role,
                heading_path=list(spec.heading_path),
                chunker=chunker,
                chunker_params=dict(params),
                meta=dict(spec.meta),
            )
        )
    return chunks


__all__ = [
    "DEFAULT_ENCODING",
    "ChunkSpec",
    "Chunker",
    "count_tokens",
    "finalize_chunks",
    "token_char_offsets",
    "token_windows",
]
