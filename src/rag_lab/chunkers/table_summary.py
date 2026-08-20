"""`table_summary` -- routes GFM pipe tables to their own chunk with an
LLM-generated summary folded into `embed_text`, and routes every other
character range through a fallback chunker's offset-aware sub-span helper
(Phase 7, Step 7.2).

Table detection reuses ``markup.find_tables`` (already exercised by
``chunk stats``'s split-table counter) rather than reimplementing it.

The fallback delegation is deliberately *not* ``split_text_recursive``'s
``protected_spans`` mechanism -- that keeps a span un-split but still
mergeable into a surrounding chunk (exactly what ``markdown_chunker.py`` does
with fenced code blocks), which is the opposite of what a table needs here:
always its own dedicated chunk. Instead every inter-table character range is
chunked independently, at its own absolute offset, and the results are
combined with one ``ChunkSpec`` per table before a single ``finalize_chunks``
call -- the same shape every other chunker already uses.

``mock_llm`` is a hashed chunker param (default ``False``), not a bypassed
runtime flag: a mock run and a real run produce materially different
``embed_text`` content (placeholder vs. real summary), so they must land at
different ``chunk_set_id``s -- otherwise a real ``chunk run`` after an
earlier ``--mock-llm`` run would silently return the cached mock output as if
it were a cache hit. This mirrors why ``role``/``parent_chunk_set_id`` are
hashed into ``chunk_set_id`` too (plan §4.7).
"""

from __future__ import annotations

from typing import Any

from rag_lab.chunkers.base import (
    ChunkSpec,
    count_tokens,
    finalize_chunks,
    token_char_offsets,
    token_windows,
)
from rag_lab.chunkers.recursive import split_text_recursive
from rag_lab.ids import sha1_hex
from rag_lab.markup import find_tables
from rag_lab.schemas import Chunk, Document

NAME = "table_summary"
DEFAULT_PARAMS: dict[str, Any] = {
    # "fixed" or "recursive" only, for v1 -- both already expose an
    # offset-aware sub-span helper (`token_windows`, `split_text_recursive`).
    # `markdown`/`sentence_window` are out of scope: chunking a table-excised
    # sub-span would break their heading-stack / sentence-continuity
    # assumptions across the removed gap.
    "fallback_chunker": "recursive",
    "fallback_chunk_tokens": 512,
    "model": "claude-haiku-4-5-20251001",
    "max_table_tokens": 2048,
    "summarize": True,
    "summary_max_tokens": 150,
    "mock_llm": False,
    "encoding": "cl100k_base",
}

_FALLBACK_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]
_SUPPORTED_FALLBACKS = ("fixed", "recursive")
_PROMPT_VERSION = "v1"

_SUMMARY_PROMPT = """You are summarizing a table extracted from a document so it \
can be found by semantic search. Describe, in prose, what the table reports, its \
dimensions (rows/columns), units, the time period it covers (if any), and the most \
notable magnitudes. Do not repeat the table verbatim. Keep it under {max_tokens} tokens.

Table:
{table_text}
"""


def _truncate_for_prompt(text: str, max_tokens: int, encoding: str) -> str:
    """Cap the table text sent to the LLM at ``max_tokens`` -- a bound on
    prompt size only. The emitted chunk's own ``text`` always stays the full,
    untruncated table (never violates the offset invariant); only what goes
    into the summarization prompt is capped, the same way a huge financial
    table shouldn't blow an LLM call's context budget."""
    if count_tokens(text, encoding) <= max_tokens:
        return text
    offsets = token_char_offsets(text, encoding)
    return text[: offsets[max_tokens]]


def _fallback_spans(
    fallback: str, text: str, base_offset: int, chunk_tokens: int, encoding: str
) -> list[tuple[int, int]]:
    """Offset-aware boundary spans for one inter-table character range,
    delegated to the named fallback chunker's low-level helper -- not its
    ``.chunk(doc, params)`` entry point, which only accepts a whole
    ``Document``, never a sub-span."""
    if not text.strip():
        return []
    if fallback == "fixed":
        return token_windows(text, base_offset, chunk_tokens, 0, encoding)
    if fallback == "recursive":
        return split_text_recursive(
            text,
            base_offset,
            chunk_tokens,
            0,
            _FALLBACK_SEPARATORS,
            lambda t: count_tokens(t, encoding),
            encoding=encoding,
        )
    raise ValueError(
        f"table_summary's fallback_chunker must be one of {_SUPPORTED_FALLBACKS}, "
        f"got {fallback!r}"
    )


def _summarize_table(
    table_text: str,
    model: str,
    max_table_tokens: int,
    summary_max_tokens: int,
    mock_llm: bool,
    encoding: str,
) -> str:
    from rag_lab.llm import cached_llm_call, call_llm

    prompt_table_text = _truncate_for_prompt(table_text, max_table_tokens, encoding)
    prompt = _SUMMARY_PROMPT.format(table_text=prompt_table_text, max_tokens=summary_max_tokens)

    def _compute() -> str:
        return call_llm(prompt, model=model, max_tokens=summary_max_tokens, mock=mock_llm)

    if mock_llm:
        # Free to recompute and must never collide with a real cache entry
        # keyed on the same table content -- see `llm.cached_llm_call`.
        return _compute()

    cache_key = sha1_hex(table_text, model, _PROMPT_VERSION)
    return cached_llm_call(cache_key, _compute)


class TableSummaryChunker:
    name = NAME
    default_params = DEFAULT_PARAMS

    def chunk(self, doc: Document, params: dict[str, Any]) -> list[Chunk]:
        resolved = {**DEFAULT_PARAMS, **params}
        fallback = str(resolved["fallback_chunker"])
        if fallback not in _SUPPORTED_FALLBACKS:
            raise ValueError(
                f"table_summary's fallback_chunker must be one of {_SUPPORTED_FALLBACKS}, "
                f"got {fallback!r}"
            )
        fallback_chunk_tokens = int(resolved["fallback_chunk_tokens"])
        model = str(resolved["model"])
        max_table_tokens = int(resolved["max_table_tokens"])
        summarize = bool(resolved["summarize"])
        summary_max_tokens = int(resolved["summary_max_tokens"])
        mock_llm = bool(resolved["mock_llm"])
        encoding = str(resolved["encoding"])

        text = doc.text
        table_spans = sorted(find_tables(text))

        specs: list[ChunkSpec] = []

        for table_start, table_end in table_spans:
            table_text = text[table_start:table_end]
            embed_text = table_text
            if summarize:
                summary = _summarize_table(
                    table_text, model, max_table_tokens, summary_max_tokens, mock_llm, encoding
                )
                embed_text = f"{summary}\n\n{table_text}"
            specs.append(
                ChunkSpec(
                    char_start=table_start,
                    char_end=table_end,
                    text=table_text,
                    embed_text=embed_text,
                    meta={"is_table": True},
                )
            )

        cursor = 0
        for table_start, table_end in table_spans:
            gap_start, gap_end = cursor, table_start
            if gap_end > gap_start:
                for p_start, p_end in _fallback_spans(
                    fallback, text[gap_start:gap_end], gap_start, fallback_chunk_tokens, encoding
                ):
                    specs.append(
                        ChunkSpec(char_start=p_start, char_end=p_end, text=text[p_start:p_end])
                    )
            cursor = table_end
        if cursor < len(text):
            for p_start, p_end in _fallback_spans(
                fallback, text[cursor:], cursor, fallback_chunk_tokens, encoding
            ):
                specs.append(
                    ChunkSpec(char_start=p_start, char_end=p_end, text=text[p_start:p_end])
                )

        specs.sort(key=lambda s: s.char_start)

        return finalize_chunks(specs, doc, NAME, resolved, encoding=encoding)


__all__ = ["DEFAULT_PARAMS", "NAME", "TableSummaryChunker"]
