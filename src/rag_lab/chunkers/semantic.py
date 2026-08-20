"""`semantic` -- embedding-based topic-shift chunker (Phase 7, Step 7.1).

Splits into sentences (reusing ``sentence_window.sentence_spans`` rather than
reimplementing sentence splitting), embeds each sentence together with
``buffer_size`` neighbouring sentences of context (context is used only to
compute inter-sentence distance -- never stored in ``text``/``embed_text``),
then cuts wherever the cosine distance between consecutive sentence
embeddings exceeds the ``breakpoint_percentile`` of the whole document's
distance distribution. Percentile-based thresholding, rather than an
absolute cosine-distance cutoff, is what makes this generalize across
corpora with different embedding-distance scales.

``min_tokens``/``max_tokens`` are enforced in the same two-phase order
``markdown_chunker.py`` already established for exactly this kind of bound:
merge undersized runs forward first, then split anything still oversized
(delegated to ``recursive.split_text_recursive``, the same fallback
``markdown_chunker.py`` uses for its own oversized sections).

Embedding a document's sentences is cheap to do once per ``chunk()`` call:
``embedders.build_embedder``'s underlying model load is process-wide
memoized (``sentence_transformer.py``'s ``_model()``), so per-document
construction here pays only the real encode cost, never a reload.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from rag_lab.chunkers.base import ChunkSpec, count_tokens, finalize_chunks
from rag_lab.chunkers.recursive import split_text_recursive
from rag_lab.chunkers.sentence_window import sentence_spans
from rag_lab.embedders import build_embedder
from rag_lab.schemas import Chunk, Document

NAME = "semantic"
DEFAULT_PARAMS: dict[str, Any] = {
    "breakpoint_percentile": 95,
    "buffer_size": 1,
    "min_tokens": 128,
    "max_tokens": 1024,
    "embedder": "bge-small",
    "encoding": "cl100k_base",
}

_FALLBACK_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def _context_text(
    text: str, sentences: list[tuple[int, int]], i: int, buffer_size: int
) -> str:
    """Sentence ``i`` concatenated with up to ``buffer_size`` neighbours on
    each side -- used only to compute a more stable embedding for breakpoint
    detection. Never stored in any chunk's ``text``/``embed_text``."""
    lo = max(0, i - buffer_size)
    hi = min(len(sentences) - 1, i + buffer_size)
    return text[sentences[lo][0] : sentences[hi][1]]


def _cosine_distances(vectors: np.ndarray) -> np.ndarray:
    """Cosine distance (``1 - cosine_similarity``) between each consecutive
    pair of rows. Vectors from ``embed_documents`` are already L2-normalized
    by convention, but the norm division here makes this correct regardless."""
    a, b = vectors[:-1], vectors[1:]
    sims = np.sum(a * b, axis=1)
    norms = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    norms = np.where(norms == 0, 1.0, norms)
    return 1.0 - (sims / norms)


def _breakpoints_after(distances: np.ndarray, percentile: float) -> set[int]:
    """Sentence indices ``i`` after which to cut, i.e. the gap between
    sentence ``i`` and ``i + 1`` exceeds the percentile threshold of the
    whole document's distance distribution."""
    if len(distances) == 0:
        return set()
    threshold = float(np.percentile(distances, percentile))
    return {i for i, d in enumerate(distances) if d > threshold}


def _sentence_runs(n: int, cut_after: set[int]) -> list[tuple[int, int]]:
    """Partition ``range(n)`` into contiguous ``[start, end)`` runs, cutting
    right after every index in ``cut_after``."""
    runs: list[tuple[int, int]] = []
    run_start = 0
    for i in range(n - 1):
        if i in cut_after:
            runs.append((run_start, i + 1))
            run_start = i + 1
    runs.append((run_start, n))
    return runs


def _merge_undersized(
    runs: list[tuple[int, int]], token_counts: list[int], min_tokens: int
) -> list[tuple[int, int]]:
    """Fold a run under ``min_tokens`` forward into the next one -- the same
    convention ``markdown_chunker.py``'s ``_merge_undersized`` uses."""
    if not runs:
        return runs
    merged = [list(runs[0])]
    for start, end in runs[1:]:
        prev = merged[-1]
        prev_tokens = sum(token_counts[prev[0] : prev[1]])
        if prev_tokens < min_tokens:
            prev[1] = end
        else:
            merged.append([start, end])
    return [(s, e) for s, e in merged]


class SemanticChunker:
    name = NAME
    default_params = DEFAULT_PARAMS

    def chunk(self, doc: Document, params: dict[str, Any]) -> list[Chunk]:
        resolved = {**DEFAULT_PARAMS, **params}
        breakpoint_percentile = float(resolved["breakpoint_percentile"])
        buffer_size = int(resolved["buffer_size"])
        min_tokens = int(resolved["min_tokens"])
        max_tokens = int(resolved["max_tokens"])
        embedder_name = str(resolved["embedder"])
        encoding = str(resolved["encoding"])

        started = time.monotonic()
        sentences = sentence_spans(doc.text)
        n = len(sentences)

        if n == 0:
            return finalize_chunks([], doc, NAME, resolved, encoding=encoding)

        if n == 1:
            start, end = sentences[0]
            specs = [
                ChunkSpec(
                    char_start=start,
                    char_end=end,
                    text=doc.text[start:end],
                    meta={"chunk_build_seconds": time.monotonic() - started},
                )
            ]
            return finalize_chunks(specs, doc, NAME, resolved, encoding=encoding)

        embedder = build_embedder(embedder_name, {})
        context_texts = [_context_text(doc.text, sentences, i, buffer_size) for i in range(n)]
        vectors = embedder.embed_documents(context_texts)
        distances = _cosine_distances(vectors)

        cut_after = _breakpoints_after(distances, breakpoint_percentile)
        runs = _sentence_runs(n, cut_after)

        token_counts = [count_tokens(doc.text[s:e], encoding) for s, e in sentences]
        runs = _merge_undersized(runs, token_counts, min_tokens)

        specs: list[ChunkSpec] = []
        for run_start, run_end in runs:
            char_start = sentences[run_start][0]
            char_end = sentences[run_end - 1][1]
            run_text = doc.text[char_start:char_end]
            if count_tokens(run_text, encoding) <= max_tokens:
                pieces = [(char_start, char_end)]
            else:
                pieces = (
                    split_text_recursive(
                        run_text,
                        char_start,
                        max_tokens,
                        0,
                        _FALLBACK_SEPARATORS,
                        lambda t: count_tokens(t, encoding),
                        encoding=encoding,
                    )
                    or [(char_start, char_end)]
                )
            for p_start, p_end in pieces:
                specs.append(
                    ChunkSpec(char_start=p_start, char_end=p_end, text=doc.text[p_start:p_end])
                )

        build_seconds = time.monotonic() - started
        for spec in specs:
            spec.meta["chunk_build_seconds"] = build_seconds

        return finalize_chunks(specs, doc, NAME, resolved, encoding=encoding)


__all__ = ["DEFAULT_PARAMS", "NAME", "SemanticChunker"]
