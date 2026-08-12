"""Chunk-set stats, lookup, and terminal rendering (`chunk stats`/`show`/`diff`).

Read side only, mirroring ``corpus.py``: it never writes artifacts, and it
goes through ``paths.resolve_artifact`` so fixture fallback (with its warning)
comes for free. The "split-X" stats are Phase 2's whole reason for existing
standalone — they're objective, zero-embedding evidence that structure-aware
chunking is doing something, computed against ground truth (``markup.py``'s
fence/table detection, ``sentence_window``'s sentence splitter) that is
itself chunker-agnostic, so the same stats apply uniformly to any chunk set.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from rich.console import Console
from rich.markup import escape
from rich.rule import Rule
from rich.table import Table

from rag_lab.chunkers.sentence_window import sentence_spans
from rag_lab.jsonl import read_jsonl
from rag_lab.markup import CODE_FENCE_RE, find_tables
from rag_lab.paths import artifacts_dir, resolve_artifact
from rag_lab.schemas import Chunk, Document

ORPHAN_TOKEN_THRESHOLD = 32


# --------------------------------------------------------------------------- #
# Lookup
# --------------------------------------------------------------------------- #


def available_chunk_set_ids() -> list[str]:
    chunks_dir = artifacts_dir() / "chunks"
    if not chunks_dir.exists():
        return []
    return sorted(p.stem for p in chunks_dir.glob("*.jsonl"))


def load_chunk_set(chunk_set_id: str) -> list[Chunk]:
    path = resolve_artifact("chunks", chunk_set_id)
    chunks = read_jsonl(path, Chunk)
    if not chunks:
        raise LookupError(f"chunk set {chunk_set_id!r} is empty (checked {path})")
    return chunks


def documents_for_chunks(chunks: list[Chunk]) -> dict[str, Document]:
    """The source documents for a chunk set's corpus, keyed by ``doc_id``. A
    chunk set is always single-corpus (it's embedded in ``chunk_set_id``), so
    one document artifact covers every chunk in the set."""
    corpus = chunks[0].corpus
    path = resolve_artifact("documents", corpus)
    return {d.doc_id: d for d in read_jsonl(path, Document) if d.corpus == corpus}


def find_document_in(docs_by_id: dict[str, Document], doc_id: str) -> Document:
    doc = docs_by_id.get(doc_id)
    if doc is None:
        raise LookupError(f"doc_id {doc_id!r} not found among this chunk set's documents")
    return doc


# --------------------------------------------------------------------------- #
# Stats
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ChunkSetStats:
    chunk_set_id: str
    corpus: str
    chunker: str
    count: int
    token_min: int
    token_p50: float
    token_p95: float
    token_max: int
    split_code_block_count: int
    split_table_count: int
    split_sentence_rate: float
    mean_heading_path_depth: float
    orphan_rate: float
    doc_count: int = 0
    fence_total: int = 0
    table_total: int = 0


def _boundaries_by_doc(chunks: list[Chunk]) -> dict[str, set[int]]:
    by_doc: dict[str, set[int]] = {}
    for c in chunks:
        by_doc.setdefault(c.doc_id, set()).update((c.char_start, c.char_end))
    return by_doc


def _count_split_regions(
    boundaries_by_doc: dict[str, set[int]],
    docs_by_id: dict[str, Document],
    finder,
) -> tuple[int, int]:
    """(number of regions split by a chunk boundary, total regions found),
    where a "region" is whatever ``finder(doc.text)`` returns as
    ``(start, end)`` spans -- fences or tables. A region counts as split when
    some chunk boundary lands strictly inside it."""
    split = 0
    total = 0
    for doc_id, boundaries in boundaries_by_doc.items():
        doc = docs_by_id.get(doc_id)
        if doc is None:
            continue
        for start, end in finder(doc.text):
            total += 1
            if any(start < b < end for b in boundaries):
                split += 1
    return split, total


def compute_chunk_stats(chunks: list[Chunk], docs_by_id: dict[str, Document]) -> ChunkSetStats:
    if not chunks:
        raise ValueError("cannot compute stats for an empty chunk list")

    token_counts = [c.token_count for c in chunks]
    boundaries_by_doc = _boundaries_by_doc(chunks)

    def _fence_spans(t: str) -> list[tuple[int, int]]:
        return [(m.start(), m.end()) for m in CODE_FENCE_RE.finditer(t)]

    split_fences, total_fences = _count_split_regions(boundaries_by_doc, docs_by_id, _fence_spans)
    split_tables, total_tables = _count_split_regions(boundaries_by_doc, docs_by_id, find_tables)

    sentence_split_flags = []
    sentence_cache: dict[str, list[tuple[int, int]]] = {}
    for c in chunks:
        doc = docs_by_id.get(c.doc_id)
        if doc is None:
            continue
        spans = sentence_cache.setdefault(c.doc_id, sentence_spans(doc.text))
        misaligned = any((ss < c.char_start < se) or (ss < c.char_end < se) for ss, se in spans)
        sentence_split_flags.append(misaligned)

    orphan_count = sum(1 for c in chunks if c.token_count < ORPHAN_TOKEN_THRESHOLD)

    return ChunkSetStats(
        chunk_set_id=chunks[0].chunk_set_id,
        corpus=chunks[0].corpus,
        chunker=chunks[0].chunker,
        count=len(chunks),
        token_min=min(token_counts),
        token_p50=float(np.percentile(token_counts, 50)),
        token_p95=float(np.percentile(token_counts, 95)),
        token_max=max(token_counts),
        split_code_block_count=split_fences,
        split_table_count=split_tables,
        split_sentence_rate=(sum(sentence_split_flags) / len(sentence_split_flags))
        if sentence_split_flags
        else 0.0,
        mean_heading_path_depth=sum(len(c.heading_path) for c in chunks) / len(chunks),
        orphan_rate=orphan_count / len(chunks),
        doc_count=len(boundaries_by_doc),
        fence_total=total_fences,
        table_total=total_tables,
    )


def render_chunk_stats_table(stats: ChunkSetStats, console: Console) -> None:
    table = Table(title=f"chunk stats — {stats.chunk_set_id}", title_style="bold")
    table.add_column("metric")
    table.add_column("value", justify="right")

    rows = [
        ("corpus", stats.corpus),
        ("chunker", stats.chunker),
        ("documents", str(stats.doc_count)),
        ("chunks", str(stats.count)),
        (
            "tokens min/p50/p95/max",
            f"{stats.token_min}/{stats.token_p50:.0f}/{stats.token_p95:.0f}/{stats.token_max}",
        ),
        ("split code blocks", f"{stats.split_code_block_count} / {stats.fence_total}"),
        ("split tables", f"{stats.split_table_count} / {stats.table_total}"),
        ("split-sentence rate", f"{stats.split_sentence_rate:.1%}"),
        ("mean heading-path depth", f"{stats.mean_heading_path_depth:.2f}"),
        (f"orphan rate (< {ORPHAN_TOKEN_THRESHOLD} tok)", f"{stats.orphan_rate:.1%}"),
    ]
    for name, value in rows:
        table.add_row(name, value)
    console.print(table)


# --------------------------------------------------------------------------- #
# `chunk show` — boundaries as rules in the terminal
# --------------------------------------------------------------------------- #


def render_chunk_boundaries(doc: Document, chunks: list[Chunk], console: Console) -> None:
    ordered = sorted(chunks, key=lambda c: (c.char_start, c.char_end))
    console.print(f"[bold]{escape(doc.title)}[/bold] ({doc.doc_id}) — {len(ordered)} chunks")
    prev_end: int | None = None
    for c in ordered:
        overlap = prev_end is not None and c.char_start < prev_end
        label = f"chunk {c.ordinal} [{c.char_start}:{c.char_end}] {c.token_count} tok" + (
            "  [yellow]«overlaps previous»[/yellow]" if overlap else ""
        )
        console.print(Rule(label, style="cyan"))
        console.print(escape(c.text))
        prev_end = c.char_end
    console.print(Rule(style="cyan"))


# --------------------------------------------------------------------------- #
# `chunk diff` — two chunkers' boundaries on the same document, side by side
# --------------------------------------------------------------------------- #


def _preview(text: str, width: int = 60) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= width else flat[: width - 1] + "…"


def _column_lines(chunks: list[Chunk]) -> list[str]:
    lines = []
    for c in sorted(chunks, key=lambda c: (c.char_start, c.char_end)):
        lines.append(
            f"[bold]#{c.ordinal}[/bold] [{c.char_start}:{c.char_end}] "
            f"({c.token_count} tok)\n{escape(_preview(c.text))}"
        )
    return lines


def render_chunk_diff(
    doc: Document,
    chunks_a: list[Chunk],
    chunks_b: list[Chunk],
    console: Console,
    label_a: str,
    label_b: str,
) -> None:
    console.print(f"[bold]{escape(doc.title)}[/bold] ({doc.doc_id})")
    table = Table(show_lines=True, expand=True)
    table.add_column(escape(label_a), ratio=1, overflow="fold")
    table.add_column(escape(label_b), ratio=1, overflow="fold")

    lines_a = _column_lines(chunks_a)
    lines_b = _column_lines(chunks_b)
    for i in range(max(len(lines_a), len(lines_b))):
        table.add_row(
            lines_a[i] if i < len(lines_a) else "",
            lines_b[i] if i < len(lines_b) else "",
        )
    console.print(table)


__all__ = [
    "ORPHAN_TOKEN_THRESHOLD",
    "ChunkSetStats",
    "available_chunk_set_ids",
    "compute_chunk_stats",
    "documents_for_chunks",
    "find_document_in",
    "load_chunk_set",
    "render_chunk_boundaries",
    "render_chunk_diff",
    "render_chunk_stats_table",
]
