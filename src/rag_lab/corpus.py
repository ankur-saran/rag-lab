"""Corpus statistics, document lookup, and stats rendering.

This is a read side only — it never writes artifacts. ``list_documents_by_corpus``
and ``find_document`` both go through ``paths.resolve_artifact``, so ``corpus
stats``/``corpus show`` get the fixture-fallback-with-warning behavior for free
and never need their own copy of that logic.
"""

from __future__ import annotations

import functools
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from rag_lab.jsonl import iter_jsonl, read_jsonl
from rag_lab.markup import count_code_blocks, count_tables, find_headings
from rag_lab.paths import artifacts_dir, resolve_artifact
from rag_lab.schemas import Document

# Matches Phase 2's `fixed` chunker default (cl100k_base), so a corpus's token
# stats and its later chunk counts are directly comparable. Hardcoded rather
# than `tiktoken.encoding_for_model` — the latter is a moving target across
# model names and library versions.
DEFAULT_ENCODING = "cl100k_base"


# --------------------------------------------------------------------------- #
# Document lookup — real artifacts first, fixture fallback via paths.py
# --------------------------------------------------------------------------- #


def _real_document_files() -> list[Path]:
    """Sorted per-corpus artifact files under artifacts/documents/, if any have
    been built yet. Empty on a clean checkout."""
    docs_dir = artifacts_dir() / "documents"
    if not docs_dir.exists():
        return []
    return sorted(docs_dir.glob("*.jsonl"))


def list_documents_by_corpus(corpus: str | None = None) -> dict[str, list[Document]]:
    """Group documents by corpus.

    ``corpus`` given: resolves that corpus's real artifact, or the shared
    fixture with its existing warning, and filters to matching records.

    ``corpus`` omitted: reads every real per-corpus file if any exist; only
    when none have been built does it fall back to the shared fixture, grouped
    by whatever corpora that fixture happens to contain.

    Raises ``LookupError`` when nothing at all is available for the request.
    """
    if corpus is not None:
        path = resolve_artifact("documents", corpus)
        docs = [d for d in read_jsonl(path, Document) if d.corpus == corpus]
        if not docs:
            raise LookupError(f"no documents found for corpus {corpus!r} (checked {path})")
        return {corpus: docs}

    real_files = _real_document_files()
    by_corpus: dict[str, list[Document]] = {}
    if real_files:
        for file in real_files:
            for doc in read_jsonl(file, Document):
                by_corpus.setdefault(doc.corpus, []).append(doc)
        return by_corpus

    path = resolve_artifact("documents", None)
    for doc in read_jsonl(path, Document):
        by_corpus.setdefault(doc.corpus, []).append(doc)
    if not by_corpus:
        raise LookupError("no documents available — build a corpus or provide fixtures")
    return by_corpus


def find_document(doc_id: str) -> Document | None:
    """Look up one document by id across real artifacts, or the fixture when
    nothing has been built yet."""
    real_files = _real_document_files()
    files = real_files if real_files else [resolve_artifact("documents", None)]
    for file in files:
        for doc in iter_jsonl(file, Document):
            if doc.doc_id == doc_id:
                return doc
    return None


# --------------------------------------------------------------------------- #
# Language-mix heuristic
# --------------------------------------------------------------------------- #

# Unicode-script / distinctive-marker detection, first match wins. This is
# script detection, not language identification — it cannot distinguish, say,
# Portuguese from Spanish in the general case, and it leans on markers chosen
# to work for *this project's own authored content* (catalog corpus) rather
# than arbitrary text. A real detector (langdetect, fastText lid.176) would be
# more accurate but is a new hard dependency, deliberately not added here.
_SCRIPT_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("ja", re.compile(r"[぀-ヿㇰ-ㇿ]")),
    ("zh", re.compile(r"[一-鿿]")),
    ("ko", re.compile(r"[가-힣]")),
    ("ru", re.compile(r"[Ѐ-ӿ]")),
    ("ar", re.compile(r"[؀-ۿ]")),
    ("hi", re.compile(r"[ऀ-ॿ]")),
    ("de", re.compile(r"[ß]|[äöüÄÖÜ]")),
    ("es", re.compile(r"[ñÑ¿¡]")),
    ("fr", re.compile(r"[œŒ]|[çàâêîïôûéèÇÀÂÊÎÏÔÛÉÈ]")),
]


def classify_script(text: str) -> str:
    """Best-effort script/language label for a document; falls through to
    ``"en"`` when no distinctive marker is found."""
    for label, pattern in _SCRIPT_PATTERNS:
        if pattern.search(text):
            return label
    return "en"


# --------------------------------------------------------------------------- #
# Stats
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CorpusStats:
    corpus: str
    doc_count: int
    total_tokens: int
    token_min: int
    token_p50: float
    token_p95: float
    token_max: int
    heading_count: int
    headings_per_1k_tokens: float
    code_block_count: int
    table_count: int
    lang_mix: dict[str, int] = field(default_factory=dict)


@functools.lru_cache(maxsize=1)
def _encoding():
    import tiktoken

    return tiktoken.get_encoding(DEFAULT_ENCODING)


def compute_corpus_stats(corpus: str, docs: list[Document]) -> CorpusStats:
    """Pure function over already-loaded documents — no I/O, no fixture
    fallback logic. Callers resolve documents first via ``list_documents_by_corpus``.
    """
    if not docs:
        return CorpusStats(
            corpus=corpus,
            doc_count=0,
            total_tokens=0,
            token_min=0,
            token_p50=0.0,
            token_p95=0.0,
            token_max=0,
            heading_count=0,
            headings_per_1k_tokens=0.0,
            code_block_count=0,
            table_count=0,
        )

    encoding = _encoding()
    token_counts = [len(encoding.encode(doc.text)) for doc in docs]
    heading_count = sum(len(find_headings(doc.text)) for doc in docs)
    code_block_count = sum(count_code_blocks(doc.text) for doc in docs)
    table_count = sum(count_tables(doc.text) for doc in docs)

    lang_mix: dict[str, int] = {}
    for doc in docs:
        label = classify_script(doc.text)
        lang_mix[label] = lang_mix.get(label, 0) + 1

    total_tokens = sum(token_counts)
    return CorpusStats(
        corpus=corpus,
        doc_count=len(docs),
        total_tokens=total_tokens,
        token_min=min(token_counts),
        token_p50=float(np.percentile(token_counts, 50)),
        token_p95=float(np.percentile(token_counts, 95)),
        token_max=max(token_counts),
        heading_count=heading_count,
        headings_per_1k_tokens=(heading_count / total_tokens * 1000) if total_tokens else 0.0,
        code_block_count=code_block_count,
        table_count=table_count,
        lang_mix=lang_mix,
    )


def render_stats_table(stats: list[CorpusStats], console: Console) -> None:
    table = Table(title="rag-lab corpus stats", title_style="bold")
    table.add_column("corpus", style="cyan", no_wrap=True)
    table.add_column("docs", justify="right")
    table.add_column("tokens (total)", justify="right")
    table.add_column("tokens min/p50/p95/max", overflow="fold")
    table.add_column("headings/1k tok", justify="right")
    table.add_column("code blocks", justify="right")
    table.add_column("tables", justify="right")
    table.add_column("lang mix", overflow="fold")

    for s in stats:
        lang = ", ".join(
            f"{label}:{count}"
            for label, count in sorted(s.lang_mix.items(), key=lambda kv: (-kv[1], kv[0]))
        )
        table.add_row(
            escape(s.corpus),
            str(s.doc_count),
            str(s.total_tokens),
            f"{s.token_min}/{s.token_p50:.0f}/{s.token_p95:.0f}/{s.token_max}",
            f"{s.headings_per_1k_tokens:.1f}",
            str(s.code_block_count),
            str(s.table_count),
            escape(lang),
        )
    console.print(table)


__all__ = [
    "DEFAULT_ENCODING",
    "CorpusStats",
    "classify_script",
    "compute_corpus_stats",
    "find_document",
    "list_documents_by_corpus",
    "render_stats_table",
]
