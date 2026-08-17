"""Retriever run/compare orchestration (Phase 4), plus terminal rendering for
the `retrieve compare` CLI command. Mirrors ``indexing.py``: this is the one
place that knows how a named retriever gets built against an index and run,
so the CLI stays thin.

Single-retriever rendering (``retrieve query``) reuses
``indexing.render_search_results`` as-is -- it is already retriever-agnostic
(rank/score/chunk_id/text only), so no new function is needed for that case.
"""

from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.markup import escape
from rich.table import Table

from rag_lab.indexing import load_manifest_and_store
from rag_lab.retrievers import available_retrievers, build_retriever
from rag_lab.schemas import IndexManifest, ScoredChunk


def run_retriever(
    index_id: str,
    retriever_name: str,
    query: str,
    k: int,
    overrides: dict[str, Any] | None = None,
) -> tuple[IndexManifest, list[ScoredChunk]]:
    if retriever_name not in available_retrievers():
        raise ValueError(
            f"unknown retriever {retriever_name!r}; available: {available_retrievers()}"
        )
    manifest, store = load_manifest_and_store(index_id)
    retriever = build_retriever(retriever_name, overrides, manifest=manifest, store=store)
    return manifest, retriever.retrieve(query, k)


def compare_retrievers(
    index_id: str,
    retriever_names: list[str],
    query: str,
    k: int,
    overrides_by_name: dict[str, dict[str, Any]] | None = None,
) -> tuple[IndexManifest, dict[str, list[ScoredChunk]]]:
    unknown = [name for name in retriever_names if name not in available_retrievers()]
    if unknown:
        raise ValueError(f"unknown retriever(s) {unknown}; available: {available_retrievers()}")

    overrides_by_name = overrides_by_name or {}
    manifest, store = load_manifest_and_store(index_id)

    results_by_name: dict[str, list[ScoredChunk]] = {}
    for name in retriever_names:
        retriever = build_retriever(
            name, overrides_by_name.get(name), manifest=manifest, store=store
        )
        results_by_name[name] = retriever.retrieve(query, k)
    return manifest, results_by_name


# --------------------------------------------------------------------------- #
# Terminal rendering
# --------------------------------------------------------------------------- #


def render_compare_table(
    query: str,
    results_by_retriever: dict[str, list[ScoredChunk]],
    k: int,
    console: Console,
) -> None:
    table = Table(title=f"retrieve compare: {query!r}", title_style="bold")
    for name in results_by_retriever:
        table.add_column(escape(name), ratio=1, overflow="fold")

    def _cell(r: ScoredChunk) -> str:
        preview = " ".join(r.chunk.text.split())
        preview = preview if len(preview) <= 100 else preview[:99] + "…"
        return f"[bold]#{r.rank}[/bold] ({r.score:.4f})\n{escape(preview)}"

    columns = list(results_by_retriever.values())
    for i in range(k):
        table.add_row(*(_cell(col[i]) if i < len(col) else "" for col in columns))
    console.print(table)


__all__ = [
    "compare_retrievers",
    "render_compare_table",
    "run_retriever",
]
