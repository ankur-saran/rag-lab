"""Cached data-loading and discovery for the Streamlit app (plan §Phase 9).

No rendering here -- every function returns plain data (or raises
``LookupError``/``paths.ArtifactNotFoundError``, caught by ``app.ui.guarded``
at the page level, per AC-3). This module is what keeps
``report.load_run``/``worst_failures`` and ``agents.optimizer.load_trace``'s
one real gap -- both are hardcoded to ``artifacts_dir()`` with no fixture
fallback of their own, unlike ``paths.resolve_artifact`` -- from leaking into
every page: the real-then-fixture fallback lives here, once.

Discovery functions return real artifacts when any exist, and fall back to
the committed fixture (flagged ``is_fixture=True``) only when the real list is
empty -- this is what makes AC-1 ("streamlit run succeeds using fixtures
only, with an empty artifacts/") true.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st

from rag_lab.agents.optimizer import load_trace as _load_optimizer_trace
from rag_lab.chunks import documents_for_chunks, find_document_in, load_chunk_set
from rag_lab.corpus import list_documents_by_corpus
from rag_lab.experiment.config import Cell, ExperimentConfig, expand_cells
from rag_lab.experiment.report import HEADLINE_METRICS
from rag_lab.indexing import list_manifests, load_manifest, load_manifest_and_store
from rag_lab.jsonl import read_jsonl
from rag_lab.metrics import bootstrap_ci
from rag_lab.metrics.gold import flatten_gold, resolve_gold_per_span
from rag_lab.paths import (
    artifacts_dir,
    fixture_path,
    fixtures_dir,
    is_fixture,
    resolve_artifact,
)
from rag_lab.retrievers import available_retrievers
from rag_lab.schemas import Chunk, Document, EvalPair, IndexManifest, OptimizerTraceEntry, QueryTrace, RunResult

CACHE_TTL_S = 60  # short TTL: fresh enough for a live demo without hammering disk every rerun

# --------------------------------------------------------------------------- #
# Corpora / documents
# --------------------------------------------------------------------------- #


@st.cache_data(ttl=CACHE_TTL_S, show_spinner=False)
def list_corpora() -> tuple[list[str], bool]:
    """Corpus names only -- filenames or (fixture-only) a peek at the shared
    fixture's ``corpus`` field, never full document bodies (AC-4)."""
    docs_dir = artifacts_dir() / "documents"
    real = sorted(p.stem for p in docs_dir.glob("*.jsonl")) if docs_dir.exists() else []
    if real:
        return real, False
    fixture = fixture_path("documents")
    corpora = sorted({d.corpus for d in read_jsonl(fixture, Document)})
    return corpora, True


@st.cache_data(ttl=CACHE_TTL_S, show_spinner=False)
def load_documents(corpus: str) -> tuple[list[Document], bool]:
    docs = list_documents_by_corpus(corpus)[corpus]
    return docs, is_fixture(resolve_artifact("documents", corpus))


# --------------------------------------------------------------------------- #
# Chunk sets
# --------------------------------------------------------------------------- #

_FIXTURE_CHUNK_SET_NAMES = ("sample", "sample_parent", "sample_child")


@st.cache_data(ttl=CACHE_TTL_S, show_spinner=False)
def list_chunk_sets() -> tuple[list[str], bool]:
    """Chunk-set ids with a built artifact, or the fixture's chunk-set names
    when none have been built yet. There's no existing discovery helper for
    chunk sets (unlike ``indexing.list_manifests`` for indexes), so this globs
    ``artifacts/chunks/*.jsonl`` directly."""
    chunks_dir = artifacts_dir() / "chunks"
    real = sorted(p.stem for p in chunks_dir.glob("*.jsonl")) if chunks_dir.exists() else []
    if real:
        return real, False
    return list(_FIXTURE_CHUNK_SET_NAMES), True


def _fixture_chunk_set_path(chunk_set_id: str) -> Path:
    suffix = "" if chunk_set_id == "sample" else f"_{chunk_set_id.removeprefix('sample_')}"
    return fixtures_dir() / "chunks" / f"sample{suffix}.jsonl"


@st.cache_data(ttl=CACHE_TTL_S, show_spinner=False)
def load_chunks(chunk_set_id: str) -> tuple[list[Chunk], bool]:
    """``chunk_set_id`` may be a real id or one of the fixture's own names
    (``sample``/``sample_parent``/``sample_child``) -- ``chunks.load_chunk_set``
    already resolves the former through ``resolve_artifact``; the latter are
    read directly since they aren't real chunk-set ids at all, just the
    fixture files' own names."""
    if chunk_set_id in _FIXTURE_CHUNK_SET_NAMES and not (artifacts_dir() / "chunks" / f"{chunk_set_id}.jsonl").exists():
        return read_jsonl(_fixture_chunk_set_path(chunk_set_id), Chunk), True
    chunks = load_chunk_set(chunk_set_id)
    return chunks, is_fixture(resolve_artifact("chunks", chunk_set_id))


def document_for_chunk_set(chunks: list[Chunk], doc_id: str) -> Document:
    docs_by_id = documents_for_chunks(chunks)
    return find_document_in(docs_by_id, doc_id)


# --------------------------------------------------------------------------- #
# Indexes
# --------------------------------------------------------------------------- #


@st.cache_data(ttl=CACHE_TTL_S, show_spinner=False)
def list_indexes(corpus: str | None = None) -> tuple[list[IndexManifest], bool]:
    real = list_manifests(corpus)
    if real:
        return real, False
    fixture = load_manifest(fixture_path("indexes"))
    if corpus is not None and fixture.corpus != corpus:
        return [], True
    return [fixture], True


@st.cache_resource(ttl=CACHE_TTL_S, show_spinner=False)
def open_index(index_id: str | None):
    """``None`` triggers ``resolve_artifact``'s own real-then-fixture fallback
    inside ``indexing.load_manifest_and_store`` -- unlike run results and
    optimizer traces, index loading already goes through ``resolve_artifact``,
    so no local fallback is needed here. ``st.cache_resource`` (not
    ``cache_data``): this holds a live ``ChromaStore`` handle, which can't be
    pickled/deep-copied the way ``cache_data`` would try to."""
    manifest, store = load_manifest_and_store(index_id)
    return manifest, store, is_fixture(resolve_artifact("indexes", index_id))


# --------------------------------------------------------------------------- #
# Retrieval
# --------------------------------------------------------------------------- #


def list_retriever_names() -> list[str]:
    return available_retrievers()


@st.cache_data(ttl=CACHE_TTL_S, show_spinner=False)
def load_evalset(corpus: str) -> tuple[list[EvalPair], bool]:
    path = resolve_artifact("evalset", corpus)
    pairs = [p for p in read_jsonl(path, EvalPair) if p.corpus == corpus]
    return pairs, is_fixture(path)


@st.cache_data(ttl=CACHE_TTL_S, show_spinner=False)
def difficulty_by_query_id(corpus: str) -> dict[str, str]:
    pairs, _ = load_evalset(corpus)
    return {p.query_id: p.difficulty for p in pairs}


def gold_chunk_ids_for_query(query: str, corpus: str, chunk_set: list[Chunk]) -> set[str]:
    """Resolve an eval-set query's gold chunk ids against ``chunk_set``, for
    Step 9.2's gold-match badge. Returns an empty set when ``query`` doesn't
    match any pair in the corpus's eval set (i.e. it's free-text, not drawn
    from the eval set) -- exact match on ``EvalPair.query``, the same
    resolution the eval-set-sourced query picker in the UI guarantees."""
    pairs, _ = load_evalset(corpus)
    pair = next((p for p in pairs if p.query == query), None)
    if pair is None:
        return set()
    per_span = resolve_gold_per_span(pair.gold_char_spans, pair.gold_doc_id, chunk_set)
    return flatten_gold(per_span)


# --------------------------------------------------------------------------- #
# Run results (experiment/report.py hardcodes artifacts_dir() with no fixture
# fallback of its own -- the same real-then-fixture pattern applied to
# optimizer traces below is applied here too, by resolving a run directory
# locally rather than trying to make report.load_run accept one).
# --------------------------------------------------------------------------- #


@st.cache_data(ttl=CACHE_TTL_S, show_spinner=False)
def list_run_ids() -> tuple[list[str], bool]:
    root = artifacts_dir() / "results"
    real = sorted(
        (p.parent.name for p in root.glob("*/matrix.json")),
        key=lambda run_id: (root / run_id / "matrix.json").stat().st_mtime,
        reverse=True,
    ) if root.exists() else []
    if real:
        return real, False
    return ["sample_run"], True


def _run_dir_for(run_id: str) -> tuple[Path, bool]:
    real_dir = artifacts_dir() / "results" / run_id
    if (real_dir / "matrix.json").exists():
        return real_dir, False
    fixture_dir = fixtures_dir() / "results" / "sample_run"
    return fixture_dir, True


@st.cache_data(ttl=CACHE_TTL_S, show_spinner=False)
def load_run(run_id: str) -> tuple[ExperimentConfig, list[RunResult], bool]:
    run_dir, fixture = _run_dir_for(run_id)
    matrix_path = run_dir / "matrix.json"
    if not matrix_path.exists():
        raise LookupError(f"no experiment run found for run_id {run_id!r} (expected {matrix_path})")
    config = ExperimentConfig.model_validate_json(matrix_path.read_text(encoding="utf-8"))
    cells = expand_cells(config)
    results = []
    for cell in cells:
        path = run_dir / "cells" / f"{cell.cell_id}.json"
        if path.exists():
            results.append(RunResult.model_validate_json(path.read_text(encoding="utf-8")))
    if not results:
        raise LookupError(f"run {run_id!r} has no computed cells yet (checked {run_dir})")
    return config, results, fixture


def cells_for_run(config: ExperimentConfig) -> list[Cell]:
    return expand_cells(config)


def result_for_cell(results: list[RunResult], cell: Cell) -> RunResult | None:
    return next((r for r in results if r.config.get("cell_id") == cell.cell_id), None)


def metric_values(result: RunResult, metric: str) -> list[float]:
    return [t.metrics[metric] for t in result.per_query if not t.excluded and metric in t.metrics]


def metric_ci(result: RunResult, metric: str) -> tuple[float, float, float] | None:
    """``(mean, lo, hi)``, or ``None`` when the metric has no values for this
    result (e.g. a cell whose ``k`` never reached this recall depth)."""
    values = metric_values(result, metric)
    if not values:
        return None
    mean = sum(values) / len(values)
    lo, hi = bootstrap_ci(values)
    return mean, lo, hi


def headline_metrics() -> list[str]:
    return HEADLINE_METRICS


def worst_failures(
    result: RunResult, corpus: str, n: int | None = None
) -> list[tuple[QueryTrace, str]]:
    """``result.per_query`` sorted worst-first (same key as
    ``experiment.report.worst_failures``), each tagged with its difficulty
    tier via ``difficulty_by_query_id`` -- ``QueryTrace`` itself carries no
    ``difficulty`` field, only ``EvalPair`` does. ``n=None`` returns every
    query (not the CLI's default top-20) so a difficulty-tier filter applied
    afterward in the UI doesn't hide lower-ranked failures of the selected
    tier."""
    tiers = difficulty_by_query_id(corpus)

    def _sort_key(t: QueryTrace) -> tuple[int, int, float]:
        if t.excluded:
            return (2, 0, 0.0)
        return (0 if t.first_hit_rank is None else 1, 0, t.metrics.get("recall@5", 0.0))

    worst = sorted(result.per_query, key=_sort_key)
    if n is not None:
        worst = worst[:n]
    return [(t, tiers.get(t.query_id, "unknown")) for t in worst]


# --------------------------------------------------------------------------- #
# Optimizer trace (agents.optimizer.load_trace hardcodes artifacts_dir() too,
# with no fixture fallback at all -- see module docstring).
# --------------------------------------------------------------------------- #


@st.cache_data(ttl=CACHE_TTL_S, show_spinner=False)
def list_optimizer_run_ids() -> tuple[list[str], bool]:
    root = artifacts_dir() / "results"
    real = sorted(
        (p.parent.name for p in root.glob("*/optimizer_trace.jsonl")),
        key=lambda run_id: (root / run_id / "optimizer_trace.jsonl").stat().st_mtime,
        reverse=True,
    ) if root.exists() else []
    if real:
        return real, False
    return ["sample_optimizer_run"], True


@st.cache_data(ttl=CACHE_TTL_S, show_spinner=False)
def load_optimizer_trace(optimizer_run_id: str) -> tuple[list[OptimizerTraceEntry], bool]:
    try:
        return _load_optimizer_trace(optimizer_run_id), False
    except LookupError:
        fixture = fixtures_dir() / "results" / "sample_optimizer_run" / "optimizer_trace.jsonl"
        if not fixture.exists():
            raise LookupError(
                f"no optimizer trace found for run_id {optimizer_run_id!r}, "
                f"and no fixture available at {fixture}"
            ) from None
        return read_jsonl(fixture, OptimizerTraceEntry), True


__all__ = [
    "cells_for_run",
    "difficulty_by_query_id",
    "document_for_chunk_set",
    "gold_chunk_ids_for_query",
    "headline_metrics",
    "list_chunk_sets",
    "list_corpora",
    "list_indexes",
    "list_optimizer_run_ids",
    "list_retriever_names",
    "list_run_ids",
    "load_chunks",
    "load_documents",
    "load_evalset",
    "load_optimizer_trace",
    "load_run",
    "metric_ci",
    "metric_values",
    "open_index",
    "result_for_cell",
    "worst_failures",
]
