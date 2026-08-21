"""Experiment matrix runner (plan Phase 6, Step 6.4).

Two sequential stages, deliberately -- not naive per-cell parallelism:

1. **Build** (single-threaded). Every cell's ``chunk_set_id``/``index_id`` is
   already known from ``expand_cells`` (pure, no I/O), so this stage dedupes
   by id and builds each distinct chunk set / index exactly once, reusing
   ``chunkers.build_chunk_set``/``indexing.build_index``'s own caching. This
   is what avoids a race a naive "parallelize the whole cell" reading of
   Step 6.4's "parallel across cells... keep embedding batches serial" would
   hit: two workers whose cells share an index_id (very common -- every
   retriever variant over the same chunker+embedder shares one index) would
   otherwise both see a cache miss and both try to build the same Chroma
   directory.
2. **Evaluate** (``ThreadPoolExecutor``). Each cell now only does read-only
   retrieval against an already-built index, so cells run in parallel safely
   -- retrieval calls (torch, chromadb) release the GIL, so threads are the
   right tool, not processes.

Each cell's ``RunResult`` is written immediately to
``artifacts/results/<run_id>/cells/<cell_id>.json``. Resuming (rerunning with
the same explicit ``--run-id``) skips any cell whose file already exists
unless ``--force``. ``--force`` only affects that skip -- it never forces a
chunk-set/index rebuild, since those are content-addressed caches already
correct by construction; forcing them would be pure waste.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import structlog
from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn, TimeRemainingColumn

from rag_lab import indexing
from rag_lab.chunkers import build_chunk_set
from rag_lab.chunks import load_chunk_set
from rag_lab.config import params_hash
from rag_lab.evalset import load_evalset
from rag_lab.experiment.config import Cell, ExperimentConfig, expand_cells
from rag_lab.ids import make_run_id
from rag_lab.metrics import (
    all_spans_hit,
    chunk_efficiency,
    flatten_gold,
    latency_percentiles,
    mrr,
    ndcg_at_k,
    recall_at_k,
    resolve_gold_per_span,
)
from rag_lab.paths import artifacts_dir
from rag_lab.retrievers import build_retriever
from rag_lab.schemas import QueryTrace, RunResult

log = structlog.get_logger(__name__)

# The headline recall depths (plan Step 6.1). Only the ones <= a cell's own k
# are ever computed for that cell -- you can't report recall@10 from a cell
# that only ever retrieved 5.
RECALL_KS = (1, 3, 5, 10)


# --------------------------------------------------------------------------- #
# Run-directory layout -- shared with report.py
# --------------------------------------------------------------------------- #


def run_dir(run_id: str) -> Path:
    return artifacts_dir() / "results" / run_id


def cell_path(run_id: str, cell_id: str) -> Path:
    return run_dir(run_id) / "cells" / f"{cell_id}.json"


def write_matrix_manifest(run_id: str, config: ExperimentConfig) -> None:
    path = run_dir(run_id) / "matrix.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(config.model_dump_json(indent=2), encoding="utf-8")


def load_matrix_manifest(run_id: str) -> ExperimentConfig:
    path = run_dir(run_id) / "matrix.json"
    if not path.exists():
        raise LookupError(f"no experiment run found for run_id {run_id!r} (expected {path})")
    return ExperimentConfig.model_validate_json(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# Stage 1 -- build
# --------------------------------------------------------------------------- #


def _build_stage(cells: list[Cell], console: Console | None) -> None:
    """Build (or reuse) every distinct chunk set and index the matrix needs,
    each exactly once, before any evaluation happens.

    Also warms each distinct embedder's underlying model synchronously here
    on a cache hit. A *fresh* index build already loads the model as a side
    effect of embedding the chunk set, but a cache hit skips embedding
    entirely -- and without this, the model's first-ever load would happen
    lazily inside stage 2's ``ThreadPoolExecutor``, the first time two
    threads' retrievers call it concurrently.
    ``sentence_transformers.SentenceTransformerEmbedder``'s model cache
    (``functools.lru_cache``) is not safe against that race: two threads
    racing the same cache miss can both start constructing the underlying
    torch model at once and crash with a meta-tensor error. Warming here
    also simply matches this stage's own contract -- model loading is setup
    cost, and stage 2 is supposed to be pure read-only retrieval.
    """
    seen_chunk_sets: set[str] = set()
    seen_indexes: set[str] = set()
    warmed_embedders: set[tuple[str, str]] = set()
    for cell in cells:
        if cell.chunk_set_id not in seen_chunk_sets:
            build_chunk_set(cell.corpus, cell.chunker, cell.chunker_params, force=False)
            seen_chunk_sets.add(cell.chunk_set_id)
        if cell.index_id not in seen_indexes:
            _manifest, cache_hit = indexing.build_index(
                cell.chunk_set_id, cell.embedder, cell.embedder_params, force=False
            )
            seen_indexes.add(cell.index_id)
            embedder_key = (cell.embedder, params_hash(cell.embedder_params))
            if cache_hit and embedder_key not in warmed_embedders:
                indexing.embedder_for(cell.embedder, cell.embedder_params).embed_query("warmup")
                warmed_embedders.add(embedder_key)
    if console:
        console.print(
            f"[green]build[/green]  {len(seen_chunk_sets)} chunk set(s), "
            f"{len(seen_indexes)} index(es)"
        )


# --------------------------------------------------------------------------- #
# Stage 2 -- evaluate
# --------------------------------------------------------------------------- #


def _first_hit_rank(retrieved_ids: list[str], gold_ids: set[str]) -> int | None:
    for rank, cid in enumerate(retrieved_ids, start=1):
        if cid in gold_ids:
            return rank
    return None


def _evaluate_cell(cell: Cell, run_id: str) -> RunResult:
    manifest, store = indexing.load_manifest_and_store(cell.index_id)
    retriever = build_retriever(
        cell.retriever, cell.retriever_params, manifest=manifest, store=store
    )
    chunk_set = load_chunk_set(cell.chunk_set_id)
    pairs = load_evalset(cell.corpus, split=cell.eval_split)

    traces: list[QueryTrace] = []
    latencies: list[float] = []
    per_metric: dict[str, list[float]] = {f"recall@{kk}": [] for kk in RECALL_KS if kk <= cell.k}
    per_metric["mrr"] = []
    per_metric["ndcg@10"] = []
    chunk_eff_values: list[float] = []
    cross_ref_values: list[float] = []

    for pair in pairs:
        per_span = resolve_gold_per_span(pair.gold_char_spans, pair.gold_doc_id, chunk_set)
        gold_ids = flatten_gold(per_span)

        if not gold_ids:
            # Step 6.2: a pair that resolves to zero gold chunks *for this
            # chunk set* must be excluded, not silently scored as a miss.
            traces.append(
                QueryTrace(
                    query_id=pair.query_id,
                    query=pair.query,
                    retrieved_chunk_ids=[],
                    gold_chunk_ids=[],
                    excluded=True,
                    exclusion_reason="gold resolved to zero chunks for this chunk set",
                )
            )
            continue

        started = time.perf_counter()
        results = retriever.retrieve(pair.query, cell.k)
        latency_ms = (time.perf_counter() - started) * 1000.0
        latencies.append(latency_ms)

        retrieved_ids = [r.chunk.chunk_id for r in results]
        retrieved_chunks = [r.chunk for r in results]

        metrics: dict[str, float] = {}
        for kk in RECALL_KS:
            if kk <= cell.k:
                val = recall_at_k(retrieved_ids, gold_ids, kk)
                metrics[f"recall@{kk}"] = val
                per_metric[f"recall@{kk}"].append(val)

        metrics["mrr"] = mrr(retrieved_ids, gold_ids)
        per_metric["mrr"].append(metrics["mrr"])

        metrics["ndcg@10"] = ndcg_at_k(retrieved_ids, gold_ids, min(10, cell.k))
        per_metric["ndcg@10"].append(metrics["ndcg@10"])

        hit = recall_at_k(retrieved_ids, gold_ids, cell.k) == 1.0
        eff = chunk_efficiency(retrieved_chunks, cell.k, hit)
        if eff is not None:
            metrics["chunk_efficiency"] = eff
            chunk_eff_values.append(eff)

        # Secondary multi-hop diagnostic (design decision, plan §6.1's open
        # question): only meaningful for genuinely multi-span pairs.
        if len(per_span) > 1:
            all_hit = all_spans_hit(retrieved_ids, per_span, cell.k)
            metrics["cross_ref_all_spans_hit"] = 1.0 if all_hit else 0.0
            cross_ref_values.append(metrics["cross_ref_all_spans_hit"])

        traces.append(
            QueryTrace(
                query_id=pair.query_id,
                query=pair.query,
                retrieved_chunk_ids=retrieved_ids,
                retrieved_scores=[r.score for r in results],
                gold_chunk_ids=sorted(gold_ids),
                first_hit_rank=_first_hit_rank(retrieved_ids, gold_ids),
                metrics=metrics,
                latency_ms=latency_ms,
            )
        )

    if pairs and all(t.excluded for t in traces):
        # Every pair resolved to zero gold chunks against this chunk set --
        # almost always a doc_id mismatch between where the chunk set's
        # documents came from and where the eval set came from (e.g. a real
        # corpus built alongside a still-fixture-fallback eval set, which
        # describe disjoint document universes). Silently reporting empty
        # metrics here would look like "ran fine, nothing to see" rather
        # than the actual problem.
        log.warning(
            "cell_fully_excluded",
            corpus=cell.corpus,
            chunker=cell.chunker,
            embedder=cell.embedder,
            retriever=cell.retriever,
            n_pairs=len(pairs),
            hint="every eval pair resolved to zero gold chunks -- check that the "
            "chunk set's documents and the eval set agree on doc_id provenance",
        )

    aggregate: dict[str, float] = {
        key: sum(values) / len(values) for key, values in per_metric.items() if values
    }
    if chunk_eff_values:
        aggregate["chunk_efficiency"] = sum(chunk_eff_values) / len(chunk_eff_values)
    if cross_ref_values:
        aggregate["cross_ref_all_spans_hit"] = sum(cross_ref_values) / len(cross_ref_values)
    aggregate.update(latency_percentiles(latencies))

    return RunResult(
        run_id=run_id,
        config={
            "corpus": cell.corpus,
            "chunker": cell.chunker,
            "chunker_params": cell.chunker_params,
            "embedder": cell.embedder,
            "embedder_params": cell.embedder_params,
            "retriever": cell.retriever,
            "retriever_params": cell.retriever_params,
            "k": cell.k,
            "chunk_set_id": cell.chunk_set_id,
            "index_id": cell.index_id,
            "cell_id": cell.cell_id,
            "eval_split": cell.eval_split,
        },
        corpus=cell.corpus,
        metrics=aggregate,
        per_query=traces,
        chunk_stats={"count": float(len(chunk_set))},
        timings={"total_s": sum(latencies) / 1000.0},
    )


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


def run_experiment(
    config: ExperimentConfig,
    *,
    run_id: str | None = None,
    workers: int = 1,
    force: bool = False,
    console: Console | None = None,
) -> tuple[str, list[RunResult]]:
    """Expand the matrix and execute it. Returns ``(run_id, results)`` for
    every cell, in matrix order -- ``results`` includes cells resumed from a
    prior invocation, not only the ones this call actually computed.
    """
    resolved_run_id = run_id or make_run_id(config.name)
    cells = expand_cells(config)
    if not cells:
        raise ValueError("experiment config produced zero cells after exclusions")

    write_matrix_manifest(resolved_run_id, config)
    if console:
        console.print(
            f"[bold]run_id[/bold] {resolved_run_id}  ({len(cells)} cell(s)) "
            "-- pass --run-id to resume this run if interrupted"
        )

    _build_stage(cells, console)

    (run_dir(resolved_run_id) / "cells").mkdir(parents=True, exist_ok=True)
    pending = [c for c in cells if force or not cell_path(resolved_run_id, c.cell_id).exists()]
    skipped = len(cells) - len(pending)
    if console and skipped:
        console.print(f"[yellow]resume[/yellow]  {skipped} cell(s) already computed, skipping")

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeRemainingColumn(),
        console=console,
        disable=console is None or not pending,
    ) as progress:
        task = progress.add_task("evaluating cells", total=max(1, len(pending)))
        with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
            futures = {
                executor.submit(_evaluate_cell, cell, resolved_run_id): cell for cell in pending
            }
            for future in as_completed(futures):
                cell = futures[future]
                result = future.result()
                cell_path(resolved_run_id, cell.cell_id).write_text(
                    result.model_dump_json(indent=2), encoding="utf-8"
                )
                progress.update(
                    task,
                    advance=1,
                    description=f"{cell.corpus}/{cell.chunker}/{cell.embedder}/{cell.retriever}",
                )

    all_results = [
        RunResult.model_validate_json(cell_path(resolved_run_id, c.cell_id).read_text("utf-8"))
        for c in cells
    ]
    return resolved_run_id, all_results


__all__ = [
    "RECALL_KS",
    "cell_path",
    "load_matrix_manifest",
    "run_dir",
    "run_experiment",
    "write_matrix_manifest",
]
