"""Reporting for Phase 6 runs (plan Phase 6, Step 6.5): a metrics table with
bootstrap confidence intervals, run-to-run comparison, and a worst-failures
dump -- the direct input to Phase 8's optimizer agent, and how you debug a
strategy by hand.
"""

from __future__ import annotations

from rich.console import Console
from rich.markup import escape
from rich.table import Table

from rag_lab.experiment.config import Cell, ExperimentConfig, expand_cells
from rag_lab.experiment.runner import cell_path, load_matrix_manifest, run_dir
from rag_lab.metrics import bootstrap_ci
from rag_lab.schemas import QueryTrace, RunResult

HEADLINE_METRICS = ["recall@1", "recall@3", "recall@5", "recall@10", "mrr", "ndcg@10"]


def _values_for(result: RunResult, metric: str) -> list[float]:
    return [t.metrics[metric] for t in result.per_query if not t.excluded and metric in t.metrics]


def _cell_label(result: RunResult) -> str:
    cfg = result.config
    return (
        f"{result.corpus}/{cfg.get('chunker', '?')}/{cfg.get('embedder', '?')}/"
        f"{cfg.get('retriever', '?')}"
    )


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def load_run(run_id: str) -> tuple[ExperimentConfig, list[RunResult]]:
    """Every computed cell of ``run_id``, in the matrix's own order. Cells
    with no result file yet (an interrupted run) are simply absent, not an
    error -- ``experiment report`` on a partial run reports what's there.
    """
    config = load_matrix_manifest(run_id)
    cells = expand_cells(config)
    results = []
    for cell in cells:
        path = cell_path(run_id, cell.cell_id)
        if path.exists():
            results.append(RunResult.model_validate_json(path.read_text(encoding="utf-8")))
    if not results:
        raise LookupError(f"run {run_id!r} has no computed cells yet (checked {run_dir(run_id)})")
    return config, results


# --------------------------------------------------------------------------- #
# `experiment report`
# --------------------------------------------------------------------------- #


def render_report_table(results: list[RunResult], console: Console) -> None:
    table = Table(title="rag-lab experiment report", title_style="bold")
    for col in ("corpus", "chunker", "embedder", "retriever"):
        table.add_column(col)
    for metric in HEADLINE_METRICS:
        table.add_column(metric, justify="right")
    table.add_column("chunk_eff", justify="right")
    table.add_column("n", justify="right")

    cross_ref_lines: list[str] = []
    for r in results:
        cfg = r.config
        row = [
            escape(r.corpus),
            escape(str(cfg.get("chunker", "?"))),
            escape(str(cfg.get("embedder", "?"))),
            escape(str(cfg.get("retriever", "?"))),
        ]
        for metric in HEADLINE_METRICS:
            values = _values_for(r, metric)
            if not values:
                row.append("-")
                continue
            mean = sum(values) / len(values)
            lo, hi = bootstrap_ci(values)
            row.append(f"{mean:.3f} [{lo:.3f},{hi:.3f}]")
        eff = r.metrics.get("chunk_efficiency")
        row.append(f"{eff:.1f}" if eff is not None else "-")
        row.append(str(len(r.per_query)))
        table.add_row(*row)

        cross_ref = _values_for(r, "cross_ref_all_spans_hit")
        if cross_ref:
            mean = sum(cross_ref) / len(cross_ref)
            cross_ref_lines.append(
                f"  cross_reference multi-hop recall@{cfg.get('k', '?')} "
                f"({_cell_label(r)}): {mean:.1%}  "
                "(diagnostic only -- any-span headline is the table above)"
            )

    console.print(table)
    for line in cross_ref_lines:
        console.print(line)


def render_report_markdown(results: list[RunResult]) -> str:
    header = ["corpus", "chunker", "embedder", "retriever", *HEADLINE_METRICS, "chunk_eff", "n"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in results:
        cfg = r.config
        row = [r.corpus, str(cfg.get("chunker", "?")), str(cfg.get("embedder", "?")), str(cfg.get("retriever", "?"))]
        for metric in HEADLINE_METRICS:
            values = _values_for(r, metric)
            if values:
                mean = sum(values) / len(values)
                lo, hi = bootstrap_ci(values)
                row.append(f"{mean:.3f} [{lo:.3f},{hi:.3f}]")
            else:
                row.append("-")
        eff = r.metrics.get("chunk_efficiency")
        row.append(f"{eff:.1f}" if eff is not None else "-")
        row.append(str(len(r.per_query)))
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# `experiment compare`
# --------------------------------------------------------------------------- #

CellKey = tuple[str, str, str, str]  # (corpus, chunker, embedder, retriever)


def _cell_key(result: RunResult) -> CellKey:
    cfg = result.config
    return (result.corpus, str(cfg.get("chunker")), str(cfg.get("embedder")), str(cfg.get("retriever")))


def compare_runs(run_ids: list[str]) -> dict[CellKey, dict[str, RunResult]]:
    """Cells present in *every* given run, keyed by
    ``(corpus, chunker, embedder, retriever)`` -- the same config always
    expands to the same ``cell_id`` regardless of which ``run_id`` it ran
    under, so this is a straightforward intersection, not a fuzzy match.
    """
    per_run: dict[str, dict[CellKey, RunResult]] = {}
    for run_id in run_ids:
        _config, results = load_run(run_id)
        per_run[run_id] = {_cell_key(r): r for r in results}

    common = set.intersection(*(set(d) for d in per_run.values())) if per_run else set()
    return {key: {run_id: per_run[run_id][key] for run_id in run_ids} for key in sorted(common)}


def render_compare_table(
    run_ids: list[str], comparison: dict[CellKey, dict[str, RunResult]], console: Console
) -> None:
    table = Table(title=f"rag-lab experiment compare: {', '.join(run_ids)}", title_style="bold")
    table.add_column("cell")
    for run_id in run_ids:
        table.add_column(escape(run_id), overflow="fold")

    if not comparison:
        console.print("[yellow]no cells in common across every given run[/yellow]")
        return

    for key, by_run in comparison.items():
        row = [escape("/".join(key))]
        for run_id in run_ids:
            m = by_run[run_id].metrics
            row.append(f"recall@5={m.get('recall@5', 0.0):.3f}  ndcg@10={m.get('ndcg@10', 0.0):.3f}")
        table.add_row(*row)
    console.print(table)


# --------------------------------------------------------------------------- #
# `experiment failures`
# --------------------------------------------------------------------------- #


def worst_failures(run_id: str, config_idx: int, n: int = 20) -> tuple[Cell, RunResult, list[QueryTrace]]:
    """Re-expands the matrix from the run's own ``matrix.json`` and indexes
    into it -- the same config always expands the same way (Step 6.3), so
    ``config_idx`` is stable across ``report``/``compare``/``failures`` calls
    on the same run."""
    config = load_matrix_manifest(run_id)
    cells = expand_cells(config)
    if not 0 <= config_idx < len(cells):
        raise ValueError(f"--config-idx {config_idx} out of range (matrix has {len(cells)} cells)")
    cell = cells[config_idx]

    path = cell_path(run_id, cell.cell_id)
    if not path.exists():
        raise LookupError(
            f"cell {config_idx} ({cell.cell_id}) has no computed result in run {run_id!r}"
        )
    result = RunResult.model_validate_json(path.read_text(encoding="utf-8"))

    def _sort_key(t: QueryTrace) -> tuple[int, int, float]:
        if t.excluded:
            return (2, 0, 0.0)
        return (0 if t.first_hit_rank is None else 1, 0, t.metrics.get("recall@5", 0.0))

    worst = sorted(result.per_query, key=_sort_key)[:n]
    return cell, result, worst


def render_failures(cell: Cell, worst: list[QueryTrace], console: Console) -> None:
    table = Table(
        title=f"worst failures — {cell.corpus}/{cell.chunker}/{cell.embedder}/{cell.retriever}",
        title_style="bold",
        show_lines=True,
    )
    table.add_column("query", overflow="fold")
    table.add_column("gold chunks", overflow="fold")
    table.add_column("retrieved (top 5)", overflow="fold")
    table.add_column("recall@5", justify="right")

    for t in worst:
        status = "excluded" if t.excluded else f"{t.metrics.get('recall@5', 0.0):.2f}"
        table.add_row(
            escape(t.query),
            escape(", ".join(t.gold_chunk_ids) or "(none)"),
            escape(", ".join(t.retrieved_chunk_ids[:5]) or "(none)"),
            status,
        )
    console.print(table)


__all__ = [
    "HEADLINE_METRICS",
    "CellKey",
    "compare_runs",
    "load_run",
    "render_compare_table",
    "render_failures",
    "render_report_markdown",
    "render_report_table",
    "worst_failures",
]
