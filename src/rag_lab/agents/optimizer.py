"""Optimizer agent (plan §Phase 8, Step 8.2): propose / run / diagnose / mutate,
on the held-out dev split, for ``max_iterations``, then report the winner on
test once.

The outer loop (across iterations) is plain Python, not another layer of
tool-use -- each iteration is *one* ``agents/runtime.py`` tool-use conversation
(propose a config via ``run_experiment``, inspect ``get_failures``, end with a
diagnosis/mutation), and the Python ``for`` loop that drives ``max_iterations``
of those is what makes budget caps, per-iteration trace writes, and the final
test-split re-evaluation straightforward to get right without needing the
model to manage its own multi-iteration bookkeeping. Constraining the search
space (plan design point) reuses the *existing* chunker/embedder/retriever
registries rather than a bespoke hardcoded menu -- ``available_chunkers()``
etc. are already "the menu"; ``run_experiment``'s tool handler validates
against them directly.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

import structlog
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from rag_lab.agents.runtime import (
    Budget,
    ModelCaller,
    ModelTurn,
    Tool,
    ToolCall,
    ToolError,
    anthropic_model_caller,
    estimate_cost_usd,
    run_agent_loop,
)
from rag_lab.chunkers import REGISTRY as CHUNKER_REGISTRY
from rag_lab.chunkers import available_chunkers
from rag_lab.chunks import compute_chunk_stats, documents_for_chunks, load_chunk_set
from rag_lab.embedders import available_embedders
from rag_lab.experiment.config import ExperimentConfig, MatrixComponentSpec, MatrixSpec
from rag_lab.experiment.report import compare_runs as report_compare_runs
from rag_lab.experiment.report import worst_failures
from rag_lab.experiment.runner import run_dir
from rag_lab.experiment.runner import run_experiment as run_experiment_matrix
from rag_lab.ids import make_run_id
from rag_lab.jsonl import write_jsonl
from rag_lab.retrievers import available_retrievers
from rag_lab.schemas import OptimizerTraceEntry

log = structlog.get_logger(__name__)

DEFAULT_MODEL = "claude-sonnet-5"
DEFAULT_MAX_ITERATIONS = 6
DEFAULT_MAX_STEPS_PER_ITERATION = 4
DEFAULT_METRIC = "recall@5"


# --------------------------------------------------------------------------- #
# Per-iteration session state and tools
# --------------------------------------------------------------------------- #


@dataclass
class _OptimizerContext:
    corpus: str
    k: int
    split: str  # "dev" while iterating, "test" for the one final evaluation
    last_run: dict[str, Any] | None = field(default=None)


def _build_tools(ctx: _OptimizerContext) -> list[Tool]:
    def _run_experiment(args: dict[str, Any]) -> dict[str, Any]:
        chunker, embedder, retriever = args["chunker"], args["embedder"], args["retriever"]
        errors = []
        if chunker not in available_chunkers():
            errors.append(f"unknown chunker {chunker!r}; available: {available_chunkers()}")
        if embedder not in available_embedders():
            errors.append(f"unknown embedder {embedder!r}; available: {available_embedders()}")
        valid_retrievers = [r for r in available_retrievers() if r != "agent_router"]
        if retriever not in valid_retrievers:
            errors.append(f"unknown retriever {retriever!r}; available: {valid_retrievers}")
        for key in ("chunker_params", "embedder_params", "retriever_params"):
            if key in args and not isinstance(args[key], dict):
                errors.append(f"{key} must be an object")
        if errors:
            raise ToolError("invalid config: " + "; ".join(errors))

        config = ExperimentConfig(
            name=f"optimizer-{ctx.corpus}",
            corpora=[ctx.corpus],
            k=ctx.k,
            eval_split=ctx.split,  # type: ignore[arg-type]
            matrix=MatrixSpec(
                chunker=[MatrixComponentSpec(name=chunker, params=args.get("chunker_params", {}))],
                embedder=[
                    MatrixComponentSpec(name=embedder, params=args.get("embedder_params", {}))
                ],
                retriever=[
                    MatrixComponentSpec(name=retriever, params=args.get("retriever_params", {}))
                ],
            ),
        )
        try:
            run_id, results = run_experiment_matrix(config, workers=1)
        except (LookupError, FileNotFoundError, ValueError) as exc:
            raise ToolError(f"run_experiment failed: {exc}") from exc

        result = results[0]  # exactly one cell -- one chunker x one embedder x one retriever
        ctx.last_run = {
            "run_id": run_id,
            "config": dict(args),
            "metrics": dict(result.metrics),
            "chunk_set_id": result.config.get("chunk_set_id"),
        }
        return {"run_id": run_id, "metrics": result.metrics}

    def _get_failures(args: dict[str, Any]) -> list[dict[str, Any]]:
        run_id = args["run_id"]
        n = int(args.get("n", 10))
        try:
            _cell, _result, worst = worst_failures(run_id, 0, n)
        except (LookupError, ValueError) as exc:
            raise ToolError(str(exc)) from exc
        return [
            {
                "query": t.query,
                "gold_chunk_ids": t.gold_chunk_ids,
                "retrieved_chunk_ids": t.retrieved_chunk_ids[:5],
                "excluded": t.excluded,
                "recall@5": t.metrics.get("recall@5"),
            }
            for t in worst
        ]

    def _get_chunk_stats(args: dict[str, Any]) -> dict[str, Any]:
        chunk_set_id = args["chunk_set_id"]
        try:
            chunks = load_chunk_set(chunk_set_id)
            stats = compute_chunk_stats(chunks, documents_for_chunks(chunks))
        except (LookupError, FileNotFoundError, ValueError) as exc:
            raise ToolError(str(exc)) from exc
        return {
            "count": stats.count,
            "token_p50": stats.token_p50,
            "token_p95": stats.token_p95,
            "split_code_block_count": stats.split_code_block_count,
            "split_table_count": stats.split_table_count,
            "orphan_rate": stats.orphan_rate,
        }

    def _compare_runs(args: dict[str, Any]) -> list[dict[str, Any]]:
        a, b = args["a"], args["b"]
        try:
            comparison = report_compare_runs([a, b])
        except LookupError as exc:
            raise ToolError(str(exc)) from exc
        return [
            {"cell": "/".join(key), a: by_run[a].metrics, b: by_run[b].metrics}
            for key, by_run in comparison.items()
        ]

    return [
        Tool(
            name="run_experiment",
            description=(
                "Run one chunker/embedder/retriever configuration on the current split "
                "and return its run_id and metrics."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "hypothesis": {"type": "string"},
                    "chunker": {"type": "string"},
                    "chunker_params": {"type": "object"},
                    "embedder": {"type": "string"},
                    "embedder_params": {"type": "object"},
                    "retriever": {"type": "string"},
                    "retriever_params": {"type": "object"},
                },
                "required": ["hypothesis", "chunker", "embedder", "retriever"],
            },
            handler=_run_experiment,
        ),
        Tool(
            name="get_failures",
            description="Worst-performing queries for a run_id, with retrieved and gold chunk ids.",
            input_schema={
                "type": "object",
                "properties": {"run_id": {"type": "string"}, "n": {"type": "integer"}},
                "required": ["run_id"],
            },
            handler=_get_failures,
        ),
        Tool(
            name="get_chunk_stats",
            description="Token distribution, split-code/table counts, orphan rate for a chunk set.",
            input_schema={
                "type": "object",
                "properties": {"chunk_set_id": {"type": "string"}},
                "required": ["chunk_set_id"],
            },
            handler=_get_chunk_stats,
        ),
        Tool(
            name="compare_runs",
            description="Metrics for two run_ids on their shared cell.",
            input_schema={
                "type": "object",
                "properties": {"a": {"type": "string"}, "b": {"type": "string"}},
                "required": ["a", "b"],
            },
            handler=_compare_runs,
        ),
    ]


# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #


def _search_space_menu() -> str:
    lines = []
    for name in available_chunkers():
        params = sorted(CHUNKER_REGISTRY[name].default_params)
        lines.append(f"  - {name} (params: {params})" if params else f"  - {name}")
    return "\n".join(lines)


def _system_prompt(corpus: str, k: int) -> str:
    embedder_menu = ", ".join(available_embedders())
    retriever_menu = ", ".join(r for r in available_retrievers() if r != "agent_router")
    return (
        "You are the optimizer agent for a retrieval-augmented generation framework. "
        f"Propose ONE chunker+embedder+retriever configuration for the {corpus!r} corpus with a "
        "stated hypothesis, call run_experiment with it (evaluated on the held-out DEV split -- "
        "you never see the test split), call get_failures on the resulting run_id, optionally "
        "call get_chunk_stats or compare_runs, then stop calling tools and reply with plain text "
        "in exactly this form:\n"
        "DIAGNOSIS: <what the failures share>\n"
        "MUTATION: <what you would try next, and why>\n\n"
        f"Available chunkers:\n{_search_space_menu()}\n"
        f"Available embedders: {embedder_menu}\n"
        f"Available retrievers: {retriever_menu}\n"
        f"Retrieve k={k} results per query. Only propose components from these lists -- "
        "an unrecognized name is rejected and wastes the turn."
    )


def _iteration_user_message(iteration: int, history: list[OptimizerTraceEntry]) -> str:
    if not history:
        return f"Iteration {iteration}: propose your first configuration and hypothesis."
    lines = [f"Iteration {iteration}. Prior iterations:"]
    for e in history:
        headline = e.metrics.get(DEFAULT_METRIC)
        headline_str = f"{headline:.3f}" if headline is not None else "n/a"
        lines.append(
            f"  #{e.iteration} {e.config.get('chunker')}/{e.config.get('embedder')}/"
            f"{e.config.get('retriever')} -> {DEFAULT_METRIC}={headline_str} "
            f"hypothesis={e.hypothesis!r} mutation={e.mutation!r}"
        )
    lines.append("Propose a new configuration that targets the last iteration's diagnosis.")
    return "\n".join(lines)


_DIAGNOSIS_RE = re.compile(r"DIAGNOSIS:\s*(.*?)(?:\n\s*MUTATION:|\Z)", re.S | re.I)
_MUTATION_RE = re.compile(r"MUTATION:\s*(.*)\Z", re.S | re.I)


def _parse_diagnosis_and_mutation(text: str) -> tuple[str, str]:
    """Best-effort split of the final turn's free text into the two labeled
    paragraphs the system prompt asks for. Falls back to storing the whole
    text as the diagnosis (never raises) -- a model that doesn't follow the
    label convention exactly still produces a usable, if less structured,
    trace entry."""
    d = _DIAGNOSIS_RE.search(text)
    m = _MUTATION_RE.search(text)
    diagnosis = d.group(1).strip() if d else text.strip()
    mutation = m.group(1).strip() if m else ""
    return diagnosis, mutation


# --------------------------------------------------------------------------- #
# Mock model caller -- deterministic, no network, no `anthropic` import
# --------------------------------------------------------------------------- #

# A small, deliberately valid rotation -- every entry names a real registered
# chunker/embedder/retriever, so a mock run can never "propose an invalid
# config" (AC-3). Cycled by iteration index.
_MOCK_ROTATION: list[dict[str, Any]] = [
    {"chunker": "fixed", "chunker_params": {"chunk_tokens": 256, "overlap_tokens": 32}},
    {"chunker": "recursive", "chunker_params": {"chunk_tokens": 256, "overlap_tokens": 32}},
    {"chunker": "markdown", "chunker_params": {}},
]


def _mock_config_for_iteration(iteration: int) -> dict[str, Any]:
    base = dict(_MOCK_ROTATION[iteration % len(_MOCK_ROTATION)])
    base["embedder"] = "bge-small"
    base["embedder_params"] = {}
    base["retriever"] = "dense"
    base["retriever_params"] = {}
    return base


def _last_tool_result_payload(messages: list[dict[str, Any]]) -> dict[str, Any]:
    for block in reversed(messages[-1]["content"]):
        if block.get("type") == "tool_result" and not block.get("is_error"):
            return json.loads(block["content"])
    raise RuntimeError("mock optimizer caller expected a successful tool_result but found none")


def _mock_model_caller(iteration: int) -> ModelCaller:
    cfg = _mock_config_for_iteration(iteration)
    step = {"n": 0}

    def _call(messages: list[dict[str, Any]], system: str, tools: list[Tool]) -> ModelTurn:
        step["n"] += 1
        if step["n"] == 1:
            args = {
                "hypothesis": f"[mock] iteration {iteration}: try {cfg['chunker']} chunking",
                **cfg,
            }
            return ModelTurn(
                text="",
                tool_calls=[
                    ToolCall(id=f"mock-run-{iteration}", name="run_experiment", arguments=args)
                ],
                input_tokens=220,
                output_tokens=70,
            )
        if step["n"] == 2:
            run_id = _last_tool_result_payload(messages)["run_id"]
            return ModelTurn(
                text="",
                tool_calls=[
                    ToolCall(
                        id=f"mock-fail-{iteration}",
                        name="get_failures",
                        arguments={"run_id": run_id, "n": 5},
                    )
                ],
                input_tokens=90,
                output_tokens=25,
            )
        return ModelTurn(
            text=(
                f"DIAGNOSIS: [mock] iteration {iteration} failures reviewed for "
                f"{cfg['chunker']} chunking on the dev split.\n"
                f"MUTATION: [mock] try a different chunker on the next iteration."
            ),
            tool_calls=[],
            input_tokens=70,
            output_tokens=45,
        )

    return _call


# --------------------------------------------------------------------------- #
# One iteration
# --------------------------------------------------------------------------- #


def _run_iteration(
    iteration: int,
    corpus: str,
    k: int,
    history: list[OptimizerTraceEntry],
    *,
    model: str,
    mock: bool,
    budget: Budget,
) -> tuple[OptimizerTraceEntry | None, int, int]:
    """Returns ``(entry, input_tokens, output_tokens)``. ``entry`` is ``None``
    when the budget was exhausted before ``run_experiment`` was ever
    successfully called -- the caller treats that as "stop, nothing new to
    record," not an error."""
    ctx = _OptimizerContext(corpus=corpus, k=k, split="dev")
    tools = _build_tools(ctx)
    caller = _mock_model_caller(iteration) if mock else anthropic_model_caller(model=model)

    result = run_agent_loop(
        system=_system_prompt(corpus, k),
        user_message=_iteration_user_message(iteration, history),
        tools=tools,
        model_caller=caller,
        model=model,
        budget=budget,
    )

    if ctx.last_run is None:
        log.warning(
            "optimizer_iteration_no_run",
            iteration=iteration,
            stopped_reason=result.stopped_reason,
        )
        return None, result.total_input_tokens, result.total_output_tokens

    diagnosis, mutation = _parse_diagnosis_and_mutation(result.final_text)
    entry = OptimizerTraceEntry(
        iteration=iteration,
        hypothesis=str(ctx.last_run["config"].get("hypothesis", "")),
        config=ctx.last_run["config"],
        run_id=ctx.last_run["run_id"],
        split="dev",
        metrics=ctx.last_run["metrics"],
        diagnosis=diagnosis,
        mutation=mutation,
        input_tokens=result.total_input_tokens,
        output_tokens=result.total_output_tokens,
    )
    return entry, result.total_input_tokens, result.total_output_tokens


def _run_final_test_eval(
    best: OptimizerTraceEntry, corpus: str, k: int, next_iteration: int
) -> OptimizerTraceEntry:
    """Re-runs the winning dev-split config once, on the test split. Plain
    Python, not another agent-loop turn -- "report the winner on test, once"
    is mechanical, not a reasoning step, so it doesn't spend any more budget."""
    cfg = best.config
    config = ExperimentConfig(
        name=f"optimizer-{corpus}-final",
        corpora=[corpus],
        k=k,
        eval_split="test",
        matrix=MatrixSpec(
            chunker=[
                MatrixComponentSpec(name=cfg["chunker"], params=cfg.get("chunker_params", {}))
            ],
            embedder=[
                MatrixComponentSpec(name=cfg["embedder"], params=cfg.get("embedder_params", {}))
            ],
            retriever=[
                MatrixComponentSpec(name=cfg["retriever"], params=cfg.get("retriever_params", {}))
            ],
        ),
    )
    try:
        run_id, results = run_experiment_matrix(config, workers=1)
        metrics = dict(results[0].metrics)
        diagnosis = ""
    except (LookupError, FileNotFoundError, ValueError) as exc:
        log.warning("optimizer_final_test_eval_failed", corpus=corpus, error=str(exc))
        run_id = best.run_id
        metrics = {}
        diagnosis = f"final test-split evaluation failed: {exc}"
    return OptimizerTraceEntry(
        iteration=next_iteration,
        hypothesis=f"final test-split evaluation of iteration {best.iteration}'s winning config",
        config=cfg,
        run_id=run_id,
        split="test",
        metrics=metrics,
        diagnosis=diagnosis,
        mutation="",
    )


def _write_trace(optimizer_run_id: str, trace: list[OptimizerTraceEntry]) -> None:
    write_jsonl(run_dir(optimizer_run_id) / "optimizer_trace.jsonl", trace)


def load_trace(optimizer_run_id: str) -> list[OptimizerTraceEntry]:
    from rag_lab.jsonl import read_jsonl

    path = run_dir(optimizer_run_id) / "optimizer_trace.jsonl"
    if not path.exists():
        raise LookupError(
            f"no optimizer trace found for run_id {optimizer_run_id!r} (checked {path})"
        )
    return read_jsonl(path, OptimizerTraceEntry)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


def optimize(
    corpus: str,
    *,
    k: int = 10,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    max_steps_per_iteration: int = DEFAULT_MAX_STEPS_PER_ITERATION,
    model: str = DEFAULT_MODEL,
    mock: bool = False,
    max_tokens: int | None = None,
    max_wall_clock_s: float | None = None,
    budget_usd: float | None = None,
    metric: str = DEFAULT_METRIC,
) -> tuple[str, list[OptimizerTraceEntry]]:
    """Run the propose/run/diagnose/mutate loop for up to ``max_iterations``
    on the dev split, then evaluate the dev-best config once on test.

    Every cap (``max_iterations``, ``max_tokens``, ``max_wall_clock_s``,
    ``budget_usd``) is checked *before* starting the next iteration -- a
    breach stops the loop and returns normally with whatever iterations
    already completed (each already written to
    ``artifacts/results/<run_id>/optimizer_trace.jsonl`` as it finished, not
    only at the end). This never raises for a budget breach, including the
    degenerate case of zero completed iterations -- an empty trace file is
    written and ``(run_id, [])`` is returned, which is exactly what "graceful
    termination with partial results written" (plan AC-5) means at the low
    end.
    """
    optimizer_run_id = make_run_id(f"optimizer-{corpus}")
    trace: list[OptimizerTraceEntry] = []
    started = time.monotonic()
    total_input = 0
    total_output = 0

    for iteration in range(max_iterations):
        elapsed = time.monotonic() - started
        spent_usd = estimate_cost_usd(model, total_input, total_output)
        if max_wall_clock_s is not None and elapsed >= max_wall_clock_s:
            log.info("optimizer_stopping", reason="max_wall_clock_s", iteration=iteration)
            break
        if max_tokens is not None and (total_input + total_output) >= max_tokens:
            log.info("optimizer_stopping", reason="max_tokens", iteration=iteration)
            break
        if budget_usd is not None and spent_usd >= budget_usd:
            log.info("optimizer_stopping", reason="budget_usd", iteration=iteration)
            break

        inner_budget = Budget(
            max_steps=max_steps_per_iteration,
            max_total_tokens=None
            if max_tokens is None
            else max(0, max_tokens - total_input - total_output),
            max_wall_clock_s=None
            if max_wall_clock_s is None
            else max(0.0, max_wall_clock_s - elapsed),
            max_usd=None if budget_usd is None else max(0.0, budget_usd - spent_usd),
        )
        entry, in_tok, out_tok = _run_iteration(
            iteration, corpus, k, trace, model=model, mock=mock, budget=inner_budget
        )
        total_input += in_tok
        total_output += out_tok
        if entry is None:
            break  # budget ran out mid-iteration -- nothing new to record
        trace.append(entry)
        _write_trace(optimizer_run_id, trace)

    if not trace:
        _write_trace(optimizer_run_id, trace)
        return optimizer_run_id, trace

    best = max(trace, key=lambda e: e.metrics.get(metric, float("-inf")))
    test_entry = _run_final_test_eval(best, corpus, k, next_iteration=len(trace))
    trace.append(test_entry)
    _write_trace(optimizer_run_id, trace)
    return optimizer_run_id, trace


def render_trace(
    trace: list[OptimizerTraceEntry], console: Console, *, metric: str = DEFAULT_METRIC
) -> None:
    table = Table(title="optimizer trace", title_style="bold", show_lines=True)
    table.add_column("iter", justify="right")
    table.add_column("split")
    table.add_column("config", overflow="fold")
    table.add_column(metric, justify="right")
    table.add_column("hypothesis", overflow="fold")
    table.add_column("diagnosis -> mutation", overflow="fold")

    for e in trace:
        cfg = f"{e.config.get('chunker')}/{e.config.get('embedder')}/{e.config.get('retriever')}"
        metric_val = e.metrics.get(metric)
        metric_str = f"{metric_val:.3f}" if metric_val is not None else "-"
        diag = f"{e.diagnosis} -> {e.mutation}" if e.mutation else e.diagnosis
        table.add_row(
            str(e.iteration), e.split, escape(cfg), metric_str, escape(e.hypothesis), escape(diag)
        )
    console.print(table)

    dev_entries = [e for e in trace if e.split == "dev"]
    test_entries = [e for e in trace if e.split == "test"]
    if dev_entries and test_entries:
        best_dev = max(dev_entries, key=lambda e: e.metrics.get(metric, float("-inf")))
        winner_cfg = (
            f"{best_dev.config.get('chunker')}/{best_dev.config.get('embedder')}/"
            f"{best_dev.config.get('retriever')}"
        )
        console.print(
            f"[bold]winner[/bold]: iteration {best_dev.iteration} ({escape(winner_cfg)}) -- "
            f"dev {metric}={best_dev.metrics.get(metric, 0.0):.3f}, "
            f"test {metric}={test_entries[-1].metrics.get(metric, 0.0):.3f}"
        )


__all__ = [
    "DEFAULT_MAX_ITERATIONS",
    "DEFAULT_MAX_STEPS_PER_ITERATION",
    "DEFAULT_METRIC",
    "DEFAULT_MODEL",
    "load_trace",
    "optimize",
    "render_trace",
]
