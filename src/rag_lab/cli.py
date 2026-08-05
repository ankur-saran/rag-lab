"""Single Typer app, one sub-app per phase.

Every phase's subcommands are registered here from the outset, stubbed until the
phase is built. Keeping the surface visible from Phase 0 means `rag-lab --help`
is always an accurate map of the system, and it prevents the CLI shape from being
invented three different ways across three phases.
"""

from __future__ import annotations

import os
import random
import sys
from typing import Annotated

import typer
from rich.console import Console

from rag_lab import __version__
from rag_lab.config import load_config, parse_overrides
from rag_lab.logging_setup import configure_logging

console = Console()

app = typer.Typer(
    name="rag-lab",
    help="A framework for showcasing chunking and embedding strategies.",
    no_args_is_help=True,
    add_completion=False,
)

corpus_app = typer.Typer(help="Phase 1 — load and inspect corpora.", no_args_is_help=True)
chunk_app = typer.Typer(
    help="Phase 2/7 — chunk documents and inspect boundaries.", no_args_is_help=True
)
index_app = typer.Typer(help="Phase 3 — embed chunks and build indexes.", no_args_is_help=True)
retrieve_app = typer.Typer(help="Phase 4 — run and compare retrievers.", no_args_is_help=True)
evalset_app = typer.Typer(
    help="Phase 5 — generate and validate the eval set.", no_args_is_help=True
)
experiment_app = typer.Typer(help="Phase 6 — run the matrix and report.", no_args_is_help=True)
agent_app = typer.Typer(help="Phase 8 — routing and optimizer agents.", no_args_is_help=True)

app.add_typer(corpus_app, name="corpus")
app.add_typer(chunk_app, name="chunk")
app.add_typer(index_app, name="index")
app.add_typer(retrieve_app, name="retrieve")
app.add_typer(evalset_app, name="evalset")
app.add_typer(experiment_app, name="experiment")
app.add_typer(agent_app, name="agent")


def _seed_everything(seed: int) -> None:
    """Seed every RNG that could affect an artifact. Called before any subcommand."""
    random.seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:  # torch arrives with the embed extras
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def _not_implemented(phase: int, what: str) -> None:
    console.print(
        f"[yellow]{what}[/yellow] is implemented in [bold]Phase {phase}[/bold], "
        "which has not been built yet."
    )
    raise typer.Exit(code=1)


@app.callback()
def main(
    ctx: typer.Context,
    config: Annotated[
        str | None, typer.Option("--config", "-c", help="YAML config file.")
    ] = None,
    set_: Annotated[
        list[str] | None,
        typer.Option("--set", "-s", help="Override, e.g. --set seed=7 --set defaults.k=5"),
    ] = None,
    seed: Annotated[int | None, typer.Option("--seed", help="Override the RNG seed.")] = None,
    log_level: Annotated[str, typer.Option("--log-level")] = "info",
    json_logs: Annotated[bool, typer.Option("--json-logs/--text-logs")] = False,
) -> None:
    """Global options. Resolution order: --set > --config > config/default.yaml > code defaults."""
    overrides = parse_overrides(set_)
    if seed is not None:
        overrides["seed"] = seed
    cfg = load_config(config, overrides)
    cfg.log_level = log_level
    cfg.json_logs = json_logs

    configure_logging(cfg.log_level, cfg.json_logs)
    _seed_everything(cfg.seed)

    ctx.obj = cfg


@app.command()
def version() -> None:
    """Print the version."""
    console.print(f"rag-lab {__version__}")


@app.command()
def doctor() -> None:
    """Check the environment: Python, dependencies, disk, fixtures, credentials, model cache."""
    from rag_lab.doctor import doctor as run_doctor

    raise typer.Exit(code=run_doctor(console))


@app.command("config")
def show_config(ctx: typer.Context) -> None:
    """Print the fully resolved configuration as JSON."""
    cfg = ctx.obj
    console.print_json(cfg.model_dump_json(indent=2))


# --------------------------------------------------------------------------- #
# Phase stubs — real implementations land in their own phases.
# --------------------------------------------------------------------------- #


@corpus_app.command("build")
def corpus_build(
    corpus: Annotated[str | None, typer.Option("--corpus")] = None,
    all_: Annotated[bool, typer.Option("--all")] = False,
) -> None:
    """Load raw files into artifacts/documents/<corpus>.jsonl."""
    _not_implemented(1, "corpus build")


@corpus_app.command("stats")
def corpus_stats(corpus: Annotated[str | None, typer.Option("--corpus")] = None) -> None:
    """Token distribution, heading density, code blocks, tables, language mix."""
    _not_implemented(1, "corpus stats")


@corpus_app.command("show")
def corpus_show(
    doc_id: Annotated[str, typer.Option("--doc-id")],
    chars: Annotated[int, typer.Option("--chars")] = 500,
) -> None:
    """Print the head of a document."""
    _not_implemented(1, "corpus show")


@chunk_app.command("run")
def chunk_run(
    corpus: Annotated[str, typer.Option("--corpus")],
    chunker: Annotated[str, typer.Option("--chunker")],
    params: Annotated[list[str] | None, typer.Option("--params")] = None,
) -> None:
    """Chunk a corpus into artifacts/chunks/<chunk_set_id>.jsonl."""
    _not_implemented(2, "chunk run")


@chunk_app.command("stats")
def chunk_stats(chunk_set: Annotated[str, typer.Option("--chunk-set")]) -> None:
    """Token distribution, split code blocks, split tables, orphan rate."""
    _not_implemented(2, "chunk stats")


@chunk_app.command("show")
def chunk_show(
    chunk_set: Annotated[str, typer.Option("--chunk-set")],
    doc_id: Annotated[str, typer.Option("--doc-id")],
) -> None:
    """Render a document with chunk boundaries."""
    _not_implemented(2, "chunk show")


@chunk_app.command("diff")
def chunk_diff(
    a: Annotated[str, typer.Option("--a")],
    b: Annotated[str, typer.Option("--b")],
    doc_id: Annotated[str, typer.Option("--doc-id")],
) -> None:
    """Compare two chunk sets' boundaries on the same document, side by side."""
    _not_implemented(2, "chunk diff")


@index_app.command("build")
def index_build(
    chunk_set: Annotated[str, typer.Option("--chunk-set")],
    embedder: Annotated[str, typer.Option("--embedder")] = "bge-small",
) -> None:
    """Embed a chunk set and persist a vector index."""
    _not_implemented(3, "index build")


@index_app.command("list")
def index_list() -> None:
    """List built indexes and their manifests."""
    _not_implemented(3, "index list")


@index_app.command("search")
def index_search(
    index_id: Annotated[str, typer.Option("--index-id")],
    query: Annotated[str, typer.Option("--query")],
    k: Annotated[int, typer.Option("--k")] = 5,
) -> None:
    """Ad-hoc semantic search against an index."""
    _not_implemented(3, "index search")


@retrieve_app.command("query")
def retrieve_query(
    index_id: Annotated[str, typer.Option("--index-id")],
    query: Annotated[str, typer.Option("--query")],
    retriever: Annotated[str, typer.Option("--retriever")] = "dense",
    k: Annotated[int, typer.Option("--k")] = 5,
) -> None:
    """Run one retriever."""
    _not_implemented(4, "retrieve query")


@retrieve_app.command("compare")
def retrieve_compare(
    index_id: Annotated[str, typer.Option("--index-id")],
    query: Annotated[str, typer.Option("--query")],
    retrievers: Annotated[str, typer.Option("--retrievers")] = "dense,bm25,hybrid",
    k: Annotated[int, typer.Option("--k")] = 5,
) -> None:
    """Run several retrievers side by side."""
    _not_implemented(4, "retrieve compare")


@evalset_app.command("build")
def evalset_build(
    corpus: Annotated[str, typer.Option("--corpus")],
    n: Annotated[int, typer.Option("--n")] = 200,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
    mock_llm: Annotated[bool, typer.Option("--mock-llm")] = False,
) -> None:
    """Generate synthetic gold-labeled queries."""
    _not_implemented(5, "evalset build")


@evalset_app.command("validate")
def evalset_validate(corpus: Annotated[str, typer.Option("--corpus")]) -> None:
    """Run the four validation filters and report drop rates."""
    _not_implemented(5, "evalset validate")


@evalset_app.command("stats")
def evalset_stats(corpus: Annotated[str, typer.Option("--corpus")]) -> None:
    """Difficulty distribution and split sizes."""
    _not_implemented(5, "evalset stats")


@evalset_app.command("review")
def evalset_review(
    corpus: Annotated[str, typer.Option("--corpus")],
    n: Annotated[int, typer.Option("--n")] = 25,
) -> None:
    """Emit a human review document. Do not skip this."""
    _not_implemented(5, "evalset review")


@experiment_app.command("run")
def experiment_run(
    config_path: Annotated[str, typer.Option("--config")],
    workers: Annotated[int, typer.Option("--workers")] = 1,
    force: Annotated[bool, typer.Option("--force")] = False,
) -> None:
    """Expand and execute the strategy matrix."""
    _not_implemented(6, "experiment run")


@experiment_app.command("report")
def experiment_report(
    run_id: Annotated[str, typer.Option("--run-id")],
    fmt: Annotated[str, typer.Option("--format")] = "table",
) -> None:
    """Metrics table with bootstrap confidence intervals."""
    _not_implemented(6, "experiment report")


@experiment_app.command("compare")
def experiment_compare(run_ids: Annotated[str, typer.Option("--run-ids")]) -> None:
    """Compare two or more runs."""
    _not_implemented(6, "experiment compare")


@experiment_app.command("failures")
def experiment_failures(
    run_id: Annotated[str, typer.Option("--run-id")],
    n: Annotated[int, typer.Option("--n")] = 20,
) -> None:
    """Dump the worst-performing queries with retrieved chunks and gold spans."""
    _not_implemented(6, "experiment failures")


@agent_app.command("route")
def agent_route(
    query: Annotated[str, typer.Option("--query")],
    corpus: Annotated[str, typer.Option("--corpus")],
    explain: Annotated[bool, typer.Option("--explain")] = False,
) -> None:
    """Pick a retrieval strategy for one query and justify the choice."""
    _not_implemented(8, "agent route")


@agent_app.command("optimize")
def agent_optimize(
    corpus: Annotated[str, typer.Option("--corpus")],
    max_iterations: Annotated[int, typer.Option("--max-iterations")] = 6,
) -> None:
    """Propose, evaluate, diagnose, mutate — on the dev split only."""
    _not_implemented(8, "agent optimize")


@agent_app.command("trace")
def agent_trace(run_id: Annotated[str, typer.Option("--run-id")]) -> None:
    """Print an optimizer reasoning trace."""
    _not_implemented(8, "agent trace")


def entrypoint() -> None:
    try:
        app()
    except Exception as exc:  # unhandled -> exit 2, per the CLI contract
        console.print_exception()
        console.print(f"[red]unhandled error:[/red] {exc}")
        sys.exit(2)


if __name__ == "__main__":
    entrypoint()
