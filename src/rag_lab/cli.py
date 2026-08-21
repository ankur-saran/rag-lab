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
    config: Annotated[str | None, typer.Option("--config", "-c", help="YAML config file.")] = None,
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
    from rag_lab.jsonl import write_jsonl
    from rag_lab.loaders import discover_corpora, load_corpus
    from rag_lab.paths import artifact_path, corpora_dir

    if corpus and all_:
        console.print("[red]error:[/red] pass --corpus or --all, not both")
        raise typer.Exit(code=1)
    if not corpus and not all_:
        console.print("[red]error:[/red] pass --corpus <name> or --all")
        raise typer.Exit(code=1)

    available = discover_corpora()
    if not available:
        console.print(f"[red]error:[/red] no corpus directories found under {corpora_dir()}")
        raise typer.Exit(code=1)

    targets = available if all_ else [corpus]
    unknown = [t for t in targets if t not in available]
    if unknown:
        console.print(
            f"[red]error:[/red] unknown corpus/corpora: {', '.join(unknown)}. "
            f"available: {', '.join(available)}"
        )
        raise typer.Exit(code=1)

    # Per-corpus failures (a bad corpus) are expected/reportable — exit 1, not
    # the exit-2 traceback path reserved for real bugs.
    successes: list[tuple[str, int]] = []
    failures: list[tuple[str, str]] = []
    for name in targets:
        try:
            docs = load_corpus(name, corpora_dir() / name)
            if not docs:
                raise ValueError(f"corpus {name!r} produced zero documents")
            count = write_jsonl(artifact_path("documents", f"{name}.jsonl"), docs)
            successes.append((name, count))
        except Exception as exc:
            failures.append((name, str(exc)))

    for name, count in successes:
        console.print(f"[green]ok[/green]  {name}: {count} documents")
    for name, message in failures:
        console.print(f"[red]fail[/red]  {name}: {message}")

    raise typer.Exit(code=1 if failures else 0)


@corpus_app.command("stats")
def corpus_stats(corpus: Annotated[str | None, typer.Option("--corpus")] = None) -> None:
    """Token distribution, heading density, code blocks, tables, language mix."""
    from rag_lab.corpus import compute_corpus_stats, list_documents_by_corpus, render_stats_table

    try:
        by_corpus = list_documents_by_corpus(corpus)
    except LookupError as exc:
        console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    try:
        stats = [compute_corpus_stats(name, docs) for name, docs in sorted(by_corpus.items())]
    except Exception as exc:  # e.g. tiktoken's encoding file failing to download offline
        console.print(f"[red]error:[/red] could not compute stats: {exc}")
        raise typer.Exit(code=1) from exc

    render_stats_table(stats, console)


@corpus_app.command("show")
def corpus_show(
    doc_id: Annotated[str, typer.Option("--doc-id")],
    chars: Annotated[int, typer.Option("--chars")] = 500,
) -> None:
    """Print the head of a document."""
    from rich.markup import escape

    from rag_lab.corpus import find_document

    doc = find_document(doc_id)
    if doc is None:
        console.print(f"[red]error:[/red] no document with doc_id {doc_id!r}")
        raise typer.Exit(code=1)

    console.print(f"[bold]{escape(doc.title)}[/bold]  ({doc.doc_id})")
    console.print(
        f"corpus={escape(doc.corpus)}  content_type={doc.content_type}  "
        f"source={escape(doc.source_path)}"
    )
    console.print(escape(doc.text[:chars]))
    if len(doc.text) > chars:
        console.print(f"[dim]... truncated, {len(doc.text) - chars} more characters[/dim]")


@chunk_app.command("run")
def chunk_run(
    corpus: Annotated[str, typer.Option("--corpus")],
    chunker: Annotated[str, typer.Option("--chunker")],
    params: Annotated[list[str] | None, typer.Option("--params")] = None,
    role: Annotated[str, typer.Option("--role")] = "standalone",
    parent_chunk_set: Annotated[str | None, typer.Option("--parent-chunk-set")] = None,
    mock_llm: Annotated[bool, typer.Option("--mock-llm")] = False,
) -> None:
    """Chunk a corpus into artifacts/chunks/<chunk_set_id>.jsonl.

    ``--role parent``/``--role child`` build the hierarchical chunk-set pairs
    Phase 4's ``parent_doc`` retriever needs (plan §4.7). A child set's
    ``--parent-chunk-set`` is folded into the hashed params so two child sets
    built against different parents never collide on the same chunk_set_id.

    ``--mock-llm`` (Phase 7's ``table_summary`` only) makes no network call and
    never imports ``anthropic``; it sets the ``mock_llm`` chunker param, which
    is hashed into the chunk_set_id like any other param -- a mock run and a
    real run produce different summary text and must never share a cache entry.

    Delegates the actual build (and its caching) to
    ``chunkers.build_chunk_set`` -- the same function Phase 6's experiment
    runner uses, so a chunk set built via either path is byte-identical.
    """
    from rag_lab.chunkers import REGISTRY, OrphanChildError, build_chunk_set

    if role not in {"standalone", "parent", "child"}:
        console.print(
            f"[red]error:[/red] --role must be one of standalone|parent|child, got {role!r}"
        )
        raise typer.Exit(code=1)
    if role == "child" and not parent_chunk_set:
        console.print("[red]error:[/red] --role child requires --parent-chunk-set <chunk_set_id>")
        raise typer.Exit(code=1)
    if role != "child" and parent_chunk_set:
        console.print("[red]error:[/red] --parent-chunk-set only applies to --role child")
        raise typer.Exit(code=1)

    if chunker not in REGISTRY:
        console.print(
            f"[red]error:[/red] unknown chunker {chunker!r}. "
            f"available: {', '.join(sorted(REGISTRY))}"
        )
        raise typer.Exit(code=1)
    if mock_llm and chunker != "table_summary":
        console.print("[red]error:[/red] --mock-llm only applies to --chunker table_summary")
        raise typer.Exit(code=1)

    overrides = parse_overrides(params)
    if mock_llm:
        overrides["mock_llm"] = True

    try:
        chunk_set_id, all_chunks, _cache_hit = build_chunk_set(
            corpus, chunker, overrides, role=role, parent_chunk_set_id=parent_chunk_set
        )
    except (LookupError, FileNotFoundError, ValueError, OrphanChildError) as exc:
        console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    doc_count = len({c.doc_id for c in all_chunks})
    console.print(
        f"[green]ok[/green]  {chunk_set_id}: {len(all_chunks)} chunks from {doc_count} documents"
    )


@chunk_app.command("stats")
def chunk_stats(chunk_set: Annotated[str, typer.Option("--chunk-set")]) -> None:
    """Token distribution, split code blocks, split tables, orphan rate."""
    from rag_lab.chunks import (
        compute_chunk_stats,
        documents_for_chunks,
        load_chunk_set,
        render_chunk_stats_table,
    )

    try:
        chunks = load_chunk_set(chunk_set)
        docs_by_id = documents_for_chunks(chunks)
    except (LookupError, FileNotFoundError) as exc:
        console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    try:
        stats = compute_chunk_stats(chunks, docs_by_id)
    except Exception as exc:
        console.print(f"[red]error:[/red] could not compute stats: {exc}")
        raise typer.Exit(code=1) from exc

    render_chunk_stats_table(stats, console)


@chunk_app.command("show")
def chunk_show(
    chunk_set: Annotated[str, typer.Option("--chunk-set")],
    doc_id: Annotated[str, typer.Option("--doc-id")],
) -> None:
    """Render a document with chunk boundaries as rules in the terminal."""
    from rag_lab.chunks import (
        documents_for_chunks,
        find_document_in,
        load_chunk_set,
        render_chunk_boundaries,
    )

    try:
        chunks = load_chunk_set(chunk_set)
        doc = find_document_in(documents_for_chunks(chunks), doc_id)
    except (LookupError, FileNotFoundError) as exc:
        console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    doc_chunks = [c for c in chunks if c.doc_id == doc_id]
    render_chunk_boundaries(doc, doc_chunks, console)


@chunk_app.command("diff")
def chunk_diff(
    a: Annotated[str, typer.Option("--a")],
    b: Annotated[str, typer.Option("--b")],
    doc_id: Annotated[str, typer.Option("--doc-id")],
) -> None:
    """Compare two chunk sets' boundaries on the same document, side by side."""
    from rag_lab.chunks import (
        documents_for_chunks,
        find_document_in,
        load_chunk_set,
        render_chunk_diff,
    )

    try:
        chunks_a = load_chunk_set(a)
        chunks_b = load_chunk_set(b)
        doc = find_document_in(documents_for_chunks(chunks_a), doc_id)
    except (LookupError, FileNotFoundError) as exc:
        console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    a_chunks = [c for c in chunks_a if c.doc_id == doc_id]
    b_chunks = [c for c in chunks_b if c.doc_id == doc_id]
    if not a_chunks and not b_chunks:
        console.print(f"[red]error:[/red] doc_id {doc_id!r} has no chunks in either chunk set")
        raise typer.Exit(code=1)
    render_chunk_diff(doc, a_chunks, b_chunks, console, a, b)


@index_app.command("build")
def index_build(
    chunk_set: Annotated[str, typer.Option("--chunk-set")],
    embedder: Annotated[str, typer.Option("--embedder")] = "bge-small",
    truncate_dim: Annotated[int | None, typer.Option("--truncate-dim")] = None,
    params: Annotated[list[str] | None, typer.Option("--params")] = None,
    force: Annotated[bool, typer.Option("--force")] = False,
) -> None:
    """Embed a chunk set and persist a vector index."""
    from rag_lab.embedders import available_embedders
    from rag_lab.indexing import build_index

    if embedder not in available_embedders():
        console.print(
            f"[red]error:[/red] unknown embedder {embedder!r}. "
            f"available: {', '.join(available_embedders())}"
        )
        raise typer.Exit(code=1)

    overrides = parse_overrides(params)
    if truncate_dim is not None:
        overrides["truncate_dim"] = truncate_dim

    try:
        manifest, cache_hit = build_index(chunk_set, embedder, overrides, force=force)
    except (LookupError, FileNotFoundError, ValueError) as exc:
        console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    status = "cache hit" if cache_hit else "built"
    console.print(
        f"[green]ok[/green]  {manifest.index_id}: {manifest.vector_count} vectors, "
        f"dim={manifest.dim} ({status})"
    )


@index_app.command("list")
def index_list() -> None:
    """List built indexes and their manifests."""
    from rag_lab.indexing import list_manifests, render_index_list_table

    manifests = list_manifests()
    if not manifests:
        console.print("[yellow]no indexes built yet[/yellow] — run `rag-lab index build`")
        raise typer.Exit(code=0)
    render_index_list_table(manifests, console)


@index_app.command("search")
def index_search(
    index_id: Annotated[str, typer.Option("--index-id")],
    query: Annotated[str, typer.Option("--query")],
    k: Annotated[int, typer.Option("--k")] = 5,
) -> None:
    """Ad-hoc semantic search against an index."""
    from rag_lab.indexing import render_search_results, search_index

    try:
        _manifest, results = search_index(index_id, query, k)
    except (LookupError, FileNotFoundError, ValueError) as exc:
        console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    if not results:
        console.print("[yellow]no results[/yellow]")
        raise typer.Exit(code=0)
    render_search_results(query, results, console)


@retrieve_app.command("query")
def retrieve_query(
    index_id: Annotated[str, typer.Option("--index-id")],
    query: Annotated[str, typer.Option("--query")],
    retriever: Annotated[str, typer.Option("--retriever")] = "dense",
    k: Annotated[int, typer.Option("--k")] = 5,
    params: Annotated[list[str] | None, typer.Option("--params")] = None,
    parent_chunk_set: Annotated[str | None, typer.Option("--parent-chunk-set")] = None,
) -> None:
    """Run one retriever."""
    from rag_lab.indexing import render_search_results
    from rag_lab.retrieval import run_retriever
    from rag_lab.retrievers import available_retrievers

    if retriever not in available_retrievers():
        console.print(
            f"[red]error:[/red] unknown retriever {retriever!r}. "
            f"available: {', '.join(available_retrievers())}"
        )
        raise typer.Exit(code=1)

    overrides = parse_overrides(params)
    if retriever == "parent_doc":
        if not parent_chunk_set:
            console.print(
                "[red]error:[/red] --retriever parent_doc requires "
                "--parent-chunk-set <chunk_set_id>"
            )
            raise typer.Exit(code=1)
        overrides["parent_chunk_set_id"] = parent_chunk_set

    try:
        _manifest, results = run_retriever(index_id, retriever, query, k, overrides)
    except (LookupError, FileNotFoundError, ValueError) as exc:
        console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    if not results:
        console.print("[yellow]no results[/yellow]")
        raise typer.Exit(code=0)
    render_search_results(query, results, console)


@retrieve_app.command("compare")
def retrieve_compare(
    index_id: Annotated[str, typer.Option("--index-id")],
    query: Annotated[str, typer.Option("--query")],
    retrievers: Annotated[str, typer.Option("--retrievers")] = "dense,bm25,hybrid",
    k: Annotated[int, typer.Option("--k")] = 5,
    params: Annotated[list[str] | None, typer.Option("--params")] = None,
    parent_chunk_set: Annotated[str | None, typer.Option("--parent-chunk-set")] = None,
) -> None:
    """Run several retrievers side by side.

    ``--params`` is namespaced per retriever (unlike ``retrieve query``'s flat
    ``--params``, since this command builds several retrievers at once), e.g.
    ``--params hybrid.k_rrf=30 --params hybrid.weights.dense=2.0``.
    """
    from rag_lab.retrieval import compare_retrievers, render_compare_table
    from rag_lab.retrievers import available_retrievers

    names = [n.strip() for n in retrievers.split(",") if n.strip()]
    unknown = [n for n in names if n not in available_retrievers()]
    if unknown:
        console.print(
            f"[red]error:[/red] unknown retriever(s) {unknown}. "
            f"available: {', '.join(available_retrievers())}"
        )
        raise typer.Exit(code=1)

    overrides_by_name = parse_overrides(params)
    if parent_chunk_set:
        overrides_by_name.setdefault("parent_doc", {})["parent_chunk_set_id"] = parent_chunk_set
    if "parent_doc" in names and "parent_chunk_set_id" not in overrides_by_name.get(
        "parent_doc", {}
    ):
        console.print(
            "[red]error:[/red] --retrievers includes parent_doc, which requires "
            "--parent-chunk-set <chunk_set_id>"
        )
        raise typer.Exit(code=1)

    try:
        _manifest, results_by_name = compare_retrievers(
            index_id, names, query, k, overrides_by_name
        )
    except (LookupError, FileNotFoundError, ValueError) as exc:
        console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    render_compare_table(query, results_by_name, k, console)


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
    run_id: Annotated[str | None, typer.Option("--run-id")] = None,
    workers: Annotated[int, typer.Option("--workers")] = 1,
    force: Annotated[bool, typer.Option("--force")] = False,
) -> None:
    """Expand and execute the strategy matrix.

    Omit ``--run-id`` to start a fresh run (a timestamped id is minted and
    printed at the start -- capture it). Pass that id back in via ``--run-id``
    to resume an interrupted run: completed cells are skipped unless
    ``--force``.
    """
    from rag_lab.experiment import load_experiment_config, run_experiment

    try:
        config = load_experiment_config(config_path)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    try:
        resolved_run_id, results = run_experiment(
            config, run_id=run_id, workers=workers, force=force, console=console
        )
    except (LookupError, FileNotFoundError, ValueError) as exc:
        console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    console.print(f"[green]ok[/green]  {resolved_run_id}: {len(results)} cell(s) computed")


@experiment_app.command("report")
def experiment_report(
    run_id: Annotated[str, typer.Option("--run-id")],
    fmt: Annotated[str, typer.Option("--format")] = "table",
) -> None:
    """Metrics table with bootstrap confidence intervals."""
    from rag_lab.experiment import load_run, render_report_markdown, render_report_table

    if fmt not in {"table", "markdown"}:
        console.print(f"[red]error:[/red] --format must be table|markdown, got {fmt!r}")
        raise typer.Exit(code=1)

    try:
        _config, results = load_run(run_id)
    except LookupError as exc:
        console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    if fmt == "markdown":
        print(render_report_markdown(results))
    else:
        render_report_table(results, console)


@experiment_app.command("compare")
def experiment_compare(run_ids: Annotated[str, typer.Option("--run-ids")]) -> None:
    """Compare two or more runs."""
    from rag_lab.experiment import compare_runs, render_compare_table

    ids = [r.strip() for r in run_ids.split(",") if r.strip()]
    if len(ids) < 2:
        console.print("[red]error:[/red] --run-ids needs at least two comma-separated run ids")
        raise typer.Exit(code=1)

    try:
        comparison = compare_runs(ids)
    except LookupError as exc:
        console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    render_compare_table(ids, comparison, console)


@experiment_app.command("failures")
def experiment_failures(
    run_id: Annotated[str, typer.Option("--run-id")],
    config_idx: Annotated[int, typer.Option("--config-idx")] = 0,
    n: Annotated[int, typer.Option("--n")] = 20,
) -> None:
    """Dump the worst-performing queries with retrieved chunks and gold spans."""
    from rag_lab.experiment import render_failures, worst_failures

    try:
        cell, _result, worst = worst_failures(run_id, config_idx, n)
    except (LookupError, ValueError) as exc:
        console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    render_failures(cell, worst, console)


@agent_app.command("route")
def agent_route(
    query: Annotated[str, typer.Option("--query")],
    corpus: Annotated[str, typer.Option("--corpus")],
    k: Annotated[int, typer.Option("--k")] = 5,
    explain: Annotated[bool, typer.Option("--explain")] = False,
    model: Annotated[str, typer.Option("--model")] = "claude-sonnet-5",
    max_steps: Annotated[int, typer.Option("--max-steps")] = 5,
    mock_llm: Annotated[bool, typer.Option("--mock-llm")] = False,
) -> None:
    """Pick a retrieval strategy for one query and justify the choice."""
    from rag_lab.agents.router import render_decision, route_query
    from rag_lab.indexing import render_search_results

    try:
        decision, results = route_query(
            query, corpus, k, model=model, max_steps=max_steps, mock=mock_llm
        )
    except (LookupError, FileNotFoundError, ValueError) as exc:
        console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    render_decision(decision, console, explain=explain)
    if not results:
        console.print("[yellow]no results[/yellow]")
    else:
        render_search_results(query, results, console)


@agent_app.command("optimize")
def agent_optimize(
    corpus: Annotated[str, typer.Option("--corpus")],
    max_iterations: Annotated[int, typer.Option("--max-iterations")] = 6,
    k: Annotated[int, typer.Option("--k")] = 10,
    model: Annotated[str, typer.Option("--model")] = "claude-sonnet-5",
    max_steps_per_iteration: Annotated[int, typer.Option("--max-steps-per-iteration")] = 4,
    budget_usd: Annotated[float | None, typer.Option("--budget-usd")] = None,
    max_tokens: Annotated[int | None, typer.Option("--max-tokens")] = None,
    max_wall_clock: Annotated[float | None, typer.Option("--max-wall-clock")] = None,
    mock_llm: Annotated[bool, typer.Option("--mock-llm")] = False,
) -> None:
    """Propose, evaluate, diagnose, mutate — on the dev split only."""
    from rag_lab.agents.optimizer import optimize, render_trace

    try:
        run_id, trace = optimize(
            corpus,
            k=k,
            max_iterations=max_iterations,
            max_steps_per_iteration=max_steps_per_iteration,
            model=model,
            mock=mock_llm,
            max_tokens=max_tokens,
            max_wall_clock_s=max_wall_clock,
            budget_usd=budget_usd,
        )
    except (LookupError, FileNotFoundError, ValueError) as exc:
        console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    console.print(f"[green]ok[/green]  {run_id}: {len(trace)} trace entrie(s)")
    if not trace:
        console.print(
            "[yellow]budget exhausted before any iteration completed[/yellow] "
            "-- raise --budget-usd/--max-tokens/--max-wall-clock"
        )
        raise typer.Exit(code=0)
    render_trace(trace, console)


@agent_app.command("trace")
def agent_trace(run_id: Annotated[str, typer.Option("--run-id")]) -> None:
    """Print an optimizer reasoning trace."""
    from rag_lab.agents.optimizer import load_trace, render_trace

    try:
        trace = load_trace(run_id)
    except LookupError as exc:
        console.print(f"[red]error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    if not trace:
        console.print("[yellow]trace is empty[/yellow] -- optimizer stopped before any iteration completed")
        raise typer.Exit(code=0)
    render_trace(trace, console)


def entrypoint() -> None:
    try:
        app()
    except Exception as exc:  # unhandled -> exit 2, per the CLI contract
        console.print_exception()
        console.print(f"[red]unhandled error:[/red] {exc}")
        sys.exit(2)


if __name__ == "__main__":
    entrypoint()
