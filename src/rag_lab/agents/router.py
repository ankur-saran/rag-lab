"""Router agent (plan §Phase 8, Step 8.1): picks a retriever + index for one
query and justifies the choice, using ``agents/runtime.py``'s tool-use loop.

``route_query`` is the one entry point both callers use:

- ``rag-lab agent route`` (``cli.py``) calls it directly for one ad-hoc query.
- ``retrievers/agent_router.py``'s ``AgentRouterRetriever`` calls it from
  inside the Phase 6 matrix, so the router is evaluated with the exact same
  metrics/reporting pipeline as ``dense``/``bm25``/``hybrid``.

Both paths return a ``RouterDecision`` (the reasoning trace) and the
``ScoredChunk``s from whichever ``retrieve`` tool call the agent made last --
the loop's own working assumption is that the agent's *last* retrieval is its
final answer, consistent with "may retrieve, judge insufficient, try a
different strategy" (plan text).
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any

from rich.console import Console
from rich.markup import escape
from rich.table import Table

from rag_lab import indexing
from rag_lab.agents.runtime import (
    Budget,
    ModelCaller,
    ModelTurn,
    Tool,
    ToolCall,
    ToolError,
    anthropic_model_caller,
    run_agent_loop,
)
from rag_lab.corpus import compute_corpus_stats, list_documents_by_corpus
from rag_lab.retrieval import run_retriever
from rag_lab.retrievers import available_retrievers, truncate_and_rank
from rag_lab.schemas import IndexManifest, RouterDecision, ScoredChunk

DEFAULT_MODEL = "claude-sonnet-5"
DEFAULT_MAX_STEPS = 5

# What counts as "an identifier-like token" for the mock caller's heuristic
# (AC-2: BM25 for an identifier query, dense for a conceptual one) -- the same
# shape of token `verify-phase-4`'s own real query already exercises
# (`IDEMPOTENCY_KEY_CONFLICT`): all-caps words joined by underscores.
_IDENTIFIER_RE = re.compile(r"\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+\b")


@dataclass
class _RouteSession:
    """Mutable state the tool handlers close over. ``last_retrieve`` is the
    whole point of this class: the model only sees a JSON preview of each
    retrieval's results (via the tool_result block), but the caller needs the
    real ``ScoredChunk`` objects back -- this is where they're kept."""

    corpus: str
    query: str
    k: int
    exclude_index_id: str | None
    last_retrieve: tuple[str, str, dict[str, Any], list[ScoredChunk]] | None = field(default=None)


def _preview(text: str, width: int = 160) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= width else flat[: width - 1] + "…"


def _build_tools(session: _RouteSession) -> list[Tool]:
    def _list_indexes(args: dict[str, Any]) -> list[dict[str, Any]]:
        target = args.get("corpus", session.corpus)
        manifests = [
            m
            for m in indexing.list_manifests(corpus=target)
            if m.index_id != session.exclude_index_id
        ]
        return [
            {
                "index_id": m.index_id,
                "chunk_set_id": m.chunk_set_id,
                "embedder": m.embedder,
                "vector_count": m.vector_count,
                "dim": m.dim,
            }
            for m in manifests
        ]

    def _get_corpus_stats(args: dict[str, Any]) -> dict[str, Any]:
        target = args.get("corpus", session.corpus)
        try:
            docs = list_documents_by_corpus(target)[target]
        except LookupError as exc:
            raise ToolError(str(exc)) from exc
        stats = compute_corpus_stats(target, docs)
        return {
            "doc_count": stats.doc_count,
            "total_tokens": stats.total_tokens,
            "token_p50": stats.token_p50,
            "token_p95": stats.token_p95,
            "headings_per_1k_tokens": stats.headings_per_1k_tokens,
            "code_block_count": stats.code_block_count,
            "table_count": stats.table_count,
            "lang_mix": dict(stats.lang_mix),
        }

    def _retrieve(args: dict[str, Any]) -> dict[str, Any]:
        index_id = args["index_id"]
        retriever_name = args["retriever"]
        query = args.get("query") or session.query
        k = int(args.get("k") or session.k)
        if retriever_name not in available_retrievers():
            raise ToolError(
                f"unknown retriever {retriever_name!r}; available: {available_retrievers()}"
            )
        try:
            manifest, results = run_retriever(index_id, retriever_name, query, k)
        except (LookupError, FileNotFoundError, ValueError) as exc:
            raise ToolError(str(exc)) from exc
        session.last_retrieve = (manifest.index_id, retriever_name, {}, results)
        return {
            "index_id": manifest.index_id,
            "retriever": retriever_name,
            "results": [
                {
                    "chunk_id": r.chunk.chunk_id,
                    "rank": r.rank,
                    "score": r.score,
                    "preview": _preview(r.chunk.text),
                }
                for r in results
            ],
        }

    def _inspect_chunk(args: dict[str, Any]) -> dict[str, Any]:
        chunk_id = args["chunk_id"]
        for m in indexing.list_manifests(corpus=session.corpus):
            store = indexing.load_store(m.index_id)
            found = store.get([chunk_id])
            if found:
                c = found[0]
                return {
                    "chunk_id": c.chunk_id,
                    "text": c.text,
                    "heading_path": c.heading_path,
                    "token_count": c.token_count,
                    "chunker": c.chunker,
                    "index_id": m.index_id,
                }
        raise ToolError(
            f"chunk_id {chunk_id!r} not found in any index for corpus {session.corpus!r}"
        )

    return [
        Tool(
            name="list_available_indexes",
            description="List every built index for a corpus (index_id, chunk_set_id, embedder).",
            input_schema={"type": "object", "properties": {"corpus": {"type": "string"}}},
            handler=_list_indexes,
        ),
        Tool(
            name="get_corpus_stats",
            description="Token distribution, heading density, code/table counts, language mix.",
            input_schema={"type": "object", "properties": {"corpus": {"type": "string"}}},
            handler=_get_corpus_stats,
        ),
        Tool(
            name="retrieve",
            description="Run one retriever against one index for the query; return scored results.",
            input_schema={
                "type": "object",
                "properties": {
                    "index_id": {"type": "string"},
                    "retriever": {"type": "string"},
                    "query": {"type": "string"},
                    "k": {"type": "integer"},
                },
                "required": ["index_id", "retriever"],
            },
            handler=_retrieve,
        ),
        Tool(
            name="inspect_chunk",
            description="Look up one chunk's full text and metadata by chunk_id.",
            input_schema={
                "type": "object",
                "properties": {"chunk_id": {"type": "string"}},
                "required": ["chunk_id"],
            },
            handler=_inspect_chunk,
        ),
    ]


def _system_prompt(corpus: str, k: int, candidates: list[IndexManifest]) -> str:
    index_lines = "\n".join(
        f"- {m.index_id} (chunk_set={m.chunk_set_id}, embedder={m.embedder}, "
        f"vectors={m.vector_count})"
        for m in candidates
    )
    return (
        "You are the routing agent for a retrieval-augmented generation framework. "
        f"Given one user query against the {corpus!r} corpus, choose the single best "
        f"retrieval strategy and return the top {k} results.\n\n"
        "Available indexes for this corpus:\n"
        f"{index_lines}\n\n"
        "Use list_available_indexes/get_corpus_stats to orient yourself if useful, then call "
        "retrieve one or more times to try strategies (dense favors conceptual/paraphrased "
        "queries; bm25 favors exact identifiers, codes, and rare proper nouns; hybrid fuses "
        "both). Your LAST retrieve call is taken as your final answer. When you are satisfied, "
        "stop calling tools and reply with a one-paragraph justification of your final choice."
    )


def _mock_model_caller(session: _RouteSession, candidates: list[IndexManifest]) -> ModelCaller:
    """Deterministic stand-in for a real Claude call: no network, no
    ``anthropic`` import. Picks ``bm25`` when the query contains an
    identifier-like token, ``dense`` otherwise -- exactly AC-2's expected
    behavior -- against the first available index for the corpus.
    """
    target = candidates[0]
    chosen = "bm25" if _IDENTIFIER_RE.search(session.query) else "dense"
    step = {"n": 0}

    def _call(messages: list[dict[str, Any]], system: str, tools: list[Tool]) -> ModelTurn:
        step["n"] += 1
        if step["n"] == 1:
            return ModelTurn(
                text="",
                tool_calls=[
                    ToolCall(
                        id="mock-retrieve-1",
                        name="retrieve",
                        arguments={
                            "index_id": target.index_id,
                            "retriever": chosen,
                            "query": session.query,
                            "k": session.k,
                        },
                    )
                ],
                input_tokens=120,
                output_tokens=40,
            )
        reason = (
            "the query names a distinctive identifier-like token"
            if chosen == "bm25"
            else "the query is conceptual/natural-language, with no exact identifier to match on"
        )
        return ModelTurn(
            text=f"[mock] Chose {chosen} on {target.index_id} because {reason}.",
            tool_calls=[],
            input_tokens=60,
            output_tokens=30,
        )

    return _call


def route_query(
    query: str,
    corpus: str,
    k: int = 5,
    *,
    model: str = DEFAULT_MODEL,
    max_steps: int = DEFAULT_MAX_STEPS,
    mock: bool = False,
    exclude_index_id: str | None = None,
    budget: Budget | None = None,
) -> tuple[RouterDecision, list[ScoredChunk]]:
    """Route one query for ``corpus`` and return ``(decision, results)``.

    ``exclude_index_id`` is what makes this safe to call from inside the
    Phase 6 matrix: ``AgentRouterRetriever`` passes its own placeholder
    index's id here, so the precondition below ("is there anything to route
    across") never counts the router's own bookkeeping index as a real
    baseline option (plan §Phase 8, Step 8.1's sequencing requirement, AC-6).

    Raises ``LookupError`` before ever invoking the model if no index is
    available to route across -- a clear, named error rather than a router
    that silently retrieves from nothing.
    """
    candidates = [
        m for m in indexing.list_manifests(corpus=corpus) if m.index_id != exclude_index_id
    ]
    if not candidates:
        suffix = f" (excluding its own index {exclude_index_id!r})" if exclude_index_id else ""
        raise LookupError(
            f"no indexes available to route across for corpus {corpus!r}{suffix} -- "
            "build the baseline matrix first (plan §Phase 8, Step 8.1's sequencing requirement)"
        )

    session = _RouteSession(corpus=corpus, query=query, k=k, exclude_index_id=exclude_index_id)
    tools = _build_tools(session)
    caller = (
        _mock_model_caller(session, candidates) if mock else anthropic_model_caller(model=model)
    )

    started = time.monotonic()
    result = run_agent_loop(
        system=_system_prompt(corpus, k, candidates),
        user_message=query,
        tools=tools,
        model_caller=caller,
        model=model,
        budget=budget or Budget(max_steps=max_steps),
    )
    latency_ms = (time.monotonic() - started) * 1000.0

    if session.last_retrieve is None:
        raise LookupError(
            f"router agent for corpus {corpus!r} never called retrieve() before stopping "
            f"(stopped_reason={result.stopped_reason!r}) -- nothing to return"
        )

    chosen_index_id, chosen_retriever, chosen_params, chunks = session.last_retrieve
    decision = RouterDecision(
        query=query,
        corpus=corpus,
        chosen_index_id=chosen_index_id,
        chosen_retriever=chosen_retriever,
        chosen_retriever_params=chosen_params,
        justification=result.final_text or f"(agent stopped: {result.stopped_reason})",
        steps=result.transcript,
        total_input_tokens=result.total_input_tokens,
        total_output_tokens=result.total_output_tokens,
        latency_ms=latency_ms,
    )
    labeled = truncate_and_rank(chunks, k, "agent_router")
    return decision, labeled


def render_decision(decision: RouterDecision, console: Console, *, explain: bool = False) -> None:
    console.print(
        f"[green]chosen[/green]  retriever={escape(decision.chosen_retriever)}  "
        f"index={escape(decision.chosen_index_id)}"
    )
    console.print(f"[bold]justification:[/bold] {escape(decision.justification)}")
    console.print(
        f"tokens: {decision.total_input_tokens} in / {decision.total_output_tokens} out   "
        f"latency: {decision.latency_ms:.0f}ms"
    )
    if explain:
        table = Table(title="router steps", title_style="bold", show_lines=True)
        table.add_column("step", justify="right")
        table.add_column("text", overflow="fold")
        table.add_column("tool calls", overflow="fold")
        for s in decision.steps:
            calls = "; ".join(f"{c['name']}({c['arguments']})" for c in s.get("tool_calls", []))
            table.add_row(str(s["step"]), escape(s.get("text", "")), escape(calls))
        console.print(table)


__all__ = ["DEFAULT_MAX_STEPS", "DEFAULT_MODEL", "render_decision", "route_query"]
