#!/usr/bin/env python
"""Generate the committed fixtures from fixtures/raw/.

Fixtures are what make every phase independently runnable, so they must be
*valid* rather than merely plausible. Generating them programmatically from the
raw sources guarantees the offset invariant holds — a hand-written chunk fixture
with an off-by-one span would break Phase 2's property tests for reasons that
look like a chunker bug.

The chunker used here is a deliberately trivial paragraph splitter, local to this
script. It is not a Phase 2 strategy and must not be imported as one.

Usage:
    python scripts/build_fixtures.py            # write fixtures
    python scripts/build_fixtures.py --check    # verify determinism, write nothing
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag_lab.chunkers.hierarchy import assign_parents  # noqa: E402
from rag_lab.experiment.config import (  # noqa: E402
    Cell,
    ExperimentConfig,
    MatrixComponentSpec,
    MatrixSpec,
    expand_cells,
)
from rag_lab.ids import (  # noqa: E402
    chunker_signature,
    make_chunk_id,
    make_chunk_set_id,
    make_doc_id,
    make_query_id,
    split_for,
)
from rag_lab.jsonl import write_jsonl  # noqa: E402
from rag_lab.normalize import normalize_text as normalize  # noqa: E402
from rag_lab.schemas import (  # noqa: E402
    Chunk,
    ChunkRole,
    Document,
    EvalPair,
    OptimizerTraceEntry,
    QueryTrace,
    RunResult,
)

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "fixtures" / "raw"
OUT = ROOT / "fixtures"

FIXTURE_CHUNKER = "paragraph_fixture"
FIXTURE_PARAMS = {"min_chars": 40}

# A coarser merge of the same elementary paragraph spans, used to build the
# committed parent/child fixture pair (fixtures/chunks/sample_parent.jsonl,
# sample_child.jsonl) that Phase 4's ParentDocumentRetriever tests consume.
# Because both this and FIXTURE_PARAMS merge the *same* underlying spans()
# list (see paragraph_spans below) with only the threshold differing, every
# min_chars=40 span is guaranteed to nest inside exactly one min_chars=400
# span -- so assign_parents() below can never produce an orphan.
FIXTURE_PARENT_CHUNKER = "paragraph_fixture_parent"
FIXTURE_PARENT_PARAMS = {"min_chars": 400}


def build_documents() -> list[Document]:
    docs: list[Document] = []
    for path in sorted(RAW.rglob("*.md")):
        rel = path.relative_to(ROOT).as_posix()
        corpus = path.parent.name
        text = normalize(path.read_text(encoding="utf-8"))
        first_line = text.splitlines()[0] if text.strip() else path.stem
        title = first_line.lstrip("#").strip() or path.stem
        docs.append(
            Document(
                doc_id=make_doc_id(corpus, rel),
                corpus=corpus,
                source_path=rel,
                title=title,
                text=text,
                content_type="markdown",
                meta={
                    "char_count": len(text),
                    "heading_count": len(re.findall(r"^#{1,6} ", text, flags=re.M)),
                    "code_block_count": text.count("```") // 2,
                },
            )
        )
    return docs


def paragraph_spans(text: str, min_chars: int) -> list[tuple[int, int]]:
    """Blank-line separated spans, merging runts forward. Offsets are exact."""
    spans: list[tuple[int, int]] = []
    for match in re.finditer(r"[^\n].*?(?=\n\n|\Z)", text, flags=re.S):
        spans.append((match.start(), match.end()))

    merged: list[tuple[int, int]] = []
    for start, end in spans:
        if merged and (end - start) < min_chars:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged


def _build_paragraph_chunks(
    docs: list[Document],
    chunker: str,
    params: dict,
    *,
    role: ChunkRole = "standalone",
) -> list[Chunk]:
    chunks: list[Chunk] = []
    signature = chunker_signature(chunker, params)

    for doc in docs:
        chunk_set = make_chunk_set_id(doc.corpus, chunker, params)
        heading_path: list[str] = []
        for ordinal, (start, end) in enumerate(paragraph_spans(doc.text, params["min_chars"])):
            body = doc.text[start:end]

            heading = re.match(r"^(#{1,6}) (.+)$", body.strip(), flags=re.M)
            if heading and body.strip().startswith("#"):
                level = len(heading.group(1))
                heading_path = heading_path[: level - 1] + [heading.group(2).strip()]

            # embed_text carries the heading path; text stays clean. This is the
            # two-field split the whole framework depends on.
            embed_text = (" > ".join(heading_path) + "\n\n" + body) if heading_path else body

            chunks.append(
                Chunk(
                    chunk_id=make_chunk_id(doc.doc_id, start, end, signature),
                    doc_id=doc.doc_id,
                    corpus=doc.corpus,
                    chunk_set_id=chunk_set,
                    text=body,
                    embed_text=embed_text,
                    char_start=start,
                    char_end=end,
                    token_count=max(1, len(body) // 4),  # approximation; tiktoken lands in Phase 2
                    ordinal=ordinal,
                    role=role,
                    heading_path=list(heading_path),
                    chunker=chunker,
                    chunker_params=dict(params),
                    meta={"approx_tokens": True},
                )
            )
    return chunks


def build_chunks(docs: list[Document]) -> list[Chunk]:
    return _build_paragraph_chunks(docs, FIXTURE_CHUNKER, FIXTURE_PARAMS)


def build_parent_chunks(docs: list[Document]) -> list[Chunk]:
    """A coarser, ``role="parent"`` sibling of ``build_chunks``'s output --
    see the ``FIXTURE_PARENT_PARAMS`` comment for why every child chunk is
    guaranteed to nest inside exactly one of these."""
    return _build_paragraph_chunks(
        docs, FIXTURE_PARENT_CHUNKER, FIXTURE_PARENT_PARAMS, role="parent"
    )


# Queries are hand-written against known substrings so the gold spans are exact
# and independent of any chunker. (needles, question, difficulty) — a tuple of
# *one* needle for lookup/synthesis, or *two or more* for cross_reference: that
# tier's gold is genuinely multiple, often-distant spans in the same document,
# not one span that happens to mention several things, so it needs a distinct
# needle per fact instead of one string covering both.
#
# Needles must not straddle a line break — the raw fixtures are hard-wrapped, so
# a needle spanning a wrap will not be found. Keep them short and single-line.
SEEDS: list[tuple[tuple[str, ...], str, str]] = [
    (("default limit is 25",), "What is the default page size for list endpoints?", "lookup"),
    (("Cursors expire after 24 hours",), "How long is a pagination cursor valid?", "lookup"),
    (
        ("`items`, `next_cursor`",),
        "Which fields does a paginated response envelope contain?",
        "lookup",
    ),
    (
        ("access token valid for one hour",),
        "How long does an access token remain valid after issuance?",
        "lookup",
    ),
    (
        ("10 requests per minute",),
        "What is the rate limit on authentication endpoints?",
        "lookup",
    ),
    (
        ("one and one-half percent (1.5%) per month",),
        "What interest rate applies to overdue undisputed amounts?",
        "lookup",
    ),
    (
        ("disputes in good faith",),
        "Under what conditions does interest not accrue on an invoice?",
        "synthesis",
    ),
    (
        # Two spans, three paragraphs apart (Section 4.3 and Section 5.3, with
        # 5.1/5.2 in between) — a genuine cross_reference example, not a single
        # span that happens to name two things.
        ("disputes in good faith and in writing", "Sections 4, 7, and 9 shall"),
        (
            "What must Customer do to avoid interest accruing on a disputed "
            "invoice, and which sections of the agreement survive its termination?"
        ),
        "cross_reference",
    ),
]


def build_evalset(docs: list[Document], chunks: list[Chunk]) -> list[EvalPair]:
    by_doc: dict[str, list[Chunk]] = {}
    for chunk in chunks:
        by_doc.setdefault(chunk.doc_id, []).append(chunk)

    pairs: list[EvalPair] = []
    for needles, query, difficulty in SEEDS:
        host = next((d for d in docs if all(needle in d.text for needle in needles)), None)
        if host is None:
            raise SystemExit(f"fixture seed not found in any document: {needles!r}")

        spans = [(host.text.index(n), host.text.index(n) + len(n)) for n in needles]

        gold = sorted(
            {
                c.chunk_id
                for span in spans
                for c in by_doc[host.doc_id]
                if c.char_start <= span[0] < c.char_end or c.char_start < span[1] <= c.char_end
            }
        )
        query_id = make_query_id(host.corpus, host.doc_id, spans, query)

        pairs.append(
            EvalPair(
                query_id=query_id,
                corpus=host.corpus,
                query=query,
                gold_doc_id=host.doc_id,
                gold_char_spans=spans,
                sampling_chunk_ids=gold,  # derived; re-resolved per chunk set in Phase 6
                answer=None,
                supporting_quotes=list(needles),
                difficulty=difficulty,  # type: ignore[arg-type]
                split=split_for(query_id),  # type: ignore[arg-type]
                generator_model="handwritten-fixture",
                validated=True,
            )
        )
    return pairs


def _sample_experiment_config() -> ExperimentConfig:
    """The one-cell matrix behind the ``sample_run`` fixture.

    Uses real, registered component names (``fixed``/``bge-small``/``bm25``),
    never the ``FIXTURE_CHUNKER``/``"fixture"`` sentinels used above for
    chunks/evalset -- ``experiment.report.load_run``'s ``expand_cells`` call
    resolves component names through ``chunkers``/``embedders``/``retrievers``
    ``resolve_params``, all three of which raise on an unregistered name (the
    sentinels are deliberately absent from those registries; see
    ``embedders/registry.py``'s own docstring). All three registries' own
    ``resolve_params`` is a pure dict-merge against registry metadata --
    importing them, and calling this, never touches ``sentence_transformers``/
    ``rank_bm25``, so this script stays ``core``-only.
    """
    return ExperimentConfig(
        name="sample_run",
        corpora=["sample"],
        seed=42,
        k=5,
        matrix=MatrixSpec(
            chunker=[MatrixComponentSpec(name="fixed", params={})],
            embedder=[MatrixComponentSpec(name="bge-small", params={})],
            retriever=[MatrixComponentSpec(name="bm25", params={})],
        ),
    )


def build_sample_run(pairs: list[EvalPair]) -> tuple[Cell, RunResult]:
    """The single cell of ``_sample_experiment_config()``, plus its
    ``RunResult``. Returning the ``Cell`` lets ``write_all`` place the result
    at the real ``cells/<cell_id>.json`` path ``experiment.runner``/``report``
    use, alongside ``matrix.json`` -- matching the layout ``report.load_run``
    expects from every real run, so Phase 9's results dashboard can load this
    fixture through the exact same code path as a real run (plan §Phase 9).
    """
    config = _sample_experiment_config()
    cells = expand_cells(config)
    assert len(cells) == 1, f"expected exactly one cell from {config!r}, got {len(cells)}"
    cell = cells[0]

    traces = [
        QueryTrace(
            query_id=p.query_id,
            query=p.query,
            retrieved_chunk_ids=list(p.sampling_chunk_ids),
            retrieved_scores=[0.9 - 0.1 * i for i in range(len(p.sampling_chunk_ids))],
            gold_chunk_ids=list(p.sampling_chunk_ids),
            first_hit_rank=1 if p.sampling_chunk_ids else None,
            metrics={"recall@5": 1.0 if p.sampling_chunk_ids else 0.0},
            latency_ms=4.2,
        )
        for p in pairs
    ]
    result = RunResult(
        run_id="sample_run",
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
        corpus="sample",
        metrics={"recall@1": 1.0, "recall@5": 1.0, "mrr": 1.0, "ndcg@10": 1.0},
        per_query=traces,
        chunk_stats={"count": float(len(traces))},
        timings={"total_s": 0.0},
    )
    return cell, result


def build_sample_optimizer_trace() -> list[OptimizerTraceEntry]:
    """A minimal, hand-authored optimizer trace (plan §Phase 9, Step 9.4).

    ``agents.optimizer.load_trace`` has no fixture fallback of its own -- on a
    clean checkout the optimizer-trace viewer would otherwise be permanently
    blank, which defeats the point of a fixture-only demo path for the one
    page most likely to be shown to an audience without stderr visible. Two
    dev iterations (a real config change and a real metrics delta between
    them) plus the final test-split entry, mirroring the shape
    ``agents/optimizer.py::optimize`` produces for a real run.
    """
    pinned = datetime(2026, 1, 1)
    base = {
        "corpus": "sample",
        "k": 5,
        "embedder": "bge-small",
        "embedder_params": {},
        "retriever": "bm25",
        "retriever_params": {},
    }
    iter0_config = {
        **base,
        "chunker": "fixed",
        "chunker_params": {"chunk_tokens": 512, "overlap_tokens": 64},
    }
    iter1_config = {
        **base,
        "chunker": "recursive",
        "chunker_params": {"chunk_tokens": 512, "overlap_tokens": 64},
    }
    return [
        OptimizerTraceEntry(
            iteration=0,
            hypothesis="Start with a fixed-window baseline to establish a recall floor.",
            config=iter0_config,
            run_id="sample_run__opt_iter0",
            split="dev",
            metrics={"recall@5": 0.60, "mrr": 0.55},
            diagnosis=(
                "Fixed windows cut several clauses mid-sentence, splitting gold "
                "spans across chunk boundaries."
            ),
            mutation="Switch chunker from fixed to recursive at the same token budget.",
            input_tokens=812,
            output_tokens=194,
            created_at=pinned,
        ),
        OptimizerTraceEntry(
            iteration=1,
            hypothesis="Recursive splitting should keep clause boundaries intact.",
            config=iter1_config,
            run_id="sample_run__opt_iter1",
            split="dev",
            metrics={"recall@5": 0.75, "mrr": 0.68},
            diagnosis=(
                "Recall improved; remaining misses are short lookup queries near "
                "document start, likely a chunk-size effect rather than a boundary "
                "effect."
            ),
            mutation="",
            input_tokens=798,
            output_tokens=176,
            created_at=pinned,
        ),
        OptimizerTraceEntry(
            iteration=2,
            hypothesis="Final test-split evaluation of iteration 1's winning config.",
            config=iter1_config,
            run_id="sample_run__opt_iter1",
            split="test",
            metrics={"recall@5": 0.74, "mrr": 0.66},
            diagnosis="",
            mutation="",
            input_tokens=0,
            output_tokens=0,
            created_at=pinned,
        ),
    ]


def verify(docs: list[Document], chunks: list[Chunk]) -> None:
    """Assert the invariants Phase 2 will later test against these fixtures."""
    by_id = {d.doc_id: d for d in docs}
    covered: dict[str, set[int]] = {d.doc_id: set() for d in docs}

    for chunk in chunks:
        doc = by_id[chunk.doc_id]
        if doc.text[chunk.char_start : chunk.char_end] != chunk.text:
            raise SystemExit(f"offset invariant violated for chunk {chunk.chunk_id}")
        covered[chunk.doc_id].update(range(chunk.char_start, chunk.char_end))

    for doc in docs:
        non_ws = {i for i, ch in enumerate(doc.text) if not ch.isspace()}
        missed = non_ws - covered[doc.doc_id]
        ratio = 1.0 - (len(missed) / max(1, len(non_ws)))
        if ratio < 0.99:
            raise SystemExit(f"coverage invariant violated for {doc.doc_id}: {ratio:.3f}")


def write_all(out: Path) -> dict[str, int]:
    docs = build_documents()
    if not docs:
        raise SystemExit(f"no raw fixtures found under {RAW}")
    chunks = build_chunks(docs)
    pairs = build_evalset(docs, chunks)
    verify(docs, chunks)

    # Parent/child fixture pair for Phase 4's ParentDocumentRetriever tests.
    # child_chunks reuses chunks' exact spans/text/chunk_ids -- assign_parents
    # only adds role="child" + parent_id -- so it never needs its own offset
    # verification; parent_chunks are real slices too, so they do.
    parent_chunks = build_parent_chunks(docs)
    verify(docs, parent_chunks)
    child_chunks = assign_parents(chunks, parent_chunks)

    counts = {
        "documents": write_jsonl(out / "documents" / "sample.jsonl", docs),
        "chunks": write_jsonl(out / "chunks" / "sample.jsonl", chunks),
        "evalset": write_jsonl(out / "evalset" / "sample.jsonl", pairs),
        "chunks_parent": write_jsonl(out / "chunks" / "sample_parent.jsonl", parent_chunks),
        "chunks_child": write_jsonl(out / "chunks" / "sample_child.jsonl", child_chunks),
    }

    cell, run = build_sample_run(pairs)
    run_dir = out / "results" / "sample_run"
    run_dir.mkdir(parents=True, exist_ok=True)

    matrix_payload = json.loads(_sample_experiment_config().model_dump_json())
    (run_dir / "matrix.json").write_text(
        json.dumps(matrix_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    payload = json.loads(run.model_dump_json())
    payload["created_at"] = "2026-01-01T00:00:00"  # pinned, or --check always differs
    serialized = (
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    # write_text's default newline translates \n -> os.linesep on Windows; pin LF.
    (run_dir / "result.json").write_text(serialized, encoding="utf-8", newline="\n")
    cells_dir = run_dir / "cells"
    cells_dir.mkdir(parents=True, exist_ok=True)
    (cells_dir / f"{cell.cell_id}.json").write_text(serialized, encoding="utf-8", newline="\n")
    counts["results"] = 1

    trace = build_sample_optimizer_trace()
    counts["optimizer_trace"] = write_jsonl(
        out / "results" / "sample_optimizer_run" / "optimizer_trace.jsonl", trace
    )
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Regenerate into a temp dir and diff against committed fixtures.",
    )
    args = parser.parse_args()

    if not args.check:
        counts = write_all(OUT)
        for kind, n in counts.items():
            print(f"  {kind:<10} {n}")
        print(f"fixtures written to {OUT}")
        return 0

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        write_all(tmp_path)
        sample_cell_id = expand_cells(_sample_experiment_config())[0].cell_id
        mismatches = []
        for rel in (
            "documents/sample.jsonl",
            "chunks/sample.jsonl",
            "evalset/sample.jsonl",
            "chunks/sample_parent.jsonl",
            "chunks/sample_child.jsonl",
            "results/sample_run/matrix.json",
            "results/sample_run/result.json",
            f"results/sample_run/cells/{sample_cell_id}.json",
            "results/sample_optimizer_run/optimizer_trace.jsonl",
        ):
            committed, regenerated = OUT / rel, tmp_path / rel
            if not committed.exists():
                mismatches.append(f"{rel}: missing (run `make fixtures`)")
            elif committed.read_bytes() != regenerated.read_bytes():
                mismatches.append(f"{rel}: differs from regenerated output")

    if mismatches:
        print("fixture check FAILED:", file=sys.stderr)
        for m in mismatches:
            print(f"  - {m}", file=sys.stderr)
        return 1
    print("fixtures are deterministic and up to date")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
