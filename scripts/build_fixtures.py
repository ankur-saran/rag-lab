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
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

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
from rag_lab.schemas import Chunk, Document, EvalPair, QueryTrace, RunResult  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "fixtures" / "raw"
OUT = ROOT / "fixtures"

FIXTURE_CHUNKER = "paragraph_fixture"
FIXTURE_PARAMS = {"min_chars": 40}


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


def build_chunks(docs: list[Document]) -> list[Chunk]:
    chunks: list[Chunk] = []
    signature = chunker_signature(FIXTURE_CHUNKER, FIXTURE_PARAMS)

    for doc in docs:
        chunk_set = make_chunk_set_id(doc.corpus, FIXTURE_CHUNKER, FIXTURE_PARAMS)
        heading_path: list[str] = []
        for ordinal, (start, end) in enumerate(
            paragraph_spans(doc.text, FIXTURE_PARAMS["min_chars"])
        ):
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
                    heading_path=list(heading_path),
                    chunker=FIXTURE_CHUNKER,
                    chunker_params=dict(FIXTURE_PARAMS),
                    meta={"approx_tokens": True},
                )
            )
    return chunks


# Queries are hand-written against known substrings so the gold spans are exact
# and independent of any chunker. (needle, question, difficulty)
#
# Needles must not straddle a line break — the raw fixtures are hard-wrapped, so
# a needle spanning a wrap will not be found. Keep them short and single-line.
SEEDS: list[tuple[str, str, str]] = [
    ("default limit is 25", "What is the default page size for list endpoints?", "lookup"),
    ("Cursors expire after 24 hours", "How long is a pagination cursor valid?", "lookup"),
    (
        "`items`, `next_cursor`",
        "Which fields does a paginated response envelope contain?",
        "lookup",
    ),
    (
        "access token valid for one hour",
        "How long does an access token remain valid after issuance?",
        "lookup",
    ),
    (
        "10 requests per minute",
        "What is the rate limit on authentication endpoints?",
        "lookup",
    ),
    (
        "one and one-half percent (1.5%) per month",
        "What interest rate applies to overdue undisputed amounts?",
        "lookup",
    ),
    (
        "disputes in good faith",
        "Under what conditions does interest not accrue on an invoice?",
        "synthesis",
    ),
    (
        "Sections 4, 7, and 9 shall",
        "Which sections survive termination of the agreement?",
        "cross_reference",
    ),
]


def build_evalset(docs: list[Document], chunks: list[Chunk]) -> list[EvalPair]:
    by_doc: dict[str, list[Chunk]] = {}
    for chunk in chunks:
        by_doc.setdefault(chunk.doc_id, []).append(chunk)

    pairs: list[EvalPair] = []
    for needle, query, difficulty in SEEDS:
        host = next((d for d in docs if needle in d.text), None)
        if host is None:
            raise SystemExit(f"fixture seed not found in any document: {needle!r}")

        start = host.text.index(needle)
        span = (start, start + len(needle))

        gold = [
            c.chunk_id
            for c in by_doc[host.doc_id]
            if c.char_start <= span[0] < c.char_end or c.char_start < span[1] <= c.char_end
        ]
        query_id = make_query_id(host.corpus, host.doc_id, span, query)

        pairs.append(
            EvalPair(
                query_id=query_id,
                corpus=host.corpus,
                query=query,
                gold_doc_id=host.doc_id,
                gold_char_span=span,
                gold_chunk_ids=gold,  # derived; re-resolved per chunk set in Phase 6
                answer=None,
                supporting_quote=needle,
                difficulty=difficulty,  # type: ignore[arg-type]
                split=split_for(query_id),  # type: ignore[arg-type]
                generator_model="handwritten-fixture",
                validated=True,
            )
        )
    return pairs


def build_sample_run(pairs: list[EvalPair]) -> RunResult:
    traces = [
        QueryTrace(
            query_id=p.query_id,
            query=p.query,
            retrieved_chunk_ids=list(p.gold_chunk_ids),
            retrieved_scores=[0.9 - 0.1 * i for i in range(len(p.gold_chunk_ids))],
            gold_chunk_ids=list(p.gold_chunk_ids),
            first_hit_rank=1 if p.gold_chunk_ids else None,
            metrics={"recall@5": 1.0 if p.gold_chunk_ids else 0.0},
            latency_ms=4.2,
        )
        for p in pairs
    ]
    return RunResult(
        run_id="sample_run",
        config={"chunker": FIXTURE_CHUNKER, "embedder": "fixture", "retriever": "fixture"},
        corpus="sample",
        metrics={"recall@1": 1.0, "recall@5": 1.0, "mrr": 1.0, "ndcg@10": 1.0},
        per_query=traces,
        chunk_stats={"count": float(len(traces))},
        timings={"total_s": 0.0},
    )


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

    counts = {
        "documents": write_jsonl(out / "documents" / "sample.jsonl", docs),
        "chunks": write_jsonl(out / "chunks" / "sample.jsonl", chunks),
        "evalset": write_jsonl(out / "evalset" / "sample.jsonl", pairs),
    }

    run = build_sample_run(pairs)
    run_dir = out / "results" / "sample_run"
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = json.loads(run.model_dump_json())
    payload["created_at"] = "2026-01-01T00:00:00"  # pinned, or --check always differs
    (run_dir / "result.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",  # pin LF; write_text's default translates \n -> os.linesep on Windows
    )
    counts["results"] = 1
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
        mismatches = []
        for rel in (
            "documents/sample.jsonl",
            "chunks/sample.jsonl",
            "evalset/sample.jsonl",
            "results/sample_run/result.json",
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
