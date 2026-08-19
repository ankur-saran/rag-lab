#!/usr/bin/env python
"""Hand-authored eval set for the real `api_docs` corpus (Phase 6's AC-6).

Phase 5 (the LLM-based generator) hasn't shipped yet. This is the "hand-
written eval set" shortcut the master plan's own Sequencing section names as
the minimum-viable path to Phase 6 -- ~25-30 needle-anchored pairs across the
real 15-document api_docs corpus, the same technique as
scripts/build_fixtures.py's SEEDS list, scaled up and run against the real
(not fixture) corpus text.

Unlike build_fixtures.py this writes a real, regenerable *artifact*
(artifacts/evalset/api_docs.jsonl), not a committed fixture -- consistent
with how verify-phase-3/4 rebuild real corpora/chunks/indexes from scratch on
every run rather than committing them.

Usage:
    python scripts/build_api_docs_evalset.py            # write the eval set
    python scripts/build_api_docs_evalset.py --check    # verify determinism, write nothing
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag_lab.ids import make_query_id, split_for  # noqa: E402
from rag_lab.jsonl import write_jsonl  # noqa: E402
from rag_lab.loaders import load_corpus  # noqa: E402
from rag_lab.paths import artifact_path, corpora_dir  # noqa: E402
from rag_lab.schemas import EvalPair  # noqa: E402

CORPUS = "api_docs"

# (needles, question, difficulty). One needle per gold span -- two-plus for
# cross_reference, evaluated independently (never concatenated). Every needle
# must be a literal substring of its host document's *normalized* text that
# doesn't straddle a hard line-wrap -- the raw corpora/api_docs/*.md files
# wrap prose at ~78 columns, so a needle spanning a wrap silently fails to
# match (same constraint scripts/build_fixtures.py documents for its SEEDS).
# Consecutive bullet lines are safe to join with "\n" since each is already
# a complete, unwrapped physical line.
SEEDS: list[tuple[tuple[str, ...], str, str]] = [
    # -- lookup: one explicit fact, one span -------------------------------
    (
        ("Authentication endpoints are limited to 10 requests per minute per client ID.",),
        "What is the rate limit on authentication endpoints?",
        "lookup",
    ),
    (
        ("page size, default 25, maximum 100.",),
        "What is the default page size for list endpoints?",
        "lookup",
    ),
    (
        ("Cursors expire after 24 hours",),
        "How long is a pagination cursor valid before it expires?",
        "lookup",
    ),
    (
        ("Read endpoints (`GET`): 600 requests per minute",),
        "What is the rate limit for read endpoints?",
        "lookup",
    ),
    (
        ("Write endpoints (`POST`/`PATCH`/`DELETE`): 120 requests per minute",),
        "What is the rate limit for write endpoints?",
        "lookup",
    ),
    (
        ("`CURSOR_EXPIRED` — a pagination cursor older than 24 hours was used",),
        "What condition triggers a CURSOR_EXPIRED error?",
        "lookup",
    ),
    (
        ("Keys older than 24 hours are evicted",),
        "After how long are unused idempotency keys evicted?",
        "lookup",
    ),
    (
        ("A delivery that doesn't get a `2xx` response within 5 seconds is retried",),
        "How quickly must a webhook endpoint respond before a delivery is considered failed?",
        "lookup",
    ),
    (
        ("`pending` orders move to `authorized` once payment authorization succeeds,",),
        "What status does a pending order move to once payment authorization succeeds?",
        "lookup",
    ),
    (
        ("up to 20 keys",),
        "How many metadata keys can a customer have?",
        "lookup",
    ),
    (
        ("Search indexes update within 30 seconds of a write",),
        "How quickly do search indexes update after a write?",
        "lookup",
    ),
    (
        ("The batch endpoint executes up to 100 sub-requests in a single call",),
        "How many sub-requests can a single batch call contain?",
        "lookup",
    ),
    (
        ("A version is supported for at least 18 months after a newer version ships.",),
        "How long is an API version supported after a newer version ships?",
        "lookup",
    ),
    (
        ("The SDK retries `5xx` responses and connection errors up to 3 times",),
        "How many times does the official SDK retry a 5xx response?",
        "lookup",
    ),
    (
        ("`amount=100000` — always fails authorization with `card_declined`",),
        "Which magic sandbox amount always fails authorization with card_declined?",
        "lookup",
    ),
    (
        ("Maximum size: 8 MB per file",),
        "What is the maximum file size for an upload?",
        "lookup",
    ),
    (
        ("Increased default read rate limit from 300 to 600 requests per minute",),
        "According to the changelog, what was the read rate limit increased to?",
        "lookup",
    ),
    # -- synthesis: two or three facts from the same span ------------------
    (
        (
            "- `access_token` — opaque bearer token, valid for one hour\n"
            "- `refresh_token` — opaque token, valid for thirty days",
        ),
        "How long do the access token and refresh token each remain valid?",
        "synthesis",
    ),
    (
        (
            "- `items` — the page of records\n"
            "- `next_cursor` — cursor for the next page, or `null` if this is the last page\n"
            "- `has_more` — boolean mirror of whether `next_cursor` is set",
        ),
        "What three fields does a paginated response envelope contain?",
        "synthesis",
    ),
    (
        (
            "- `INVALID_GRANT` — token refresh used a rotated or expired refresh token\n"
            "- `RESOURCE_NOT_FOUND` — the id in the path does not exist in this workspace",
        ),
        "What do the INVALID_GRANT and RESOURCE_NOT_FOUND error codes mean?",
        "synthesis",
    ),
    (
        (
            "- Read endpoints (`GET`): 600 requests per minute\n"
            "- Write endpoints (`POST`/`PATCH`/`DELETE`): 120 requests per minute",
        ),
        "What are the rate limits for read versus write endpoints?",
        "synthesis",
    ),
    (
        (
            "- `amount=100000` — always fails authorization with `card_declined`\n"
            "- `amount=100001` — succeeds authorization, fails on capture",
        ),
        "What do the two sandbox magic amounts 100000 and 100001 each trigger?",
        "synthesis",
    ),
    (
        ("Batches are capped at 100 items and 5 MB of total request body.",),
        "What are the batch size limits -- item count and total body size?",
        "synthesis",
    ),
    (
        (
            "- Maximum size: 8 MB per file\n"
            "- Files are retained for 2 years, then automatically deleted",
        ),
        "What is the maximum file size, and how long are files retained?",
        "synthesis",
    ),
    (
        ("Official SDKs exist for Python, Node.js, Ruby, and Go.",),
        "Which languages have official (not community-maintained) SDKs?",
        "synthesis",
    ),
    (
        (
            "`pending` orders move to `authorized` once payment authorization succeeds,\n"
            "then to `captured` when funds are actually collected. Only `captured` orders",
        ),
        "What sequence of states does an order move through from pending to captured?",
        "synthesis",
    ),
    # -- cross_reference: two distant spans in the same document -----------
    (
        (
            "an access token valid for one hour and a refresh token valid for thirty days.",
            "Authentication endpoints are limited to 10 requests per minute per client ID.",
        ),
        "How long is a newly issued access token valid, and what's the rate limit "
        "on the endpoint used to obtain one?",
        "cross_reference",
    ),
    (
        (
            "page size, default 25, maximum 100.",
            "Cursors expire after 24 hours",
        ),
        "What is the default page size for list endpoints, and how long is a "
        "pagination cursor valid before it expires?",
        "cross_reference",
    ),
    (
        (
            "- `status` — one of `pending`, `authorized`, `captured`, `refunded`, `failed`",
            "`GET /orders` supports `status`, `customer_id`, and `created_after` /",
        ),
        "What are the possible order status values, and which filters does GET /orders support?",
        "cross_reference",
    ),
    (
        (
            "A delivery that doesn't get a `2xx` response within 5 seconds is retried",
            "Replays are marked with a `replayed: true`",
        ),
        "How quickly must a webhook respond before a delivery is retried, and how "
        "are replayed events marked in the payload?",
        "cross_reference",
    ),
]


def build_evalset() -> list[EvalPair]:
    docs = load_corpus(CORPUS, corpora_dir() / CORPUS)
    if not docs:
        raise SystemExit(f"no documents found for corpus {CORPUS!r} under {corpora_dir()}")

    pairs: list[EvalPair] = []
    for needles, query, difficulty in SEEDS:
        host = next((d for d in docs if all(needle in d.text for needle in needles)), None)
        if host is None:
            raise SystemExit(f"seed not found in any {CORPUS} document: {needles!r}")

        spans: list[tuple[int, int]] = []
        for needle in needles:
            start = host.text.index(needle)
            if host.text.find(needle, start + 1) != -1:
                raise SystemExit(f"seed matches more than once in {host.doc_id}: {needle!r}")
            spans.append((start, start + len(needle)))

        query_id = make_query_id(host.corpus, host.doc_id, spans, query)
        pairs.append(
            EvalPair(
                query_id=query_id,
                corpus=host.corpus,
                query=query,
                gold_doc_id=host.doc_id,
                gold_char_spans=spans,
                sampling_chunk_ids=[],  # hand-authored directly, no sampling chunk set
                answer=None,
                supporting_quotes=list(needles),
                difficulty=difficulty,  # type: ignore[arg-type]
                split=split_for(query_id),  # type: ignore[arg-type]
                generator_model="handwritten-api-docs",
                validated=True,
            )
        )
    return pairs


def write_all(out_path: Path) -> int:
    return write_jsonl(out_path, build_evalset())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="Verify determinism against the existing artifact."
    )
    args = parser.parse_args()

    out_path = artifact_path("evalset", f"{CORPUS}.jsonl")

    if not args.check:
        count = write_all(out_path)
        print(f"  evalset    {count} pairs -> {out_path}")
        return 0

    if not out_path.exists():
        print(f"{out_path} missing -- run `python scripts/build_api_docs_evalset.py` first", file=sys.stderr)
        return 1
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp) / f"{CORPUS}.jsonl"
        write_all(tmp_path)
        if out_path.read_bytes() != tmp_path.read_bytes():
            print(f"{out_path} differs from regenerated output", file=sys.stderr)
            return 1
    print("api_docs eval set is deterministic and up to date")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
