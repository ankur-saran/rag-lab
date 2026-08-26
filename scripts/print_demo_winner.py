#!/usr/bin/env python
"""Print the winning cell of a `make demo` run (Phase 10, Step 10.1's "launch
the viewer on the winning configuration").

Reuses `experiment.report.load_run` -- the same function `experiment
report`/`compare`/`failures` already call -- and reads each cell's plain mean
`recall@5` off `RunResult.metrics` (already computed by the runner; no CI
recomputation needed here). `app/data.py`'s `list_run_ids()` already surfaces
the newest real run first in the Streamlit "Results dashboard" page, so this
script's job is only to name the winner in `make demo`'s own console output,
not to configure the viewer.

Usage:
    python scripts/print_demo_winner.py --run-id demo__20260101T000000Z
    python scripts/print_demo_winner.py --run-id demo__20260101T000000Z --metric mrr
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag_lab.experiment.report import load_run  # noqa: E402
from rag_lab.schemas import RunResult  # noqa: E402


def pick_winner(results: list[RunResult], metric: str) -> tuple[RunResult, float] | None:
    """The cell with the highest ``metric`` among results that report it, or
    ``None`` if no cell does (an all-excluded run, or a metric name typo)."""
    scored = [(r, r.metrics[metric]) for r in results if metric in r.metrics]
    if not scored:
        return None
    return max(scored, key=lambda rv: rv[1])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--metric", default="recall@5")
    args = parser.parse_args()

    try:
        _config, results = load_run(args.run_id)
    except (LookupError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    winner = pick_winner(results, args.metric)
    if winner is None:
        print(f"no cell in run {args.run_id!r} reports {args.metric!r}", file=sys.stderr)
        return 1

    result, value = winner
    cfg = result.config
    print(
        f"winner  {result.corpus}/{cfg.get('chunker')}/{cfg.get('embedder')}/"
        f"{cfg.get('retriever')}  {args.metric}={value:.3f}"
    )
    print(f"  -> open the Streamlit 'Results dashboard' page, run {args.run_id!r}, to see it")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
