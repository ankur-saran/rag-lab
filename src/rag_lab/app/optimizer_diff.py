"""Config/metrics diffing between optimizer trace iterations (plan §Phase 9,
Step 9.4).

``agents.optimizer.render_trace`` prints each entry's raw ``config``/
``metrics`` but never diffs consecutive iterations against each other -- this
module adds that. Pure functions, no Streamlit import, so they're plain
pytest testable independent of the app.
"""

from __future__ import annotations

from typing import Any

_MISSING = object()


def diff_config(prev: dict[str, Any] | None, curr: dict[str, Any]) -> dict[str, tuple[Any, Any]]:
    """``{key: (old, new)}`` for every key added, removed, or changed between
    ``prev`` and ``curr``. ``prev=None`` (the first iteration) reports every
    key of ``curr`` as newly added (``old=None``)."""
    prev = prev or {}
    keys = set(prev) | set(curr)
    diff: dict[str, tuple[Any, Any]] = {}
    for key in keys:
        old = prev.get(key, _MISSING)
        new = curr.get(key, _MISSING)
        if old != new:
            diff[key] = (None if old is _MISSING else old, None if new is _MISSING else new)
    return diff


def diff_metrics(prev: dict[str, float] | None, curr: dict[str, float]) -> dict[str, float]:
    """``{metric: curr - prev}`` for every metric present in ``curr``. A
    metric absent from ``prev`` (or ``prev=None``) is reported as a delta from
    0.0 -- the natural reading for "this metric didn't exist before"."""
    prev = prev or {}
    return {metric: value - prev.get(metric, 0.0) for metric, value in curr.items()}


__all__ = ["diff_config", "diff_metrics"]
