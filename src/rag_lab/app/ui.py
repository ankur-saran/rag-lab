"""Shared UI chrome: the fixture-data banner and the per-page error boundary
(plan §Phase 9).

``guarded()`` is what turns AC-3 ("every page degrades gracefully with a
clear message when its artifact is missing, rather than throwing") into a
structural guarantee instead of a per-page discipline problem -- every page
wraps its body in it once, at the top.
"""

from __future__ import annotations

from contextlib import contextmanager

import streamlit as st

from rag_lab.paths import ArtifactNotFoundError


def fixture_banner(is_fixture: bool, label: str) -> None:
    """Visible "you're viewing sample data" banner (plan §0's independence
    contract: fixture fallback must be visible to a demo audience, not only
    logged via structlog on stderr, which an audience never sees)."""
    if is_fixture:
        st.info(f"📎 {label} is sample fixture data, not a real build. Results are not meaningful.")


@contextmanager
def guarded(what: str):
    """Catch the "nothing built yet" family of exceptions this codebase uses
    (``ArtifactNotFoundError`` is a ``FileNotFoundError`` subclass;
    ``LookupError`` is what ``report.load_run``/``worst_failures`` and
    ``agents.optimizer.load_trace`` raise for "run/trace not found") and show
    a clear ``st.warning`` instead of an uncaught traceback, then stop the
    page there. A bare ``ValueError`` (e.g. an unregistered component name
    from a stale selection) is treated the same way -- still a "nothing
    sensible to show" condition, not a bug to hide.
    """
    try:
        yield
    except (ArtifactNotFoundError, LookupError, FileNotFoundError, ValueError) as exc:
        st.warning(f"Couldn't load {what}: {exc}")
        st.stop()


__all__ = ["fixture_banner", "guarded"]
