"""Phase 9 acceptance tests (plan §Phase 9: Visual explorer).

Streamlit pages aren't Typer subcommands, so there's no ``CliRunner`` to
drive them -- ``streamlit.testing.v1.AppTest`` runs each page script
headlessly, in-process, without a live server.

No ``conftest.py``, matching every other phase's self-contained-file
convention. ``isolated_artifacts`` mirrors ``test_phase_6.py``/
``test_phase_8.py``'s own fixture of the same name (monkeypatching each
module's own ``artifacts_dir`` binding -- a plain ``from rag_lab.paths import
artifacts_dir`` binds a *new* name per importing module, so patching
``rag_lab.paths.artifacts_dir`` alone would not affect
``rag_lab.app.data.artifacts_dir``), extended with ``rag_lab.app.data``'s own
binding. Real committed ``fixtures/`` is deliberately left untouched by that
fixture -- only ``artifacts/`` is redirected to an empty ``tmp_path`` -- which
is exactly AC-1's "fixtures only, empty artifacts/" scenario.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from rag_lab.app import boundaries, data, optimizer_diff
from rag_lab.jsonl import read_jsonl
from rag_lab.schemas import Chunk, Document

REPO_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = REPO_ROOT / "src" / "rag_lab" / "app"
PAGES = sorted((APP_DIR / "pages").glob("*.py"))


@pytest.fixture(autouse=True)
def _clear_streamlit_cache():
    """``st.cache_data``/``st.cache_resource`` are process-global, keyed only
    by call args -- without clearing between tests, a value cached under one
    test's ``isolated_artifacts``/``broken_fixtures`` tmp_path would leak into
    a later test calling the same function with the same corpus/run_id."""
    st.cache_data.clear()
    st.cache_resource.clear()
    yield


@pytest.fixture
def isolated_artifacts(monkeypatch, tmp_path):
    for target in (
        "rag_lab.paths.artifacts_dir",
        "rag_lab.corpus.artifacts_dir",
        "rag_lab.chunks.artifacts_dir",
        "rag_lab.indexing.artifacts_dir",
        "rag_lab.experiment.runner.artifacts_dir",
        "rag_lab.app.data.artifacts_dir",
    ):
        monkeypatch.setattr(target, lambda p=tmp_path: p)
    return tmp_path


@pytest.fixture
def broken_fixtures(monkeypatch, tmp_path, isolated_artifacts):
    """``isolated_artifacts`` (empty ``artifacts/``) plus a *copy* of the
    committed ``fixtures/`` with the run-result and optimizer-trace fixtures
    removed -- forces the true "nothing available at all" branch AC-3
    requires. ``experiment.report``/``agents.optimizer`` have no fixture
    fallback of their own (unlike ``resolve_artifact``-backed kinds); this
    empties the one thing ``app.data``'s own real-then-fixture fallback would
    otherwise still find.
    """
    fixtures_copy = tmp_path / "fixtures"
    shutil.copytree(REPO_ROOT / "fixtures", fixtures_copy)
    shutil.rmtree(fixtures_copy / "results" / "sample_run")
    shutil.rmtree(fixtures_copy / "results" / "sample_optimizer_run")
    for target in ("rag_lab.paths.fixtures_dir", "rag_lab.app.data.fixtures_dir"):
        monkeypatch.setattr(target, lambda p=fixtures_copy: p)
    return fixtures_copy


# --------------------------------------------------------------------------- #
# AC-1 / AC-3 happy path: every page boots against an empty artifacts/,
# fixtures only.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.stem)
def test_page_boots_on_fixtures_only_with_empty_artifacts(page, isolated_artifacts):  # AC-1, AC-3
    at = AppTest.from_file(str(page))
    at.run(timeout=60)
    assert not at.exception, [str(e.value) for e in at.exception]


# --------------------------------------------------------------------------- #
# AC-3: graceful degradation when nothing at all is available -- the two
# pages whose underlying loader (experiment.report / agents.optimizer) has no
# fixture fallback of its own.
# --------------------------------------------------------------------------- #


def test_results_dashboard_degrades_gracefully_with_nothing_available(broken_fixtures):  # AC-3
    at = AppTest.from_file(str(APP_DIR / "pages" / "3_results_dashboard.py"))
    at.run(timeout=60)
    assert not at.exception, [str(e.value) for e in at.exception]
    assert any("Couldn't load" in w.value for w in at.warning)


def test_optimizer_trace_degrades_gracefully_with_nothing_available(broken_fixtures):  # AC-3
    at = AppTest.from_file(str(APP_DIR / "pages" / "4_optimizer_trace.py"))
    at.run(timeout=60)
    assert not at.exception, [str(e.value) for e in at.exception]
    assert any("Couldn't load" in w.value for w in at.warning)


# --------------------------------------------------------------------------- #
# AC-2, as a data-layer assertion rather than screen-scraped HTML:
# boundaries.segment_document reconstructs the document exactly, and every
# segment's covering chunk(s) satisfy the same offset invariant
# scripts/build_fixtures.py's own verify() enforces elsewhere.
# --------------------------------------------------------------------------- #


def test_segment_document_reconstructs_document_text_exactly():  # AC-2
    docs = {d.doc_id: d for d in read_jsonl(REPO_ROOT / "fixtures" / "documents" / "sample.jsonl", Document)}
    chunks = read_jsonl(REPO_ROOT / "fixtures" / "chunks" / "sample.jsonl", Chunk)
    by_doc: dict[str, list[Chunk]] = {}
    for c in chunks:
        by_doc.setdefault(c.doc_id, []).append(c)
    assert by_doc, "fixture chunk set is unexpectedly empty"

    for doc_id, doc_chunks in by_doc.items():
        doc = docs[doc_id]
        segments = boundaries.segment_document(len(doc.text), doc_chunks)

        reconstructed = "".join(doc.text[s.start : s.end] for s in segments)
        assert reconstructed == doc.text, f"segments don't reconstruct doc {doc_id!r} exactly"

        for seg in segments:
            for chunk in seg.covering:
                assert chunk.char_start <= seg.start and seg.end <= chunk.char_end
                assert doc.text[chunk.char_start : chunk.char_end] == chunk.text


def test_segment_document_flags_overlap_and_gap_correctly():
    doc_text = "0123456789"
    a = Chunk(
        chunk_id="a", doc_id="d", corpus="c", chunk_set_id="cs",
        text=doc_text[0:5], embed_text=doc_text[0:5],
        char_start=0, char_end=5, token_count=1, ordinal=0, chunker="x",
    )
    b = Chunk(
        chunk_id="b", doc_id="d", corpus="c", chunk_set_id="cs",
        text=doc_text[3:8], embed_text=doc_text[3:8],
        char_start=3, char_end=8, token_count=1, ordinal=1, chunker="x",
    )
    segments = boundaries.segment_document(len(doc_text), [a, b])
    reconstructed = "".join(doc_text[s.start : s.end] for s in segments)
    assert reconstructed == doc_text

    overlap_segments = [s for s in segments if s.overlapping]
    assert overlap_segments and all(s.start >= 3 and s.end <= 5 for s in overlap_segments)

    gap_segments = [s for s in segments if not s.covering]
    assert gap_segments and all(s.start >= 8 for s in gap_segments)


# --------------------------------------------------------------------------- #
# Plain-pytest unit tests for the pure helpers (independent of Streamlit).
# --------------------------------------------------------------------------- #


def test_diff_config_reports_added_changed_and_removed_keys():
    prev = {"a": 1, "b": 2}
    curr = {"a": 1, "b": 3, "c": 4}
    assert optimizer_diff.diff_config(prev, curr) == {"b": (2, 3), "c": (None, 4)}


def test_diff_config_first_iteration_reports_everything_as_added():
    assert optimizer_diff.diff_config(None, {"a": 1}) == {"a": (None, 1)}


def test_diff_metrics_computes_delta_from_previous():
    prev = {"recall@5": 0.5}
    curr = {"recall@5": 0.6, "mrr": 0.4}
    delta = optimizer_diff.diff_metrics(prev, curr)
    assert delta["recall@5"] == pytest.approx(0.1)
    assert delta["mrr"] == pytest.approx(0.4)


def test_difficulty_by_query_id_covers_real_fixture_queries(isolated_artifacts):
    tiers = data.difficulty_by_query_id("api_docs")
    assert tiers
    assert set(tiers.values()) <= {"lookup", "synthesis", "cross_reference"}
