"""Phase 7 acceptance tests -- `semantic` and `table_summary` chunkers.

The six numbered criteria from the plan are marked with `# AC-n`. A
module-scoped autouse fixture runs `corpus build --all` once, mirroring
`test_phase_2.py`, so this file is self-contained on a clean checkout.

Tiering mirrors `test_phase_3.py`/`test_phase_4.py`:
- Tier 1 (default): structural correctness (offsets, coverage, determinism,
  param -> chunk_set_id, bound enforcement) via `fast_semantic_embedder`
  (swaps in the dependency-free `HashEmbedder`, same technique as
  `test_phase_3.py`'s `fast_embedder`) and `table_summary`'s `--mock-llm`
  (no network, no `anthropic` import). Neither needs `sentence-transformers`,
  `chromadb`, or an API key.
- `@requires_sentence_transformers` (AC-3): the topic-shift precision test is
  inherently about real embedding quality -- `HashEmbedder` disclaims
  semantic meaning by design and cannot stand in for it.
- `@requires_anthropic_api_key` (AC-5): proving summarized beats
  not-summarized needs a real, semantically useful summary; a mock response
  is deliberately meaningless text and would not demonstrate the effect.
  Skipped by default, exactly like the other optional-dependency tiers.
"""

from __future__ import annotations

import os
import re
import sys

import pytest
from typer.testing import CliRunner

from rag_lab.chunkers import REGISTRY, available_chunkers, resolve_params, run_chunker
from rag_lab.chunkers.base import count_tokens
from rag_lab.chunkers.table_summary import _truncate_for_prompt
from rag_lab.chunks import compute_chunk_stats
from rag_lab.cli import app
from rag_lab.embedders.fixture import HashEmbedder
from rag_lab.experiment.config import ExperimentConfig, expand_cells
from rag_lab.ids import make_chunk_set_id
from rag_lab.jsonl import read_jsonl
from rag_lab.llm import call_llm, cached_llm_call
from rag_lab.markup import find_tables
from rag_lab.paths import artifact_path, fixture_path
from rag_lab.schemas import Chunk, Document

try:
    import sentence_transformers as _sentence_transformers  # noqa: F401

    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

requires_sentence_transformers = pytest.mark.skipif(
    not HAS_SENTENCE_TRANSFORMERS, reason="sentence-transformers not installed (embed extra)"
)
requires_anthropic_api_key = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set"
)

runner = CliRunner()

NEW_CHUNKER_NAMES = ["semantic", "table_summary"]
ALL_CORPORA = ["api_docs", "catalog", "contracts", "filings", "transcripts"]


@pytest.fixture(scope="module", autouse=True)
def _build_all_corpora():
    result = runner.invoke(app, ["corpus", "build", "--all"])
    assert result.exit_code == 0, result.output
    yield


@pytest.fixture
def fast_semantic_embedder(monkeypatch):
    """Swap `semantic.py`'s embedder construction for the dependency-free
    `HashEmbedder`, mirroring `test_phase_3.py`'s `fast_embedder`, so
    structural tests run in milliseconds without `sentence-transformers` or a
    model download. AC-3's topic-shift precision test needs the real model
    instead -- `HashEmbedder` disclaims semantic meaning by design."""

    def _build(name, overrides=None):
        return HashEmbedder(overrides)

    monkeypatch.setattr("rag_lab.chunkers.semantic.build_embedder", _build)


def _fixture_documents() -> list[Document]:
    return read_jsonl(fixture_path("documents"), Document)


def _real_documents(corpus: str) -> list[Document]:
    return read_jsonl(artifact_path("documents", f"{corpus}.jsonl"), Document)


def _coverage_ratio(doc: Document, chunks: list[Chunk]) -> float:
    covered: set[int] = set()
    for c in chunks:
        covered.update(range(c.char_start, c.char_end))
    non_ws = {i for i, ch in enumerate(doc.text) if not ch.isspace()}
    if not non_ws:
        return 1.0
    return 1.0 - len(non_ws - covered) / len(non_ws)


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #


def test_registry_includes_the_phase_7_chunkers():
    assert set(NEW_CHUNKER_NAMES) <= set(REGISTRY)
    assert available_chunkers() == sorted(
        ["fixed", "recursive", "markdown", "sentence_window", *NEW_CHUNKER_NAMES]
    )


# --------------------------------------------------------------------------- #
# AC-1: both run against fixtures, no prior phase; table_summary --mock-llm
# makes no network call and never imports anthropic
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", NEW_CHUNKER_NAMES)
def test_phase7_chunkers_run_against_fixture_documents(name, fast_semantic_embedder):  # AC-1
    docs = _fixture_documents()
    assert docs
    overrides = {"mock_llm": True} if name == "table_summary" else {}
    for doc in docs:
        chunks = run_chunker(name, doc, overrides)
        assert chunks
        assert all(isinstance(c, Chunk) for c in chunks)
        assert all(c.chunker == name for c in chunks)


def test_cli_chunk_run_table_summary_mock_llm_works_with_no_prior_phase(monkeypatch, tmp_path):
    """The independence contract, exercised end to end through the CLI --
    mirrors `test_phase_2.py`'s equivalent for the baseline chunkers."""
    monkeypatch.setattr("rag_lab.paths.artifacts_dir", lambda: tmp_path)
    monkeypatch.setattr("rag_lab.corpus.artifacts_dir", lambda: tmp_path)

    result = runner.invoke(
        app, ["chunk", "run", "--corpus", "api_docs", "--chunker", "table_summary", "--mock-llm"]
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "chunks").exists()


def test_call_llm_mock_never_imports_anthropic(monkeypatch):  # AC-1
    monkeypatch.delitem(sys.modules, "anthropic", raising=False)
    result = call_llm("describe this table", model="whatever", mock=True)
    assert result
    assert "anthropic" not in sys.modules


def test_call_llm_mock_makes_no_network_call_and_is_deterministic():  # AC-1
    a = call_llm("summarize this table", model="whatever", mock=True)
    b = call_llm("summarize this table", model="whatever", mock=True)
    assert a == b


def test_call_llm_mock_differs_for_different_prompts():
    a = call_llm("prompt one", model="whatever", mock=True)
    b = call_llm("prompt two", model="whatever", mock=True)
    assert a != b


def test_cached_llm_call_hits_on_second_call(monkeypatch, tmp_path):
    monkeypatch.setattr("rag_lab.llm.artifacts_dir", lambda: tmp_path)
    calls = []

    def compute():
        calls.append(1)
        return "real result"

    first = cached_llm_call("key1", compute)
    second = cached_llm_call("key1", compute)
    assert first == second == "real result"
    assert len(calls) == 1  # second call was a cache hit, `compute` not re-invoked


def test_cached_llm_call_writes_under_artifacts_llm_cache(monkeypatch, tmp_path):
    monkeypatch.setattr("rag_lab.llm.artifacts_dir", lambda: tmp_path)
    cached_llm_call("key2", lambda: "value")
    assert (tmp_path / "llm_cache" / "key2.txt").read_text(encoding="utf-8") == "value"


# --------------------------------------------------------------------------- #
# Offset / coverage invariants, swept across every real corpus
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("corpus", ALL_CORPORA)
def test_semantic_offset_and_coverage_invariants_hold_everywhere(fast_semantic_embedder, corpus):
    for doc in _real_documents(corpus):
        chunks = run_chunker("semantic", doc, {})
        for c in chunks:
            assert doc.text[c.char_start : c.char_end] == c.text
        assert _coverage_ratio(doc, chunks) >= 0.99


@pytest.mark.parametrize("corpus", ALL_CORPORA)
def test_table_summary_offset_and_coverage_invariants_hold_everywhere(corpus):
    for doc in _real_documents(corpus):
        chunks = run_chunker("table_summary", doc, {"mock_llm": True})
        for c in chunks:
            assert doc.text[c.char_start : c.char_end] == c.text
        assert _coverage_ratio(doc, chunks) >= 0.99


# --------------------------------------------------------------------------- #
# Determinism and identity
# --------------------------------------------------------------------------- #


def test_semantic_determinism_same_params_same_output(fast_semantic_embedder):
    doc = _real_documents("transcripts")[0]
    a = run_chunker("semantic", doc, {})
    b = run_chunker("semantic", doc, {})
    assert [c.model_dump_json() for c in a] == [c.model_dump_json() for c in b]
    assert a[0].chunk_set_id == b[0].chunk_set_id


def test_table_summary_determinism_same_params_same_output():
    doc = _real_documents("filings")[0]
    a = run_chunker("table_summary", doc, {"mock_llm": True})
    b = run_chunker("table_summary", doc, {"mock_llm": True})
    assert [c.model_dump_json() for c in a] == [c.model_dump_json() for c in b]
    assert a[0].chunk_set_id == b[0].chunk_set_id


def test_different_summarize_param_yields_different_chunk_set_id():
    doc = _real_documents("filings")[0]
    a = run_chunker("table_summary", doc, {"mock_llm": True, "summarize": True})
    b = run_chunker("table_summary", doc, {"summarize": False})
    assert a[0].chunk_set_id != b[0].chunk_set_id


def test_mock_llm_and_real_never_share_a_chunk_set_id():
    """Guards the cache-poisoning bug the module docstring describes: since
    `mock_llm` is hashed like any other param, a mock run and the
    corresponding real run resolve to different chunk_set_ids, so a later
    real `chunk run` can never silently read back cached mock output as if it
    were a legitimate cache hit."""
    doc = _real_documents("filings")[0]
    mock_chunks = run_chunker("table_summary", doc, {"mock_llm": True})
    default_params = resolve_params("table_summary", {})
    assert default_params["mock_llm"] is False
    assert mock_chunks[0].chunker_params["mock_llm"] is True
    real_chunk_set_id = make_chunk_set_id(doc.corpus, "table_summary", default_params)
    assert mock_chunks[0].chunk_set_id != real_chunk_set_id


# --------------------------------------------------------------------------- #
# AC-2: semantic respects min_tokens/max_tokens strictly
# --------------------------------------------------------------------------- #


def test_semantic_merges_undersized_runs_forward(fast_semantic_embedder):  # AC-2
    doc = _real_documents("transcripts")[0]
    unmerged = run_chunker("semantic", doc, {"min_tokens": 0, "max_tokens": 100_000})
    merged = run_chunker("semantic", doc, {"min_tokens": 64, "max_tokens": 100_000})
    assert len(merged) <= len(unmerged)
    assert all(c.token_count >= 64 for c in merged[:-1])  # trailing run may stay small


def test_semantic_splits_oversized_runs(fast_semantic_embedder):  # AC-2
    doc = _real_documents("transcripts")[0]
    # breakpoint_percentile=100 -> zero cuts (nothing exceeds the max
    # distance), so the whole document becomes one run and must go through
    # the oversized-split fallback -- deterministic regardless of embedder.
    chunks = run_chunker(
        "semantic", doc, {"min_tokens": 0, "max_tokens": 80, "breakpoint_percentile": 100}
    )
    assert len(chunks) > 1
    assert all(c.token_count <= 80 for c in chunks)


def test_semantic_single_sentence_document_is_one_chunk(fast_semantic_embedder):
    doc = Document(
        doc_id="one-sentence",
        corpus="test",
        source_path="synthetic.txt",
        title="One Sentence",
        text="Just one sentence here.",
    )
    chunks = run_chunker("semantic", doc, {})
    assert len(chunks) == 1
    assert chunks[0].text == doc.text


# --------------------------------------------------------------------------- #
# AC-3: boundaries on a synthetic three-topic document land within +/-1
# sentence of the true topic shifts (needs a real embedder)
# --------------------------------------------------------------------------- #

_TOPIC_A = (
    "Pour-over coffee relies on a slow, even saturation of the grounds. "
    "Water at roughly two hundred degrees Fahrenheit extracts oils and acids "
    "without scorching the bean. Most baristas recommend a ratio of one gram "
    "of coffee to sixteen grams of water. Grinding the beans too fine will "
    "clog the filter and produce a bitter, over-extracted cup."
)
_TOPIC_B = (
    "A satellite in geostationary orbit circles the Earth once every "
    "twenty-four hours, matching the planet's own rotation. This orbit sits "
    "roughly thirty-six thousand kilometers above the equator. Because the "
    "satellite appears fixed relative to the ground, it is ideal for "
    "television broadcasting and weather monitoring. Small deviations in "
    "altitude or inclination require periodic thruster corrections to keep "
    "the satellite in its assigned slot."
)
_TOPIC_C = (
    "A castle's curtain wall was built thick enough to resist battering rams "
    "and early siege engines. Arrow slits allowed defenders to fire on "
    "attackers while remaining shielded behind stone. Many castles also "
    "included a moat, which slowed any assault on the main gate. The keep, "
    "usually the tallest structure, served as the final refuge if the outer "
    "walls were breached."
)


@requires_sentence_transformers
def test_semantic_boundaries_land_near_the_true_topic_shifts():  # AC-3
    from rag_lab.chunkers.sentence_window import sentence_spans

    text = f"{_TOPIC_A} {_TOPIC_B} {_TOPIC_C}"
    spans = sentence_spans(text)
    b_start = len(_TOPIC_A) + 1
    c_start = len(_TOPIC_A) + 1 + len(_TOPIC_B) + 1
    # `sentence_spans` attaches trailing whitespace to the preceding sentence
    # (recursive.py's convention too), so the last sentence of one topic ends
    # exactly where the next topic's text begins.
    true_shift_after = {
        next(i for i, (_, e) in enumerate(spans) if e == b_start),
        next(i for i, (_, e) in enumerate(spans) if e == c_start),
    }

    doc = Document(
        doc_id="three-topic-synthetic",
        corpus="test",
        source_path="synthetic.txt",
        title="Three Topic Synthetic",
        text=text,
    )
    chunks = sorted(
        run_chunker(
            "semantic",
            doc,
            {
                "buffer_size": 1,
                "breakpoint_percentile": 75,
                "min_tokens": 0,
                "max_tokens": 100_000,
                "embedder": "bge-small",
            },
        ),
        key=lambda c: c.char_start,
    )
    found_shift_after = set()
    for c in chunks[1:]:
        idx = next(i for i, (s, _) in enumerate(spans) if s == c.char_start)
        found_shift_after.add(idx - 1)

    for true_idx in true_shift_after:
        assert any(abs(true_idx - found) <= 1 for found in found_shift_after), (
            f"no detected breakpoint within +/-1 sentence of the true shift after "
            f"sentence {true_idx}; found breakpoints after {sorted(found_shift_after)}"
        )


# --------------------------------------------------------------------------- #
# AC-4: table_summary's embed_text/text split
# --------------------------------------------------------------------------- #


def _first_filings_doc_with_a_table() -> Document:
    docs = [d for d in _real_documents("filings") if find_tables(d.text)]
    assert docs, "filings corpus must contain at least one real GFM table"
    return docs[0]


def test_table_summary_embed_text_differs_from_text_when_summarizing():  # AC-4
    doc = _first_filings_doc_with_a_table()
    chunks = run_chunker("table_summary", doc, {"mock_llm": True})
    table_chunks = [c for c in chunks if c.meta.get("is_table")]
    assert table_chunks
    assert all(c.embed_text != c.text for c in table_chunks)


def test_table_summary_embed_text_equals_text_when_not_summarizing():  # AC-4
    doc = _first_filings_doc_with_a_table()
    chunks = run_chunker("table_summary", doc, {"summarize": False})
    table_chunks = [c for c in chunks if c.meta.get("is_table")]
    assert table_chunks
    assert all(c.embed_text == c.text for c in table_chunks)


def test_table_summary_rejects_unsupported_fallback_chunker():
    doc = _fixture_documents()[0]
    with pytest.raises(ValueError):
        run_chunker("table_summary", doc, {"fallback_chunker": "markdown"})


def test_truncate_for_prompt_leaves_short_text_untouched():
    text = "a short table"
    assert _truncate_for_prompt(text, max_tokens=1000, encoding="cl100k_base") == text


def test_truncate_for_prompt_shortens_long_text():
    text = "word " * 5000
    truncated = _truncate_for_prompt(text, max_tokens=10, encoding="cl100k_base")
    assert len(truncated) < len(text)
    assert count_tokens(truncated, "cl100k_base") <= 10


# --------------------------------------------------------------------------- #
# AC-5: on filings, summarizing beats not summarizing on table-targeted
# queries -- needs a real, semantically useful summary (real API key)
# --------------------------------------------------------------------------- #


@requires_sentence_transformers
@requires_anthropic_api_key
def test_table_summary_beats_no_summary_on_a_table_targeted_query():  # AC-5
    from rag_lab import indexing, retrieval
    from rag_lab.chunks import load_chunk_set
    from rag_lab.jsonl import write_jsonl
    from rag_lab.paths import artifact_path as _artifact_path

    doc = _first_filings_doc_with_a_table()
    table_start, table_end = sorted(find_tables(doc.text))[0]
    table_text = doc.text[table_start:table_end]

    with_summary = run_chunker("table_summary", doc, {"summarize": True})
    without_summary = run_chunker("table_summary", doc, {"summarize": False})
    write_jsonl(_artifact_path("chunks", f"{with_summary[0].chunk_set_id}.jsonl"), with_summary)
    write_jsonl(
        _artifact_path("chunks", f"{without_summary[0].chunk_set_id}.jsonl"), without_summary
    )

    target_chunk_id = next(c.chunk_id for c in with_summary if table_text in c.text)

    # A query about what the table reports, phrased the way a user would ask
    # it -- answerable from the LLM summary's prose, not from table syntax.
    query = "What financial figures does this table break down, and for what period?"

    def _rank(chunk_set_id: str) -> int:
        manifest, _hit = indexing.build_index(chunk_set_id, "bge-small", {})
        _m, results = retrieval.run_retriever(
            manifest.index_id, "dense", query, k=len(load_chunk_set(chunk_set_id))
        )
        ranks = [r.rank for r in results if r.chunk.chunk_id == target_chunk_id]
        assert ranks, "the table chunk was never retrieved at all"
        return ranks[0]

    summary_rank = _rank(with_summary[0].chunk_set_id)
    no_summary_rank = _rank(without_summary[0].chunk_set_id)
    assert summary_rank < no_summary_rank


# --------------------------------------------------------------------------- #
# AC-6: chunk stats reports chunk_build_seconds for semantic, omits it
# elsewhere
# --------------------------------------------------------------------------- #


def test_compute_chunk_stats_reports_build_seconds_for_semantic_only(fast_semantic_embedder):
    doc = _real_documents("transcripts")[0]
    docs_by_id = {doc.doc_id: doc}

    semantic_stats = compute_chunk_stats(run_chunker("semantic", doc, {}), docs_by_id)
    fixed_stats = compute_chunk_stats(run_chunker("fixed", doc, {}), docs_by_id)
    table_summary_stats = compute_chunk_stats(
        run_chunker("table_summary", doc, {"mock_llm": True}), docs_by_id
    )

    assert semantic_stats.chunk_build_seconds is not None
    assert semantic_stats.chunk_build_seconds >= 0.0
    assert fixed_stats.chunk_build_seconds is None
    assert table_summary_stats.chunk_build_seconds is None


def test_cli_chunk_stats_shows_build_time_row_only_for_semantic(fast_semantic_embedder):  # AC-6
    semantic_run = runner.invoke(
        app, ["chunk", "run", "--corpus", "transcripts", "--chunker", "semantic"]
    )
    assert semantic_run.exit_code == 0, semantic_run.output
    semantic_id = re.search(r"(transcripts__semantic__\w+):", semantic_run.output).group(1)

    fixed_run = runner.invoke(
        app, ["chunk", "run", "--corpus", "transcripts", "--chunker", "fixed"]
    )
    fixed_id = re.search(r"(transcripts__fixed__\w+):", fixed_run.output).group(1)

    semantic_stats = runner.invoke(app, ["chunk", "stats", "--chunk-set", semantic_id])
    fixed_stats = runner.invoke(app, ["chunk", "stats", "--chunk-set", fixed_id])
    assert semantic_stats.exit_code == 0
    assert fixed_stats.exit_code == 0
    assert "chunk build time" in semantic_stats.output
    assert "chunk build time" not in fixed_stats.output


# --------------------------------------------------------------------------- #
# CLI: --mock-llm flag validation
# --------------------------------------------------------------------------- #


def test_cli_mock_llm_flag_rejected_for_non_table_summary_chunker():
    result = runner.invoke(
        app, ["chunk", "run", "--corpus", "filings", "--chunker", "fixed", "--mock-llm"]
    )
    assert result.exit_code == 1
    assert "--mock-llm" in result.output


def test_cli_chunk_run_table_summary_mock_llm():
    result = runner.invoke(
        app, ["chunk", "run", "--corpus", "filings", "--chunker", "table_summary", "--mock-llm"]
    )
    assert result.exit_code == 0, result.output
    match = re.search(r"(filings__table_summary__\w+):\s+(\d+) chunks", result.output)
    assert match
    chunk_set_id, count = match.group(1), int(match.group(2))
    path = artifact_path("chunks", f"{chunk_set_id}.jsonl")
    assert path.exists()
    assert len(read_jsonl(path, Chunk)) == count


def test_cli_chunk_run_semantic(fast_semantic_embedder):
    result = runner.invoke(
        app, ["chunk", "run", "--corpus", "transcripts", "--chunker", "semantic"]
    )
    assert result.exit_code == 0, result.output
    match = re.search(r"(transcripts__semantic__\w+):\s+(\d+) chunks", result.output)
    assert match


# --------------------------------------------------------------------------- #
# experiment/config.py: per-matrix-entry corpora allow-list
# --------------------------------------------------------------------------- #


class TestMatrixCorporaAllowList:
    def _config(self, chunker_entries: list[dict]) -> ExperimentConfig:
        return ExperimentConfig(
            name="test",
            corpora=["api_docs", "transcripts", "filings"],
            k=10,
            matrix={
                "chunker": chunker_entries,
                "embedder": [{"name": "bge-small", "params": {}}],
                "retriever": [{"name": "dense", "params": {}}],
            },
        )

    def test_corpora_none_is_unrestricted(self):
        cfg = self._config([{"name": "fixed", "params": {}}])
        cells = expand_cells(cfg)
        assert {c.corpus for c in cells} == {"api_docs", "transcripts", "filings"}

    def test_corpora_allow_list_restricts_cells(self):
        cfg = self._config(
            [
                {"name": "fixed", "params": {}},
                {"name": "semantic", "params": {}, "corpora": ["transcripts"]},
            ]
        )
        cells = expand_cells(cfg)
        semantic_corpora = {c.corpus for c in cells if c.chunker == "semantic"}
        assert semantic_corpora == {"transcripts"}
        fixed_corpora = {c.corpus for c in cells if c.chunker == "fixed"}
        assert fixed_corpora == {"api_docs", "transcripts", "filings"}

    def test_full_matrix_yaml_scopes_semantic_and_table_summary(self):
        from rag_lab.experiment.config import load_experiment_config

        config = load_experiment_config("config/experiments/full_matrix.yaml")
        cells = expand_cells(config)
        semantic_corpora = {c.corpus for c in cells if c.chunker == "semantic"}
        table_summary_corpora = {c.corpus for c in cells if c.chunker == "table_summary"}
        assert semantic_corpora == {"transcripts", "api_docs"}
        assert table_summary_corpora == {"filings"}
