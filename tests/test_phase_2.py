"""Phase 2 acceptance tests.

The five numbered criteria from the plan are marked with `# AC-n`. A
module-scoped autouse fixture runs `corpus build --all` once, so this file is
self-contained and doesn't depend on `make verify-phase-1` having run first —
mirrors `test_phase_1.py`.
"""

from __future__ import annotations

import re

import pytest
from typer.testing import CliRunner

from rag_lab.chunkers import REGISTRY, available_chunkers, resolve_params, run_chunker
from rag_lab.chunkers.base import (
    ChunkSpec,
    count_tokens,
    finalize_chunks,
    token_char_offsets,
    token_windows,
)
from rag_lab.chunkers.markdown_chunker import MarkdownChunker
from rag_lab.chunkers.recursive import split_text_recursive
from rag_lab.chunkers.sentence_window import sentence_spans
from rag_lab.chunks import (
    compute_chunk_stats,
    documents_for_chunks,
    load_chunk_set,
)
from rag_lab.cli import app
from rag_lab.jsonl import read_jsonl
from rag_lab.markup import CODE_FENCE_RE, find_tables
from rag_lab.paths import artifact_path, fixture_path
from rag_lab.schemas import Chunk, Document

runner = CliRunner()

CHUNKER_NAMES = ["fixed", "recursive", "markdown", "sentence_window"]

# Chosen empirically (see the token-count survey below `_api_docs_docs`): every
# api_docs document is 279-378 tokens, so the *default* chunk_tokens (512)
# never produces more than one chunk per doc and can never split a fence.
# 128 is small enough to force a split for `fixed` while `markdown` still
# never does, at 0 overlap and at the default 32.
DEMO_FIXED_PARAMS = {"chunk_tokens": 128, "overlap_tokens": 32}
DEMO_MARKDOWN_PARAMS = {"max_tokens": 128}


@pytest.fixture(scope="module", autouse=True)
def _build_all_corpora():
    result = runner.invoke(app, ["corpus", "build", "--all"])
    assert result.exit_code == 0, result.output
    yield


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


def _split_fence_count(doc: Document, chunks: list[Chunk]) -> int:
    boundaries = sorted({c.char_start for c in chunks} | {c.char_end for c in chunks})
    return sum(
        1
        for m in CODE_FENCE_RE.finditer(doc.text)
        if any(m.start() < b < m.end() for b in boundaries)
    )


# --------------------------------------------------------------------------- #
# AC-1: all four chunkers run against the fixture, no prior phase
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", CHUNKER_NAMES)
def test_all_chunkers_run_against_fixture_documents(name):  # AC-1
    docs = _fixture_documents()
    assert docs
    for doc in docs:
        chunks = run_chunker(name, doc, {})
        assert chunks
        assert all(isinstance(c, Chunk) for c in chunks)
        assert all(c.chunker == name for c in chunks)


def test_registry_exposes_exactly_the_four_baseline_chunkers():
    assert available_chunkers() == sorted(CHUNKER_NAMES)


def test_run_chunker_rejects_unknown_name():
    with pytest.raises(ValueError):
        run_chunker("not_a_real_chunker", _fixture_documents()[0], {})


# --------------------------------------------------------------------------- #
# AC-2 / AC-3: offset and coverage invariants, swept across every real corpus
# --------------------------------------------------------------------------- #


ALL_CORPORA = ["api_docs", "catalog", "contracts", "filings", "transcripts"]


@pytest.mark.parametrize("chunker_name", CHUNKER_NAMES)
@pytest.mark.parametrize("corpus", ALL_CORPORA)
def test_offset_and_coverage_invariants_hold_everywhere(chunker_name, corpus):  # AC-2, AC-3
    for doc in _real_documents(corpus):
        chunks = run_chunker(chunker_name, doc, {})
        for c in chunks:
            # AC-2: offset invariant. Every chunker here produces genuine
            # slices (see chunkers/base.py's ChunkSpec docstring), so this is
            # the strict equality, not the plan's "contains" fallback.
            assert doc.text[c.char_start : c.char_end] == c.text
        # AC-3: coverage invariant, deduped (overlapping chunkers would
        # otherwise double-count and exceed 100%, per the plan's own pitfall
        # note -- `_coverage_ratio` already unions before measuring).
        assert _coverage_ratio(doc, chunks) >= 0.99


def test_finalize_chunks_rejects_a_spec_whose_text_does_not_match_the_slice():
    doc = _fixture_documents()[0]
    bad = [ChunkSpec(char_start=0, char_end=5, text="wrong")]
    with pytest.raises(ValueError):
        finalize_chunks(bad, doc, "fixed", {})


# --------------------------------------------------------------------------- #
# AC-4: determinism
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", CHUNKER_NAMES)
def test_determinism_same_params_same_output(name):  # AC-4
    doc = _real_documents("api_docs")[0]
    a = run_chunker(name, doc, {})
    b = run_chunker(name, doc, {})
    assert [c.model_dump_json() for c in a] == [c.model_dump_json() for c in b]
    assert a[0].chunk_set_id == b[0].chunk_set_id


def test_cli_chunk_run_is_byte_identical_across_runs():  # AC-4
    result1 = runner.invoke(app, ["chunk", "run", "--corpus", "api_docs", "--chunker", "recursive"])
    assert result1.exit_code == 0, result1.output
    chunk_set_id = re.search(r"(api_docs__recursive__\w+):", result1.output).group(1)
    path = artifact_path("chunks", f"{chunk_set_id}.jsonl")
    before = path.read_bytes()

    result2 = runner.invoke(app, ["chunk", "run", "--corpus", "api_docs", "--chunker", "recursive"])
    assert result2.exit_code == 0
    assert path.read_bytes() == before


def test_different_params_yield_different_chunk_set_ids():  # AC-4
    doc = _real_documents("api_docs")[0]
    a = run_chunker("fixed", doc, {"chunk_tokens": 256})
    b = run_chunker("fixed", doc, {"chunk_tokens": 128})
    assert a[0].chunk_set_id != b[0].chunk_set_id


# --------------------------------------------------------------------------- #
# AC-5: the punchline -- markdown never splits a fence, fixed does
# --------------------------------------------------------------------------- #


def test_fixed_splits_code_blocks_on_api_docs_markdown_never_does():  # AC-5
    docs = _real_documents("api_docs")
    fixed_splits = sum(
        _split_fence_count(doc, run_chunker("fixed", doc, DEMO_FIXED_PARAMS)) for doc in docs
    )
    markdown_splits = sum(
        _split_fence_count(doc, run_chunker("markdown", doc, DEMO_MARKDOWN_PARAMS)) for doc in docs
    )
    assert fixed_splits > 0
    assert markdown_splits == 0


def test_api_docs_default_chunk_tokens_never_splits_a_fence_either():
    """Documents the reason AC-5's test above uses chunk_tokens=128 rather
    than the default 512: every api_docs doc is well under 512 tokens, so the
    default produces exactly one chunk per doc and the split can't occur at
    all. If this ever fails, the corpora changed and DEMO_FIXED_PARAMS above
    should be revisited.
    """
    docs = _real_documents("api_docs")
    assert all(count_tokens(d.text) < 512 for d in docs)


# --------------------------------------------------------------------------- #
# fixed / base token-window mechanics
# --------------------------------------------------------------------------- #


def test_token_char_offsets_round_trips_multibyte_text():
    text = "hello café 世界 \U0001f389 world, repeated repeated repeated"
    offsets = token_char_offsets(text)
    assert offsets[0] == 0
    assert offsets[-1] == len(text)
    assert offsets == sorted(offsets)


def test_token_windows_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValueError):
        token_windows("some text here", 0, chunk_tokens=10, overlap_tokens=10)


def test_token_windows_empty_text_yields_no_windows():
    assert token_windows("", 0, chunk_tokens=10) == []


def test_fixed_chunker_rejects_overlap_not_smaller_than_chunk_tokens():
    doc = _fixture_documents()[0]
    with pytest.raises(ValueError):
        run_chunker("fixed", doc, {"chunk_tokens": 50, "overlap_tokens": 50})


def test_fixed_chunker_windows_advance_and_respect_budget():
    doc = _real_documents("api_docs")[0]
    chunks = run_chunker("fixed", doc, {"chunk_tokens": 100, "overlap_tokens": 20})
    assert len(chunks) > 1
    assert all(c.token_count <= 100 for c in chunks)
    ordinals = [c.ordinal for c in chunks]
    assert ordinals == sorted(ordinals)


# --------------------------------------------------------------------------- #
# recursive: separator cascade + greedy merge with overlap
# --------------------------------------------------------------------------- #


def test_recursive_zero_overlap_produces_no_overlapping_spans():
    text = "Sentence one. Sentence two. Sentence three. Sentence four. Sentence five."
    spans = split_text_recursive(text, 0, chunk_tokens=6, overlap_tokens=0, separators=[". ", ""])
    for (_, e1), (s2, _) in zip(spans, spans[1:], strict=False):
        assert e1 <= s2  # strictly non-overlapping, contiguous or gapless


def test_recursive_nonzero_overlap_actually_overlaps_when_it_should():
    text = " ".join(f"word{i}" for i in range(200))
    spans = split_text_recursive(text, 0, chunk_tokens=20, overlap_tokens=10, separators=[" ", ""])
    assert len(spans) > 1
    assert any(s2 < e1 for (_, e1), (s2, _) in zip(spans, spans[1:], strict=False))


def test_recursive_protected_span_is_never_cut_even_when_oversized():
    fence = "```\n" + ("x " * 200) + "\n```"
    text = f"intro text.\n\n{fence}\n\noutro text."
    fence_start = text.index("```")
    fence_end = text.rindex("```") + 3
    spans = split_text_recursive(
        text,
        0,
        chunk_tokens=10,
        overlap_tokens=0,
        separators=["\n\n", " ", ""],
        protected_spans=[(fence_start, fence_end)],
    )
    # the fence must appear whole, inside exactly one span
    assert any(s <= fence_start and fence_end <= e for s, e in spans)


def test_recursive_falls_back_through_separators_that_do_not_occur():
    # no "\n\n" or ". " in this text at all -- must fall through to " " and "".
    text = "a very long run of words with no sentence punctuation " * 10
    spans = split_text_recursive(
        text, 0, chunk_tokens=8, overlap_tokens=0, separators=["\n\n", ". ", " ", ""]
    )
    assert spans
    assert spans[0][0] == 0
    assert spans[-1][1] == len(text)


# --------------------------------------------------------------------------- #
# markdown: heading_path, prepend_heading_path, keep_code_blocks_intact,
# min_tokens merging
# --------------------------------------------------------------------------- #


def test_markdown_heading_path_excludes_own_heading_includes_ancestors():
    doc = next(d for d in _real_documents("api_docs") if d.title == "Authentication")
    chunks = run_chunker("markdown", doc, {"min_tokens": 0})
    with_path = [c for c in chunks if c.heading_path]
    assert with_path
    for c in with_path:
        # the section's own heading text is visible in `text`, not repeated
        # as the last element of heading_path
        assert c.heading_path[-1] != c.text.splitlines()[0].lstrip("#").strip()


def test_markdown_prepend_heading_path_leaves_text_clean():
    doc = next(d for d in _real_documents("api_docs") if d.title == "Authentication")
    chunks = run_chunker("markdown", doc, {"min_tokens": 0, "prepend_heading_path": True})
    for c in chunks:
        if c.heading_path:
            assert c.embed_text == " > ".join(c.heading_path) + "\n\n" + c.text
            assert c.embed_text != c.text
        else:
            assert c.embed_text == c.text


def test_markdown_prepend_heading_path_false_keeps_embed_text_equal_to_text():
    doc = next(d for d in _real_documents("api_docs") if d.title == "Authentication")
    chunks = run_chunker("markdown", doc, {"prepend_heading_path": False})
    assert all(c.embed_text == c.text for c in chunks)


def test_markdown_keep_code_blocks_intact_never_splits_a_fence_under_pressure():
    doc = next(d for d in _real_documents("api_docs") if d.title == "Authentication")
    # a max_tokens far smaller than any fence forces the fallback splitter to
    # confront every fence in the document
    chunks = run_chunker("markdown", doc, {"max_tokens": 16, "keep_code_blocks_intact": True})
    assert _split_fence_count(doc, chunks) == 0


def test_markdown_min_tokens_merges_undersized_sections_forward():
    doc = next(d for d in _real_documents("api_docs") if d.title == "Authentication")
    unmerged = run_chunker("markdown", doc, {"min_tokens": 0})
    merged = run_chunker("markdown", doc, {"min_tokens": 64})
    assert len(merged) < len(unmerged)
    assert all(c.token_count >= 64 for c in merged[:-1])  # trailing section may stay small


def test_markdown_heading_levels_filters_boundary_depth():
    doc = next(d for d in _real_documents("api_docs") if d.title == "Authentication")
    only_h1 = run_chunker("markdown", doc, {"heading_levels": [1], "min_tokens": 0})
    h1_and_h2 = run_chunker("markdown", doc, {"heading_levels": [1, 2], "min_tokens": 0})
    assert len(only_h1) <= len(h1_and_h2)


def test_markdown_chunker_handles_a_document_with_no_headings():
    doc = _real_documents("transcripts")[0]
    chunks = run_chunker("markdown", doc, {})
    assert chunks
    assert _coverage_ratio(doc, chunks) >= 0.99


def test_markdown_default_params_are_exposed_on_the_registry():
    assert REGISTRY["markdown"] is not None
    md = MarkdownChunker()
    assert md.default_params["max_tokens"] == 768
    assert md.default_params["heading_levels"] == [1, 2, 3]


# --------------------------------------------------------------------------- #
# sentence_window: splitter correctness, contiguous coverage, embed/text split
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "text,expected",
    [
        (
            "Dr. Smith arrived early. She left late.",
            ["Dr. Smith arrived early. ", "She left late."],
        ),
        ("J. Smith wrote it. It was thorough.", ["J. Smith wrote it. ", "It was thorough."]),
        (
            "Solstice Analytics, Inc. Q1 results follow.",
            ["Solstice Analytics, Inc. Q1 results follow."],
        ),
        ('He said "Stop!" Then he left.', ['He said "Stop!" ', "Then he left."]),
    ],
)
def test_sentence_spans_abbreviation_and_initial_guards(text, expected):
    spans = sentence_spans(text)
    assert [text[s:e] for s, e in spans] == expected


def test_sentence_spans_cover_the_whole_text_contiguously():
    doc = _real_documents("transcripts")[0]
    spans = sentence_spans(doc.text)
    assert spans[0][0] == 0
    assert spans[-1][1] == len(doc.text)
    for (_, e1), (s2, _) in zip(spans, spans[1:], strict=False):
        assert e1 == s2  # no gaps, no overlap


def test_sentence_window_embed_text_is_bare_sentence_text_is_widened():
    doc = _real_documents("transcripts")[0]
    chunks = run_chunker("sentence_window", doc, {"window_size": 2})
    widened = [c for c in chunks if len(c.embed_text) < len(c.text)]
    assert widened  # window_size > 0 must actually widen at least some chunks
    for c in chunks:
        assert c.embed_text in c.text


def test_sentence_window_clamps_at_document_edges():
    doc = _real_documents("transcripts")[0]
    chunks = run_chunker("sentence_window", doc, {"window_size": 3})
    chunks = sorted(chunks, key=lambda c: c.ordinal)
    assert chunks[0].char_start == 0
    assert chunks[-1].char_end == len(doc.text)


def test_sentence_window_stride_reduces_chunk_count():
    doc = _real_documents("transcripts")[0]
    dense = run_chunker("sentence_window", doc, {"stride": 1})
    sparse = run_chunker("sentence_window", doc, {"stride": 2})
    assert len(sparse) < len(dense)


def test_sentence_window_rejects_unsupported_splitter():
    doc = _fixture_documents()[0]
    with pytest.raises(ValueError):
        run_chunker("sentence_window", doc, {"sentence_splitter": "pysbd"})


# --------------------------------------------------------------------------- #
# markup.find_tables
# --------------------------------------------------------------------------- #


def test_find_tables_locates_a_real_table():
    text = "# Doc\n\n| a | b |\n| --- | --- |\n| 1 | 2 |\n\nTrailing prose.\n"
    spans = find_tables(text)
    assert len(spans) == 1
    start, end = spans[0]
    assert text[start:end] == "| a | b |\n| --- | --- |\n| 1 | 2 |"


def test_find_tables_ignores_a_table_inside_a_fence():
    text = "```\n| a | b |\n| --- | --- |\n```\n"
    assert find_tables(text) == []


def test_find_tables_offsets_are_valid_after_an_earlier_fence():
    # regression guard: offsets must be resolved against the *original* text,
    # not the (shorter) fence-masked copy -- a table after an earlier fence is
    # exactly the case that would drift if that weren't handled correctly.
    text = (
        "intro\n\n```python\nprint('this fence is much longer than the table below')\n```"
        "\n\nmore text\n\n| x | y |\n| --- | --- |\n| 1 | 2 |\n"
    )
    spans = find_tables(text)
    assert len(spans) == 1
    start, end = spans[0]
    assert text[start:end] == "| x | y |\n| --- | --- |\n| 1 | 2 |"


# --------------------------------------------------------------------------- #
# chunkers/__init__.py registry
# --------------------------------------------------------------------------- #


def test_resolve_params_layers_overrides_on_defaults():
    resolved = resolve_params("fixed", {"chunk_tokens": 100})
    assert resolved["chunk_tokens"] == 100
    assert resolved["overlap_tokens"] == REGISTRY["fixed"].default_params["overlap_tokens"]


def test_resolve_params_rejects_unknown_chunker():
    with pytest.raises(ValueError):
        resolve_params("nope", {})


# --------------------------------------------------------------------------- #
# CLI: chunk run / stats / show / diff
# --------------------------------------------------------------------------- #


def test_cli_chunk_run_writes_the_expected_artifact_path():
    result = runner.invoke(app, ["chunk", "run", "--corpus", "api_docs", "--chunker", "fixed"])
    assert result.exit_code == 0, result.output
    match = re.search(r"(api_docs__fixed__\w+):\s+(\d+) chunks", result.output)
    assert match
    chunk_set_id, count = match.group(1), int(match.group(2))
    path = artifact_path("chunks", f"{chunk_set_id}.jsonl")
    assert path.exists()
    assert len(read_jsonl(path, Chunk)) == count


def test_cli_chunk_run_rejects_unknown_chunker():
    result = runner.invoke(app, ["chunk", "run", "--corpus", "api_docs", "--chunker", "nope"])
    assert result.exit_code == 1
    assert "unknown chunker" in result.output.lower()


def test_cli_chunk_run_rejects_unknown_corpus():
    result = runner.invoke(app, ["chunk", "run", "--corpus", "nope", "--chunker", "fixed"])
    assert result.exit_code == 1


def test_cli_chunk_run_honours_params_flag():
    result = runner.invoke(
        app,
        [
            "chunk",
            "run",
            "--corpus",
            "api_docs",
            "--chunker",
            "fixed",
            "--params",
            "chunk_tokens=64",
            "--params",
            "overlap_tokens=8",
        ],
    )
    assert result.exit_code == 0, result.output
    chunk_set_id = re.search(r"(api_docs__fixed__\w+):", result.output).group(1)
    chunks = read_jsonl(artifact_path("chunks", f"{chunk_set_id}.jsonl"), Chunk)
    assert all(c.chunker_params["chunk_tokens"] == 64 for c in chunks)


def test_cli_chunk_stats_reports_the_ac5_punchline():
    fixed_run = runner.invoke(
        app,
        [
            "chunk",
            "run",
            "--corpus",
            "api_docs",
            "--chunker",
            "fixed",
            "--params",
            "chunk_tokens=128",
            "--params",
            "overlap_tokens=32",
        ],
    )
    markdown_run = runner.invoke(
        app,
        [
            "chunk",
            "run",
            "--corpus",
            "api_docs",
            "--chunker",
            "markdown",
            "--params",
            "max_tokens=128",
        ],
    )
    fixed_id = re.search(r"(api_docs__fixed__\w+):", fixed_run.output).group(1)
    markdown_id = re.search(r"(api_docs__markdown__\w+):", markdown_run.output).group(1)

    fixed_result = runner.invoke(app, ["chunk", "stats", "--chunk-set", fixed_id])
    markdown_result = runner.invoke(app, ["chunk", "stats", "--chunk-set", markdown_id])
    assert fixed_result.exit_code == 0
    assert markdown_result.exit_code == 0

    fixed_chunks = load_chunk_set(fixed_id)
    fixed_stats = compute_chunk_stats(fixed_chunks, documents_for_chunks(fixed_chunks))
    markdown_chunks = load_chunk_set(markdown_id)
    markdown_stats = compute_chunk_stats(markdown_chunks, documents_for_chunks(markdown_chunks))
    assert fixed_stats.split_code_block_count > 0
    assert markdown_stats.split_code_block_count == 0


def test_cli_chunk_stats_unknown_chunk_set_falls_back_to_fixture_with_warning():
    # CliRunner merges the process's stdout/stderr into `result.output` by
    # default, so the structlog warning shows up there rather than under
    # capsys — unlike test_phase_0.py's direct (non-CLI) equivalent.
    result = runner.invoke(app, ["chunk", "stats", "--chunk-set", "does_not_exist"])
    assert result.exit_code == 0
    assert "using_fixture" in result.output or "artifact_missing_using_fixture" in result.output


def test_cli_chunk_show_renders_boundaries_for_a_document():
    run_result = runner.invoke(
        app, ["chunk", "run", "--corpus", "api_docs", "--chunker", "markdown"]
    )
    chunk_set_id = re.search(r"(api_docs__markdown__\w+):", run_result.output).group(1)
    chunks = read_jsonl(artifact_path("chunks", f"{chunk_set_id}.jsonl"), Chunk)
    doc_id = chunks[0].doc_id

    result = runner.invoke(app, ["chunk", "show", "--chunk-set", chunk_set_id, "--doc-id", doc_id])
    assert result.exit_code == 0, result.output
    assert doc_id in result.output
    assert "chunk 0" in result.output


def test_cli_chunk_show_unknown_doc_id_exits_1():
    run_result = runner.invoke(app, ["chunk", "run", "--corpus", "api_docs", "--chunker", "fixed"])
    chunk_set_id = re.search(r"(api_docs__fixed__\w+):", run_result.output).group(1)
    result = runner.invoke(
        app, ["chunk", "show", "--chunk-set", chunk_set_id, "--doc-id", "does_not_exist"]
    )
    assert result.exit_code == 1


def test_cli_chunk_diff_renders_both_chunk_sets_side_by_side():
    fixed_run = runner.invoke(app, ["chunk", "run", "--corpus", "api_docs", "--chunker", "fixed"])
    markdown_run = runner.invoke(
        app, ["chunk", "run", "--corpus", "api_docs", "--chunker", "markdown"]
    )
    fixed_id = re.search(r"(api_docs__fixed__\w+):", fixed_run.output).group(1)
    markdown_id = re.search(r"(api_docs__markdown__\w+):", markdown_run.output).group(1)
    doc_id = read_jsonl(artifact_path("chunks", f"{fixed_id}.jsonl"), Chunk)[0].doc_id

    result = runner.invoke(
        app, ["chunk", "diff", "--a", fixed_id, "--b", markdown_id, "--doc-id", doc_id]
    )
    assert result.exit_code == 0, result.output
    assert fixed_id in result.output
    assert markdown_id in result.output


def test_cli_chunk_run_works_with_no_prior_phase(monkeypatch, tmp_path):
    """The independence contract, exercised end to end through the CLI: point
    both `paths` and `corpus`'s bound reference at an empty artifacts dir, so
    `chunk run` has nothing to resolve except the committed fixture."""
    monkeypatch.setattr("rag_lab.paths.artifacts_dir", lambda: tmp_path)
    monkeypatch.setattr("rag_lab.corpus.artifacts_dir", lambda: tmp_path)

    result = runner.invoke(app, ["chunk", "run", "--corpus", "api_docs", "--chunker", "markdown"])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "chunks").exists()


# --------------------------------------------------------------------------- #
# chunks.py stats plumbing
# --------------------------------------------------------------------------- #


def test_compute_chunk_stats_orphan_rate_and_token_distribution():
    doc = _real_documents("api_docs")[0]
    chunks = run_chunker("fixed", doc, {"chunk_tokens": 128, "overlap_tokens": 0})
    docs_by_id = {doc.doc_id: doc}
    stats = compute_chunk_stats(chunks, docs_by_id)
    assert stats.count == len(chunks)
    assert stats.token_min <= stats.token_p50 <= stats.token_p95 <= stats.token_max
    assert 0.0 <= stats.orphan_rate <= 1.0
    assert 0.0 <= stats.split_sentence_rate <= 1.0


def test_compute_chunk_stats_rejects_empty_input():
    with pytest.raises(ValueError):
        compute_chunk_stats([], {})
