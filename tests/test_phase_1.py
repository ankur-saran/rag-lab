"""Phase 1 acceptance tests.

The acceptance criteria from the plan are marked with `# AC-n`. A
module-scoped autouse fixture runs `corpus build --all` once, so this file is
self-contained and doesn't depend on `make verify-phase-1` having run first.
"""

from __future__ import annotations

import importlib.util
import shutil
import uuid

import pytest
from typer.testing import CliRunner

from rag_lab.cli import app
from rag_lab.corpus import (
    classify_script,
    compute_corpus_stats,
    find_document,
    list_documents_by_corpus,
)
from rag_lab.ids import make_doc_id
from rag_lab.jsonl import read_jsonl
from rag_lab.loaders import (
    REGISTRY,
    MarkdownLoader,
    TextLoader,
    discover_corpora,
    humanize,
    load_corpus,
)
from rag_lab.markup import count_code_blocks, count_tables, find_headings, mask_code_blocks
from rag_lab.normalize import normalize_text
from rag_lab.paths import artifact_path, corpora_dir, repo_root
from rag_lab.schemas import Document

runner = CliRunner()

EXPECTED_CORPORA = {"api_docs", "catalog", "contracts", "filings", "transcripts"}

# authentication.md's known values, cross-checked against the committed
# fixtures/documents/sample.jsonl -- ground truth with zero new content.
FIXTURE_DOC_ID = "409159541a40"


@pytest.fixture(scope="module", autouse=True)
def _build_all_corpora():
    result = runner.invoke(app, ["corpus", "build", "--all"])
    assert result.exit_code == 0, result.output
    yield


@pytest.fixture
def repo_scratch_corpus():
    """A throwaway corpus directory under corpora/.

    relative_source_path requires files live under the repo root, so pytest's
    tmp_path (usually outside the repo) doesn't work for loader tests -- this
    creates a scratch directory in the right place and removes it after.
    """
    name = f"_test_scratch_{uuid.uuid4().hex[:8]}"
    path = corpora_dir() / name
    path.mkdir(parents=True)
    try:
        yield name, path
    finally:
        shutil.rmtree(path, ignore_errors=True)


# --------------------------------------------------------------------------- #
# AC-1: build --all produces five valid Document JSONL files
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("corpus", sorted(EXPECTED_CORPORA))
def test_corpus_build_all_produces_valid_documents(corpus):  # AC-1
    docs = read_jsonl(artifact_path("documents", f"{corpus}.jsonl"), Document)
    assert 12 <= len(docs) <= 20
    assert all(d.corpus == corpus for d in docs)


def test_corpus_build_single_corpus():  # AC-1
    result = runner.invoke(app, ["corpus", "build", "--corpus", "api_docs"])
    assert result.exit_code == 0, result.output


def test_corpus_build_rejects_unknown_corpus():  # AC-1
    result = runner.invoke(app, ["corpus", "build", "--corpus", "does_not_exist"])
    assert result.exit_code == 1
    assert "unknown" in result.output.lower()


def test_corpus_build_requires_corpus_or_all():  # AC-1
    result = runner.invoke(app, ["corpus", "build"])
    assert result.exit_code == 1


def test_corpus_build_rejects_both_corpus_and_all():  # AC-1
    result = runner.invoke(app, ["corpus", "build", "--corpus", "api_docs", "--all"])
    assert result.exit_code == 1


def test_discover_corpora_finds_all_five():  # AC-1
    assert EXPECTED_CORPORA <= set(discover_corpora())


# --------------------------------------------------------------------------- #
# AC-2: doc_id formula and byte-identical rebuilds
# --------------------------------------------------------------------------- #


def test_corpus_build_is_byte_identical_across_runs():  # AC-2
    path = artifact_path("documents", "api_docs.jsonl")
    before = path.read_bytes()
    result = runner.invoke(app, ["corpus", "build", "--corpus", "api_docs"])
    assert result.exit_code == 0
    assert path.read_bytes() == before


def test_doc_id_matches_make_doc_id_formula():  # AC-2
    docs = read_jsonl(artifact_path("documents", "api_docs.jsonl"), Document)
    for doc in docs:
        assert doc.doc_id == make_doc_id(doc.corpus, doc.source_path)


def test_load_corpus_output_is_sorted_by_source_path():  # AC-2
    docs = load_corpus("api_docs", corpora_dir() / "api_docs")
    paths = [d.source_path for d in docs]
    assert paths == sorted(paths)


def test_document_text_is_reproducible():  # AC-2
    a = load_corpus("api_docs", corpora_dir() / "api_docs")
    b = load_corpus("api_docs", corpora_dir() / "api_docs")
    assert [d.text for d in a] == [d.text for d in b]


# --------------------------------------------------------------------------- #
# AC-3: stats report the signal each corpus was designed to exhibit
# --------------------------------------------------------------------------- #


def test_corpus_stats_api_docs_has_code_blocks():  # AC-3
    by_corpus = list_documents_by_corpus("api_docs")
    stats = compute_corpus_stats("api_docs", by_corpus["api_docs"])
    assert stats.code_block_count > 0


def test_corpus_stats_filings_has_tables():  # AC-3
    by_corpus = list_documents_by_corpus("filings")
    stats = compute_corpus_stats("filings", by_corpus["filings"])
    assert stats.table_count > 0


def test_corpus_stats_transcripts_have_no_headings():  # AC-3
    # Load-bearing for Phase 7's semantic-chunker demo -- see corpora/transcripts/SOURCE.md.
    by_corpus = list_documents_by_corpus("transcripts")
    stats = compute_corpus_stats("transcripts", by_corpus["transcripts"])
    assert stats.heading_count == 0


def test_corpus_stats_token_distribution_is_monotonic():  # AC-3
    by_corpus = list_documents_by_corpus("contracts")
    stats = compute_corpus_stats("contracts", by_corpus["contracts"])
    assert stats.token_min <= stats.token_p50 <= stats.token_p95 <= stats.token_max


def test_language_mix_detects_multiple_scripts_on_catalog():  # AC-3
    by_corpus = list_documents_by_corpus("catalog")
    stats = compute_corpus_stats("catalog", by_corpus["catalog"])
    assert len(stats.lang_mix) >= 3


def test_corpus_stats_cli_all_corpora():  # AC-3
    result = runner.invoke(app, ["corpus", "stats"])
    assert result.exit_code == 0
    for corpus in EXPECTED_CORPORA:
        assert corpus in result.output


def test_corpus_stats_cli_single_corpus():  # AC-3
    result = runner.invoke(app, ["corpus", "stats", "--corpus", "filings"])
    assert result.exit_code == 0
    assert "filings" in result.output


def test_corpus_stats_cli_unknown_corpus_exits_1():  # AC-3
    result = runner.invoke(app, ["corpus", "stats", "--corpus", "does_not_exist"])
    assert result.exit_code == 1


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Just plain English text.", "en"),
        ("これは日本語のテキストです。", "ja"),
        ("Это русский текст.", "ru"),
        ("Das ist ein deutscher Text mit ü.", "de"),
        ("Esto es un texto en español, ¿verdad?", "es"),
        ("Ceci est un texte en français avec des œufs.", "fr"),
    ],
)
def test_classify_script(text, expected):  # AC-3
    assert classify_script(text) == expected


# --------------------------------------------------------------------------- #
# AC-4 / AC-5: normalize-before-offset, the explicit pitfall
# --------------------------------------------------------------------------- #


def test_normalize_is_idempotent():  # AC-4
    sample = "line one\r\n\r\n\r\n\r\nline two   \r\n"
    once = normalize_text(sample)
    assert once == normalize_text(once)


def test_normalization_order_does_not_shift_offsets():  # AC-5, the pitfall test
    # An NFD-composed 'é' (e + combining acute accent), CRLF, extra blank
    # lines, and trailing whitespace -- exactly what would break if someone
    # later reordered normalize-vs-offset logic.
    nfd_e = "é"
    raw = f"before\r\n\r\n\r\n\r\ncaf{nfd_e} marker   \r\nafter"
    normalized = normalize_text(raw)

    assert "café marker" in normalized  # NFC 'é' == é
    idx = normalized.find("marker")
    assert normalized[idx : idx + len("marker")] == "marker"
    assert "\n\n\n" not in normalized  # 4 blank lines collapsed to at most 1


def test_markdown_loader_title_is_nfc(repo_scratch_corpus):  # AC-5
    name, path = repo_scratch_corpus
    nfd_e = "é"
    (path / "doc.md").write_text(f"# Caf{nfd_e} Notes\n\nBody.\n", encoding="utf-8")
    docs = load_corpus(name, path)
    assert docs[0].title == "Café Notes"


# --------------------------------------------------------------------------- #
# markup.py -- code-fence masking must actually suppress false positives
# --------------------------------------------------------------------------- #


def test_mask_code_blocks_preserves_line_count():
    text = "a\n```\nfake heading\n| x | y |\n```\nb\n"
    assert len(text.split("\n")) == len(mask_code_blocks(text).split("\n"))


def test_count_tables_ignores_tables_inside_code_fences():
    text = "# Doc\n\n```\n| a | b |\n| --- | --- |\n```\n"
    assert count_tables(text) == 0


def test_find_headings_ignores_hash_inside_code_fence():
    text = "# Real\n\n```\n# not real\n```\n"
    assert find_headings(text) == [(1, "Real")]


def test_count_code_blocks_counts_fences_on_unmasked_text():
    text = "```\ncode\n```\n\n```\nmore\n```\n"
    assert count_code_blocks(text) == 2


def test_count_tables_detects_a_real_table_outside_fences():
    text = "# Doc\n\n| a | b |\n| --- | --- |\n| 1 | 2 |\n"
    assert count_tables(text) == 1


def test_bare_horizontal_rule_is_not_a_table():
    text = "Some text\n\n---\n\nMore text\n"
    assert count_tables(text) == 0


# --------------------------------------------------------------------------- #
# Loaders -- validated first against the existing fixtures/raw files
# --------------------------------------------------------------------------- #


def test_markdown_loader_matches_known_fixture_values():
    docs = load_corpus("api_docs", repo_root() / "fixtures" / "raw" / "api_docs")
    by_name = {d.source_path.rsplit("/", 1)[-1]: d for d in docs}
    auth = by_name["authentication.md"]
    assert auth.title == "Authentication"
    assert auth.meta["section_count"] == 4
    assert auth.meta["char_count"] == 903


def test_loader_registry_dispatches_by_extension():
    assert isinstance(REGISTRY[".md"], MarkdownLoader)
    assert isinstance(REGISTRY[".txt"], TextLoader)


def test_humanize_formats_filename_stems():
    assert humanize("auth-token_lifecycle") == "Auth Token Lifecycle"
    assert humanize("sdks_and_clients") == "Sdks And Clients"


def test_load_corpus_skips_source_md(repo_scratch_corpus):
    name, path = repo_scratch_corpus
    (path / "SOURCE.md").write_text("# provenance\n", encoding="utf-8")
    (path / "doc.md").write_text("# Real Doc\n\nBody.\n", encoding="utf-8")
    docs = load_corpus(name, path)
    assert len(docs) == 1
    assert docs[0].title == "Real Doc"


def test_load_corpus_warns_and_skips_unknown_extension(repo_scratch_corpus, capsys):
    name, path = repo_scratch_corpus
    (path / "note.txt").write_text("hello\n", encoding="utf-8")
    (path / "image.png").write_bytes(b"\x89PNG stub")
    docs = load_corpus(name, path)
    assert len(docs) == 1
    assert "unrecognized_corpus_file_skipped" in capsys.readouterr().err


def test_markdown_loader_falls_back_to_filename_when_no_h1(repo_scratch_corpus):
    name, path = repo_scratch_corpus
    (path / "no_title_here.md").write_text("Just prose, no heading.\n", encoding="utf-8")
    docs = load_corpus(name, path)
    assert docs[0].title == humanize("no_title_here")


def test_catalog_no_h1_edge_cases_use_filename_fallback():
    """desk_lamp and water_bottle deliberately omit an H1 -- see corpora/catalog/SOURCE.md."""
    docs = read_jsonl(artifact_path("documents", "catalog.jsonl"), Document)
    by_name = {d.source_path.rsplit("/", 1)[-1]: d for d in docs}
    assert by_name["01_desk_lamp_en.md"].title == humanize("01_desk_lamp_en")
    assert by_name["02_water_bottle_en.md"].title == humanize("02_water_bottle_en")


def test_text_loader_sets_content_type_text():
    docs = read_jsonl(artifact_path("documents", "transcripts.jsonl"), Document)
    assert all(d.content_type == "text" for d in docs)


# --------------------------------------------------------------------------- #
# corpus show
# --------------------------------------------------------------------------- #


def test_corpus_show_prints_document_head():
    docs = read_jsonl(artifact_path("documents", "api_docs.jsonl"), Document)
    doc_id = docs[0].doc_id
    result = runner.invoke(app, ["corpus", "show", "--doc-id", doc_id, "--chars", "50"])
    assert result.exit_code == 0
    assert doc_id in result.output


def test_corpus_show_unknown_doc_id_exits_1():
    result = runner.invoke(app, ["corpus", "show", "--doc-id", "does_not_exist_000"])
    assert result.exit_code == 1


def test_find_document_falls_back_to_fixture_when_nothing_built(monkeypatch, tmp_path):
    monkeypatch.setattr("rag_lab.corpus.artifacts_dir", lambda: tmp_path)
    doc = find_document(FIXTURE_DOC_ID)
    assert doc is not None
    assert doc.title == "Authentication"


# --------------------------------------------------------------------------- #
# Fixture-determinism guard -- normalize.py extraction must not have drifted
# --------------------------------------------------------------------------- #


def test_build_fixtures_normalize_delegates_to_shared_function():
    spec = importlib.util.spec_from_file_location(
        "build_fixtures", repo_root() / "scripts" / "build_fixtures.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.normalize is normalize_text
