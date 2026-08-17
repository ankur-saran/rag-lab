"""Phase 0 acceptance tests.

The four numbered criteria from the plan are marked with `# AC-n`. Everything
else guards an invariant that later phases will silently depend on.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest
from typer.testing import CliRunner

from rag_lab import __version__
from rag_lab.cli import app
from rag_lab.config import canonical_json, load_config, params_hash, parse_overrides
from rag_lab.ids import (
    make_chunk_id,
    make_chunk_set_id,
    make_doc_id,
    make_index_id,
    split_for,
)
from rag_lab.jsonl import read_jsonl
from rag_lab.paths import (
    ArtifactNotFoundError,
    artifact_path,
    fixture_path,
    is_fixture,
    resolve_artifact,
)
from rag_lab.schemas import Chunk, Document, EvalPair

runner = CliRunner()

PHASE_SUBCOMMANDS = [
    "corpus",
    "chunk",
    "index",
    "retrieve",
    "evalset",
    "experiment",
    "agent",
    "doctor",
]


# --------------------------------------------------------------------------- #
# AC-1: doctor exits 0 with only core installed
# --------------------------------------------------------------------------- #


def test_doctor_exits_zero_with_core_only():  # AC-1
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0, result.output
    assert "rag-lab doctor" in result.output


def test_doctor_reports_optional_groups_without_failing():  # AC-1
    from rag_lab.doctor import run_checks

    checks = {c.name: c for c in run_checks()}
    assert checks["deps:core"].status == "ok"
    # Optional groups may be absent; absence must never be fatal.
    for group in ("embed", "agents", "app"):
        assert checks[f"deps:{group}"].status in {"ok", "warn"}


def test_doctor_never_fails_on_missing_api_key():  # AC-1
    from rag_lab.doctor import run_checks

    key_check = next(c for c in run_checks() if c.name == "anthropic_api_key")
    assert key_check.status in {"ok", "warn"}


# --------------------------------------------------------------------------- #
# AC-2: --help lists every phase subcommand, stubs included
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", PHASE_SUBCOMMANDS)
def test_help_lists_every_phase_subcommand(name):  # AC-2
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert name in result.output


def test_stubs_exit_with_code_one_and_name_their_phase():  # AC-2
    # `chunk run` was Phase 2's example, then `index build` was Phase 3's,
    # then `retrieve query` was Phase 4's -- all three have since been
    # implemented, so this now exercises the next real stub instead:
    # `evalset build` (Phase 5). The point of the test is unchanged: an
    # unimplemented phase command exits 1 and names its phase.
    result = runner.invoke(app, ["evalset", "build", "--corpus", "api_docs"])
    assert result.exit_code == 1
    assert "Phase 5" in result.output


def test_version_command():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.output


# --------------------------------------------------------------------------- #
# AC-3: params_hash is order-independent
# --------------------------------------------------------------------------- #


def test_params_hash_is_key_order_independent():  # AC-3
    assert params_hash({"a": 1, "b": 2}) == params_hash({"b": 2, "a": 1})


def test_params_hash_is_value_sensitive():  # AC-3
    assert params_hash({"a": 1}) != params_hash({"a": 2})


def test_params_hash_ignores_annotation_keys():  # AC-3
    base = {"chunk_tokens": 512}
    annotated = {"chunk_tokens": 512, "description": "the default", "comment": "x"}
    assert params_hash(base) == params_hash(annotated)


def test_params_hash_honours_local_nohash_list():  # AC-3
    a = {"chunk_tokens": 512, "label": "v1", "_nohash": ["label"]}
    b = {"chunk_tokens": 512, "label": "v2", "_nohash": ["label"]}
    assert params_hash(a) == params_hash(b)


def test_canonical_json_is_stable_across_nesting():
    left = {"outer": {"b": [3, 2], "a": 1}}
    right = {"outer": {"a": 1, "b": [3, 2]}}
    assert canonical_json(left) == canonical_json(right)


# --------------------------------------------------------------------------- #
# AC-4: resolve_artifact falls back to the fixture and warns
# --------------------------------------------------------------------------- #


def test_resolve_artifact_returns_fixture_when_no_id_given():  # AC-4
    path = resolve_artifact("chunks", None)
    assert path == fixture_path("chunks")
    assert is_fixture(path)


def test_resolve_artifact_warns_on_fallback(capsys):  # AC-4
    resolve_artifact("chunks", "does_not_exist")
    stderr = capsys.readouterr().err
    assert "artifact_missing_using_fixture" in stderr or "using_fixture" in stderr


def test_resolve_artifact_raises_when_fixtures_disabled():  # AC-4
    with pytest.raises(ArtifactNotFoundError):
        resolve_artifact("chunks", "does_not_exist", allow_fixture=False)


def test_artifact_path_rejects_unknown_kind():
    with pytest.raises(ValueError):
        artifact_path("nonsense", "x.jsonl")


# --------------------------------------------------------------------------- #
# Identity determinism — caching depends entirely on this
# --------------------------------------------------------------------------- #


def test_ids_are_deterministic():
    assert make_doc_id("api_docs", "a/b.md") == make_doc_id("api_docs", "a/b.md")
    assert make_doc_id("api_docs", "a/b.md") != make_doc_id("contracts", "a/b.md")


def test_doc_id_normalizes_path_separators():
    assert make_doc_id("c", "a/b.md") == make_doc_id("c", "a\\b.md")


def test_sha1_delimiter_prevents_concatenation_collisions():
    assert make_doc_id("ab", "c.md") != make_doc_id("a", "bc.md")


def test_chunk_set_and_index_ids_track_params():
    a = make_chunk_set_id("api_docs", "fixed", {"chunk_tokens": 512})
    b = make_chunk_set_id("api_docs", "fixed", {"chunk_tokens": 256})
    assert a != b
    assert a.startswith("api_docs__fixed__")
    assert make_index_id(a, "bge-small", {}) != make_index_id(a, "bge-small", {"truncate_dim": 256})


def test_chunk_id_varies_with_span():
    assert make_chunk_id("d", 0, 10, "sig") != make_chunk_id("d", 0, 11, "sig")


def test_split_is_deterministic_and_roughly_balanced():
    ids = [f"q{i:04d}" for i in range(2000)]
    splits = [split_for(i) for i in ids]
    assert splits == [split_for(i) for i in ids]
    train = splits.count("train") / len(splits)
    assert 0.55 < train < 0.65


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #


def test_chunk_rejects_inverted_span():
    with pytest.raises(ValueError):
        Chunk(
            chunk_id="c",
            doc_id="d",
            corpus="x",
            chunk_set_id="s",
            text="t",
            embed_text="t",
            char_start=10,
            char_end=5,
            token_count=1,
            ordinal=0,
            chunker="fixed",
        )


def test_schemas_forbid_unknown_fields():
    """Typos in an artifact writer must fail loudly, not vanish into the record."""
    with pytest.raises(ValueError):
        Document(
            doc_id="d",
            corpus="c",
            source_path="p",
            title="t",
            text="x",
            typo_field="oops",
        )


# --------------------------------------------------------------------------- #
# Fixtures validate and satisfy the invariants Phase 2 will test
# --------------------------------------------------------------------------- #


def test_fixture_documents_validate():
    docs = read_jsonl(fixture_path("documents"), Document)
    assert len(docs) >= 3
    assert all(d.text for d in docs)


def test_fixture_chunks_validate_and_respect_offsets():
    docs = {d.doc_id: d for d in read_jsonl(fixture_path("documents"), Document)}
    chunks = read_jsonl(fixture_path("chunks"), Chunk)
    assert chunks
    for chunk in chunks:
        assert docs[chunk.doc_id].text[chunk.char_start : chunk.char_end] == chunk.text


def test_fixture_chunks_cover_the_documents():
    docs = {d.doc_id: d for d in read_jsonl(fixture_path("documents"), Document)}
    chunks = read_jsonl(fixture_path("chunks"), Chunk)
    covered: dict[str, set[int]] = {doc_id: set() for doc_id in docs}
    for chunk in chunks:
        covered[chunk.doc_id].update(range(chunk.char_start, chunk.char_end))
    for doc_id, doc in docs.items():
        non_ws = {i for i, ch in enumerate(doc.text) if not ch.isspace()}
        assert len(non_ws - covered[doc_id]) / max(1, len(non_ws)) < 0.01


def test_fixture_evalset_gold_spans_resolve_to_their_quote():
    docs = {d.doc_id: d for d in read_jsonl(fixture_path("documents"), Document)}
    pairs = read_jsonl(fixture_path("evalset"), EvalPair)
    assert pairs
    for pair in pairs:
        start, end = pair.gold_char_span
        assert docs[pair.gold_doc_id].text[start:end] == pair.supporting_quote


def test_fixture_evalset_gold_chunks_are_derivable():
    """gold_chunk_ids must be re-derivable from the span, never authoritative."""
    chunks = read_jsonl(fixture_path("chunks"), Chunk)
    by_id = {c.chunk_id: c for c in chunks}
    for pair in read_jsonl(fixture_path("evalset"), EvalPair):
        assert pair.gold_chunk_ids
        for chunk_id in pair.gold_chunk_ids:
            chunk = by_id[chunk_id]
            start, end = pair.gold_char_span
            assert chunk.char_start < end and start < chunk.char_end


def test_fixture_results_are_readable():
    payload = json.loads((fixture_path("results") / "result.json").read_text(encoding="utf-8"))
    assert payload["run_id"] == "sample_run"
    assert "recall@5" in payload["metrics"]


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #


def test_config_defaults_load():
    cfg = load_config()
    assert cfg.seed == 42
    assert cfg.defaults.k == 10


def test_overrides_are_typed_and_nested():
    parsed = parse_overrides(["seed=7", "defaults.k=3", "json_logs=true", "x.y=hello"])
    assert parsed == {"seed": 7, "defaults": {"k": 3}, "json_logs": True, "x": {"y": "hello"}}


def test_override_precedence_beats_file():
    cfg = load_config(overrides=parse_overrides(["seed=99"]))
    assert cfg.seed == 99


def test_malformed_override_raises():
    with pytest.raises(ValueError):
        parse_overrides(["not-a-pair"])


# --------------------------------------------------------------------------- #
# Fixture determinism (mirrors `make verify-phase-0`)
# --------------------------------------------------------------------------- #


def test_fixture_regeneration_is_byte_identical():
    from rag_lab.paths import repo_root

    result = subprocess.run(
        [sys.executable, str(repo_root() / "scripts" / "build_fixtures.py"), "--check"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
