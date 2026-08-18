"""Phase 3 acceptance tests.

Split into three dependency tiers, mirroring the plan's own split:

- **Always collected** (pure Python/numpy): ``HashEmbedder``, the
  truncate/renormalize math, the embedder registry, and
  ``SentenceTransformerEmbedder``'s *construction*-time prefix validation —
  none of that touches ``sentence_transformers`` or ``chromadb`` (the model
  itself is lazily loaded only by ``embed_documents``/``embed_query``).
- ``@requires_chromadb``: ``ChromaStore`` mechanics and ``indexing.py``'s
  build/cache/search orchestration, exercised through a monkeypatched
  ``HashEmbedder`` (the ``fast_embedder`` fixture) so these run in
  milliseconds without ``sentence-transformers`` or a model download.
- ``@requires_sentence_transformers``: the three ACs that are inherently
  about the real model — asymmetry, normalization, and the
  truncation-vs-recall smoke check. Only these trigger a one-time ~133MB
  download of ``BAAI/bge-small-en-v1.5`` (see ``doctor.py``'s
  ``model_cache`` check) the first time they run.

Plain ``pytest.importorskip`` isn't used here because it aborts collection of
the *entire file* the moment it's missing, which would also skip the
always-safe tier-1 tests; ``skipif`` marks skip only the tests that actually
need the dependency.

The six numbered acceptance criteria from the plan are marked ``# AC-n``.
"""

from __future__ import annotations

import re
import time
import uuid
from datetime import datetime, timezone

import numpy as np
import pytest
from typer.testing import CliRunner

from rag_lab import indexing
from rag_lab.cli import app
from rag_lab.config import params_hash
from rag_lab.embedders import (
    NAMED_EMBEDDERS,
    available_embedders,
    build_embedder,
    resolve_params,
)
from rag_lab.embedders.base import truncate_and_renormalize
from rag_lab.embedders.fixture import HashEmbedder, hash_embed
from rag_lab.embedders.sentence_transformer import SentenceTransformerEmbedder
from rag_lab.ids import make_index_id
from rag_lab.jsonl import read_jsonl
from rag_lab.paths import artifact_path, fixture_path
from rag_lab.schemas import Chunk, EvalPair, IndexManifest
from rag_lab.stores import ChromaStore

runner = CliRunner()

try:
    import chromadb as _chromadb  # noqa: F401

    HAS_CHROMADB = True
except ImportError:
    HAS_CHROMADB = False

try:
    import sentence_transformers as _sentence_transformers  # noqa: F401

    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

requires_chromadb = pytest.mark.skipif(
    not HAS_CHROMADB, reason="chromadb not installed (embed extra)"
)
requires_sentence_transformers = pytest.mark.skipif(
    not HAS_SENTENCE_TRANSFORMERS, reason="sentence-transformers not installed (embed extra)"
)


def _fixture_chunks() -> list[Chunk]:
    return read_jsonl(fixture_path("chunks"), Chunk)


def _unique_chunk_set_id(label: str) -> str:
    """A chunk_set_id guaranteed not to be a real artifact, so
    `load_chunk_set` always falls back to `fixtures/chunks/sample.jsonl` — and
    unique per call, so repeated local test runs never collide with a
    previous run's leftover `artifacts/indexes/<id>/` from an earlier
    session (which would otherwise turn an expected fresh build into an
    unexpected cache hit)."""
    return f"test-phase3-{label}-{uuid.uuid4().hex[:8]}"


# --------------------------------------------------------------------------- #
# Tier 1 -- pure Python/numpy, always collected
# --------------------------------------------------------------------------- #


class TestHashEmbedder:
    def test_shape_and_unit_norm(self):
        embedder = HashEmbedder()
        vecs = embedder.embed_documents(["hello world", "another chunk of text"])
        assert vecs.shape == (2, HashEmbedder.dim)
        assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-5)

    def test_deterministic_across_calls(self):
        assert np.array_equal(hash_embed("some fixed text"), hash_embed("some fixed text"))

    def test_different_text_yields_different_vectors(self):
        assert not np.array_equal(hash_embed("alpha"), hash_embed("beta"))

    def test_empty_text_is_well_defined(self):
        v = hash_embed("")
        assert v.shape == (HashEmbedder.dim,)
        assert np.linalg.norm(v) == pytest.approx(1.0)

    def test_query_and_document_prefixes_differ(self):
        embedder = HashEmbedder()
        q = embedder.embed_query("paginate results")
        d = embedder.embed_documents(["paginate results"])[0]
        assert not np.array_equal(q, d)

    def test_not_registered_as_a_real_embedder(self):
        assert HashEmbedder.name not in NAMED_EMBEDDERS
        assert HashEmbedder.name not in available_embedders()


class TestTruncateAndRenormalize:
    def test_slices_and_renormalizes(self):
        vecs = np.random.default_rng(0).normal(size=(4, 16)).astype(np.float32)
        vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
        truncated = truncate_and_renormalize(vecs, 8)
        expected = vecs[:, :8] / np.linalg.norm(vecs[:, :8], axis=1, keepdims=True)
        assert truncated.shape == (4, 8)
        assert np.allclose(truncated, expected)
        assert np.allclose(np.linalg.norm(truncated, axis=1), 1.0, atol=1e-6)

    def test_rejects_dim_larger_than_native(self):
        with pytest.raises(ValueError):
            truncate_and_renormalize(np.zeros((2, 8), dtype=np.float32), 16)

    def test_rejects_non_positive_dim(self):
        with pytest.raises(ValueError):
            truncate_and_renormalize(np.zeros((2, 8), dtype=np.float32), 0)


class TestEmbedderRegistry:
    def test_available_embedders_matches_plan_table(self):
        expected = ["bge-small", "e5-base", "e5-multilingual", "minilm"]
        assert available_embedders() == sorted(expected)

    def test_resolve_params_layers_overrides_on_defaults(self):
        resolved = resolve_params("bge-small", {"truncate_dim": 256})
        assert resolved["truncate_dim"] == 256
        assert resolved["model_name"] == NAMED_EMBEDDERS["bge-small"].model_name

    def test_resolve_params_rejects_unknown_name(self):
        with pytest.raises(ValueError):
            resolve_params("not-a-real-embedder", {})

    def test_batch_size_is_excluded_from_the_hash(self):
        a = resolve_params("bge-small", {"batch_size": 8})
        b = resolve_params("bge-small", {"batch_size": 64})
        assert params_hash(a) == params_hash(b)

    def test_index_id_changes_when_truncate_dim_changes(self):
        a = resolve_params("bge-small", {})
        b = resolve_params("bge-small", {"truncate_dim": 256})
        assert make_index_id("cs", "bge-small", a) != make_index_id("cs", "bge-small", b)


class TestSentenceTransformerEmbedderConstruction:
    """Construction never touches the network or `sentence_transformers` --
    only `_encode()` (called by `embed_documents`/`embed_query`) does, via a
    lazily-imported, `lru_cache`'d model loader. So all of this is safe to
    run unconditionally, even without the `embed` extra installed."""

    def test_strict_prefixes_raises_for_asymmetric_model_with_empty_prefixes(self):
        params = resolve_params("bge-small", {"query_prefix": "", "doc_prefix": ""})
        with pytest.raises(ValueError, match="asymmetric"):
            SentenceTransformerEmbedder("bge-small", NAMED_EMBEDDERS["bge-small"], params)

    def test_strict_prefixes_false_opts_out(self):
        params = resolve_params(
            "bge-small", {"query_prefix": "", "doc_prefix": "", "strict_prefixes": False}
        )
        embedder = SentenceTransformerEmbedder("bge-small", NAMED_EMBEDDERS["bge-small"], params)
        assert embedder.query_prefix == ""

    def test_symmetric_model_never_raises_on_empty_prefixes(self):
        params = resolve_params("minilm", {})
        embedder = SentenceTransformerEmbedder("minilm", NAMED_EMBEDDERS["minilm"], params)
        assert embedder.query_prefix == "" and embedder.doc_prefix == ""

    def test_advertised_dim_reflects_truncate_dim(self):
        params = resolve_params("bge-small", {"truncate_dim": 256})
        embedder = SentenceTransformerEmbedder("bge-small", NAMED_EMBEDDERS["bge-small"], params)
        assert embedder.dim == 256


class TestIndexManifestSchema:
    def test_round_trips_through_json(self):
        manifest = IndexManifest(
            index_id="cs__bge-small__deadbeef",
            chunk_set_id="cs",
            embedder="bge-small",
            embedder_params={"model_name": "x"},
            vector_count=5,
            dim=384,
            build_duration_s=1.23,
            library_versions={"chromadb": "1.0.0"},
            created_at=datetime.now(timezone.utc),
        )
        assert IndexManifest.model_validate_json(manifest.model_dump_json()) == manifest


# --------------------------------------------------------------------------- #
# Tier 2 -- needs chromadb (not sentence_transformers)
# --------------------------------------------------------------------------- #


@pytest.fixture
def fast_embedder(monkeypatch):
    """Swap `indexing.py`'s embedder construction for the dependency-free
    `HashEmbedder`, so orchestration/caching/CLI tests run in milliseconds
    without `sentence-transformers` or a model download. The embedder *name*
    passed in still has to be a real `NAMED_EMBEDDERS` key -- CLI validation
    and `resolve_params` both check that for real -- only the vectors
    themselves are faked."""

    def _build(name, overrides=None):
        return HashEmbedder(overrides)

    monkeypatch.setattr("rag_lab.indexing.build_embedder", _build)


@requires_chromadb
class TestChromaStore:
    def test_upsert_search_get_round_trip(self, tmp_path):
        chunks = _fixture_chunks()[:5]
        embedder = HashEmbedder()
        vectors = embedder.embed_documents([c.embed_text for c in chunks])

        store = ChromaStore(tmp_path / "idx")
        store.upsert(chunks, vectors)
        assert store.count() == 5

        assert store.get([chunks[0].chunk_id]) == [chunks[0]]

        results = store.search(vectors[0], k=3)
        assert results[0].chunk.chunk_id == chunks[0].chunk_id
        assert [r.rank for r in results] == list(range(1, len(results) + 1))
        assert results[0].score >= results[-1].score
        store.close()

    def test_search_respects_metadata_filters(self, tmp_path):
        chunks = _fixture_chunks()
        embedder = HashEmbedder()
        vectors = embedder.embed_documents([c.embed_text for c in chunks])
        store = ChromaStore(tmp_path / "idx")
        store.upsert(chunks, vectors)

        target_doc = chunks[0].doc_id
        results = store.search(vectors[0], k=len(chunks), filters={"doc_id": target_doc})
        assert results
        assert all(r.chunk.doc_id == target_doc for r in results)
        store.close()

    def test_collection_is_created_with_cosine_space(self, tmp_path):
        # plan's own pitfall note: Chroma defaults to L2, not cosine, and
        # this can only be set at collection-creation time.
        store = ChromaStore(tmp_path / "idx")
        assert store._collection.configuration_json["hnsw"]["space"] == "cosine"
        store.close()

    def test_get_returns_empty_list_for_empty_input(self, tmp_path):
        store = ChromaStore(tmp_path / "idx")
        assert store.get([]) == []
        store.close()

    def test_upsert_rejects_mismatched_lengths(self, tmp_path):
        chunks = _fixture_chunks()[:2]
        store = ChromaStore(tmp_path / "idx")
        with pytest.raises(ValueError):
            store.upsert(chunks, np.zeros((1, HashEmbedder.dim), dtype=np.float32))
        store.close()


@requires_chromadb
def test_ac1_build_index_from_fixture_chunk_set(fast_embedder):  # AC-1
    chunk_set_id = _unique_chunk_set_id("ac1")
    manifest, cache_hit = indexing.build_index(chunk_set_id, "minilm", {})
    assert cache_hit is False
    assert manifest.vector_count == len(_fixture_chunks())
    assert manifest.dim == HashEmbedder.dim  # faked by fast_embedder


@requires_chromadb
def test_ac5_second_build_is_a_cache_hit_and_fast(fast_embedder):  # AC-5
    chunk_set_id = _unique_chunk_set_id("ac5")
    manifest1, hit1 = indexing.build_index(chunk_set_id, "minilm", {})
    assert hit1 is False

    started = time.monotonic()
    manifest2, hit2 = indexing.build_index(chunk_set_id, "minilm", {})
    elapsed = time.monotonic() - started

    assert hit2 is True
    assert elapsed < 1.0
    assert manifest2 == manifest1


@requires_chromadb
def test_force_bypasses_the_cache(fast_embedder):
    chunk_set_id = _unique_chunk_set_id("force")
    manifest1, _ = indexing.build_index(chunk_set_id, "minilm", {})
    manifest2, hit2 = indexing.build_index(chunk_set_id, "minilm", {}, force=True)
    assert hit2 is False
    assert manifest2.index_id == manifest1.index_id


@requires_chromadb
def test_different_embedder_params_produce_different_index_ids(fast_embedder):
    chunk_set_id = _unique_chunk_set_id("params")
    a, _ = indexing.build_index(chunk_set_id, "minilm", {})
    b, _ = indexing.build_index(chunk_set_id, "minilm", {"truncate_dim": 16})
    assert a.index_id != b.index_id


@requires_chromadb
def test_search_index_returns_scored_chunks_ranked_by_score(fast_embedder):
    chunk_set_id = _unique_chunk_set_id("search")
    manifest, _ = indexing.build_index(chunk_set_id, "minilm", {})
    _manifest, results = indexing.search_index(manifest.index_id, "how do I paginate?", k=5)
    assert results
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


@requires_chromadb
def test_cli_index_build_and_search_work_with_no_prior_phase(monkeypatch, tmp_path, fast_embedder):
    """The independence contract this phase set out to close for `indexes`
    (mirrors test_phase_2's equivalent for `chunks`): point `artifacts_dir` at
    an empty directory so `index build` has nothing but the fixture chunk set
    to work from, then confirm `index search` against a never-built index_id
    falls back to the *real* committed `fixtures/indexes/sample/` -- built
    with the real `HashEmbedder`, unaffected by this test's mock."""
    monkeypatch.setattr("rag_lab.paths.artifacts_dir", lambda: tmp_path)
    monkeypatch.setattr("rag_lab.indexing.artifacts_dir", lambda: tmp_path)

    build_result = runner.invoke(
        app, ["index", "build", "--chunk-set", "whatever-not-real", "--embedder", "minilm"]
    )
    assert build_result.exit_code == 0, build_result.output
    assert (tmp_path / "indexes").exists()

    search_result = runner.invoke(
        app, ["index", "search", "--index-id", "also-not-real", "--query", "how do I paginate?"]
    )
    assert search_result.exit_code == 0, search_result.output


def test_cli_index_build_rejects_unknown_embedder():
    result = runner.invoke(app, ["index", "build", "--chunk-set", "whatever", "--embedder", "nope"])
    assert result.exit_code == 1
    assert "unknown embedder" in result.output.lower()


@requires_chromadb
def test_cli_index_list_reports_built_indexes(fast_embedder):
    chunk_set_id = _unique_chunk_set_id("list")
    build_result = runner.invoke(
        app, ["index", "build", "--chunk-set", chunk_set_id, "--embedder", "minilm"]
    )
    assert build_result.exit_code == 0, build_result.output
    index_id = re.search(r"ok\s+(\S+):", build_result.output).group(1)

    list_result = runner.invoke(app, ["index", "list"])
    assert list_result.exit_code == 0
    assert index_id in list_result.output


@requires_chromadb
def test_cli_index_search_reports_ranked_results(fast_embedder):
    chunk_set_id = _unique_chunk_set_id("cli-search")
    build_result = runner.invoke(
        app, ["index", "build", "--chunk-set", chunk_set_id, "--embedder", "minilm"]
    )
    index_id = re.search(r"ok\s+(\S+):", build_result.output).group(1)

    search_result = runner.invoke(
        app, ["index", "search", "--index-id", index_id, "--query", "pagination cursor", "--k", "3"]
    )
    assert search_result.exit_code == 0, search_result.output


@requires_chromadb
def test_cli_index_build_truncate_dim_flag_threads_through_to_the_manifest(fast_embedder):
    chunk_set_id = _unique_chunk_set_id("trunc")
    result = runner.invoke(
        app,
        [
            "index", "build", "--chunk-set", chunk_set_id, "--embedder", "minilm",
            "--truncate-dim", "8",
        ],
    )
    assert result.exit_code == 0, result.output
    index_id = re.search(r"ok\s+(\S+):", result.output).group(1)
    manifest = indexing.load_manifest(artifact_path("indexes", index_id))
    assert manifest.embedder_params["truncate_dim"] == 8


# --------------------------------------------------------------------------- #
# Tier 3 -- needs the real sentence-transformers model
# --------------------------------------------------------------------------- #


@requires_sentence_transformers
def test_ac2_bge_asymmetry_query_and_document_embeddings_differ():  # AC-2
    embedder = build_embedder("bge-small", {})
    query = "how do I paginate results?"
    query_vec = embedder.embed_query(query)
    doc_vec = embedder.embed_documents([query])[0]
    assert not np.allclose(query_vec, doc_vec)


@requires_sentence_transformers
@requires_chromadb
def test_ac3_stored_vectors_are_unit_normalized(tmp_path):  # AC-3
    chunks = _fixture_chunks()
    embedder = build_embedder("bge-small", {})
    vectors = embedder.embed_documents([c.embed_text for c in chunks])
    store = ChromaStore(tmp_path / "idx")
    store.upsert(chunks, vectors)

    stored = store._collection.get(ids=[c.chunk_id for c in chunks], include=["embeddings"])
    norms = np.linalg.norm(np.asarray(stored["embeddings"]), axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)
    store.close()


@requires_sentence_transformers
@requires_chromadb
def test_ac6_searching_a_chunks_own_text_returns_it_at_rank_1():  # AC-6
    chunk_set_id = _unique_chunk_set_id("sanity")
    manifest, _ = indexing.build_index(chunk_set_id, "bge-small", {})
    target = _fixture_chunks()[3]
    _manifest, results = indexing.search_index(manifest.index_id, target.text, k=3)
    assert results[0].chunk.chunk_id == target.chunk_id


@requires_sentence_transformers
@requires_chromadb
def test_ac4_truncation_recall_within_ten_points_of_full_dim():  # AC-4
    pairs = [p for p in read_jsonl(fixture_path("evalset"), EvalPair) if p.sampling_chunk_ids]
    assert pairs  # the fixture evalset must have gold to be a useful check at all

    chunk_set_id = _unique_chunk_set_id("recall")
    full_manifest, _ = indexing.build_index(chunk_set_id, "bge-small", {})
    trunc_manifest, _ = indexing.build_index(chunk_set_id, "bge-small", {"truncate_dim": 256})
    assert full_manifest.dim == 384
    assert trunc_manifest.dim == 256

    def _recall_at_5(manifest: IndexManifest) -> float:
        # A minimal, test-local sanity metric -- NOT the production recall@k,
        # which is Phase 6's `metrics/recall_at_k`. With only 8 fixture
        # queries this is a coarse smoke check, not a statistically
        # meaningful truncation-quality claim.
        hits = 0
        for pair in pairs:
            _m, results = indexing.search_index(manifest.index_id, pair.query, k=5)
            if {r.chunk.chunk_id for r in results} & set(pair.sampling_chunk_ids):
                hits += 1
        return hits / len(pairs)

    full_recall = _recall_at_5(full_manifest)
    trunc_recall = _recall_at_5(trunc_manifest)
    assert trunc_recall >= full_recall - 0.10
