"""Phase 4 acceptance tests.

Split into three dependency tiers, same spirit as ``test_phase_3.py`` but with
a boundary drawn from what actually requires which import, not just "is this
retrieval-related":

- **Always collected** (pure Python/numpy, no ``embed`` extras): RRF fusion
  math, ``chunkers.hierarchy`` (parent/child span matching), the BM25
  tokenizer (a bare regex -- ``rank_bm25`` itself is only imported lazily
  inside ``BM25Retriever``), ``ParentDocumentRetriever``/
  ``SentenceWindowRetriever`` exercised against a small in-file
  ``FakeRetriever`` double (neither class touches a store or embedder), the
  retriever registry's name/param resolution, and every CLI validation path
  that fails before a ``ChromaStore``/``BM25Okapi``/``SentenceTransformer`` is
  ever constructed -- including the full ``chunk run --role parent`` ->
  ``--role child --parent-chunk-set`` round trip, since chunking itself never
  touches embeddings.
- ``@requires_chromadb`` / ``@requires_rank_bm25``: actual retrieval
  execution (``dense``/``bm25``/``hybrid``/``sentence_window`` against a real
  or the committed fixture index), exercised through the ``fast_embedder``
  fixture (same monkeypatch as ``test_phase_3.py``) so these run in
  milliseconds without ``sentence-transformers`` or a model download.
- ``@requires_sentence_transformers``: AC-4 (BM25 vs. dense on a real
  identifier query) and the hierarchical parent_doc end-to-end check, both
  inherently about real embedding signal -- the committed fixture index is a
  32-dim ``HashEmbedder`` over 19 vectors and cannot support either claim.

The acceptance criteria from the plan are marked ``# AC-n``.
"""

from __future__ import annotations

import re

import pytest
from typer.testing import CliRunner

from rag_lab import indexing, retrieval
from rag_lab.chunkers.hierarchy import OrphanChildError, assign_parents, find_parent
from rag_lab.chunks import load_chunk_set
from rag_lab.cli import app
from rag_lab.config import deep_merge
from rag_lab.embedders.fixture import HashEmbedder
from rag_lab.jsonl import read_jsonl, write_jsonl
from rag_lab.paths import artifact_path, fixture_path, fixtures_dir
from rag_lab.retrievers import available_retrievers, build_retriever, resolve_params
from rag_lab.retrievers.bm25 import _tokenize
from rag_lab.retrievers.hybrid import reciprocal_rank_fusion
from rag_lab.retrievers.parent_doc import ParentDocumentRetriever
from rag_lab.retrievers.reranker import NoOpReranker
from rag_lab.retrievers.sentence_window import SentenceWindowRetriever
from rag_lab.schemas import Chunk, ScoredChunk

runner = CliRunner()

try:
    import chromadb as _chromadb  # noqa: F401

    HAS_CHROMADB = True
except ImportError:
    HAS_CHROMADB = False

try:
    import rank_bm25 as _rank_bm25  # noqa: F401

    HAS_RANK_BM25 = True
except ImportError:
    HAS_RANK_BM25 = False

try:
    import sentence_transformers as _sentence_transformers  # noqa: F401

    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

requires_chromadb = pytest.mark.skipif(
    not HAS_CHROMADB, reason="chromadb not installed (embed extra)"
)
requires_rank_bm25 = pytest.mark.skipif(
    not HAS_RANK_BM25, reason="rank_bm25 not installed (embed extra)"
)
requires_sentence_transformers = pytest.mark.skipif(
    not HAS_SENTENCE_TRANSFORMERS, reason="sentence-transformers not installed (embed extra)"
)


def _chunk(
    chunk_id: str,
    *,
    doc_id: str = "doc-1",
    char_start: int = 0,
    char_end: int = 10,
    text: str | None = None,
) -> Chunk:
    body = text if text is not None else "x" * max(1, char_end - char_start)
    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        corpus="test",
        chunk_set_id="test__fake__00000000",
        text=body,
        embed_text=body,
        char_start=char_start,
        char_end=char_end,
        token_count=max(1, len(body)),
        ordinal=0,
        chunker="fake",
        chunker_params={},
    )


def _scored(chunk_id: str, *, rank: int, score: float = 0.0, retriever: str = "fake", **kw) -> ScoredChunk:
    return ScoredChunk(chunk=_chunk(chunk_id, **kw), score=score, rank=rank, retriever=retriever)


class FakeRetriever:
    """A canned ``Retriever`` double for unit-testing the wrapper retrievers
    (``parent_doc``, ``sentence_window``) without a real store/embedder."""

    def __init__(self, results: list[ScoredChunk], name: str = "fake") -> None:
        self._results = results
        self.name = name

    def retrieve(self, query: str, k: int) -> list[ScoredChunk]:
        return self._results[:k]


def _fixture_chunks() -> list[Chunk]:
    return read_jsonl(fixture_path("chunks"), Chunk)


def _fixture_parent_child_pair() -> tuple[list[Chunk], list[Chunk]]:
    parents = read_jsonl(fixtures_dir() / "chunks" / "sample_parent.jsonl", Chunk)
    children = read_jsonl(fixtures_dir() / "chunks" / "sample_child.jsonl", Chunk)
    return parents, children


def _unique_chunk_set_id(label: str) -> str:
    """A chunk_set_id guaranteed not to be a real artifact, so
    `load_chunk_set` always falls back to `fixtures/chunks/sample.jsonl` --
    and unique per call, so repeated local test runs never collide with a
    previous run's leftover `artifacts/indexes/<id>/`."""
    import uuid

    return f"test-phase4-{label}-{uuid.uuid4().hex[:8]}"


# --------------------------------------------------------------------------- #
# Tier 1 -- pure Python, always collected
# --------------------------------------------------------------------------- #


class TestReciprocalRankFusion:
    def test_hand_computed_toy_case_exact_match(self):  # AC-2
        # dense: A(1) B(2) C(3); sparse: A(1) C(2) -- B absent from sparse.
        dense = [_scored("A", rank=1), _scored("B", rank=2), _scored("C", rank=3)]
        sparse = [_scored("A", rank=1), _scored("C", rank=2)]

        fused = reciprocal_rank_fusion({"dense": dense, "sparse": sparse}, k_rrf=60)

        expected = {"A": 1 / 61 + 1 / 61, "B": 1 / 62, "C": 1 / 63 + 1 / 62}
        by_id = {r.chunk.chunk_id: r.score for r in fused}
        assert by_id == pytest.approx(expected)

        # A (0.03279) > C (0.03200) > B (0.01613); ranks assigned accordingly.
        assert [r.chunk.chunk_id for r in fused] == ["A", "C", "B"]
        assert [r.rank for r in fused] == [1, 2, 3]
        assert all(r.retriever == "hybrid" for r in fused)

    def test_a_document_missing_from_one_component_sums_only_the_other(self):
        fused = reciprocal_rank_fusion({"dense": [_scored("A", rank=1)], "sparse": []}, k_rrf=60)
        assert fused[0].score == pytest.approx(1 / 61)
        assert fused[0].debug["components"] == {"dense": {"rank": 1, "score": 0.0}}

    def test_weights_scale_each_components_contribution(self):
        dense = [_scored("A", rank=1)]
        sparse = [_scored("A", rank=1)]
        equal = reciprocal_rank_fusion({"dense": dense, "sparse": sparse}, k_rrf=60)
        weighted = reciprocal_rank_fusion(
            {"dense": dense, "sparse": sparse}, k_rrf=60, weights={"dense": 2.0, "sparse": 1.0}
        )
        assert equal[0].score == pytest.approx(1 / 61 + 1 / 61)
        assert weighted[0].score == pytest.approx(2 / 61 + 1 / 61)

    def test_empty_input_yields_empty_output(self):
        assert reciprocal_rank_fusion({"dense": [], "sparse": []}) == []


class TestFindParent:
    def test_returns_none_when_no_candidate_contains_the_midpoint(self):
        child = _chunk("c", char_start=500, char_end=520)
        assert find_parent(child, [_chunk("p", char_start=0, char_end=100)]) is None

    def test_picks_the_containing_parent(self):
        child = _chunk("c", char_start=10, char_end=30)  # midpoint 20
        parents = [_chunk("p1", char_start=0, char_end=100), _chunk("p2", char_start=200, char_end=300)]
        assert find_parent(child, parents).chunk_id == "p1"

    def test_boundary_straddling_child_resolves_by_midpoint(self):
        # span [60, 140) straddles the 0-100/100-200 boundary; midpoint 100
        # falls in the *second* parent ([100, 200), start inclusive).
        child = _chunk("c", char_start=60, char_end=140)
        parents = [_chunk("p1", char_start=0, char_end=100), _chunk("p2", char_start=100, char_end=200)]
        assert find_parent(child, parents).chunk_id == "p2"

    def test_ties_broken_by_smallest_span_then_earliest_start(self):
        child = _chunk("c", char_start=25, char_end=35)  # midpoint 30
        big = _chunk("big", char_start=0, char_end=100)  # span 100, contains 30
        small_later = _chunk("small-later", char_start=20, char_end=60)  # span 40
        small_earlier = _chunk("small-earlier", char_start=10, char_end=50)  # span 40
        assert find_parent(child, [big, small_later, small_earlier]).chunk_id == "small-earlier"


class TestAssignParents:
    def test_all_children_matched_no_orphans_no_duplicates(self):  # AC-3
        p1 = _chunk("p1", char_start=0, char_end=100)
        p2 = _chunk("p2", char_start=100, char_end=200)
        c1 = _chunk("c1", char_start=0, char_end=50)  # mid 25 -> p1
        c2 = _chunk("c2", char_start=60, char_end=140)  # mid 100 -> p2 (straddles)
        c3 = _chunk("c3", char_start=150, char_end=190)  # mid 170 -> p2

        resolved = assign_parents([c1, c2, c3], [p1, p2])

        by_id = {c.chunk_id: c for c in resolved}
        assert by_id["c1"].parent_id == "p1"
        assert by_id["c2"].parent_id == "p2"
        assert by_id["c3"].parent_id == "p2"
        assert all(c.role == "child" for c in resolved)
        assert len(resolved) == 3
        assert len({c.chunk_id for c in resolved}) == 3  # no duplicates

    def test_inputs_are_not_mutated(self):
        p1 = _chunk("p1", char_start=0, char_end=100)
        c1 = _chunk("c1", char_start=0, char_end=50)
        assign_parents([c1], [p1])
        assert c1.role == "standalone"
        assert c1.parent_id is None

    def test_raises_and_names_every_orphan_without_dropping_silently(self):
        p1 = _chunk("p1", char_start=0, char_end=100)
        orphan = _chunk("orphan-1", char_start=500, char_end=520)
        matched = _chunk("matched", char_start=10, char_end=30)
        with pytest.raises(OrphanChildError, match="orphan-1"):
            assign_parents([matched, orphan], [p1])

    def test_matching_is_scoped_per_doc_id(self):
        parent_other_doc = _chunk("p-other", char_start=0, char_end=1000, doc_id="d2")
        child = _chunk("c", char_start=10, char_end=30, doc_id="d1")
        with pytest.raises(OrphanChildError):
            assign_parents([child], [parent_other_doc])

    def test_round_trip_over_the_committed_fixture_pair_has_zero_orphans(self):
        parents, committed_children = _fixture_parent_child_pair()
        resolved = assign_parents(_fixture_chunks(), parents)  # sample.jsonl: same spans as sample_child.jsonl
        assert len(resolved) == len(committed_children)
        assert {(c.chunk_id, c.parent_id) for c in resolved} == {
            (c.chunk_id, c.parent_id) for c in committed_children
        }


class TestBM25Tokenizer:
    def test_identifier_with_underscores_is_one_token(self):
        tokens = _tokenize("Returns IDEMPOTENCY_KEY_CONFLICT on retry")
        assert "idempotency_key_conflict" in tokens
        assert "idempotency" not in tokens  # not shredded into pieces

    def test_punctuation_is_stripped(self):
        assert _tokenize("cursor, expired!") == ["cursor", "expired"]

    def test_no_stemming(self):
        assert _tokenize("cursors cursor") == ["cursors", "cursor"]  # kept distinct, not stemmed


class TestParentDocumentRetrieverDedup:
    def test_dedupes_to_the_best_ranked_child_per_parent(self):
        parents, children = _fixture_parent_child_pair()
        by_parent: dict[str, list[Chunk]] = {}
        for c in children:
            by_parent.setdefault(c.parent_id, []).append(c)
        parent_id, siblings = next((pid, sibs) for pid, sibs in by_parent.items() if len(sibs) >= 2)
        child_a, child_b = siblings[0], siblings[1]

        base = FakeRetriever(
            [
                ScoredChunk(chunk=child_a, score=0.5, rank=2, retriever="fake"),
                ScoredChunk(chunk=child_b, score=0.9, rank=1, retriever="fake"),
            ]
        )
        results = ParentDocumentRetriever(base, parents, fanout=1).retrieve("q", k=5)

        assert len(results) == 1
        assert results[0].chunk.chunk_id == parent_id
        assert results[0].chunk.role == "parent"
        assert results[0].debug["n_children_matched"] == 2
        assert results[0].debug["child_chunk_id"] == child_b.chunk_id  # rank 1, the better one
        assert results[0].rank == 1

    def test_raises_when_no_retrieved_chunk_has_a_parent_id(self):
        parents, _ = _fixture_parent_child_pair()
        standalone = _fixture_chunks()[0]  # role="standalone", parent_id=None
        base = FakeRetriever([ScoredChunk(chunk=standalone, score=1.0, rank=1, retriever="fake")])
        with pytest.raises(ValueError, match="parent_id"):
            ParentDocumentRetriever(base, parents, fanout=1).retrieve("q", k=5)

    def test_raises_when_parent_id_is_not_in_the_loaded_parent_set(self):
        _, children = _fixture_parent_child_pair()
        base = FakeRetriever([ScoredChunk(chunk=children[0], score=1.0, rank=1, retriever="fake")])
        wrong_parents = [_chunk("not-the-real-parent", char_start=0, char_end=10)]
        with pytest.raises(ValueError, match="not in the loaded parent chunk set"):
            ParentDocumentRetriever(base, wrong_parents, fanout=1).retrieve("q", k=5)

    def test_empty_base_results_return_empty_not_an_error(self):
        parents, _ = _fixture_parent_child_pair()
        assert ParentDocumentRetriever(FakeRetriever([]), parents, fanout=1).retrieve("q", k=5) == []

    def test_rejects_empty_parent_chunks(self):
        with pytest.raises(ValueError):
            ParentDocumentRetriever(FakeRetriever([]), [], fanout=1)


class TestSentenceWindowDedup:
    def test_suppresses_windows_overlapping_more_than_fifty_percent(self):  # AC-5
        a = _chunk("a", char_start=0, char_end=100, doc_id="d")
        b = _chunk("b", char_start=40, char_end=140, doc_id="d")  # overlap 60/100 = 0.6 > 0.5
        c = _chunk("c", char_start=200, char_end=300, doc_id="d")  # no overlap
        base = FakeRetriever(
            [
                ScoredChunk(chunk=a, score=0.9, rank=1, retriever="fake"),
                ScoredChunk(chunk=b, score=0.8, rank=2, retriever="fake"),
                ScoredChunk(chunk=c, score=0.7, rank=3, retriever="fake"),
            ]
        )
        results = SentenceWindowRetriever(base, overlap_threshold=0.5, fanout=1).retrieve("q", k=5)
        assert [r.chunk.chunk_id for r in results] == ["a", "c"]  # b suppressed, a (better rank) kept

    def test_exactly_fifty_percent_overlap_is_kept(self):
        a = _chunk("a", char_start=0, char_end=100, doc_id="d")
        b = _chunk("b", char_start=50, char_end=150, doc_id="d")  # overlap 50/100 = 0.5, not > 0.5
        base = FakeRetriever(
            [
                ScoredChunk(chunk=a, score=0.9, rank=1, retriever="fake"),
                ScoredChunk(chunk=b, score=0.8, rank=2, retriever="fake"),
            ]
        )
        results = SentenceWindowRetriever(base, overlap_threshold=0.5, fanout=1).retrieve("q", k=5)
        assert [r.chunk.chunk_id for r in results] == ["a", "b"]

    def test_different_docs_never_suppress_each_other(self):
        a = _chunk("a", char_start=0, char_end=100, doc_id="d1")
        b = _chunk("b", char_start=0, char_end=100, doc_id="d2")  # identical span, different doc
        base = FakeRetriever(
            [
                ScoredChunk(chunk=a, score=0.9, rank=1, retriever="fake"),
                ScoredChunk(chunk=b, score=0.8, rank=2, retriever="fake"),
            ]
        )
        results = SentenceWindowRetriever(base, fanout=1).retrieve("q", k=5)
        assert len(results) == 2


class TestNoOpReranker:
    def test_returns_input_unchanged(self):
        results = [_scored("a", rank=1)]
        assert NoOpReranker().rerank("q", results) == results


class TestRetrieverRegistry:
    def test_available_retrievers_matches_the_five_implemented_names(self):
        assert available_retrievers() == sorted(
            ["dense", "bm25", "hybrid", "parent_doc", "sentence_window"]
        )
        assert "auto_merging" not in available_retrievers()  # deliberately deferred

    def test_resolve_params_layers_overrides_on_defaults(self):
        resolved = resolve_params("hybrid", {"k_rrf": 30})
        assert resolved["k_rrf"] == 30
        assert resolved["weights"] == {"dense": 1.0, "sparse": 1.0}

    def test_resolve_params_deep_merges_nested_weights(self):
        # a shallow {**defaults, **overrides} would drop "sparse" entirely here.
        resolved = resolve_params("hybrid", {"weights": {"dense": 2.0}})
        assert resolved["weights"] == {"dense": 2.0, "sparse": 1.0}

    def test_resolve_params_rejects_unknown_name(self):
        with pytest.raises(ValueError):
            resolve_params("not-a-real-retriever", {})

    def test_build_retriever_rejects_unknown_name(self):
        with pytest.raises(ValueError):
            build_retriever("nope", {}, manifest=None, store=None)  # errors before touching either

    def test_build_retriever_parent_doc_requires_parent_chunk_set_id(self):
        with pytest.raises(ValueError, match="parent_chunk_set_id"):
            build_retriever("parent_doc", {}, manifest=None, store=None)

    def test_deep_merge_used_by_resolve_params_is_the_shared_config_helper(self):
        # not a retriever-specific reimplementation -- confirms the reuse.
        assert deep_merge({"a": {"x": 1}}, {"a": {"y": 2}}) == {"a": {"x": 1, "y": 2}}


# --------------------------------------------------------------------------- #
# Tier 1 -- CLI validation paths that never touch a store/embedder
# --------------------------------------------------------------------------- #


def test_cli_retrieve_query_rejects_unknown_retriever():
    result = runner.invoke(
        app, ["retrieve", "query", "--index-id", "whatever", "--query", "q", "--retriever", "nope"]
    )
    assert result.exit_code == 1
    assert "unknown retriever" in result.output.lower()


def test_cli_retrieve_compare_rejects_unknown_retriever():
    result = runner.invoke(
        app,
        ["retrieve", "compare", "--index-id", "whatever", "--query", "q", "--retrievers", "dense,nope"],
    )
    assert result.exit_code == 1
    assert "unknown retriever" in result.output.lower()


def test_cli_retrieve_query_parent_doc_without_parent_chunk_set_is_rejected():
    result = runner.invoke(
        app, ["retrieve", "query", "--index-id", "whatever", "--query", "q", "--retriever", "parent_doc"]
    )
    assert result.exit_code == 1
    assert "--parent-chunk-set" in result.output


def test_cli_retrieve_compare_parent_doc_without_parent_chunk_set_is_rejected():
    result = runner.invoke(
        app,
        [
            "retrieve", "compare", "--index-id", "whatever", "--query", "q",
            "--retrievers", "dense,parent_doc",
        ],
    )
    assert result.exit_code == 1
    assert "--parent-chunk-set" in result.output


def test_cli_chunk_run_rejects_invalid_role():
    result = runner.invoke(
        app, ["chunk", "run", "--corpus", "api_docs", "--chunker", "recursive", "--role", "grandparent"]
    )
    assert result.exit_code == 1
    assert "--role" in result.output


def test_cli_chunk_run_role_child_without_parent_chunk_set_is_rejected():
    result = runner.invoke(
        app, ["chunk", "run", "--corpus", "api_docs", "--chunker", "recursive", "--role", "child"]
    )
    assert result.exit_code == 1
    assert "--parent-chunk-set" in result.output


def test_cli_chunk_run_parent_chunk_set_without_role_child_is_rejected():
    result = runner.invoke(
        app,
        [
            "chunk", "run", "--corpus", "api_docs", "--chunker", "recursive",
            "--parent-chunk-set", "whatever",
        ],
    )
    assert result.exit_code == 1
    assert "--parent-chunk-set" in result.output


def test_cli_chunk_run_role_parent_then_child_round_trip():
    parent_result = runner.invoke(
        app,
        [
            "chunk", "run", "--corpus", "api_docs", "--chunker", "markdown",
            "--params", "max_tokens=2048", "--role", "parent",
        ],
    )
    assert parent_result.exit_code == 0, parent_result.output
    parent_id = re.search(r"ok\s+(\S+):", parent_result.output).group(1)

    child_result = runner.invoke(
        app,
        [
            "chunk", "run", "--corpus", "api_docs", "--chunker", "recursive",
            "--params", "chunk_tokens=256", "--role", "child", "--parent-chunk-set", parent_id,
        ],
    )
    assert child_result.exit_code == 0, child_result.output
    child_id = re.search(r"ok\s+(\S+):", child_result.output).group(1)

    children = load_chunk_set(child_id)
    parent_ids = {c.chunk_id for c in load_chunk_set(parent_id)}
    assert children
    assert all(c.role == "child" for c in children)
    assert all(c.parent_id in parent_ids for c in children)


def test_cli_chunk_run_role_and_standalone_get_different_chunk_set_ids():
    # Same corpus/chunker/params, only --role differs -- must not collide
    # (chunkers/hierarchy.py's docstring / plan §4.7 correction).
    standalone = runner.invoke(
        app, ["chunk", "run", "--corpus", "api_docs", "--chunker", "markdown", "--params", "max_tokens=999"]
    )
    parent = runner.invoke(
        app,
        [
            "chunk", "run", "--corpus", "api_docs", "--chunker", "markdown",
            "--params", "max_tokens=999", "--role", "parent",
        ],
    )
    assert standalone.exit_code == 0, standalone.output
    assert parent.exit_code == 0, parent.output
    standalone_id = re.search(r"ok\s+(\S+):", standalone.output).group(1)
    parent_id = re.search(r"ok\s+(\S+):", parent.output).group(1)
    assert standalone_id != parent_id


# --------------------------------------------------------------------------- #
# Tier 2 -- needs chromadb + rank_bm25 (not sentence_transformers)
# --------------------------------------------------------------------------- #


@pytest.fixture
def fast_embedder(monkeypatch):
    """Swap `indexing.py`'s embedder construction for the dependency-free
    `HashEmbedder` (same fixture as `test_phase_3.py`), so retrievers built
    through the registry's `dense`/`hybrid` path run in milliseconds without
    `sentence-transformers` or a model download."""

    def _build(name, overrides=None):
        return HashEmbedder(overrides)

    monkeypatch.setattr("rag_lab.indexing.build_embedder", _build)


@requires_chromadb
class TestDenseRetriever:
    def test_relabels_to_dense_and_preserves_score_and_rank(self, fast_embedder):
        chunk_set_id = _unique_chunk_set_id("dense")
        manifest, _ = indexing.build_index(chunk_set_id, "minilm", {})
        _m, results = retrieval.run_retriever(manifest.index_id, "dense", "pagination cursor", k=3)
        assert results
        assert all(r.retriever == "dense" for r in results)
        assert [r.rank for r in results] == list(range(1, len(results) + 1))


@requires_chromadb
@requires_rank_bm25
@pytest.mark.parametrize("retriever_name", ["dense", "bm25", "hybrid", "sentence_window"])
def test_ac1_retriever_runs_against_the_fixture_index_with_no_prior_phase(retriever_name):  # AC-1
    manifest, results = retrieval.run_retriever("sample", retriever_name, "pagination cursor", k=3)
    assert manifest.chunk_set_id == "sample"
    assert results
    assert all(r.retriever == retriever_name for r in results)


@requires_chromadb
def test_parent_doc_against_the_fixture_index_fails_with_a_clear_error_not_a_crash():
    # the committed fixture index's chunks are 100% role="standalone" -- no
    # role="child" chunk set was ever built against it.
    with pytest.raises(ValueError, match="parent_id"):
        retrieval.run_retriever(
            "sample", "parent_doc", "pagination cursor", k=3,
            overrides={"parent_chunk_set_id": "sample"},
        )


@requires_chromadb
@requires_rank_bm25
def test_bm25_lazily_builds_and_persists_a_pickle_for_a_real_index(fast_embedder):
    chunk_set_id = _unique_chunk_set_id("bm25-persist")
    manifest, _ = indexing.build_index(chunk_set_id, "minilm", {})
    pickle_path = artifact_path("indexes", manifest.index_id) / "bm25.pkl"
    assert not pickle_path.exists()

    _m, results = retrieval.run_retriever(manifest.index_id, "bm25", "pagination cursor", k=3)
    assert results
    assert pickle_path.exists()

    mtime_before = pickle_path.stat().st_mtime
    retrieval.run_retriever(manifest.index_id, "bm25", "pagination cursor", k=3)
    assert pickle_path.stat().st_mtime == mtime_before  # reused, not rebuilt


@requires_chromadb
@requires_rank_bm25
def test_bm25_never_writes_into_the_committed_fixture_index_dir():
    fixture_pickle = fixtures_dir() / "indexes" / "sample" / "bm25.pkl"
    assert not fixture_pickle.exists()
    _m, results = retrieval.run_retriever("sample", "bm25", "pagination cursor", k=3)
    assert results
    assert not fixture_pickle.exists()


@requires_chromadb
@requires_rank_bm25
def test_cli_retrieve_query_and_compare_smoke(fast_embedder):
    chunk_set_id = _unique_chunk_set_id("cli-retrieve")
    build_result = runner.invoke(
        app, ["index", "build", "--chunk-set", chunk_set_id, "--embedder", "minilm"]
    )
    assert build_result.exit_code == 0, build_result.output
    index_id = re.search(r"ok\s+(\S+):", build_result.output).group(1)

    query_result = runner.invoke(
        app,
        [
            "retrieve", "query", "--index-id", index_id, "--query", "pagination cursor",
            "--retriever", "hybrid",
        ],
    )
    assert query_result.exit_code == 0, query_result.output

    compare_result = runner.invoke(
        app,
        [
            "retrieve", "compare", "--index-id", index_id, "--query", "pagination cursor",
            "--retrievers", "dense,bm25,hybrid",
        ],
    )
    assert compare_result.exit_code == 0, compare_result.output
    assert "dense" in compare_result.output
    assert "bm25" in compare_result.output


@requires_chromadb
@requires_rank_bm25
def test_cli_retrieve_compare_params_are_namespaced_per_retriever(fast_embedder):
    chunk_set_id = _unique_chunk_set_id("cli-namespaced-params")
    build_result = runner.invoke(
        app, ["index", "build", "--chunk-set", chunk_set_id, "--embedder", "minilm"]
    )
    index_id = re.search(r"ok\s+(\S+):", build_result.output).group(1)

    result = runner.invoke(
        app,
        [
            "retrieve", "compare", "--index-id", index_id, "--query", "pagination cursor",
            "--retrievers", "hybrid", "--params", "hybrid.k_rrf=5",
        ],
    )
    assert result.exit_code == 0, result.output


@requires_chromadb
@requires_rank_bm25
def test_cli_retrieve_query_and_compare_work_with_no_prior_phase(monkeypatch, tmp_path):
    """Independence contract for `retrieve` (mirrors test_phase_2/3's
    equivalent): point every phase's artifacts_dir at an empty directory so
    these have nothing but the committed fixture index/chunk set to work
    from."""
    monkeypatch.setattr("rag_lab.paths.artifacts_dir", lambda: tmp_path)
    monkeypatch.setattr("rag_lab.corpus.artifacts_dir", lambda: tmp_path)
    monkeypatch.setattr("rag_lab.chunks.artifacts_dir", lambda: tmp_path)
    monkeypatch.setattr("rag_lab.indexing.artifacts_dir", lambda: tmp_path)

    query_result = runner.invoke(
        app,
        [
            "retrieve", "query", "--index-id", "not-real", "--query", "pagination cursor",
            "--retriever", "hybrid",
        ],
    )
    assert query_result.exit_code == 0, query_result.output

    compare_result = runner.invoke(
        app,
        [
            "retrieve", "compare", "--index-id", "not-real", "--query", "pagination cursor",
            "--retrievers", "dense,bm25,hybrid,sentence_window",
        ],
    )
    assert compare_result.exit_code == 0, compare_result.output


# --------------------------------------------------------------------------- #
# Tier 3 -- needs the real sentence-transformers model
# --------------------------------------------------------------------------- #


API_DOCS_IDENTIFIER_QUERIES = [
    ("IDEMPOTENCY_KEY_CONFLICT", "What does the error code IDEMPOTENCY_KEY_CONFLICT mean?"),
    ("CURSOR_EXPIRED", "What does the error code CURSOR_EXPIRED mean?"),
    ("INVALID_GRANT", "What does the error code INVALID_GRANT mean?"),
]


@pytest.fixture(scope="module")
def real_api_docs_recursive_index():
    """A real `api_docs` `recursive` chunk set + real `bge-small` index. AC-4
    needs genuine embedding signal, which the committed fixture index (a
    32-dim HashEmbedder over 19 vectors) can't provide -- same reasoning as
    the Makefile's `verify-phase-4` target."""
    from rag_lab.chunkers import run_chunker
    from rag_lab.corpus import list_documents_by_corpus

    build_result = runner.invoke(app, ["corpus", "build", "--corpus", "api_docs"])
    assert build_result.exit_code == 0, build_result.output

    docs = list_documents_by_corpus("api_docs")["api_docs"]
    all_chunks: list[Chunk] = []
    for doc in docs:
        all_chunks.extend(run_chunker("recursive", doc, {}))
    chunk_set_id = all_chunks[0].chunk_set_id
    write_jsonl(artifact_path("chunks", f"{chunk_set_id}.jsonl"), all_chunks)

    manifest, _ = indexing.build_index(chunk_set_id, "bge-small", {})
    return manifest


def _rank_of_chunk_containing(manifest, chunks, identifier: str, query: str, retriever_name: str) -> int:
    target = next((c for c in chunks if identifier in c.text), None)
    assert target is not None, f"no chunk in {manifest.chunk_set_id} contains {identifier!r}"
    _m, results = retrieval.run_retriever(manifest.index_id, retriever_name, query, k=len(chunks))
    ranks = [r.rank for r in results if r.chunk.chunk_id == target.chunk_id]
    assert ranks, f"{identifier} chunk never retrieved by {retriever_name}"
    return ranks[0]


@requires_sentence_transformers
@requires_chromadb
@requires_rank_bm25
def test_ac4_bm25_ranks_the_identifier_chunk_higher_than_dense(real_api_docs_recursive_index):  # AC-4
    """Plan AC-4 is existential ("for an exact-identifier query...") --
    IDEMPOTENCY_KEY_CONFLICT is the demonstrative query (also the Makefile's
    verify-phase-4 choice), so this is the hard, single-query assertion."""
    manifest = real_api_docs_recursive_index
    chunks = load_chunk_set(manifest.chunk_set_id)
    identifier, query = API_DOCS_IDENTIFIER_QUERIES[0]

    bm25_rank = _rank_of_chunk_containing(manifest, chunks, identifier, query, "bm25")
    dense_rank = _rank_of_chunk_containing(manifest, chunks, identifier, query, "dense")
    assert bm25_rank < dense_rank


@requires_sentence_transformers
@requires_chromadb
@requires_rank_bm25
@pytest.mark.parametrize("identifier, query", API_DOCS_IDENTIFIER_QUERIES)
def test_bm25_is_never_worse_than_dense_on_identifier_queries(real_api_docs_recursive_index, identifier, query):
    """A softer, non-flaky companion to AC-4 across more identifiers: real
    embeddings don't cooperate identically for every identifier (the plan's
    own "if this doesn't reproduce" caveat), so this only requires BM25 to
    never rank the identifier chunk *worse* than dense, rather than strictly
    better for all three."""
    manifest = real_api_docs_recursive_index
    chunks = load_chunk_set(manifest.chunk_set_id)

    bm25_rank = _rank_of_chunk_containing(manifest, chunks, identifier, query, "bm25")
    dense_rank = _rank_of_chunk_containing(manifest, chunks, identifier, query, "dense")
    assert bm25_rank <= dense_rank


@requires_sentence_transformers
@requires_chromadb
def test_hierarchical_parent_doc_retrieval_end_to_end():
    from rag_lab.chunkers import run_chunker
    from rag_lab.chunkers.hierarchy import assign_parents
    from rag_lab.corpus import list_documents_by_corpus

    build_result = runner.invoke(app, ["corpus", "build", "--corpus", "api_docs"])
    assert build_result.exit_code == 0, build_result.output
    docs = list_documents_by_corpus("api_docs")["api_docs"]

    parent_chunks: list[Chunk] = []
    for doc in docs:
        parent_chunks.extend(run_chunker("markdown", doc, {"max_tokens": 2048, "role": "parent"}))
    parent_chunks = [c.model_copy(update={"role": "parent"}) for c in parent_chunks]
    parent_set_id = parent_chunks[0].chunk_set_id
    write_jsonl(artifact_path("chunks", f"{parent_set_id}.jsonl"), parent_chunks)

    raw_children: list[Chunk] = []
    for doc in docs:
        raw_children.extend(
            run_chunker(
                "recursive", doc,
                {"chunk_tokens": 256, "role": "child", "parent_chunk_set_id": parent_set_id},
            )
        )
    child_chunks = assign_parents(raw_children, parent_chunks)
    child_set_id = child_chunks[0].chunk_set_id
    write_jsonl(artifact_path("chunks", f"{child_set_id}.jsonl"), child_chunks)

    manifest, _ = indexing.build_index(child_set_id, "bge-small", {})

    _m, results = retrieval.run_retriever(
        manifest.index_id, "parent_doc",
        "What does the error code IDEMPOTENCY_KEY_CONFLICT mean?", k=3,
        overrides={"parent_chunk_set_id": parent_set_id},
    )
    assert results
    assert all(r.chunk.role == "parent" for r in results)
