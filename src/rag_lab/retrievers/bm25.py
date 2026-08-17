"""``bm25`` — lexical retrieval over ``Chunk.text`` (never ``embed_text`` —
lexical matching wants the real content, not embedding prefixes/heading
paths). Plan §Phase 4, Step 4.3.

Lowercasing, punctuation stripped, **no stemming** (stemming hurts on
identifier-heavy corpora like ``api_docs``), and underscores kept as
word-characters so an identifier like ``IDEMPOTENCY_KEY_CONFLICT`` tokenizes
as a single token rather than being shredded into ``idempotency``, ``key``,
``conflict`` — the latter would blur BM25's advantage on exact-identifier
queries, which is the entire point of keeping this retriever.

Persistence is lazy, not upfront: a BM25 index is built in memory from the
chunk set every time ``bm25.pkl`` is missing, and persisted back to
``index_dir`` only when that directory is writable (``not
paths.is_fixture(index_dir)``). This is the general mechanism, not a fixture
special case — a real ``index build`` output has no ``bm25.pkl`` either, until
the first ``bm25``/``hybrid`` query builds and caches one.
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path

import structlog

from rag_lab.chunks import load_chunk_set
from rag_lab.paths import is_fixture
from rag_lab.retrievers.base import truncate_and_rank
from rag_lab.schemas import Chunk, ScoredChunk

NAME = "bm25"
PICKLE_FILENAME = "bm25.pkl"

log = structlog.get_logger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9_]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _pickle_path(index_dir: Path) -> Path:
    return index_dir / PICKLE_FILENAME


def _build_bm25(chunks: list[Chunk], *, k1: float, b: float):
    from rank_bm25 import BM25Okapi

    corpus = [_tokenize(c.text) for c in chunks]
    return BM25Okapi(corpus, k1=k1, b=b)


def _load_or_build(
    chunk_set_id: str, index_dir: Path, chunks: list[Chunk], *, k1: float, b: float
):
    """Return a ``(bm25, chunk_ids)`` pair whose order matches the BM25 corpus.

    ``chunk_ids`` is persisted alongside the model so a stored pickle can be
    matched back up against freshly-loaded ``Chunk`` objects even if the
    on-disk chunk set's line order ever changes.
    """
    pickle_path = _pickle_path(index_dir)
    if pickle_path.exists():
        try:
            with pickle_path.open("rb") as fh:
                payload = pickle.load(fh)
            return payload["bm25"], payload["chunk_ids"]
        except Exception:
            log.warning("bm25_pickle_invalid_rebuilding", chunk_set_id=chunk_set_id)

    bm25 = _build_bm25(chunks, k1=k1, b=b)
    chunk_ids = [c.chunk_id for c in chunks]

    if not is_fixture(index_dir):
        try:
            pickle_path.parent.mkdir(parents=True, exist_ok=True)
            with pickle_path.open("wb") as fh:
                pickle.dump({"chunk_ids": chunk_ids, "bm25": bm25}, fh)
        except OSError as exc:
            log.warning("bm25_pickle_write_failed", chunk_set_id=chunk_set_id, error=str(exc))

    return bm25, chunk_ids


class BM25Retriever:
    name = NAME

    def __init__(
        self,
        chunk_set_id: str,
        index_dir: Path,
        *,
        k1: float = 1.5,
        b: float = 0.75,
        name: str = NAME,
    ) -> None:
        self.chunk_set_id = chunk_set_id
        self.index_dir = Path(index_dir)
        self.name = name

        chunks = load_chunk_set(chunk_set_id)
        self._chunks_by_id = {c.chunk_id: c for c in chunks}
        self._bm25, self._chunk_ids = _load_or_build(
            chunk_set_id, self.index_dir, chunks, k1=k1, b=b
        )

    def retrieve(self, query: str, k: int) -> list[ScoredChunk]:
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(zip(self._chunk_ids, scores), key=lambda pair: pair[1], reverse=True)

        results = [
            ScoredChunk(
                chunk=self._chunks_by_id[chunk_id],
                score=float(score),
                rank=i,
                retriever=self.name,
                debug={"bm25_score": float(score)},
            )
            for i, (chunk_id, score) in enumerate(ranked[:k], start=1)
            if chunk_id in self._chunks_by_id
        ]
        return truncate_and_rank(results, k, self.name)


__all__ = ["NAME", "PICKLE_FILENAME", "BM25Retriever"]
