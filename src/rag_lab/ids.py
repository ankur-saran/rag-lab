"""Deterministic identity.

Stable IDs are what make caching and cross-phase joins work. Every ID in the
system is a pure function of its semantic inputs, so re-running any phase with
unchanged inputs reproduces byte-identical artifacts.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any

from rag_lab.config import params_hash

DOC_ID_LEN = 12
CHUNK_ID_LEN = 16


def sha1_hex(*parts: Any) -> str:
    h = hashlib.sha1()
    for part in parts:
        h.update(str(part).encode("utf-8"))
        h.update(b"\x00")  # delimiter, so ("ab","c") != ("a","bc")
    return h.hexdigest()


def make_doc_id(corpus: str, relative_path: str) -> str:
    return sha1_hex(corpus, relative_path.replace("\\", "/"))[:DOC_ID_LEN]


def chunker_signature(chunker: str, params: dict[str, Any]) -> str:
    return f"{chunker}:{params_hash(params)}"


def make_chunk_id(doc_id: str, char_start: int, char_end: int, signature: str) -> str:
    return sha1_hex(doc_id, char_start, char_end, signature)[:CHUNK_ID_LEN]


def make_chunk_set_id(corpus: str, chunker: str, params: dict[str, Any]) -> str:
    return f"{corpus}__{chunker}__{params_hash(params)}"


def make_index_id(chunk_set_id: str, embedder: str, params: dict[str, Any]) -> str:
    return f"{chunk_set_id}__{embedder}__{params_hash(params)}"


def make_run_id(experiment_name: str, now: datetime | None = None) -> str:
    stamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    return f"{experiment_name}__{stamp}"


def make_query_id(corpus: str, doc_id: str, span: tuple[int, int], query: str) -> str:
    return sha1_hex(corpus, doc_id, span[0], span[1], query)[:CHUNK_ID_LEN]


def split_for(query_id: str, ratios: tuple[float, float, float] = (0.6, 0.2, 0.2)) -> str:
    """Deterministic train/dev/test assignment by ID hash.

    Hash-based rather than random so the split survives regeneration, and so the
    optimizer agent (Phase 8) can never accidentally see the test split.
    """
    bucket = int(sha1_hex(query_id)[:8], 16) / 0xFFFFFFFF
    train, dev, _ = ratios
    if bucket < train:
        return "train"
    if bucket < train + dev:
        return "dev"
    return "test"


__all__ = [
    "chunker_signature",
    "make_chunk_id",
    "make_chunk_set_id",
    "make_doc_id",
    "make_index_id",
    "make_query_id",
    "make_run_id",
    "sha1_hex",
    "split_for",
]
