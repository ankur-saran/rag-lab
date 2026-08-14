"""Index build/cache/load/search orchestration (Phase 3), plus terminal
rendering for the `index build`/`list`/`search` CLI commands. Mirrors
`chunks.py`/`corpus.py`: this is the one place that knows how a chunk set
becomes a persisted Chroma index and back, so the CLI stays thin.
"""

from __future__ import annotations

import importlib.metadata
import time
from pathlib import Path
from typing import Any

import structlog
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from rag_lab.chunks import load_chunk_set
from rag_lab.embedders import build_embedder
from rag_lab.embedders import resolve_params as resolve_embedder_params
from rag_lab.embedders.fixture import HashEmbedder
from rag_lab.ids import make_index_id
from rag_lab.paths import artifact_path, artifacts_dir, resolve_artifact
from rag_lab.schemas import IndexManifest, ScoredChunk
from rag_lab.stores import ChromaStore

log = structlog.get_logger(__name__)


def _manifest_path(index_dir: Path) -> Path:
    return index_dir / "manifest.json"


def _library_versions(*names: str) -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return versions


def load_manifest(index_dir: Path) -> IndexManifest:
    return IndexManifest.model_validate_json(_manifest_path(index_dir).read_text(encoding="utf-8"))


def write_manifest(index_dir: Path, manifest: IndexManifest) -> None:
    index_dir.mkdir(parents=True, exist_ok=True)
    _manifest_path(index_dir).write_text(manifest.model_dump_json(indent=2), encoding="utf-8")


def _embedder_for(name: str, params: dict[str, Any]):
    """Reconstruct the embedder that produced a given index's vectors.

    ``HashEmbedder`` is deliberately absent from ``embedders.NAMED_EMBEDDERS``
    (fixture/test-only, never a real ``--embedder`` choice — see
    ``embedders/fixture.py``), but the *committed* fixture index was built
    with it, and searching that index needs to reconstruct the exact same
    embedder to embed the query. This is the one place that special-cases it,
    driven entirely by what ``manifest.embedder`` says was actually used.
    """
    if name == HashEmbedder.name:
        return HashEmbedder(params)
    return build_embedder(name, params)


def build_index(
    chunk_set_id: str,
    embedder_name: str,
    overrides: dict[str, Any] | None = None,
    force: bool = False,
) -> tuple[IndexManifest, bool]:
    """Build (or reuse) the index for ``(chunk_set_id, embedder_name, overrides)``.

    Returns ``(manifest, cache_hit)``. The cache check happens *before* the
    embedder is constructed, so a cache hit never pays model-load cost — this
    is what makes a repeat ``index build`` complete in well under a second
    regardless of how slow the underlying model is to load (plan §3.5's
    caching acceptance criterion).
    """
    chunks = load_chunk_set(chunk_set_id)
    embedder_params = resolve_embedder_params(embedder_name, overrides or {})
    index_id = make_index_id(chunk_set_id, embedder_name, embedder_params)
    index_dir = artifact_path("indexes", index_id)

    if not force and _manifest_path(index_dir).exists():
        try:
            manifest = load_manifest(index_dir)
        except Exception:
            log.warning("index_manifest_invalid_rebuilding", index_id=index_id)
        else:
            log.info("index_cache_hit", index_id=index_id)
            return manifest, True

    embedder = build_embedder(embedder_name, overrides or {})
    started = time.monotonic()

    texts = [c.embed_text for c in chunks]  # embed_text, never .text
    vectors = embedder.embed_documents(texts)

    store = ChromaStore(index_dir)
    store.upsert(chunks, vectors)

    manifest = IndexManifest(
        index_id=index_id,
        chunk_set_id=chunk_set_id,
        embedder=embedder_name,
        embedder_params=embedder_params,
        vector_count=len(chunks),
        dim=int(vectors.shape[1]),
        build_duration_s=time.monotonic() - started,
        library_versions=_library_versions("sentence-transformers", "chromadb", "torch"),
    )
    write_manifest(index_dir, manifest)
    log.info("index_built", index_id=index_id, vector_count=len(chunks), dim=manifest.dim)
    return manifest, False


def _index_dir_for(index_id: str | None, allow_fixture: bool = True) -> Path:
    return resolve_artifact("indexes", index_id, allow_fixture=allow_fixture)


def load_store(index_id: str) -> ChromaStore:
    return ChromaStore(_index_dir_for(index_id))


def search_index(
    index_id: str, query: str, k: int = 5
) -> tuple[IndexManifest, list[ScoredChunk]]:
    index_dir = _index_dir_for(index_id)
    manifest = load_manifest(index_dir)
    embedder = _embedder_for(manifest.embedder, manifest.embedder_params)
    vector = embedder.embed_query(query)
    return manifest, ChromaStore(index_dir).search(vector, k)


def available_index_ids() -> list[str]:
    indexes_dir = artifacts_dir() / "indexes"
    if not indexes_dir.exists():
        return []
    return sorted(p.parent.name for p in indexes_dir.glob("*/manifest.json"))


def list_manifests() -> list[IndexManifest]:
    indexes_dir = artifacts_dir() / "indexes"
    return [load_manifest(indexes_dir / index_id) for index_id in available_index_ids()]


# --------------------------------------------------------------------------- #
# Terminal rendering
# --------------------------------------------------------------------------- #


def render_index_list_table(manifests: list[IndexManifest], console: Console) -> None:
    table = Table(title="rag-lab indexes", title_style="bold")
    table.add_column("index_id", no_wrap=True)
    table.add_column("chunk_set_id", no_wrap=True)
    table.add_column("embedder")
    table.add_column("dim", justify="right")
    table.add_column("vectors", justify="right")
    table.add_column("build (s)", justify="right")
    for m in manifests:
        table.add_row(
            escape(m.index_id),
            escape(m.chunk_set_id),
            escape(m.embedder),
            str(m.dim),
            str(m.vector_count),
            f"{m.build_duration_s:.2f}",
        )
    console.print(table)


def render_search_results(query: str, results: list[ScoredChunk], console: Console) -> None:
    table = Table(title=f"search: {query!r}", title_style="bold")
    table.add_column("rank", justify="right")
    table.add_column("score", justify="right")
    table.add_column("chunk_id", no_wrap=True)
    table.add_column("text", overflow="fold")
    for r in results:
        preview = " ".join(r.chunk.text.split())
        preview = preview if len(preview) <= 120 else preview[:119] + "…"
        table.add_row(str(r.rank), f"{r.score:.4f}", r.chunk.chunk_id, escape(preview))
    console.print(table)


__all__ = [
    "available_index_ids",
    "build_index",
    "list_manifests",
    "load_manifest",
    "load_store",
    "render_index_list_table",
    "render_search_results",
    "search_index",
    "write_manifest",
]
