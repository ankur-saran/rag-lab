#!/usr/bin/env python
"""Generate fixtures/indexes/sample/ from fixtures/chunks/sample.jsonl.

Uses the deterministic, dependency-free ``HashEmbedder``
(``rag_lab.embedders.fixture``) rather than a real model: that keeps this
fixture reproducible across machines/torch versions and keeps
``scripts/build_fixtures.py`` (used by ``make fixtures``, which
``verify-phase-0/1/2`` depend on) from needing the ``embed`` extra. Building
*this* fixture does need ``chromadb`` installed, same as ``verify-phase-3``
already requires.

``fixtures/chunks/sample.jsonl`` spans multiple corpora (one chunk set per
corpus — see ``build_fixtures.py``), so there is no single real
``chunk_set_id`` to attach to this index. The manifest instead uses the
sentinel ``"sample"``, mirroring the sentinel ``corpus="sample"`` the
``RunResult`` fixture already uses for the same reason.

Usage:
    python scripts/build_index_fixture.py            # write the fixture
    python scripts/build_index_fixture.py --check     # verify, write nothing
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag_lab.embedders.fixture import HashEmbedder  # noqa: E402
from rag_lab.ids import make_index_id  # noqa: E402
from rag_lab.indexing import write_manifest  # noqa: E402
from rag_lab.jsonl import read_jsonl  # noqa: E402
from rag_lab.schemas import Chunk, IndexManifest  # noqa: E402
from rag_lab.stores import ChromaStore  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CHUNKS_FIXTURE = ROOT / "fixtures" / "chunks" / "sample.jsonl"
OUT = ROOT / "fixtures" / "indexes" / "sample"

EMBEDDER_NAME = HashEmbedder.name  # "hash-fixture" -- never a real --embedder choice
SAMPLE_CHUNK_SET_ID = "sample"


def build(out_dir: Path) -> IndexManifest:
    chunks = read_jsonl(CHUNKS_FIXTURE, Chunk)
    if not chunks:
        raise SystemExit(f"no chunks found at {CHUNKS_FIXTURE}")

    embedder = HashEmbedder()
    started = time.monotonic()
    vectors = embedder.embed_documents([c.embed_text for c in chunks])  # embed_text, never .text

    if out_dir.exists():
        shutil.rmtree(out_dir)
    store = ChromaStore(out_dir)
    store.upsert(chunks, vectors)

    manifest = IndexManifest(
        index_id=make_index_id(SAMPLE_CHUNK_SET_ID, EMBEDDER_NAME, embedder.default_params),
        chunk_set_id=SAMPLE_CHUNK_SET_ID,
        # Same "sample" sentinel as chunk_set_id, for the same reason: the
        # underlying chunks span multiple real corpora, so there is no single
        # real corpus to attach (plan §Phase 8, Step 8.0.1's `corpus` field).
        corpus=SAMPLE_CHUNK_SET_ID,
        embedder=EMBEDDER_NAME,
        embedder_params=dict(embedder.default_params),
        vector_count=len(chunks),
        dim=int(vectors.shape[1]),
        build_duration_s=time.monotonic() - started,
        library_versions={},
    )
    write_manifest(out_dir, manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Rebuild into a temp dir and verify the committed fixture reproduces the "
        "same chunk count, dim, and chunk_ids -- not a byte diff, since Chroma's "
        "on-disk format isn't guaranteed byte-stable across runs the way JSONL is.",
    )
    args = parser.parse_args()

    if not args.check:
        manifest = build(OUT)
        print(f"  indexes    1 ({manifest.vector_count} vectors, dim={manifest.dim})")
        print(f"fixture written to {OUT}")
        return 0

    if not (OUT / "manifest.json").exists():
        print(
            f"fixture check FAILED: {OUT} has no manifest.json (run `make fixtures-index`)",
            file=sys.stderr,
        )
        return 1

    committed_manifest = IndexManifest.model_validate_json(
        (OUT / "manifest.json").read_text(encoding="utf-8")
    )
    all_chunk_ids = [c.chunk_id for c in read_jsonl(CHUNKS_FIXTURE, Chunk)]
    committed_store = ChromaStore(OUT)
    committed_ids = {c.chunk_id for c in committed_store.get(all_chunk_ids)}

    mismatches = []
    with tempfile.TemporaryDirectory() as tmp:
        regenerated_manifest = build(Path(tmp))
        regenerated_store = ChromaStore(Path(tmp))
        regenerated_ids = {c.chunk_id for c in regenerated_store.get(all_chunk_ids)}
        # Release Chroma's file handles before the `with` block tries to
        # delete `tmp` -- on Windows, an open sqlite/HNSW file blocks rmtree.
        regenerated_store.close()
        committed_store.close()

        if committed_manifest.vector_count != regenerated_manifest.vector_count:
            mismatches.append(
                f"vector_count differs: committed={committed_manifest.vector_count} "
                f"regenerated={regenerated_manifest.vector_count}"
            )
        if committed_manifest.dim != regenerated_manifest.dim:
            mismatches.append(f"dim differs: committed={committed_manifest.dim} "
                               f"regenerated={regenerated_manifest.dim}")
        if committed_ids != regenerated_ids:
            mismatches.append("stored chunk_ids differ")

    if mismatches:
        print("fixture check FAILED:", file=sys.stderr)
        for m in mismatches:
            print(f"  - {m}", file=sys.stderr)
        return 1
    print("index fixture is reproducible and up to date")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
