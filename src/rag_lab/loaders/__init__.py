"""Document loaders. Phase 1.

``load_corpus`` is the single entry point ``corpus build`` uses: walk a
corpus directory, dispatch each file to the loader registered for its
extension, skip the provenance doc, warn and skip anything unrecognized, and
return documents sorted by ``source_path``. That sort is not cosmetic —
filesystem walk order is not guaranteed to be stable across OSes, and
byte-identical rebuilds (Phase 1 acceptance criterion 2) depend on stable line
order in the written JSONL as much as they depend on a stable ``doc_id``.
"""

from __future__ import annotations

from pathlib import Path

import structlog

from rag_lab.loaders.base import Loader, humanize, relative_source_path
from rag_lab.loaders.markdown_loader import MarkdownLoader
from rag_lab.loaders.text_loader import TextLoader
from rag_lab.paths import corpora_dir
from rag_lab.schemas import Document

log = structlog.get_logger(__name__)

# Provenance doc filename, matched case-insensitively. Anything else
# unrecognized is a loader gap, not a convention — it gets a loud warning.
SKIP_FILENAMES = {"source.md"}

_LOADERS: list[Loader] = [MarkdownLoader(), TextLoader()]
REGISTRY: dict[str, Loader] = {ext: loader for loader in _LOADERS for ext in loader.extensions}


def discover_corpora(root: Path | None = None) -> list[str]:
    """Sorted names of corpus subdirectories under ``corpora/``."""
    base = root if root is not None else corpora_dir()
    if not base.exists():
        return []
    return sorted(p.name for p in base.iterdir() if p.is_dir())


def load_corpus(corpus: str, corpus_dir: Path) -> list[Document]:
    """Load every recognized file under ``corpus_dir`` into a ``Document``.

    Returns documents sorted by ``source_path`` — see the module docstring for
    why that sort is load-bearing, not cosmetic.
    """
    docs: list[Document] = []
    for path in corpus_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.name.lower() in SKIP_FILENAMES:
            continue
        loader = REGISTRY.get(path.suffix.lower())
        if loader is None:
            log.warning(
                "unrecognized_corpus_file_skipped",
                corpus=corpus,
                path=str(path),
                suffix=path.suffix,
            )
            continue
        docs.append(loader.load(path, corpus))
    docs.sort(key=lambda d: d.source_path)
    return docs


__all__ = [
    "REGISTRY",
    "SKIP_FILENAMES",
    "Loader",
    "MarkdownLoader",
    "TextLoader",
    "discover_corpora",
    "humanize",
    "load_corpus",
    "relative_source_path",
]
