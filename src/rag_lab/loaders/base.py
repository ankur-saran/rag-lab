"""Shared loader primitives: the ``Loader`` protocol and small helpers every
concrete loader needs."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Protocol

from rag_lab.paths import repo_root
from rag_lab.schemas import Document


class Loader(Protocol):
    extensions: set[str]

    def load(self, path: Path, corpus: str) -> Document: ...


def relative_source_path(path: Path) -> str:
    """``path`` relative to the repo root, as a forward-slash string.

    This is the exact string ``make_doc_id`` hashes and ``Document.source_path``
    stores, so it must be stable across OSes — ``as_posix()`` normalizes the
    separator regardless of platform.
    """
    return path.resolve().relative_to(repo_root().resolve()).as_posix()


_WORD_SPLIT_RE = re.compile(r"[-_]+")


def humanize(stem: str) -> str:
    """``'auth-token_lifecycle'`` -> ``'Auth Token Lifecycle'``.

    Filename-derived title fallback for documents with no other title signal
    (a ``TextLoader`` document, or a markdown document with no H1). Purely
    cosmetic — it doesn't know about acronyms ("Sdks" not "SDKs") — which is an
    acceptable tradeoff for a fallback path rather than a dependency.
    """
    words = [w for w in _WORD_SPLIT_RE.split(stem) if w]
    return " ".join(w.capitalize() for w in words) or stem


__all__ = ["Loader", "humanize", "relative_source_path"]
