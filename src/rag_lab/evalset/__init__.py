"""Synthetic eval set generation and validation. Phase 5.

Generation/validation (``evalset build``/``validate``/``review``) lands with
the rest of Phase 5. ``load_evalset`` is a deliberate exception: Phase 6's
runner (and, later, Phase 8's optimizer) need to *read* an already-produced
eval set, and that's a pure read over the frozen ``EvalPair`` schema -- the
same kind of cross-phase read ``retrieval.py`` already does against
``indexing.py``. It lives here rather than being duplicated inside
``experiment/`` so every future consumer shares one implementation.
"""

from __future__ import annotations

from rag_lab.jsonl import read_jsonl
from rag_lab.paths import resolve_artifact
from rag_lab.schemas import EvalPair


def load_evalset(corpus: str) -> list[EvalPair]:
    """The eval set for one corpus: real artifact if built, else the shared
    fixture filtered to this corpus -- mirrors
    ``corpus.list_documents_by_corpus``'s shape exactly, including the
    warning-on-fixture-fallback behavior that comes for free from
    ``resolve_artifact``.

    Raises ``LookupError`` when nothing at all is available for ``corpus``.
    """
    path = resolve_artifact("evalset", corpus)
    pairs = [p for p in read_jsonl(path, EvalPair) if p.corpus == corpus]
    if not pairs:
        raise LookupError(f"no eval pairs found for corpus {corpus!r} (checked {path})")
    return pairs


__all__ = ["load_evalset"]
