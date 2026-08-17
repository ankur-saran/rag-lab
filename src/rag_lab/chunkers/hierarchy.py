"""Parent/child chunk-set matching by span containment (plan §Phase 4, Step 4.7).

Lives in ``chunkers/`` rather than ``retrievers/``: this is pure
chunk-boundary-relationship logic operating on ``list[Chunk]``, no
retrieval/store/embedder dependency -- the same domain as ``chunkers/base.py``'s
``ChunkSpec``/``finalize_chunks``. Keeping it here means ``cli.py::chunk_run``
(a Phase 2 command) never has to import ``retrievers/`` (a Phase 4 package)
to support ``--role child``.
"""

from __future__ import annotations

from rag_lab.schemas import Chunk


class OrphanChildError(ValueError):
    """One or more children matched zero parents. Never silently dropped --
    plan Phase 4 AC-3 requires "no orphans, no duplicates"."""


def find_parent(child: Chunk, candidates: list[Chunk]) -> Chunk | None:
    """The parent whose span contains ``child``'s midpoint.

    Containment by midpoint (rather than full overlap) handles boundary
    straddling without ambiguity (plan §4.7). Ties -- only possible when
    candidate parent spans themselves overlap -- are broken by smallest span,
    then earliest ``char_start``, so the result is deterministic.
    """
    midpoint = (child.char_start + child.char_end) / 2
    matches = [p for p in candidates if p.char_start <= midpoint < p.char_end]
    if not matches:
        return None
    return min(matches, key=lambda p: (p.char_end - p.char_start, p.char_start))


def assign_parents(children: list[Chunk], parents: list[Chunk]) -> list[Chunk]:
    """Match each child to a parent by midpoint containment, per ``doc_id``.

    Returns new ``Chunk`` records with ``role="child"`` and ``parent_id`` set --
    inputs are never mutated, and each child resolves to at most one parent by
    construction (``find_parent`` returns a single ``Chunk | None``), so "no
    duplicates" holds structurally. Raises ``OrphanChildError`` naming every
    unmatched child if any child resolves to zero parents, rather than
    dropping it or leaving it silently parentless.
    """
    parents_by_doc: dict[str, list[Chunk]] = {}
    for p in parents:
        parents_by_doc.setdefault(p.doc_id, []).append(p)

    resolved: list[Chunk] = []
    orphans: list[str] = []
    for child in children:
        candidates = parents_by_doc.get(child.doc_id, [])
        parent = find_parent(child, candidates)
        if parent is None:
            orphans.append(
                f"{child.chunk_id} (doc_id={child.doc_id}, "
                f"span=[{child.char_start}:{child.char_end}])"
            )
            continue
        resolved.append(child.model_copy(update={"role": "child", "parent_id": parent.chunk_id}))

    if orphans:
        raise OrphanChildError(
            f"{len(orphans)} child chunk(s) matched zero parents: " + "; ".join(orphans)
        )
    return resolved


__all__ = ["OrphanChildError", "assign_parents", "find_parent"]
