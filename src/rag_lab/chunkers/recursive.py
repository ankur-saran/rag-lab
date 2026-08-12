"""`recursive` — separator-cascade chunker: try the coarsest separator first,
recurse into any piece still over budget with the next separator, then
greedily merge adjacent pieces back up to the token limit so structure isn't
paid for with a swarm of one-sentence chunks.

``split_text_recursive`` is the reusable half of this — Phase 2's `markdown`
chunker calls it as the fallback for a leaf section that's still oversized
after structural splitting, passing the section's fenced-code spans as
``protected_spans`` so a fence is never cut even under a token-budget squeeze.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from rag_lab.chunkers.base import ChunkSpec, count_tokens, finalize_chunks, token_windows
from rag_lab.schemas import Chunk, Document

NAME = "recursive"
DEFAULT_PARAMS: dict[str, Any] = {
    "chunk_tokens": 512,
    "overlap_tokens": 64,
    "separators": ["\n\n", "\n", ". ", " ", ""],
    "encoding": "cl100k_base",
}

CountFn = Callable[[str], int]


# --------------------------------------------------------------------------- #
# Separator cascade -> atomic pieces
# --------------------------------------------------------------------------- #


def _split_points(text: str, sep: str) -> list[int]:
    """End-offsets (within ``text``) of every occurrence of ``sep``."""
    points: list[int] = []
    start = 0
    while True:
        idx = text.find(sep, start)
        if idx == -1:
            return points
        start = idx + len(sep)
        points.append(start)


def _raw_pieces(text: str, sep: str) -> list[tuple[int, int]]:
    """Contiguous ``(start, end)`` spans of ``text`` split on ``sep``, with
    the separator attached to the piece that precedes it. Attaching rather
    than dropping the separator is what keeps every piece boundary a clean
    slice — no whitespace or content ever falls in a gap between two pieces.
    """
    bounds = sorted({0, *_split_points(text, sep), len(text)})
    return [(a, b) for a, b in zip(bounds, bounds[1:], strict=False) if b > a]


def _protected_at(span: tuple[int, int], protected_spans: list[tuple[int, int]]) -> bool:
    return any(ps <= span[0] and span[1] <= pe for ps, pe in protected_spans)


def _touches_protected(span: tuple[int, int], protected_spans: list[tuple[int, int]]) -> bool:
    """True if ``span`` overlaps a protected span *at all* -- looser than
    ``_protected_at``'s full-containment check. Used only as the last-resort
    guard right before the "" fallback would slice raw token windows: by that
    point every separator has already been tried and ``_drop_protected_boundaries``
    has already shrunk the piece as far as any of them could, so what's left
    is the protected content plus, at most, a little unsplittable fringe
    (typically a trailing separator `_raw_pieces` had to attach to the
    preceding piece) — not a large stretch of free text that would be worth
    peeling off. Using the same full-containment check here instead would
    fail to catch exactly that fringe case and let the fence get sliced.
    """
    s, e = span
    return any(ps < e and s < pe for ps, pe in protected_spans)


def _drop_protected_boundaries(
    pieces: list[tuple[int, int]], offset: int, protected_spans: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    """Merge consecutive pieces across any boundary that would fall strictly
    inside a protected span.

    Checking whether one whole candidate piece happens to sit inside a
    protected span (``_protected_at``) isn't enough on its own: a piece
    routinely carries a little trailing separator text past the fence's own
    edge (``_raw_pieces`` attaches the separator to the piece before it), so
    "piece == fence" is the exception, not the rule. Dropping every forbidden
    boundary directly, at each cascade level, is what actually keeps a fence
    whole regardless of where the surrounding whitespace happens to fall —
    the piece the fence lives in shrinks back to (approximately) just the
    fence as separators are tried, until ``_protected_at`` catches it cleanly.
    """
    if not protected_spans or len(pieces) <= 1:
        return pieces
    merged = [pieces[0]]
    for p_start, p_end in pieces[1:]:
        boundary = offset + p_start  # == offset + merged[-1][1]
        if any(ps < boundary < pe for ps, pe in protected_spans):
            merged[-1] = (merged[-1][0], p_end)
        else:
            merged.append((p_start, p_end))
    return merged


def split_text_recursive(
    text: str,
    base_offset: int,
    chunk_tokens: int,
    overlap_tokens: int,
    separators: list[str],
    count_fn: CountFn = count_tokens,
    protected_spans: list[tuple[int, int]] | None = None,
    encoding: str = "cl100k_base",
) -> list[tuple[int, int]]:
    """Absolute ``(char_start, char_end)`` spans covering ``text`` (a slice of
    a larger document starting at ``base_offset``), each within
    ``chunk_tokens`` wherever the text admits a split at all, then greedily
    merged back up to the limit with ``overlap_tokens`` of overlap between
    neighbours.

    A span wholly inside one of ``protected_spans`` (document-absolute
    offsets — e.g. a fenced code block) is treated as atomic: it is never cut
    further, even if it alone exceeds ``chunk_tokens``. That oversized-chunk
    cost is accepted deliberately; a bisected fence is not.
    """
    protected = protected_spans or []
    pieces = _atomic_pieces(
        (0, len(text)),
        text,
        base_offset,
        chunk_tokens,
        separators,
        0,
        count_fn,
        protected,
        encoding,
    )
    if not pieces:
        return []
    return _greedy_merge_with_overlap(
        pieces, chunk_tokens, overlap_tokens, count_fn, text, base_offset
    )


def _atomic_pieces(
    local_span: tuple[int, int],
    text: str,
    base_offset: int,
    chunk_tokens: int,
    separators: list[str],
    sep_idx: int,
    count_fn: CountFn,
    protected_spans: list[tuple[int, int]],
    encoding: str,
) -> list[tuple[int, int]]:
    start, end = local_span
    piece_text = text[start:end]
    if not piece_text:
        return []

    abs_span = (base_offset + start, base_offset + end)
    if count_fn(piece_text) <= chunk_tokens or _protected_at(abs_span, protected_spans):
        return [abs_span]
    if sep_idx >= len(separators):
        return [abs_span]  # nothing left to split on; accept the oversized piece

    sep = separators[sep_idx]
    if sep == "":
        if _touches_protected(abs_span, protected_spans):
            # Every separator has been tried; what's left is protected
            # content plus at most a little fringe `_raw_pieces` couldn't
            # detach (see `_touches_protected`). Accept it whole rather than
            # let raw token windows below cut into it.
            return [abs_span]
        # No textual structure left, and nothing protected here. Bound the
        # piece by raw token windows instead — the only place overlap is
        # *not* applied; the overlap the caller asked for is applied once,
        # at the merge step.
        return token_windows(piece_text, base_offset + start, chunk_tokens, 0, encoding)

    raw = _raw_pieces(piece_text, sep)
    raw = _drop_protected_boundaries(raw, base_offset + start, protected_spans)
    if len(raw) <= 1:
        # `sep` doesn't occur in this piece at all (or every occurrence sat
        # inside a protected span and got merged away) -- try the next one.
        return _atomic_pieces(
            local_span,
            text,
            base_offset,
            chunk_tokens,
            separators,
            sep_idx + 1,
            count_fn,
            protected_spans,
            encoding,
        )

    out: list[tuple[int, int]] = []
    for p_start, p_end in raw:
        out.extend(
            _atomic_pieces(
                (start + p_start, start + p_end),
                text,
                base_offset,
                chunk_tokens,
                separators,
                sep_idx + 1,
                count_fn,
                protected_spans,
                encoding,
            )
        )
    return out


# --------------------------------------------------------------------------- #
# Greedy merge with overlap
# --------------------------------------------------------------------------- #


def _greedy_merge_with_overlap(
    pieces: list[tuple[int, int]],
    chunk_tokens: int,
    overlap_tokens: int,
    count_fn: CountFn,
    text: str,
    base_offset: int,
) -> list[tuple[int, int]]:
    """Pack adjacent atomic pieces into windows up to ``chunk_tokens``, each
    new window seeded with a token-budgeted tail of the previous one.

    Overlap is computed at piece granularity (a piece is a line/sentence/etc
    from the separator cascade, or a token window from the "" fallback) —
    standard practice for recursive splitters, since re-slicing mid-piece
    would defeat the point of splitting on structure in the first place.
    """
    piece_tokens = [count_fn(text[p[0] - base_offset : p[1] - base_offset]) for p in pieces]

    windows: list[list[int]] = []
    current: list[int] = []
    current_tokens = 0
    i = 0
    n = len(pieces)
    while i < n:
        t = piece_tokens[i]
        if current and current_tokens + t > chunk_tokens:
            windows.append(current)
            current, current_tokens = _overlap_tail(current, piece_tokens, overlap_tokens)
            continue
        current.append(i)
        current_tokens += t
        i += 1
    if current:
        windows.append(current)

    return [(pieces[w[0]][0], pieces[w[-1]][1]) for w in windows]


def _overlap_tail(
    closed_window: list[int], piece_tokens: list[int], overlap_tokens: int
) -> tuple[list[int], int]:
    """The trailing sub-list of ``closed_window`` whose combined tokens are
    <= ``overlap_tokens``, used to seed the next window.

    A single-piece window can't usefully overlap with itself — carrying it
    forward would just re-close on the same piece and never advance — so it
    seeds nothing, regardless of ``overlap_tokens``. For a multi-piece window,
    accumulating from the end and stopping the instant the running total
    would exceed ``overlap_tokens`` (so ``overlap_tokens=0`` legitimately
    yields an empty tail, not "one piece minimum") is always a *proper*
    subset: the window's total exceeded ``chunk_tokens`` > ``overlap_tokens``
    to have been closed at all, so it's guaranteed to hit the threshold
    before consuming every piece -- progress on the next outer-loop iteration
    is never at risk.
    """
    if len(closed_window) <= 1:
        return [], 0
    tail: list[int] = []
    tail_tokens = 0
    for idx in reversed(closed_window):
        t = piece_tokens[idx]
        if tail_tokens + t > overlap_tokens:
            break
        tail.insert(0, idx)
        tail_tokens += t
    return tail, tail_tokens


# --------------------------------------------------------------------------- #
# Chunker
# --------------------------------------------------------------------------- #


class RecursiveChunker:
    name = NAME
    default_params = DEFAULT_PARAMS

    def chunk(self, doc: Document, params: dict[str, Any]) -> list[Chunk]:
        resolved = {**DEFAULT_PARAMS, **params}
        chunk_tokens = int(resolved["chunk_tokens"])
        overlap_tokens = int(resolved["overlap_tokens"])
        separators = list(resolved["separators"])
        encoding = str(resolved["encoding"])

        if overlap_tokens >= chunk_tokens:
            raise ValueError(
                f"overlap_tokens ({overlap_tokens}) must be < chunk_tokens ({chunk_tokens})"
            )

        def count_fn(t: str) -> int:
            return count_tokens(t, encoding)

        spans = split_text_recursive(
            doc.text,
            0,
            chunk_tokens,
            overlap_tokens,
            separators,
            count_fn,
            encoding=encoding,
        )
        specs = [
            ChunkSpec(char_start=start, char_end=end, text=doc.text[start:end])
            for start, end in spans
        ]
        return finalize_chunks(specs, doc, NAME, resolved, encoding=encoding)


__all__ = ["DEFAULT_PARAMS", "NAME", "RecursiveChunker", "split_text_recursive"]
