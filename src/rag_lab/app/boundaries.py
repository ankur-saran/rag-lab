"""Chunk-boundary segmentation for the Streamlit viewer (plan §Phase 9, Step 9.1).

Pure data transformation, no Streamlit import -- this is what makes it plain
pytest testable (AC-2: "boundary offsets align exactly with document text")
without ever touching a browser or screen-scraping HTML.

This is deliberately *not* a port of ``chunks.render_chunk_boundaries``'s loop
(that prints a `Rule` per chunk boundary in document order, re-printing the
overlapping slice once per overlapping chunk -- fine for a terminal
transcript, impossible to turn into one non-overlapping run of HTML `<span>`s).
Instead this does interval-overlay segmentation: cut the document into the
maximal runs of text covered by an unchanging set of chunks, reusing
``render_chunk_boundaries``'s own sort key and overlap predicate
(``c.char_start < prev_end``) as the ordering/overlap definition.
"""

from __future__ import annotations

import html
from dataclasses import dataclass

from rag_lab.schemas import Chunk


@dataclass(frozen=True)
class Segment:
    start: int
    end: int
    covering: tuple[Chunk, ...]  # chunks whose span contains [start, end); order matches sort key

    @property
    def overlapping(self) -> bool:
        return len(self.covering) > 1


def segment_document(text_len: int, chunks: list[Chunk]) -> list[Segment]:
    """Cut ``[0, text_len)`` at every chunk boundary, so each returned
    ``Segment`` has a fixed, unambiguous set of covering chunks.

    Segments are contiguous and gap-free by construction (breakpoints always
    include 0 and ``text_len``), so concatenating ``doc.text[s.start:s.end]``
    across the result reconstructs ``doc.text`` exactly -- the property
    ``tests/test_phase_9.py`` checks directly for AC-2.
    """
    ordered = sorted(chunks, key=lambda c: (c.char_start, c.char_end))  # chunks.render_chunk_boundaries's key
    breakpoints: set[int] = {0, text_len}
    for c in ordered:
        breakpoints.add(max(0, min(c.char_start, text_len)))
        breakpoints.add(max(0, min(c.char_end, text_len)))
    sorted_bp = sorted(breakpoints)

    segments: list[Segment] = []
    for a, b in zip(sorted_bp, sorted_bp[1:]):
        if a >= b:
            continue
        covering = tuple(c for c in ordered if c.char_start <= a < c.char_end)
        segments.append(Segment(a, b, covering))
    return segments


def heading_path_str(chunk: Chunk) -> str:
    return " > ".join(chunk.heading_path) if chunk.heading_path else "(no heading)"


def segment_title(segment: Segment) -> str:
    """Hover text (``title`` attribute): chunk id(s), token count(s), heading
    path(s) -- plan Step 9.1's "hovering shows chunk ID, token count, and
    heading path"."""
    parts = [
        f"{c.chunk_id} | {c.token_count} tok | {heading_path_str(c)}" for c in segment.covering
    ]
    return html.escape(" || ".join(parts)) if parts else ""


def render_segments_html(doc_text: str, segments: list[Segment], *, shade_class_prefix: str = "shade") -> str:
    """One escaped, alternating-shade (and overlap-hatched) `<span>` per
    segment. Text is always sliced from ``doc_text`` by offset -- never from
    any ``Chunk.text`` -- which is what makes the rendered highlight
    mechanically aligned with the document rather than merely asserted to be.
    """
    pieces: list[str] = []
    for i, seg in enumerate(segments):
        text = html.escape(doc_text[seg.start : seg.end])
        classes = [f"{shade_class_prefix}-{i % 2}"]
        if seg.overlapping:
            classes.append("overlap")
        elif not seg.covering:
            classes.append("gap")
        title = segment_title(seg)
        title_attr = f' title="{title}"' if title else ""
        pieces.append(f'<span class="{" ".join(classes)}"{title_attr}>{text}</span>')
    return "".join(pieces)


__all__ = ["Segment", "heading_path_str", "render_segments_html", "segment_document", "segment_title"]
