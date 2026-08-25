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
import itertools
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
    # sort key matches chunks.render_chunk_boundaries's own ordering
    ordered = sorted(chunks, key=lambda c: (c.char_start, c.char_end))
    breakpoints: set[int] = {0, text_len}
    for c in ordered:
        breakpoints.add(max(0, min(c.char_start, text_len)))
        breakpoints.add(max(0, min(c.char_end, text_len)))
    sorted_bp = sorted(breakpoints)

    segments: list[Segment] = []
    for a, b in itertools.pairwise(sorted_bp):
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


def render_segments_html(
    doc_text: str, segments: list[Segment], *, shade_class_prefix: str = "shade"
) -> str:
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


# --------------------------------------------------------------------------- #
# HTML wrapping -- still pure string building, no Streamlit import. The single
# vs. dual rendering split (plan Step 9.1) is load-bearing, not cosmetic:
# `st.markdown(unsafe_allow_html=True)` renders via `dangerouslySetInnerHTML`,
# and browsers never execute a `<script>` injected that way, so real
# scroll-sync between two panes is only reachable inside a real iframe
# (`st.components.v1.html`), where injected scripts do execute.
# --------------------------------------------------------------------------- #

PANE_CSS = """
<style>
  body { font-family: ui-monospace, "Cascadia Code", Consolas, monospace; margin: 0; }
  .doc-pane { white-space: pre-wrap; word-wrap: break-word; line-height: 1.5; padding: 12px; }
  .shade-0 { background: rgba(99, 102, 241, 0.16); }
  .shade-1 { background: rgba(16, 185, 129, 0.16); }
  .overlap {
    background: repeating-linear-gradient(
      45deg, rgba(245, 158, 11, 0.35), rgba(245, 158, 11, 0.35) 6px,
      rgba(245, 158, 11, 0.12) 6px, rgba(245, 158, 11, 0.12) 12px
    );
  }
  .gap { background: rgba(239, 68, 68, 0.12); }
</style>
"""


def wrap_single_pane_html(inner_html: str) -> str:
    """Standalone fragment for a single chunk-set view, safe to pass to
    ``st.markdown(..., unsafe_allow_html=True)``."""
    return f"{PANE_CSS}<div class=\"doc-pane\">{inner_html}</div>"


def wrap_dual_pane_html(inner_html_a: str, inner_html_b: str, label_a: str, label_b: str) -> str:
    """A full standalone HTML document -- two scroll-synced panes -- for
    ``st.components.v1.html``. Scroll position is mirrored by *ratio*
    (``scrollTop / (scrollHeight - clientHeight)``), not raw pixels, since the
    two chunk sets' rendered lengths generally differ; a ``syncing`` guard
    flag prevents the mirrored scroll event from re-triggering its own
    listener and looping."""
    return f"""<!doctype html>
<html><head>{PANE_CSS}
<style>
  .panes {{ display: flex; gap: 8px; height: 100%; }}
  .pane-col {{ flex: 1; min-width: 0; display: flex; flex-direction: column; }}
  .pane-label {{ font: 600 13px ui-sans-serif, system-ui; padding: 6px 12px; opacity: 0.7; }}
  .pane-scroll {{ overflow-y: auto; flex: 1; border: 1px solid rgba(128,128,128,0.25); }}
</style></head>
<body>
<div class="panes">
  <div class="pane-col">
    <div class="pane-label">{html.escape(label_a)}</div>
    <div class="pane-scroll" id="pane-a"><div class="doc-pane">{inner_html_a}</div></div>
  </div>
  <div class="pane-col">
    <div class="pane-label">{html.escape(label_b)}</div>
    <div class="pane-scroll" id="pane-b"><div class="doc-pane">{inner_html_b}</div></div>
  </div>
</div>
<script>
  const a = document.getElementById("pane-a");
  const b = document.getElementById("pane-b");
  let syncing = false;
  function mirror(src, dst) {{
    if (syncing) return;
    syncing = true;
    const ratio = src.scrollTop / Math.max(1, src.scrollHeight - src.clientHeight);
    dst.scrollTop = ratio * Math.max(1, dst.scrollHeight - dst.clientHeight);
    syncing = false;
  }}
  a.addEventListener("scroll", () => mirror(a, b));
  b.addEventListener("scroll", () => mirror(b, a));
</script>
</body></html>"""


__all__ = [
    "Segment",
    "heading_path_str",
    "render_segments_html",
    "segment_document",
    "segment_title",
    "wrap_dual_pane_html",
    "wrap_single_pane_html",
]
