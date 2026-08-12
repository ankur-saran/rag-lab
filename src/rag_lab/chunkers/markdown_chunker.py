"""`markdown` — structure-aware chunker: one chunk per leaf section (the
content directly under a heading, up to the next heading), fenced code blocks
never split, undersized sections folded forward, oversized ones handed to the
`recursive` splitter as a fallback.

A real parser (markdown-it-py) rather than the regex detection in
``markup.py`` is what makes this tractable: its block tokens carry line
ranges (``token.map``), so headings and fences are found correctly even when
nested inside a list or blockquote, and content inside a fence can never be
mistaken for a heading -- the fenced-code false positive `markup.py` guards
against by masking simply cannot occur here, because the parser already knows
it's looking at raw fence content.
"""

from __future__ import annotations

from typing import Any

from markdown_it import MarkdownIt

from rag_lab.chunkers.base import ChunkSpec, count_tokens, finalize_chunks
from rag_lab.chunkers.recursive import split_text_recursive
from rag_lab.schemas import Chunk, Document

NAME = "markdown"
DEFAULT_PARAMS: dict[str, Any] = {
    "max_tokens": 768,
    "min_tokens": 64,
    "heading_levels": [1, 2, 3],
    "prepend_heading_path": True,
    "keep_code_blocks_intact": True,
    "encoding": "cl100k_base",
}

# Used only for the oversized-section fallback split; not a user-facing param
# -- the plan doesn't expose separate separator control for this chunker.
_FALLBACK_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

_MD = MarkdownIt("commonmark")


def _line_offsets(text: str) -> list[int]:
    """``offsets[k]`` = char offset of the start of ``text.split("\\n")[k]``,
    for ``k`` in ``[0, n_lines]`` (the last entry is ``len(text)``) -- matches
    markdown-it-py's own line numbering, which splits on ``"\\n"`` the same way.
    """
    raw_lines = text.split("\n")
    offsets = [0]
    for line in raw_lines[:-1]:
        offsets.append(offsets[-1] + len(line) + 1)
    offsets.append(len(text))
    return offsets


def _parse_structure(
    text: str, heading_levels: set[int]
) -> tuple[list[tuple[int, int, int, str]], list[tuple[int, int]]]:
    """Returns (boundary headings, fence spans).

    Boundary headings are ``(start_line, end_line, level, heading_text)`` for
    headings whose level is in ``heading_levels``, in document order. Fence
    spans are absolute ``(char_start, char_end)`` pairs.
    """
    offsets = _line_offsets(text)
    tokens = _MD.parse(text)

    headings: list[tuple[int, int, int, str]] = []
    fence_spans: list[tuple[int, int]] = []
    for i, tok in enumerate(tokens):
        if tok.type == "heading_open" and tok.map:
            level = int(tok.tag[1:])
            if level in heading_levels:
                start_line, end_line = tok.map
                content = ""
                if i + 1 < len(tokens) and tokens[i + 1].type == "inline":
                    content = tokens[i + 1].content.strip()
                headings.append((start_line, end_line, level, content))
        elif tok.type == "fence" and tok.map:
            start_line, end_line = tok.map
            fence_spans.append((offsets[start_line], offsets[end_line]))

    return headings, fence_spans


def _leaf_sections(text: str, headings: list[tuple[int, int, int, str]]) -> list[dict[str, Any]]:
    """One section per boundary heading -- spanning from the heading line
    itself through the content up to the next boundary heading -- plus a
    leading section for any content before the first one. Zero-length spans
    are dropped; near-empty ones are left for the min_tokens merge pass.

    A section's span starts *at* its own heading line, not after it: the
    heading's text has to live somewhere in the returned ``text``, and this
    is what keeps the coverage invariant intact -- excluding it left every
    heading line uncovered by any chunk. ``heading_path`` is therefore just
    the *ancestor* headings; the section's own heading is already visible
    literally in its ``text``, so repeating it there would be redundant.
    """
    offsets = _line_offsets(text)
    sections: list[dict[str, Any]] = []
    stack: list[str] = []

    pre_end = offsets[headings[0][0]] if headings else len(text)
    if pre_end > 0 and text[:pre_end].strip():
        sections.append({"char_start": 0, "char_end": pre_end, "heading_path": []})

    for idx, (start_line, _end_line, level, heading_text) in enumerate(headings):
        ancestors = stack[: level - 1]
        stack = [*ancestors, heading_text]  # kept for any deeper headings that follow
        body_start = offsets[start_line]
        body_end = offsets[headings[idx + 1][0]] if idx + 1 < len(headings) else len(text)
        if body_end > body_start:
            sections.append(
                {"char_start": body_start, "char_end": body_end, "heading_path": list(ancestors)}
            )

    return sections


def _merge_undersized(
    sections: list[dict[str, Any]], text: str, min_tokens: int, encoding: str
) -> list[dict[str, Any]]:
    """Fold a section under ``min_tokens`` forward into the next one, so a
    heading with almost no body of its own (immediately followed by a
    subheading) doesn't become its own near-empty chunk. The merged span
    keeps the earlier section's ``heading_path`` -- it's still where the
    combined content starts.
    """
    if not sections:
        return sections
    merged = [dict(sections[0])]
    for sec in sections[1:]:
        prev = merged[-1]
        prev_tokens = count_tokens(text[prev["char_start"] : prev["char_end"]], encoding)
        if prev_tokens < min_tokens:
            prev["char_end"] = sec["char_end"]
        else:
            merged.append(dict(sec))
    return merged


class MarkdownChunker:
    name = NAME
    default_params = DEFAULT_PARAMS

    def chunk(self, doc: Document, params: dict[str, Any]) -> list[Chunk]:
        resolved = {**DEFAULT_PARAMS, **params}
        max_tokens = int(resolved["max_tokens"])
        min_tokens = int(resolved["min_tokens"])
        heading_levels = {int(x) for x in resolved["heading_levels"]}
        prepend_heading_path = bool(resolved["prepend_heading_path"])
        keep_code_blocks_intact = bool(resolved["keep_code_blocks_intact"])
        encoding = str(resolved["encoding"])

        text = doc.text
        headings, fence_spans = _parse_structure(text, heading_levels)
        sections = _leaf_sections(text, headings)
        sections = _merge_undersized(sections, text, min_tokens, encoding)

        protected = fence_spans if keep_code_blocks_intact else []

        specs: list[ChunkSpec] = []
        for sec in sections:
            start, end = sec["char_start"], sec["char_end"]
            heading_path: list[str] = sec["heading_path"]
            section_text = text[start:end]

            if count_tokens(section_text, encoding) <= max_tokens:
                pieces = [(start, end)]
            else:
                pieces = split_text_recursive(
                    section_text,
                    start,
                    max_tokens,
                    0,
                    _FALLBACK_SEPARATORS,
                    lambda t: count_tokens(t, encoding),
                    protected_spans=protected,
                    encoding=encoding,
                ) or [(start, end)]

            for p_start, p_end in pieces:
                piece_text = text[p_start:p_end]
                embed_text = None
                if prepend_heading_path and heading_path:
                    embed_text = " > ".join(heading_path) + "\n\n" + piece_text
                specs.append(
                    ChunkSpec(
                        char_start=p_start,
                        char_end=p_end,
                        text=piece_text,
                        embed_text=embed_text,
                        heading_path=list(heading_path),
                    )
                )

        return finalize_chunks(specs, doc, NAME, resolved, encoding=encoding)


__all__ = ["DEFAULT_PARAMS", "NAME", "MarkdownChunker"]
