"""Shared, regex-based markdown structure detection.

Both ``MarkdownLoader`` (title / section_count) and ``corpus.py`` (heading
density, code-block count, table count) need to answer the same three
questions — "where are the headings", "where are the fenced code blocks", and
"where are the tables" — so the logic lives here once instead of twice, where
it could quietly drift.

The one rule that matters: heading and table detection always run on
``mask_code_blocks(text)``, never on raw text. A fenced example that happens to
contain a ``# comment`` or a fake ``| a | b |`` table must never be counted as
real document structure. Code-block counting is the one exception — it counts
the fences themselves, so it runs on the unmasked text.
"""

from __future__ import annotations

import re

CODE_FENCE_RE = re.compile(r"```.*?```", re.S)
ATX_HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*$", re.M)
_TABLE_CELL_RE = re.compile(r"^:?-{3,}:?$")


def mask_code_blocks(text: str) -> str:
    """Replace each fenced code block with blank lines, preserving line count.

    Fence markers and their contents are all blanked — headings/tables can
    only be found outside of them. Line count is preserved so any later
    line-oriented logic stays aligned with the original text.
    """

    def _blank(match: re.Match[str]) -> str:
        return "\n".join("" for _ in match.group(0).split("\n"))

    return CODE_FENCE_RE.sub(_blank, text)


def count_code_blocks(text: str) -> int:
    """Number of fenced code blocks. Runs on unmasked text — it's counting the
    fences themselves, not looking past them."""
    return len(CODE_FENCE_RE.findall(text))


def find_headings(text: str) -> list[tuple[int, str]]:
    """(level, heading_text) pairs in document order, from ATX (``#``) headings
    outside of fenced code blocks."""
    masked = mask_code_blocks(text)
    return [(len(m.group(1)), m.group(2).strip()) for m in ATX_HEADING_RE.finditer(masked)]


def _is_table_separator_line(line: str) -> bool:
    """True for a GFM table separator row, e.g. ``| --- | :---: | ---: |``.

    Requires at least one ``|`` (so a bare ``---`` horizontal rule / setext
    heading underline is never mistaken for a one-column table), and every
    pipe-delimited cell must be a dash run with optional colons.
    """
    line = line.strip()
    if "|" not in line:
        return False
    inner = line.strip("|")
    if not inner:
        return False
    cells = [c.strip() for c in inner.split("|")]
    return all(_TABLE_CELL_RE.match(c) for c in cells)


def count_tables(text: str) -> int:
    """Number of GFM pipe tables — one per separator row found outside of
    fenced code blocks."""
    masked = mask_code_blocks(text)
    return sum(1 for line in masked.split("\n") if _is_table_separator_line(line))


def find_tables(text: str) -> list[tuple[int, int]]:
    """Char spans of GFM pipe tables (header row through the last contiguous
    pipe-delimited row), used by Phase 2's ``chunk stats`` to tell whether a
    chunk boundary lands inside a table.

    Row/separator detection runs on ``mask_code_blocks(text)`` so a table-like
    block inside a fenced example is never matched — the same rule as
    ``count_tables``. The returned offsets are read off *unmasked* ``text``,
    not the masked copy: masking blanks fenced lines down to length zero, so
    it preserves line *count* but not line *length*, and an offset computed
    against the shortened copy would drift for anything after an earlier
    fence. A real table can only be found outside a fence, and outside a
    fence the two copies are character-for-character identical — so line
    indices found via the masked copy are safe to resolve against the
    original text's own line offsets.
    """
    masked_lines = mask_code_blocks(text).split("\n")
    orig_lines = text.split("\n")

    is_sep = [_is_table_separator_line(line) for line in masked_lines]
    is_row = [("|" in line and line.strip() != "") for line in masked_lines]

    line_offsets = [0]
    for line in orig_lines[:-1]:
        line_offsets.append(line_offsets[-1] + len(line) + 1)

    spans: list[tuple[int, int]] = []
    covered: set[int] = set()
    for i, is_separator_row in enumerate(is_sep):
        if not is_separator_row or i in covered:
            continue
        top = i
        while top - 1 >= 0 and is_row[top - 1]:
            top -= 1
        bottom = i
        while bottom + 1 < len(masked_lines) and is_row[bottom + 1]:
            bottom += 1
        covered.update(range(top, bottom + 1))
        spans.append((line_offsets[top], line_offsets[bottom] + len(orig_lines[bottom])))
    return spans


__all__ = [
    "ATX_HEADING_RE",
    "CODE_FENCE_RE",
    "count_code_blocks",
    "count_tables",
    "find_headings",
    "find_tables",
    "mask_code_blocks",
]
