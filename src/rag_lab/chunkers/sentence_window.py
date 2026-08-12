"""`sentence_window` — the inverse of the markdown chunker's heading-path
split: embed narrow, return wide. Each sentence gets its own chunk whose
``text`` widens to include ``window_size`` neighbouring sentences on each
side (for the consumer), while ``embed_text`` stays the bare sentence (for
the embedder).

``sentence_spans`` is exposed for reuse by ``chunks.py``'s split-sentence-rate
stat, which needs the same sentence boundaries independent of which chunker
produced a given chunk set.
"""

from __future__ import annotations

import re
from typing import Any

from rag_lab.chunkers.base import ChunkSpec, finalize_chunks
from rag_lab.schemas import Chunk, Document

NAME = "sentence_window"
DEFAULT_PARAMS: dict[str, Any] = {
    "window_size": 3,
    "stride": 1,
    "sentence_splitter": "regex",
    "encoding": "cl100k_base",
}

# A sentence boundary is `.`/`!`/`?` (optionally followed by a closing quote
# or paren), then whitespace, then something that looks like the start of a
# new sentence or markdown block: a capital letter, digit, opening
# quote/paren, or one of the markdown block markers (`#` heading, `` ` ``
# fence, `-`/`*` list item, `>` blockquote, `|` table row). Requiring that
# lookahead is what keeps "3.5 inches" and "e.g. widgets" from splitting on a
# bare heuristic of "any period followed by space" -- and including the
# block markers is what keeps a paragraph that ends right before a heading or
# a fenced code block from being treated as one giant unclosed "sentence".
_BOUNDARY_RE = re.compile(r"[.!?]+[\"'’”)]*\s+(?=[A-Z0-9\"'‘“(#`\-*>|])")
# Matched against the text *before* the punctuation (`m.start()` is the
# punctuation's own position, so it's already excluded) -- no literal period
# in this pattern, since the boundary regex already established one is there.
_TRAILING_WORD_RE = re.compile(r"(\w+)$")

# Case-insensitive; checked against the single word immediately before the
# terminal period. A single-letter word is also guarded (initials, "J. Smith").
_ABBREVIATIONS = {
    "mr",
    "mrs",
    "ms",
    "dr",
    "prof",
    "sr",
    "jr",
    "st",
    "vs",
    "etc",
    "eg",
    "ie",
    "inc",
    "ltd",
    "co",
    "corp",
    "no",
    "fig",
    "vol",
    "approx",
    "gen",
    "rep",
    "sen",
    "gov",
    "dept",
    "univ",
    "assoc",
    "bros",
}


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """Contiguous ``(start, end)`` sentence spans covering the whole of
    ``text`` with no gaps -- each span's trailing whitespace is attached to
    the sentence that precedes it, the same convention `recursive.py` uses,
    so any window built by slicing between two span endpoints is an exact
    slice of the source text.
    """
    spans: list[tuple[int, int]] = []
    pos = 0
    for m in _BOUNDARY_RE.finditer(text):
        preceding = text[pos : m.start()]
        word_match = _TRAILING_WORD_RE.search(preceding)
        if word_match:
            word = word_match.group(1).lower()
            if word in _ABBREVIATIONS or len(word) == 1:
                continue  # false boundary (abbreviation or initial) -- keep going
        spans.append((pos, m.end()))
        pos = m.end()
    if pos < len(text):
        spans.append((pos, len(text)))
    return spans


class SentenceWindowChunker:
    name = NAME
    default_params = DEFAULT_PARAMS

    def chunk(self, doc: Document, params: dict[str, Any]) -> list[Chunk]:
        resolved = {**DEFAULT_PARAMS, **params}
        window_size = int(resolved["window_size"])
        stride = max(1, int(resolved["stride"]))
        encoding = str(resolved["encoding"])
        if resolved["sentence_splitter"] != "regex":
            raise ValueError(
                f"unsupported sentence_splitter {resolved['sentence_splitter']!r}; "
                "only 'regex' is implemented"
            )

        sentences = sentence_spans(doc.text)
        n = len(sentences)

        specs: list[ChunkSpec] = []
        for i in range(0, n, stride):
            lo = max(0, i - window_size)
            hi = min(n - 1, i + window_size)
            char_start, char_end = sentences[lo][0], sentences[hi][1]
            sent_start, sent_end = sentences[i]
            specs.append(
                ChunkSpec(
                    char_start=char_start,
                    char_end=char_end,
                    text=doc.text[char_start:char_end],
                    embed_text=doc.text[sent_start:sent_end],
                    meta={"center_sentence_index": i},
                )
            )

        return finalize_chunks(specs, doc, NAME, resolved, encoding=encoding)


__all__ = ["DEFAULT_PARAMS", "NAME", "SentenceWindowChunker", "sentence_spans"]
