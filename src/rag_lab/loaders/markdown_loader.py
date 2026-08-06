"""MarkdownLoader — reads markdown as-is, title from the first H1 (or the
filename when there isn't one), records structural stats in ``meta``."""

from __future__ import annotations

from pathlib import Path

from rag_lab.ids import make_doc_id
from rag_lab.loaders.base import humanize, relative_source_path
from rag_lab.markup import find_headings
from rag_lab.normalize import normalize_text
from rag_lab.schemas import Document


class MarkdownLoader:
    extensions = {".md", ".markdown"}

    def load(self, path: Path, corpus: str) -> Document:
        text = normalize_text(path.read_text(encoding="utf-8"))
        rel = relative_source_path(path)
        headings = find_headings(text)
        title = next((h_text for level, h_text in headings if level == 1), None)
        return Document(
            doc_id=make_doc_id(corpus, rel),
            corpus=corpus,
            source_path=rel,
            title=title or humanize(path.stem),
            text=text,
            content_type="markdown",
            meta={"char_count": len(text), "section_count": len(headings)},
        )


__all__ = ["MarkdownLoader"]
