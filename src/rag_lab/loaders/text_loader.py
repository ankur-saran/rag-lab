"""TextLoader — plain text passthrough with whitespace normalization. Plain
text has no heading concept, so the title is always the humanized filename."""

from __future__ import annotations

from pathlib import Path

from rag_lab.ids import make_doc_id
from rag_lab.loaders.base import humanize, relative_source_path
from rag_lab.normalize import normalize_text
from rag_lab.schemas import Document


class TextLoader:
    extensions = {".txt"}

    def load(self, path: Path, corpus: str) -> Document:
        text = normalize_text(path.read_text(encoding="utf-8"))
        rel = relative_source_path(path)
        return Document(
            doc_id=make_doc_id(corpus, rel),
            corpus=corpus,
            source_path=rel,
            title=humanize(path.stem),
            text=text,
            content_type="text",
            meta={"char_count": len(text)},
        )


__all__ = ["TextLoader"]
