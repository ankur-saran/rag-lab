"""Vector stores behind a common protocol (Phase 3).

Nothing imported here touches ``chromadb`` until ``ChromaStore`` is actually
instantiated, so importing this package stays safe under a ``core``-only
install.
"""

from __future__ import annotations

from rag_lab.stores.base import VectorStore
from rag_lab.stores.chroma_store import ChromaStore

__all__ = ["ChromaStore", "VectorStore"]
