"""``agent_router`` -- wraps ``agents.router.route_query`` as a ``Retriever``
(plan §Phase 8, Step 8.1).

Unlike every other retriever in this package, it does not search the index
it's constructed with. ``manifest``/``store`` exist only because
``build_retriever``'s signature requires them, matching every other
retriever's construction -- this class reads ``manifest.corpus`` from them
and otherwise ignores both. Instead it surveys every *other* index for that
corpus at query time via ``route_query``, excluding its own ``manifest.index_id``
so the matrix's placeholder chunker/embedder pairing (Step 6.3's "some
chunker/embedder pair for cell-id bookkeeping") never counts as a real
baseline option to route across.
"""

from __future__ import annotations

from rag_lab.agents.router import DEFAULT_MAX_STEPS, DEFAULT_MODEL, route_query
from rag_lab.schemas import IndexManifest, ScoredChunk
from rag_lab.stores.base import VectorStore

NAME = "agent_router"


class AgentRouterRetriever:
    name = NAME

    def __init__(
        self,
        manifest: IndexManifest,
        store: VectorStore,
        *,
        model: str = DEFAULT_MODEL,
        max_steps: int = DEFAULT_MAX_STEPS,
        mock: bool = False,
    ) -> None:
        self.manifest = manifest  # read for .corpus only -- see module docstring
        self.store = store  # unused -- see module docstring
        self.model = model
        self.max_steps = max_steps
        self.mock = mock

    def retrieve(self, query: str, k: int) -> list[ScoredChunk]:
        _decision, results = route_query(
            query,
            self.manifest.corpus,
            k,
            model=self.model,
            max_steps=self.max_steps,
            mock=self.mock,
            exclude_index_id=self.manifest.index_id,
        )
        return results


__all__ = ["NAME", "AgentRouterRetriever"]
