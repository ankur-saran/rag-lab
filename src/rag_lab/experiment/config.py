"""Experiment matrix configuration (plan Phase 6, Step 6.3).

This is a different thing from ``config.Config`` (global seed/paths/defaults):
it's the richer, separate structure describing a cartesian product of
chunker x embedder x retriever cells per corpus, with an ``exclude`` list that
prunes redundant combinations before anything gets built.

``expand_cells`` is pure (no I/O) -- every id a ``Cell`` carries
(``chunk_set_id``, ``index_id``, ``cell_id``) is a deterministic function of
its resolved config, using the exact same ``ids.py``/``resolve_params``
machinery every earlier phase already relies on. The runner uses those ids to
dedupe build work and to decide what a resume can skip.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from rag_lab import chunkers, embedders, retrievers
from rag_lab.ids import make_cell_id, make_chunk_set_id, make_index_id

# Top-level YAML keys that annotate rather than configure -- stripped before
# model construction, the same spirit as config.py's GLOBAL_NOHASH_KEYS
# (smoke.yaml uses `_nohash: [description]` + `description:` exactly this way).
_ANNOTATION_KEYS = frozenset({"_nohash", "description", "comment", "note"})


class MatrixComponentSpec(BaseModel):
    """One named component with its (unresolved) override params, as written
    in the matrix YAML -- e.g. ``{name: bge-small, params: {truncate_dim: 256}}``.

    Named to avoid colliding with ``retrievers.base.RetrieverSpec``, which is
    a different thing (a retriever's own fixed identity, not a matrix entry).

    ``corpora`` is an optional allow-list (``None`` = unrestricted, matching
    every corpus in the experiment's own ``corpora:`` list) -- what Phase 7's
    ``semantic``/``table_summary`` entries need to stay scoped to the corpora
    they were validated on (a 3-5x-cost or LLM-calling chunker silently
    running against every corpus is a real cost risk, not just noise). A
    first-class allow-list here, checked before cell expansion, rather than
    the only mechanism this class had before -- ``ExcludeRule.corpus``, a
    denylist that would need one line per *disallowed* corpus.
    """

    model_config = ConfigDict(extra="forbid")
    name: str
    params: dict[str, Any] = Field(default_factory=dict)
    corpora: list[str] | None = None


class MatrixSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    chunker: list[MatrixComponentSpec]
    embedder: list[MatrixComponentSpec]
    retriever: list[MatrixComponentSpec]


class EmbedderExcludeSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    params: dict[str, Any] | None = None  # None = match this name at any params


class ExcludeRule(BaseModel):
    """Prunes cells before they're built (Step 6.3: "BM25 ignores the
    embedder, so don't run it three times"). Every field given must match; an
    omitted field matches anything. Matching is against the *raw*
    ``MatrixComponentSpec`` entries the cartesian product was built from (name
    + as-written params), not resolved params -- a rule references a specific
    matrix entry, e.g. "bge-small at truncate_dim=256" vs. "bge-small at
    defaults."
    """

    model_config = ConfigDict(extra="forbid")
    corpus: str | None = None
    chunker: str | None = None
    embedder: EmbedderExcludeSpec | None = None
    retriever: str | None = None


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    corpora: list[str]
    seed: int = 42
    k: int = 10
    matrix: MatrixSpec
    exclude: list[ExcludeRule] = Field(default_factory=list)
    # `None` (default) = every split, exactly today's behavior for every
    # config written before Phase 8 -- `full_matrix.yaml`/`smoke.yaml` never
    # set this. Phase 8's optimizer is the one caller that sets it: "dev"
    # while iterating, "test" once for the final report (plan §Phase 8, Step
    # 8.0.3), threaded through `Cell` into `_evaluate_cell`'s
    # `load_evalset(cell.corpus, split=cell.eval_split)` call below.
    eval_split: Literal["train", "dev", "test"] | None = None


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"experiment config not found: {p}")
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    data = {k: v for k, v in data.items() if k not in _ANNOTATION_KEYS}
    return ExperimentConfig(**data)


@dataclass(frozen=True)
class Cell:
    """One fully-resolved cell of the expanded matrix: everything the build
    stage and the evaluate stage need, with every id already computed."""

    corpus: str
    chunker: str
    chunker_params: dict[str, Any]
    embedder: str
    embedder_params: dict[str, Any]
    retriever: str
    retriever_params: dict[str, Any]
    k: int
    chunk_set_id: str
    index_id: str
    cell_id: str
    eval_split: Literal["train", "dev", "test"] | None = None


def _rule_matches(
    rule: ExcludeRule,
    corpus: str,
    chunker: MatrixComponentSpec,
    embedder: MatrixComponentSpec,
    retriever: MatrixComponentSpec,
) -> bool:
    if rule.corpus is not None and rule.corpus != corpus:
        return False
    if rule.chunker is not None and rule.chunker != chunker.name:
        return False
    if rule.retriever is not None and rule.retriever != retriever.name:
        return False
    if rule.embedder is not None:
        if rule.embedder.name != embedder.name:
            return False
        if rule.embedder.params is not None and rule.embedder.params != embedder.params:
            return False
    return True


def _corpora_allowed(component: MatrixComponentSpec, corpus: str) -> bool:
    """``component.corpora is None`` means unrestricted -- matches every
    corpus in the experiment's own ``corpora:`` list, preserving today's
    behaviour for every matrix entry that doesn't set it."""
    return component.corpora is None or corpus in component.corpora


def expand_cells(config: ExperimentConfig) -> list[Cell]:
    """Cartesian product of ``corpora x matrix.chunker x matrix.embedder x
    matrix.retriever``, with each component's own ``corpora`` allow-list
    applied first and ``exclude`` rules applied after. On the documented
    ``full_matrix.yaml`` example this cuts a naive 4x3x3 by roughly a third.

    Deterministic and I/O-free, so the same config always expands to the same
    list of ``Cell``s in the same order -- what makes ``--config-idx``
    (``experiment failures``) and matrix-content-addressed resume both work.
    """
    cells: list[Cell] = []
    for corpus in config.corpora:
        for chunker in config.matrix.chunker:
            if not _corpora_allowed(chunker, corpus):
                continue
            for embedder in config.matrix.embedder:
                if not _corpora_allowed(embedder, corpus):
                    continue
                for retriever in config.matrix.retriever:
                    if not _corpora_allowed(retriever, corpus):
                        continue
                    if any(
                        _rule_matches(rule, corpus, chunker, embedder, retriever)
                        for rule in config.exclude
                    ):
                        continue

                    chunker_params = chunkers.resolve_params(chunker.name, chunker.params)
                    embedder_params = embedders.resolve_params(embedder.name, embedder.params)
                    retriever_params = retrievers.resolve_params(retriever.name, retriever.params)

                    chunk_set_id = make_chunk_set_id(corpus, chunker.name, chunker_params)
                    index_id = make_index_id(chunk_set_id, embedder.name, embedder_params)
                    cell_id = make_cell_id(
                        corpus,
                        chunk_set_id,
                        index_id,
                        retriever.name,
                        retriever_params,
                        config.k,
                    )

                    cells.append(
                        Cell(
                            corpus=corpus,
                            chunker=chunker.name,
                            chunker_params=chunker_params,
                            embedder=embedder.name,
                            embedder_params=embedder_params,
                            retriever=retriever.name,
                            retriever_params=retriever_params,
                            k=config.k,
                            chunk_set_id=chunk_set_id,
                            index_id=index_id,
                            cell_id=cell_id,
                            eval_split=config.eval_split,
                        )
                    )
    return cells


__all__ = [
    "Cell",
    "EmbedderExcludeSpec",
    "ExcludeRule",
    "ExperimentConfig",
    "MatrixComponentSpec",
    "MatrixSpec",
    "expand_cells",
    "load_experiment_config",
]
