"""Experiment matrix configuration, the runner, and reporting. Phase 6.

Re-export surface only, mirroring ``retrievers/__init__.py``'s shape.
"""

from __future__ import annotations

from rag_lab.experiment.config import (
    Cell,
    EmbedderExcludeSpec,
    ExcludeRule,
    ExperimentConfig,
    MatrixComponentSpec,
    MatrixSpec,
    expand_cells,
    load_experiment_config,
)
from rag_lab.experiment.report import (
    compare_runs,
    load_run,
    render_compare_table,
    render_failures,
    render_report_markdown,
    render_report_table,
    worst_failures,
)
from rag_lab.experiment.runner import run_experiment

__all__ = [
    "Cell",
    "EmbedderExcludeSpec",
    "ExcludeRule",
    "ExperimentConfig",
    "MatrixComponentSpec",
    "MatrixSpec",
    "compare_runs",
    "expand_cells",
    "load_experiment_config",
    "load_run",
    "render_compare_table",
    "render_failures",
    "render_report_markdown",
    "render_report_table",
    "run_experiment",
    "worst_failures",
]
