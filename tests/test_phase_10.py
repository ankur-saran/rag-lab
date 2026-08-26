"""Phase 10 acceptance tests (Packaging and the demo path).

Tiered like every other phase's test module:

- **Always collected** (pure Python, no ``embed`` extras): ``demo.yaml``
  expands to the two ``api_docs`` cells Step 10.1 describes, and
  ``scripts/print_demo_winner.py``'s ``pick_winner`` picks correctly and
  degrades to ``None`` rather than raising when nothing reports the metric.
- ``@requires_chromadb``: the demo matrix runs end to end through the same
  ``fast_embedder``/``isolated_artifacts`` pattern ``test_phase_6.py`` uses
  (dependency-free ``HashEmbedder``, isolated artifact root), then
  ``print_demo_winner.py`` is invoked as a real subprocess against that run,
  pointed at the isolated root via ``RAG_LAB_ROOT`` -- the same env var
  ``paths.repo_root()`` already reads, so no new plumbing is needed for a
  subprocess (which can't be monkeypatched in-process) to see the isolated
  artifacts.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DEMO_CONFIG_PATH = REPO_ROOT / "config" / "experiments" / "demo.yaml"

try:
    import chromadb as _chromadb  # noqa: F401

    HAS_CHROMADB = True
except ImportError:
    HAS_CHROMADB = False

requires_chromadb = pytest.mark.skipif(
    not HAS_CHROMADB, reason="chromadb not installed (embed extra)"
)


def _load_print_demo_winner():
    spec = importlib.util.spec_from_file_location(
        "print_demo_winner", REPO_ROOT / "scripts" / "print_demo_winner.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------- #
# Tier 1 -- pure Python, always collected
# --------------------------------------------------------------------------- #


def test_demo_config_expands_to_two_api_docs_cells():
    from rag_lab.experiment.config import expand_cells, load_experiment_config

    config = load_experiment_config(DEMO_CONFIG_PATH)
    assert config.corpora == ["api_docs"]

    cells = expand_cells(config)
    assert len(cells) == 2
    assert all(c.corpus == "api_docs" for c in cells)
    assert sorted(c.chunker for c in cells) == ["fixed", "markdown"]
    assert {c.embedder for c in cells} == {"bge-small"}
    assert {c.retriever for c in cells} == {"dense"}


def test_pick_winner_selects_higher_recall_cell():
    from rag_lab.schemas import RunResult

    module = _load_print_demo_winner()

    low = RunResult(
        run_id="demo__test",
        config={"chunker": "fixed", "embedder": "bge-small", "retriever": "dense"},
        corpus="api_docs",
        metrics={"recall@5": 0.4},
    )
    high = RunResult(
        run_id="demo__test",
        config={"chunker": "markdown", "embedder": "bge-small", "retriever": "dense"},
        corpus="api_docs",
        metrics={"recall@5": 0.8},
    )

    winner = module.pick_winner([low, high], "recall@5")
    assert winner is not None
    result, value = winner
    assert result.config["chunker"] == "markdown"
    assert value == pytest.approx(0.8)


def test_pick_winner_returns_none_when_metric_absent():
    from rag_lab.schemas import RunResult

    module = _load_print_demo_winner()
    result = RunResult(
        run_id="demo__test", config={}, corpus="api_docs", metrics={"mrr": 0.5}
    )
    assert module.pick_winner([result], "recall@5") is None


# --------------------------------------------------------------------------- #
# Tier 2 -- needs chromadb
# --------------------------------------------------------------------------- #


@pytest.fixture
def fast_embedder(monkeypatch):
    """Same fixture as ``test_phase_6.py``: swap in the dependency-free
    ``HashEmbedder`` so the demo matrix runs in milliseconds, no model
    download."""
    from rag_lab.embedders.fixture import HashEmbedder

    def _build(name, overrides=None):
        return HashEmbedder(overrides)

    monkeypatch.setattr("rag_lab.indexing.build_embedder", _build)


@pytest.fixture
def isolated_artifacts(monkeypatch, tmp_path):
    """Points every module's ``artifacts_dir()`` at an empty ``<tmp_path>/
    artifacts`` directory -- the ``/artifacts`` suffix (unlike
    ``test_phase_6.py``'s own fixture, which patches ``artifacts_dir()`` to
    return ``tmp_path`` directly) is what lets a real subprocess, which can't
    see these in-process monkeypatches, land on the exact same directory by
    being given ``RAG_LAB_ROOT=str(tmp_path)`` alone: its unpatched
    ``artifacts_dir()`` computes ``repo_root() / "artifacts"`` the normal way.
    Deliberately does not set ``RAG_LAB_ROOT`` here, only ``artifacts_dir`` --
    the former also relocates ``fixtures_dir()``, which would make the
    fixture-fallback this run depends on unresolvable."""
    artifacts = tmp_path / "artifacts"
    for target in (
        "rag_lab.paths.artifacts_dir",
        "rag_lab.corpus.artifacts_dir",
        "rag_lab.chunks.artifacts_dir",
        "rag_lab.indexing.artifacts_dir",
        "rag_lab.experiment.runner.artifacts_dir",
    ):
        monkeypatch.setattr(target, lambda p=artifacts: p)
    return tmp_path


@requires_chromadb
def test_demo_matrix_runs_end_to_end(fast_embedder, isolated_artifacts):
    from rag_lab.experiment.config import expand_cells, load_experiment_config
    from rag_lab.experiment.runner import run_experiment

    config = load_experiment_config(DEMO_CONFIG_PATH)
    run_id, results = run_experiment(config, workers=2)
    assert len(results) == len(expand_cells(config))
    assert any("recall@5" in r.metrics for r in results)

    module = _load_print_demo_winner()
    winner = module.pick_winner(results, "recall@5")
    assert winner is not None


@requires_chromadb
def test_print_demo_winner_cli_names_the_higher_recall_cell(fast_embedder, isolated_artifacts):
    from rag_lab.experiment.config import load_experiment_config
    from rag_lab.experiment.runner import run_experiment

    config = load_experiment_config(DEMO_CONFIG_PATH)
    run_id, results = run_experiment(config, workers=2)
    by_chunker = {r.config["chunker"]: r for r in results}
    expected_winner = max(by_chunker.values(), key=lambda r: r.metrics.get("recall@5", -1.0))

    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "print_demo_winner.py"), "--run-id", run_id],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env={**os.environ, "RAG_LAB_ROOT": str(isolated_artifacts)},
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert f"winner  api_docs/{expected_winner.config['chunker']}/" in proc.stdout


def test_print_demo_winner_cli_unknown_run_id_errors():
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "print_demo_winner.py"), "--run-id", "no-such-run"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 1
    assert "error" in (proc.stdout + proc.stderr).lower()
