"""Artifact path resolution with fixture fallback.

``resolve_artifact`` is what implements the independence contract. Every phase
resolves its inputs through it, so any phase runs on a clean checkout with no
prior phase ever having executed — it falls back to the committed fixture.

The warning on fallback is not decoration. Silently benchmarking a 12-record
fixture would produce plausible-looking, meaningless numbers, and that failure is
very hard to notice after the fact.
"""

from __future__ import annotations

import os
from pathlib import Path

import structlog

log = structlog.get_logger(__name__)


def repo_root() -> Path:
    """Repo root, overridable for installed (non-editable) deployments."""
    env = os.environ.get("RAG_LAB_ROOT")
    if env:
        return Path(env).resolve()
    return Path(__file__).resolve().parents[2]


ARTIFACT_KINDS = ("documents", "chunks", "indexes", "evalset", "results")

# Committed fixture standing in for each artifact kind.
FIXTURE_FALLBACK = {
    "documents": "documents/sample.jsonl",
    "chunks": "chunks/sample.jsonl",
    "evalset": "evalset/sample.jsonl",
    "results": "results/sample_run",
}


class ArtifactNotFoundError(FileNotFoundError):
    pass


def artifacts_dir() -> Path:
    return repo_root() / "artifacts"


def fixtures_dir() -> Path:
    return repo_root() / "fixtures"


def corpora_dir() -> Path:
    return repo_root() / "corpora"


def artifact_path(kind: str, name: str) -> Path:
    """Path an artifact *would* occupy. Does not check existence."""
    if kind not in ARTIFACT_KINDS:
        raise ValueError(f"unknown artifact kind {kind!r}; expected one of {ARTIFACT_KINDS}")
    return artifacts_dir() / kind / name


def fixture_path(kind: str) -> Path:
    rel = FIXTURE_FALLBACK.get(kind)
    if rel is None:
        raise ValueError(f"no fixture registered for artifact kind {kind!r}")
    return fixtures_dir() / rel


def resolve_artifact(
    kind: str,
    artifact_id: str | None = None,
    allow_fixture: bool = True,
) -> Path:
    """Return the real artifact if it exists, else the committed fixture.

    Args:
        kind: one of ARTIFACT_KINDS.
        artifact_id: filename or directory name under ``artifacts/<kind>/``.
            ``.jsonl`` is appended for file-based kinds when absent.
        allow_fixture: set False in production paths where a silent fixture
            substitution would be a correctness bug.

    Raises:
        ArtifactNotFoundError: nothing found and fallback disabled or unavailable.
    """
    if artifact_id:
        name = artifact_id
        if kind != "results" and not name.endswith(".jsonl"):
            name = f"{name}.jsonl"
        candidate = artifact_path(kind, name)
        if candidate.exists():
            return candidate
        if not allow_fixture:
            raise ArtifactNotFoundError(f"{kind} artifact not found: {candidate}")
        log.warning(
            "artifact_missing_using_fixture",
            kind=kind,
            requested=artifact_id,
            expected_at=str(candidate),
            hint="Run the producing phase, or pass --no-fixture to fail instead.",
        )

    if not allow_fixture:
        raise ArtifactNotFoundError(f"no {kind} artifact id given and fixtures disabled")

    fallback = fixture_path(kind)
    if not fallback.exists():
        raise ArtifactNotFoundError(
            f"neither artifact nor fixture available for {kind!r} "
            f"(expected fixture at {fallback}); run `make fixtures`"
        )
    log.warning(
        "using_fixture",
        kind=kind,
        path=str(fallback),
        hint="This is sample data. Results computed from it are not meaningful.",
    )
    return fallback


def ensure_dirs() -> None:
    for kind in ARTIFACT_KINDS:
        (artifacts_dir() / kind).mkdir(parents=True, exist_ok=True)


def is_fixture(path: Path) -> bool:
    try:
        Path(path).resolve().relative_to(fixtures_dir().resolve())
        return True
    except ValueError:
        return False


__all__ = [
    "ARTIFACT_KINDS",
    "ArtifactNotFoundError",
    "artifact_path",
    "artifacts_dir",
    "corpora_dir",
    "ensure_dirs",
    "fixture_path",
    "fixtures_dir",
    "is_fixture",
    "repo_root",
    "resolve_artifact",
]
