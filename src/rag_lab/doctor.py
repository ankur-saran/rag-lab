"""Health check.

Nothing here does retrieval. The entire purpose is to let every later phase start
from a known-good state, and to surface the two things that reliably derail a
live demo: a missing model download, and a missing API key.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from rich.console import Console
from rich.markup import escape
from rich.table import Table

from rag_lab import __version__
from rag_lab.paths import artifacts_dir, fixture_path, repo_root

Status = Literal["ok", "warn", "fail"]

MIN_PYTHON = (3, 10)
MIN_FREE_GB = 2.0

# Approximate on-disk size of each model's weights, for the download warning.
# Keys match embedders.registry.NAMED_EMBEDDERS' model_name values.
MODEL_SIZES_MB = {
    "BAAI/bge-small-en-v1.5": 133,
    "sentence-transformers/all-MiniLM-L6-v2": 91,
    "intfloat/e5-base-v2": 438,
    "intfloat/multilingual-e5-base": 1110,
}
DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"

DEPENDENCY_GROUPS: dict[str, list[str]] = {
    "core": ["pydantic", "typer", "yaml", "structlog", "numpy", "pandas", "rich", "tiktoken"],
    "embed": ["sentence_transformers", "chromadb", "rank_bm25"],
    "agents": ["anthropic"],
    "app": ["streamlit", "plotly"],
    "dev": ["pytest", "ruff"],
}

# Groups whose absence is fatal rather than merely informational.
REQUIRED_GROUPS = {"core"}


@dataclass
class Check:
    name: str
    status: Status
    detail: str

    @property
    def needed_for(self) -> str:
        return ""


def _module_present(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _hf_cache_dir() -> Path:
    env = os.environ.get("HF_HOME")
    if env:
        return Path(env) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def _model_cached(model_name: str) -> bool:
    folder = "models--" + model_name.replace("/", "--")
    return (_hf_cache_dir() / folder).is_dir()


def run_checks() -> list[Check]:
    checks: list[Check] = []

    # Python version
    v = sys.version_info
    checks.append(
        Check(
            "python",
            "ok" if v[:2] >= MIN_PYTHON else "fail",
            f"{v.major}.{v.minor}.{v.micro} (need >= {MIN_PYTHON[0]}.{MIN_PYTHON[1]})",
        )
    )

    # Dependency groups
    for group, modules in DEPENDENCY_GROUPS.items():
        missing = [m for m in modules if not _module_present(m)]
        if not missing:
            status: Status = "ok"
            detail = f"{len(modules)} module(s) present"
        elif group in REQUIRED_GROUPS:
            status = "fail"
            detail = f"missing: {', '.join(missing)} — run: pip install -e '.[core]'"
        else:
            status = "warn"
            detail = f"absent ({', '.join(missing)}) — install with: pip install -e '.[{group}]'"
        checks.append(Check(f"deps:{group}", status, detail))

    # Repo layout
    root = repo_root()
    checks.append(
        Check(
            "repo_root",
            "ok" if (root / "pyproject.toml").exists() else "warn",
            str(root),
        )
    )

    # Writable artifacts dir. The check is "can I write artifacts", not "can I
    # delete them" — some mounts (Windows shares, FUSE) permit create and write
    # but deny unlink, and that is not a blocking condition for any phase.
    try:
        artifacts_dir().mkdir(parents=True, exist_ok=True)
        probe = artifacts_dir() / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        try:
            probe.unlink()
            detail = str(artifacts_dir())
        except OSError:
            detail = f"{artifacts_dir()} (writable; unlink denied by the filesystem)"
        checks.append(Check("artifacts_writable", "ok", detail))
    except OSError as exc:
        checks.append(Check("artifacts_writable", "fail", f"{artifacts_dir()}: {exc}"))

    # Disk space
    try:
        free_gb = shutil.disk_usage(root).free / 1024**3
        checks.append(
            Check(
                "disk_space",
                "ok" if free_gb >= MIN_FREE_GB else "warn",
                f"{free_gb:.1f} GB free (need >= {MIN_FREE_GB:.0f} GB for models + indexes)",
            )
        )
    except OSError as exc:
        checks.append(Check("disk_space", "warn", f"could not determine: {exc}"))

    # Fixtures
    missing_fixtures = [
        kind
        for kind in ("documents", "chunks", "evalset", "indexes")
        if not fixture_path(kind).exists()
    ]
    checks.append(
        Check(
            "fixtures",
            "ok" if not missing_fixtures else "warn",
            "present"
            if not missing_fixtures
            else f"missing {', '.join(missing_fixtures)} — run: make fixtures",
        )
    )

    # API key — a warning, never a failure. Phases 1-4 do not need it.
    has_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
    checks.append(
        Check(
            "anthropic_api_key",
            "ok" if has_key else "warn",
            "set" if has_key else "not set — needed for Phases 5 and 8 (or use --mock-llm)",
        )
    )

    # Model cache. Discovering a 1.1 GB download five minutes into a demo is
    # avoidable, so surface it here.
    if _module_present("sentence_transformers"):
        if _model_cached(DEFAULT_MODEL):
            checks.append(Check("model_cache", "ok", f"{DEFAULT_MODEL} cached"))
        else:
            size = MODEL_SIZES_MB.get(DEFAULT_MODEL, 0)
            checks.append(
                Check(
                    "model_cache",
                    "warn",
                    f"{DEFAULT_MODEL} not cached — first index build downloads ~{size} MB",
                )
            )
    else:
        checks.append(Check("model_cache", "warn", "skipped (embed extras not installed)"))

    return checks


def render(checks: list[Check], console: Console | None = None) -> None:
    console = console or Console()
    table = Table(title=f"rag-lab doctor  (v{__version__})", title_style="bold")
    table.add_column("check", style="cyan", no_wrap=True)
    table.add_column("status", no_wrap=True)
    table.add_column("detail", overflow="fold")

    glyph = {"ok": "[green]ok[/green]", "warn": "[yellow]warn[/yellow]", "fail": "[red]FAIL[/red]"}
    for check in checks:
        # escape() matters: details contain extras like ".[embed]", which rich
        # would otherwise swallow as a markup tag.
        table.add_row(check.name, glyph[check.status], escape(check.detail))

    console.print(table)

    failures = [c for c in checks if c.status == "fail"]
    warnings = [c for c in checks if c.status == "warn"]
    if failures:
        console.print(f"[red]{len(failures)} blocking issue(s).[/red] Fix before running phases.")
    else:
        console.print(
            f"[green]Environment OK.[/green] {len(warnings)} warning(s) — "
            "optional extras and credentials are not required for Phases 1-4."
        )


def doctor(console: Console | None = None) -> int:
    """Returns a process exit code: 0 when no blocking issue, 1 otherwise."""
    checks = run_checks()
    render(checks, console)
    return 1 if any(c.status == "fail" for c in checks) else 0


__all__ = ["Check", "doctor", "render", "run_checks"]
