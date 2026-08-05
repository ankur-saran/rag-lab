"""Configuration loading and parameter hashing.

``params_hash`` is the entire caching strategy. Because every artifact ID embeds
the hash of the parameters that produced it, a changed parameter yields a new ID
and therefore a cache miss, while unchanged parameters yield a hit. No cache
invalidation logic is needed anywhere in the system.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

# Keys ignored when hashing, so you can annotate a config without invalidating
# every cached artifact. A dict may also carry its own "_nohash": [...] list.
GLOBAL_NOHASH_KEYS = frozenset({"_nohash", "description", "comment", "note", "name"})

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "default.yaml"


class PathsConfig(BaseModel):
    artifacts: str = "artifacts"
    fixtures: str = "fixtures"
    corpora: str = "corpora"
    results: str = "artifacts/results"


class DefaultsConfig(BaseModel):
    chunker: str = "recursive"
    embedder: str = "bge-small"
    retriever: str = "dense"
    k: int = 10


class Config(BaseModel):
    seed: int = 42
    paths: PathsConfig = Field(default_factory=PathsConfig)
    defaults: DefaultsConfig = Field(default_factory=DefaultsConfig)
    log_level: str = "info"
    json_logs: bool = False
    extra: dict[str, Any] = Field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Hashing
# --------------------------------------------------------------------------- #


def _strip_nohash(obj: Any) -> Any:
    """Recursively drop non-semantic keys so annotations never affect the hash."""
    if isinstance(obj, dict):
        local_nohash = set(obj.get("_nohash", []) or [])
        drop = GLOBAL_NOHASH_KEYS | local_nohash
        return {k: _strip_nohash(v) for k, v in obj.items() if k not in drop}
    if isinstance(obj, (list, tuple)):
        return [_strip_nohash(v) for v in obj]
    return obj


def canonical_json(obj: Any) -> str:
    """Deterministic JSON: sorted keys, no insignificant whitespace.

    Key order must not affect the hash, or logically identical configs written by
    two different people would miss each other's caches.
    """
    return json.dumps(
        _strip_nohash(obj),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )


def params_hash(params: Any, length: int = 8) -> str:
    """Stable short hash of a parameter dict."""
    digest = hashlib.sha1(canonical_json(params).encode("utf-8")).hexdigest()
    return digest[:length]


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _coerce(value: str) -> Any:
    """Parse a CLI ``key=value`` string into a typed scalar."""
    lowered = value.strip().lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none", ""}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def parse_overrides(pairs: list[str] | None) -> dict[str, Any]:
    """Turn ``["a.b=1", "c=true"]`` into a nested dict."""
    result: dict[str, Any] = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise ValueError(f"override must be key=value, got: {pair!r}")
        key, _, raw = pair.partition("=")
        cursor = result
        parts = key.strip().split(".")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = _coerce(raw)
    return result


def load_config(
    path: Path | str | None = None,
    overrides: dict[str, Any] | None = None,
) -> Config:
    """Resolution order: CLI overrides > given file > config/default.yaml > code defaults."""
    data: dict[str, Any] = {}

    if DEFAULT_CONFIG_PATH.exists():
        data = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8")) or {}

    if path is not None:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"config file not found: {p}")
        data = deep_merge(data, yaml.safe_load(p.read_text(encoding="utf-8")) or {})

    if overrides:
        data = deep_merge(data, overrides)

    known = set(Config.model_fields)
    extra = {k: v for k, v in data.items() if k not in known}
    known_data = {k: v for k, v in data.items() if k in known}
    known_data.setdefault("extra", {})
    known_data["extra"] = deep_merge(known_data["extra"], extra)
    return Config(**known_data)


__all__ = [
    "Config",
    "DefaultsConfig",
    "PathsConfig",
    "REPO_ROOT",
    "canonical_json",
    "deep_merge",
    "load_config",
    "params_hash",
    "parse_overrides",
]
