"""Minimal, non-agentic LLM call helper (Phase 7, Step 7.0).

Deliberately outside ``agents/`` -- Phase 8 reserves that package for its
tool-use agent loop; this is a single one-shot prompt -> text call, plus an
on-disk response cache. ``table_summary`` (Phase 7.2) is the first real LLM
integration in this project (Phase 5's real eval-set generator is still a
stub), so this module is where that pattern gets established; Phase 5 is
expected to reuse it later rather than the other way around.

``anthropic`` is imported lazily, inside ``call_llm``'s non-mock branch only,
so importing this module -- and calling ``call_llm(mock=True)`` -- never
requires the ``agents`` extra. That mirrors ``embedders/sentence_transformer.py``'s
lazy ``sentence_transformers`` import: a ``core``-only install stays import-safe,
and only the real (non-mock) path pays the dependency cost.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import structlog

from rag_lab.paths import artifacts_dir

log = structlog.get_logger(__name__)

MOCK_RESPONSE_PREFIX = "[mock-llm response]"


def call_llm(prompt: str, *, model: str, max_tokens: int = 1024, mock: bool = False) -> str:
    """One-shot prompt -> text.

    ``mock=True`` returns a deterministic canned response derived from a hash
    of the prompt, and never imports ``anthropic`` or touches the network --
    what ``--mock-llm`` and CI without ``ANTHROPIC_API_KEY`` both need. The
    mock response is stable across repeated calls with the same prompt, so
    callers relying on ``cached_llm_call``'s hit/miss behaviour still see
    consistent results in mock mode.
    """
    if mock:
        return _mock_response(prompt)

    from anthropic import Anthropic  # lazy: keep `core`/`embed` installs import-safe

    client = Anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(
        block.text for block in response.content if getattr(block, "type", None) == "text"
    )


def _mock_response(prompt: str) -> str:
    from rag_lab.ids import sha1_hex

    return f"{MOCK_RESPONSE_PREFIX} {sha1_hex(prompt)[:12]}"


def llm_cache_dir() -> Path:
    """``artifacts/llm_cache/`` -- not one of ``paths.ARTIFACT_KINDS``: this is
    a cross-cutting response cache, not a phase-output artifact subject to the
    fixture-fallback contract, so it bypasses ``paths.artifact_path`` and sits
    directly under ``artifacts/`` (already gitignored as a whole)."""
    return artifacts_dir() / "llm_cache"


def cached_llm_call(cache_key: str, compute: Callable[[], str]) -> str:
    """Read ``artifacts/llm_cache/<cache_key>.txt`` if present, else call
    ``compute()`` and write the result.

    Callers build ``cache_key`` with ``ids.sha1_hex(...)`` -- the same generic
    hash primitive every other cache key in this codebase uses. Mock calls
    should bypass this helper entirely (call ``call_llm(mock=True)`` directly)
    rather than caching through it: a mock response is free to recompute, and
    caching it under the same key a real call would use risks a stale mock
    response masquerading as real output on a later, non-mock run.
    """
    cache_dir = llm_cache_dir()
    cache_path = cache_dir / f"{cache_key}.txt"
    if cache_path.exists():
        log.debug("llm_cache_hit", cache_key=cache_key)
        return cache_path.read_text(encoding="utf-8")

    result = compute()
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(result, encoding="utf-8")
    log.debug("llm_cache_miss", cache_key=cache_key)
    return result


__all__ = ["MOCK_RESPONSE_PREFIX", "call_llm", "cached_llm_call", "llm_cache_dir"]
