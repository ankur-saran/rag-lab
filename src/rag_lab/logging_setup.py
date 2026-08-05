"""structlog configuration.

Logs go to stderr, always. stdout is reserved for machine-consumable output
(tables, JSON, artifact paths) so `rag-lab ... | jq` stays usable.
"""

from __future__ import annotations

import logging
import sys

import structlog

_LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
}


def _stderr_logger_factory(*args: object) -> structlog.PrintLogger:
    """Resolve ``sys.stderr`` at call time, not at configuration time.

    Binding the stream eagerly means any harness that swaps stderr — pytest's
    capsys, Typer's CliRunner — leaves the cached logger holding a closed file,
    and the next log call raises instead of logging.
    """
    return structlog.PrintLogger(sys.stderr)


def configure_logging(level: str = "info", json_logs: bool = False) -> None:
    log_level = _LEVELS.get(level.lower(), logging.INFO)

    logging.basicConfig(format="%(message)s", stream=sys.stderr, level=log_level, force=True)

    processors: list = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
    ]
    if json_logs:
        processors.append(structlog.processors.JSONRenderer())
    else:
        colors = getattr(sys.stderr, "isatty", lambda: False)()
        processors.append(structlog.dev.ConsoleRenderer(colors=colors))

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        logger_factory=_stderr_logger_factory,
        # Caching would re-introduce the stale-stream problem the factory avoids.
        cache_logger_on_first_use=False,
    )


__all__ = ["configure_logging"]
