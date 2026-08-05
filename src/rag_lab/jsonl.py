"""JSONL read/write for artifact records.

Deliberately strict: a malformed line fails loudly with its line number rather
than being skipped. A silently dropped chunk becomes a mysterious recall gap
three phases later.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


class ArtifactValidationError(ValueError):
    pass


def write_jsonl(path: Path | str, records: Iterable[BaseModel]) -> int:
    """Write records, creating parent directories. Returns the count written."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with p.open("w", encoding="utf-8", newline="\n") as fh:
        for record in records:
            fh.write(record.model_dump_json())
            fh.write("\n")
            count += 1
    return count


def iter_jsonl(path: Path | str, model: type[T]) -> Iterator[T]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield model.model_validate_json(line)
            except ValidationError as exc:
                raise ArtifactValidationError(
                    f"{p}:{lineno} failed to validate as {model.__name__}\n{exc}"
                ) from exc
            except json.JSONDecodeError as exc:
                raise ArtifactValidationError(f"{p}:{lineno} is not valid JSON: {exc}") from exc


def read_jsonl(path: Path | str, model: type[T]) -> list[T]:
    return list(iter_jsonl(path, model))


def count_lines(path: Path | str) -> int:
    with Path(path).open("r", encoding="utf-8") as fh:
        return sum(1 for line in fh if line.strip())


__all__ = [
    "ArtifactValidationError",
    "count_lines",
    "iter_jsonl",
    "read_jsonl",
    "write_jsonl",
]
