"""Text normalization — the one routine every loader runs before anything else.

Every chunk offset in the system (from Phase 2 onward) indexes into
``Document.text``. Normalizing *after* offsets are computed would silently shift
every span by a variable amount, and the resulting failures look like retrieval
quality problems three phases later rather than what they actually are. So this
module exists to make "normalize first" the only code path available: loaders
call it before constructing a ``Document``, and nothing downstream touches
``Document.text`` again.

This function was originally inlined in ``scripts/build_fixtures.py``; it now
lives here and that script imports it, so there is exactly one implementation.
"""

from __future__ import annotations

import re
import unicodedata


def normalize_text(text: str) -> str:
    """NFC-normalize, convert CRLF/CR to LF, collapse 3+ blank lines to 2, strip
    trailing whitespace per line, strip the whole document, end with one newline.

    Idempotent: ``normalize_text(normalize_text(x)) == normalize_text(x)``.
    """
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


__all__ = ["normalize_text"]
