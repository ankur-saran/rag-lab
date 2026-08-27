# Adding a chunker

A chunker only ever produces boundary decisions. Identity (`chunk_id`,
`ordinal`), token counts, the `embed_text` default, and the offset-invariant
check are all handled once, centrally, by `finalize_chunks()`
([`src/rag_lab/chunkers/base.py`](../src/rag_lab/chunkers/base.py)) — that's
what makes a new chunker "genuinely just the boundary logic" (plan §Phase 2,
Step 2.1).

## The protocol

```python
class Chunker(Protocol):
    name: str
    default_params: dict[str, Any]

    def chunk(self, doc: Document, params: dict[str, Any]) -> list[Chunk]: ...
```

`chunk()` returns a `list[ChunkSpec]` turned into `Chunk`s by `finalize_chunks`
— you never construct a `Chunk` by hand.

```python
@dataclass
class ChunkSpec:
    char_start: int
    char_end: int
    text: str                       # must equal doc.text[char_start:char_end]
    embed_text: str | None = None   # None defaults to `text` in finalize_chunks
    heading_path: list[str] = field(default_factory=list)
    parent_id: str | None = None
    role: ChunkRole = "standalone"
    meta: dict[str, Any] = field(default_factory=dict)
```

## A worked example: `paragraph`

The simplest real chunker in the registry, `fixed`
([`src/rag_lab/chunkers/fixed.py`](../src/rag_lab/chunkers/fixed.py)), is the
template. Here's a new one that splits on blank lines:

```python
from rag_lab.chunkers.base import ChunkSpec, finalize_chunks
from rag_lab.schemas import Chunk, Document

NAME = "paragraph"
DEFAULT_PARAMS: dict = {}


class ParagraphChunker:
    name = NAME
    default_params = DEFAULT_PARAMS

    def chunk(self, doc: Document, params: dict) -> list[Chunk]:
        specs: list[ChunkSpec] = []
        pos = 0
        for para in doc.text.split("\n\n"):
            end = pos + len(para)
            if para.strip():
                specs.append(ChunkSpec(char_start=pos, char_end=end, text=para))
            pos = end + 2  # skip the "\n\n" separator
        return finalize_chunks(specs, doc, NAME, params)
```

Register it in `REGISTRY`
([`src/rag_lab/chunkers/__init__.py`](../src/rag_lab/chunkers/__init__.py)):

```python
from rag_lab.chunkers.paragraph import ParagraphChunker

REGISTRY: dict[str, Chunker] = {
    ...,
    "paragraph": ParagraphChunker(),
}
```

That's the whole extension surface. `chunk run --chunker paragraph`,
`experiment run`'s matrix, and every CLI command that takes `--chunker` pick
it up immediately — nothing else in the codebase is chunker-specific.

## Two invariants you cannot violate

**The offset invariant.** `spec.text` must equal
`doc.text[spec.char_start:spec.char_end]` exactly. `finalize_chunks` raises
`ValueError` if it doesn't — this is what stops a chunker from silently
drifting off the document it claims to be chunking. If your boundary logic
needs to *widen* what's returned to the consumer (the way `sentence_window`
does), keep `char_start`/`char_end`/`text` describing the widened span
consistently; don't fake a narrower span with wider text.

**The `text`/`embed_text` split.** `text` is what a retriever returns to the
consumer; `embed_text` is what gets embedded. Leave `embed_text=None` and it
defaults to `text` — only set it explicitly when they should differ (heading
prefixes, asymmetric embedding prefixes, summary-indexing). Never collapse
the two fields into one; every advanced chunker in this codebase
(`markdown`'s heading-path prefix, `sentence_window`'s embed-narrow/return-wide,
`table_summary`'s LLM summary) exists because this split is expressible.

## Verifying a new chunker

```bash
rag-lab chunk run --corpus api_docs --chunker paragraph
rag-lab chunk stats --chunk-set <chunk_set_id>     # token distribution, split-code/table counts, orphan rate
rag-lab chunk show  --chunk-set <chunk_set_id> --doc-id <doc_id>
rag-lab chunk diff  --a <chunk_set_id> --b api_docs__markdown__<hash> --doc-id <doc_id>
```

Add a `TestParagraphChunker` class to a new `tests/test_phase_2.py`-style
module following the pattern every baseline chunker's tests already use: the
offset invariant and coverage invariant as property tests over
`fixtures/documents/sample.jsonl`, plus determinism (two runs, identical
`chunk_set_id` and byte-identical output).
