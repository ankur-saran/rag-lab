# rag-lab

A framework for showcasing chunking and embedding strategies, with an agentic
optimization layer.

**Status: Phase 0 complete.** The skeleton, artifact schemas, config system, CLI
surface and health check are in place. Every later phase's subcommands are
registered and stubbed, so `rag-lab --help` is an accurate map of the system.

---

## Quick start

```bash
python -m pip install -e ".[core,dev]"   # ~15s, no torch
rag-lab doctor                           # environment health check
make verify-phase-0                      # full Phase 0 acceptance run
```

`core` is deliberately light — no PyTorch, no ChromaDB. Phases 1, 2 and 5
(with `--mock-llm`) run on it alone. Install `.[embed]` when you reach Phase 3.

## The independence contract

Every phase is runnable on its own. That is enforced architecturally, not by
convention:

1. **Phases communicate only through on-disk artifacts** with frozen schemas
   (`src/rag_lab/schemas.py`). No phase imports another phase's runtime state.
2. **Every phase ships fixtures.** `fixtures/` holds a small, valid, committed
   example of every artifact any phase consumes, so a phase works on a clean
   checkout with no prior phase ever having run.
3. **Every phase has one verification command** that exits non-zero on failure:
   `make verify-phase-N`.

`paths.resolve_artifact()` is what implements this — it returns the real artifact
when it exists and the fixture otherwise, logging a loud warning on fallback.
The warning matters: silently benchmarking a 12-record fixture would produce
plausible-looking, meaningless numbers.

## Layout

```
config/          default.yaml + experiment matrices
corpora/         raw source documents (Phase 1)
fixtures/        committed sample artifacts — the independence contract
  raw/           source markdown the fixtures are generated from
artifacts/       generated output (gitignored)
scripts/         build_fixtures.py
src/rag_lab/
  schemas.py     frozen artifact models — the interface contract
  ids.py         deterministic IDs
  config.py      YAML loading + params_hash (the entire caching strategy)
  paths.py       artifact resolution with fixture fallback
  jsonl.py       strict JSONL read/write
  doctor.py      environment health check
  cli.py         Typer app, one sub-app per phase
tests/           one test module per phase
```

## Two things worth knowing before extending this

**`Chunk.text` vs `Chunk.embed_text`.** One field is returned to the consumer,
the other is embedded. Heading-path prefixing, asymmetric query/document
prefixes, table summary-indexing and parent-document retrieval are all
expressible through that single split without special-casing any of them
downstream. Do not collapse the two fields.

**Gold labels are document character offsets, never chunk IDs.** A label
expressed as a chunk ID is only valid for the chunk set that produced it, which
would make chunker comparison circular — the exact thing this project exists to
measure. `EvalPair.gold_char_span` is authoritative; `gold_chunk_ids` is derived
and re-resolved per chunk set at evaluation time.

## Caching

Every artifact ID embeds the hash of the parameters that produced it
(`config.params_hash`). A changed parameter yields a new ID and therefore a cache
miss; unchanged parameters yield a hit. There is no cache invalidation logic
anywhere in the system, by design.

Key order never affects the hash, and annotation keys (`description`, `comment`,
`note`, plus anything listed in a local `_nohash`) are excluded — so you can
document a config without invalidating every cached artifact.

## CLI

```bash
rag-lab --help                    # every phase, stubs included
rag-lab doctor                    # environment check
rag-lab config                    # fully resolved configuration
rag-lab --set seed=7 config       # override resolution: --set > --config > default.yaml
```

Stubs exit 1 and name the phase that will implement them.

## Requirements

Python 3.10+. No GPU, Docker, database or search service. An `ANTHROPIC_API_KEY`
is needed for Phases 5 and 8 only, and both support `--mock-llm` for testing
without one.

## License

MIT.
