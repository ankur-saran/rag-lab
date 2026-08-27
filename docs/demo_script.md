# Demo script

A run-of-show for presenting rag-lab live. Every command below was run for
real to write this document — the IDs are deterministic (a pure function of
corpus + chunker/embedder + params, per the caching strategy in
`config.params_hash`), so running the same commands against a clean checkout
reproduces the same `chunk_set_id`/`index_id` and the numbers next to each
beat. If a number has drifted, something changed — that's the point of
writing them down.

Start with `make demo` running in the background (or already finished) so
the Streamlit viewer is open in a browser tab before beat 3.

## 5-minute version

### 1. Boundary diff — no embeddings, no eval set, no API key

The single most persuasive artifact in the project, per the plan, and it
costs an hour to build. Force both chunkers down to a small enough window
that `fixed` actually has to cut through content:

```bash
rag-lab chunk run --corpus api_docs --chunker fixed --params chunk_tokens=128 --params overlap_tokens=32
rag-lab chunk run --corpus api_docs --chunker markdown --params max_tokens=128
rag-lab chunk diff --a api_docs__fixed__8b51dccd --b api_docs__markdown__581d4e51 --doc-id e0e95f816e69
```

**Narrate:** `fixed`'s chunk #2 cuts through the middle of a code block
(the boundary lands right after `` handle(order) ``` `` — inside a fence).
`markdown`'s boundaries line up with `##` headings every time.

**Expected number:** at this window, `fixed` splits **20 of 29** fenced
code blocks in `api_docs`; `markdown` splits **0**. (`chunk stats
--chunk-set api_docs__fixed__8b51dccd` / `...__markdown__581d4e51` to show
the counters directly instead of eyeballing the diff.)

### 2. Retrieval comparison — BM25 vs. dense on an exact identifier

```bash
rag-lab retrieve compare \
    --index-id api_docs__markdown__1253e256__bge-small__5b10c141 \
    --retrievers dense,bm25,hybrid \
    --query "What does the error code IDEMPOTENCY_KEY_CONFLICT mean?" --k 6
```

**Narrate:** the chunk that actually defines `IDEMPOTENCY_KEY_CONFLICT`
(the `# Errors` section) lands at **rank 3 under BM25 and hybrid**, but
**rank 6 under dense alone** — dense's embedding pulls in topically-related
but less exact sections first (`Conflicting bodies`, `Endpoints that
require a key`). This is the honest version of the story: not a #1 hit
either way at this chunk size, but a real, reproducible gap in BM25's favor
on an exact-identifier query — the reason `hybrid` (RRF fusion) exists in
this framework at all.

### 3. Results table — the demo matrix, with confidence intervals

Already computed by `make demo` (`config/experiments/demo.yaml`: `fixed` vs
`markdown`, `bge-small`, `dense`, against the real 30-pair `api_docs` eval
set):

```bash
rag-lab experiment report --run-id demo__20260826T030253Z
```

**Narrate:** recall@5/@10 are **tied at 1.000** for both chunkers — the eval
set is easy enough, and the corpus small enough, to saturate that metric.
The real differentiator is **chunk_efficiency**: `markdown` returns
**3.4x fewer tokens per correct answer** (980.5 vs. 3321.0) for the same
recall, plus a small edge on recall@1 (0.867 vs 0.833) and MRR (0.928 vs
0.917). This is deliberately not a clean "markdown wins" slide — it's a
more honest one, and it's the point of tracking chunk_efficiency at all
(see [`findings.md`](findings.md) §1 for the full table).

Or skip the CLI and point at the already-open Streamlit **Results
dashboard** page, which renders the same numbers with CI error bars and a
chunker × embedder heatmap.

## 15-minute extended version

Everything above, plus:

### 4. Optimizer trace — a real agentic iteration loop, credential-free

```bash
rag-lab agent optimize --corpus api_docs --max-iterations 3 --mock-llm
rag-lab agent trace --run-id optimizer-api_docs__20260827T022312Z
```

**Narrate:** iteration 0 proposes `fixed`/`bge-small`/`dense`, evaluates on
the **dev** split (never `test`, until the very end — see
[`findings.md`](findings.md) and the plan's overfitting mitigation),
diagnoses, and mutates toward markdown chunking on the next iteration. The
final line reports the winning config's **test**-split number exactly once:

```
winner: iteration 0 (fixed/bge-small/dense) -- dev recall@5=1.000, test recall@5=1.000
```

Or point at the Streamlit **Optimizer trace** page for the same data as an
iteration timeline with hypothesis → config diff → metrics delta →
diagnosis cards.

### 5. The negative result — a fancier chunker that doesn't pay for itself

```bash
rag-lab chunk run --corpus api_docs --chunker semantic
rag-lab chunk stats --chunk-set api_docs__semantic__9f87aab8
```

**Narrate:** `semantic` costs **42.55 seconds** to chunk 15 `api_docs`
documents (the embedding pass inside `.chunk()`), versus **0.77 seconds**
for the same chunker on 13 `transcripts` documents — a 55x difference in
build cost for a comparable document count, because `api_docs`'s
code-punctuated prose fragments the sentence splitter far more
aggressively. `api_docs` already has real markdown structure `markdown`
exploits for free; `semantic` buys nothing here and costs dramatically more
to try. This is the intellectually honest part of the demo: a framework
that only shows wins is a sales deck (see
[`findings.md`](findings.md) §3 for the full comparison, including the
caveat that a `transcripts`-side recall comparison isn't available yet —
no eval set exists for that corpus).

## If a number doesn't match

Every ID above is deterministic — if `chunk diff`/`index build`/`experiment
report` produce a different ID or a materially different number than what's
written here, something upstream changed (a corpus document, a default
param, a dependency version). That's a signal to stop and check
`docs/findings.md` and the relevant `verify-phase-N` target before
presenting, not to edit this document to match.
