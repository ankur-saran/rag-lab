# Strategies

One page per chunker and embedder: what it does, when it wins, when it
loses, and the real number behind that claim. Every number here comes from
an actual local run against the real corpora — reproduce with the commands
shown, or with `make demo` for the `api_docs` fixed/markdown comparison.

Two honest scoping notes before the numbers:

- A real, hand-authored eval set (§ [`findings.md`](findings.md)) exists
  only for `api_docs` today — Phase 5's LLM-based generator, which would
  produce one for every corpus, is still a stub. So recall/MRR/nDCG numbers
  below are `api_docs`-only; `semantic`, `table_summary`, and the embedders'
  behavior on `transcripts`/`filings`/`contracts`/`catalog` are evidenced by
  chunk-structure stats (`chunk stats`'s split-code/table counts, build
  time, orphan rate) rather than retrieval quality, because there is nothing
  to retrieve against on those corpora yet.
- All chunk counts and stats below are against the real, committed corpora
  under `corpora/` (`api_docs`: 15 docs / 4,965 tokens; `transcripts`: 13
  docs; `filings`: 14 docs), not fixtures.

## Chunkers

### `fixed` — token windows, no structural awareness

Slides a fixed-size token window over the whole document with no regard for
any structural signal (`src/rag_lab/chunkers/fixed.py`). The baseline every
other chunker is measured against.

**Wins:** simplicity, zero build cost, works on any text.
**Loses:** blind to structure — a window boundary can land mid-sentence,
mid-table, or mid-fenced-code-block with equal probability.

At its default params (`chunk_tokens=512`) every `api_docs` document is
short enough to fit in one window, so it never actually cuts a fence at that
size — the split only shows up once the window gets small enough to force a
cut inside a real document:

```bash
rag-lab chunk run --corpus api_docs --chunker fixed --params chunk_tokens=128 --params overlap_tokens=32
rag-lab chunk stats --chunk-set api_docs__fixed__<hash>
```

| window | chunks | split code blocks (of 29) |
|---|---|---|
| 512 tok (default) | 15 | 0 |
| 128 tok | 56 | **20** |

### `recursive` — separator cascade

Tries the coarsest separator first (`\n\n`), recurses into any piece still
over budget with the next separator, then greedily merges adjacent pieces
back up to the token limit (`src/rag_lab/chunkers/recursive.py`).

**Wins:** respects paragraph/sentence boundaries without needing a real
parser; the reusable half (`split_text_recursive`) is `markdown`'s own
fallback for oversized sections.
**Loses:** still has no concept of "this is a code fence" or "this is a
table" on its own — that's `markdown`'s and `table_summary`'s job
respectively.

On `api_docs` at default params it produces byte-identical chunk boundaries
to `fixed` (15 chunks, 0/29 split code, same 279/335/367/378 token
distribution) — every document in this corpus is short enough that the
separator cascade never needs to actually cut. The two chunkers only
diverge once documents exceed the window, which is why `filings`
(268–418 tok/doc, closer to the 512-tok default) is a better corpus to see
`recursive` do real work.

### `markdown` — structure-aware, the demo's headline chunker

One chunk per leaf section (the content directly under a heading, up to the
next heading); fenced code blocks are never split even if that pushes a
chunk past `max_tokens`; undersized sections merge forward; oversized ones
fall back to `recursive` (`src/rag_lab/chunkers/markdown_chunker.py`). Uses
a real parser (`markdown-it-py`), not regex detection, so a fence can never
be mistaken for a heading.

**Wins:** the fence-preservation punchline is unconditional, not just a
large-window artifact:

| window | chunks | split code blocks (of 29) |
|---|---|---|
| 768 tok (default) | 51 | **0** |
| 128 tok | 58 | **0** |

vs. `fixed`'s 20/29 at the same 128-tok window. On the real `api_docs` eval
set (30 hand-authored pairs, `bge-small`, `dense`, `k=10` — the `make demo`
matrix), `markdown` and `fixed` **tie** on recall@5/recall@10 (both 1.000,
CI [1.000, 1.000] — the eval set's easy questions saturate both at this
corpus size), but `markdown` returns **3.4x fewer tokens per correct
answer** (chunk_efficiency 980.5 vs. fixed's 3321.0) and edges ahead on the
harder-to-saturate metrics: recall@1 0.867 vs 0.833, MRR 0.928 vs 0.917. See
[`findings.md`](findings.md) for the full table and the caveat about what
this eval set can and can't distinguish.

**Loses:** needs real markdown structure to have anything to key off —
`transcripts` (unstructured, no headings) gets no benefit at all, which is
exactly why that corpus was chosen to stress `semantic` instead.

### `sentence_window` — embed narrow, return wide

The inverse of `markdown`'s heading-path split: each sentence gets its own
chunk whose `text` widens to include `window_size` neighboring sentences
(for the consumer) while `embed_text` stays the bare sentence (for the
embedder) (`src/rag_lab/chunkers/sentence_window.py`).

**Wins:** precise retrieval (embedding a single sentence is a sharp target)
combined with enough surrounding context in what's actually returned.
**Loses:** many more chunks than any other strategy — 99 on `api_docs`
(vs. 15–58 for the others) — so it costs more to embed and index, and
overlapping windows need dedup at retrieval time
(`retrievers/sentence_window.py`).

```
rag-lab chunk run --corpus api_docs --chunker sentence_window
# 99 chunks, 73/279/353/378 tok (min/p50/p95/max), 0/29 split code blocks
```

### `semantic` — embedding-distance topic-shift detection

Splits into sentences, embeds each with neighboring context, cuts wherever
consecutive-sentence cosine distance exceeds the `breakpoint_percentile` of
the document's own distance distribution (`src/rag_lab/chunkers/semantic.py`).
Percentile-based thresholding is what makes this generalize across corpora
with different embedding-distance scales.

**Wins:** on headingless, topic-shifting text — `transcripts` is exactly
that: no markdown structure for `markdown` to key off, but real topic
boundaries `semantic` can detect.
**Loses / costs:** the plan's own estimate is "3–5x the indexing cost of
`recursive`" — the real number is corpus-dependent and can be far worse than
that on code-punctuated text:

| corpus | chunks | chunk build time (embedding pass) | orphan rate |
|---|---|---|---|
| `transcripts` (13 docs) | 29 | **0.77s** | 6.9% |
| `api_docs` (15 docs) | 28 | **42.55s** | 14.3% |

`api_docs`'s prose is dense with code identifiers, inline backticks, and
abbreviations, which the sentence splitter fragments far more aggressively
than natural language — many more sentence-pair embeddings per document than
on `transcripts`, despite a comparable document count. That's the honest
version of the plan's "expect it to lose on `api_docs`" — on this codebase's
real corpora it doesn't just lose on relevance, it costs 55x more to build.

```bash
rag-lab chunk run --corpus transcripts --chunker semantic
rag-lab chunk stats --chunk-set transcripts__semantic__<hash>   # chunk build time row
```

### `table_summary` — tables get their own chunk, LLM summary in `embed_text`

Every GFM pipe table becomes its own chunk (`text` = the full untruncated
table); every other character range is delegated to a fallback chunker's
offset-aware sub-span helper (`src/rag_lab/chunkers/table_summary.py`).
`summarize=true` (default) sets `embed_text = summary + "\n\n" + text`, an
LLM-generated description of what the table reports.

**Wins:** on `filings` (14 docs, 15 tables), it produces **44 chunks**
against `recursive`'s **14** on the same corpus — every table gets isolated
into its own retrievable unit instead of living embedded inside a
paragraph-sized chunk. (At this corpus's scale `recursive`'s 512-token
default window happens to be large enough that it never literally *splits*
a table mid-row — the real difference isn't split-avoidance here, it's
dedicated indexing.)
**Loses / costs:** an LLM call per table when `summarize=true` (cached by
table content, `--mock-llm` for credential-free testing/CI).

```bash
rag-lab chunk run --corpus filings --chunker table_summary --mock-llm
# 44 chunks, 0/15 split tables, 93.2% split-sentence rate (short table-summary chunks)
```

## Embedders

All four are `sentence-transformers` models behind one asymmetric
query/document protocol (`embed_query`/`embed_documents`, not a single
`embed`) — see `src/rag_lab/embedders/registry.py`. Getting the
query/passage prefix wrong is silent and costs several points of recall;
Phase 3's own asymmetry test (`embed_query(q) != embed_documents([q])[0]`
for an asymmetric model) is what catches that class of bug.

| name | model | dim | query prefix | doc prefix |
|---|---|---|---|---|
| `bge-small` | `BAAI/bge-small-en-v1.5` | 384 | `Represent this sentence for searching relevant passages: ` | *(none)* |
| `e5-base` | `intfloat/e5-base-v2` | 768 | `query: ` | `passage: ` |
| `e5-multilingual` | `intfloat/multilingual-e5-base` | 768 | `query: ` | `passage: ` |
| `minilm` | `sentence-transformers/all-MiniLM-L6-v2` | 384 | *(none — symmetric)* | *(none)* |

### `bge-small` — the default, and every number in this document

384-dim, asymmetric (real query prefix, empty doc prefix). What `make demo`
and every `verify-phase-N` target use. On `api_docs__fixed` (15 chunks):
built in a few seconds on CPU, 15 vectors, dim=384. This is the model behind
every recall/MRR/nDCG number in this document and in
[`findings.md`](findings.md).

### `minilm` — the symmetric baseline

384-dim, no query/document distinction at all (`strict_prefixes` would
reject an *asymmetric* model configured with empty prefixes, but `minilm`
genuinely has none — it's the control case). Builds at the same speed as
`bge-small` on the same chunk set (15 vectors, dim=384). Not yet
recall-benchmarked against `bge-small` locally — the interesting comparison
(does BGE's asymmetric prefix actually buy recall on this corpus) is a
`full_matrix.yaml` run away, not yet captured in this document.

### `e5-base` / `e5-multilingual` — larger, asymmetric, not yet locally benchmarked

768-dim, `query: `/`passage: ` prefixes (the E5 family's convention).
`e5-multilingual` is the one embedder in the registry actually relevant to
`catalog` (short multilingual product descriptions) — none of the other
three claim multilingual competence. Neither has been pulled/benchmarked in
the run this document's numbers come from (a larger download, skipped to
keep this pass fast); mechanism and prefix behavior above is accurate from
the registry, but there is no local cost/recall number for either yet. Run
`rag-lab index build --chunk-set <id> --embedder e5-base` to fill this in.

## Retrieval note (not a chunker/embedder, but the numbers earned it)

On the identifier query `"What does the error code IDEMPOTENCY_KEY_CONFLICT
mean?"` against `api_docs__markdown__<768-tok hash>` (`bge-small`), the
chunk that actually defines the error code ranks **#3 under BM25/hybrid** vs
**#6 under dense** (`k=6`):

```bash
rag-lab retrieve compare --index-id <markdown+bge-small index> \
    --retrievers dense,bm25,hybrid \
    --query "What does the error code IDEMPOTENCY_KEY_CONFLICT mean?" --k 6
```

Not a #1 hit for either — this markdown chunk set's 768-token sections merge
several related subsections together, diluting the exact match — but a real,
reproducible rank-3-vs-rank-6 gap in BM25's favor on an exact-identifier
query, which is the whole reason `hybrid` (RRF) exists in this framework
rather than picking one retriever and calling it done.
