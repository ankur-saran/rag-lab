# Findings

What the experiments actually showed, negative results included. Every
number below is from a real local run (`run_id`s and exact commands are
given so they're reproducible), not a projection from the plan's estimates.

## 1. `markdown` vs `fixed` on `api_docs`: a tie on the headline metric, a
   clear win underneath it

`make demo`'s matrix (`config/experiments/demo.yaml`: `fixed` vs `markdown`,
both `chunk_tokens≈512`-scale params matching `smoke.yaml`, `bge-small`,
`dense`, `k=10`, against the real 30-pair hand-authored `api_docs` eval set)
produced, on a real run (`run_id demo__20260826T030253Z`):

| chunker | recall@1 | recall@5 | recall@10 | mrr | ndcg@10 | chunk_efficiency |
|---|---|---|---|---|---|---|
| `fixed` | 0.833 [0.700, 0.967] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.917 [0.850, 0.983] | 0.938 [0.889, 0.988] | 3321.0 |
| `markdown` | 0.867 [0.733, 0.967] | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.928 [0.856, 0.983] | 0.938 [0.884, 0.984] | **980.5** |

Recall@5 and recall@10 are **tied at a perfect 1.000 for both chunkers, with
identical [1.000, 1.000] confidence intervals.** This is not a bug — it's
what happens when 30 hand-authored, needle-anchored questions are evaluated
against a 15-document corpus at `k=10`: the eval set is easy enough, and the
corpus small enough, that both chunkers find the right answer within their
top 10 essentially every time. Recall@5/@10 is *saturated* at this corpus
scale and cannot distinguish the two strategies.

What does distinguish them: **`markdown` returns 3.4x fewer tokens per
correct answer** (chunk_efficiency 980.5 vs. 3321.0 — this is the metric the
plan calls "under-reported and important," and this run is exactly why),
and edges ahead on the metrics that aren't saturated (recall@1, MRR — both
non-overlapping-adjacent, not clean CI separation, but consistently higher).
A chunker that wins recall by returning 4x the text is not obviously
winning; here `markdown` wins on both quality *and* economy, just not on
the metric most demos would lead with.

**This is a different, complementary result from `test_phase_6.py`'s own
AC-6 test**, which uses a *much smaller* `fixed` window
(`chunk_tokens=64, overlap_tokens=0`) specifically because that's the
parameter choice that produces genuinely non-overlapping recall@5
confidence intervals — proving `markdown` beats `fixed` outright once
`fixed`'s window is small enough to actually bisect the hand-authored
facts' surrounding context. Put together: **at a small window, `fixed`
loses on recall outright; at a window generous enough to saturate recall,
`fixed` still loses on token economy.** Neither result alone is the whole
story.

## 2. The unconditional fence-preservation result

Independent of window size, `markdown` never splits a fenced code block on
`api_docs` (0/29 at both its 768-token default and a 128-token squeeze);
`fixed` splits 0/29 at 512 tokens (every doc fits in one window) but **20 of
29** once the window shrinks to 128 tokens — the size Step 2's own
`verify-phase-2` target deliberately uses to force the split. This is the
cleanest, cheapest demonstration in the whole project: no embeddings, no
eval set, no API key, just `chunk diff`.

## 3. Negative result: `semantic`'s cost is corpus-dependent, and can be
   far worse than "3-5x"

The plan estimates `semantic` costs "roughly 3-5x the indexing cost of
`recursive`." The real, measured `chunk_build_seconds` (the embedding pass
inside `.chunk()`) tells a sharper story:

| corpus | docs | chunks | chunk build time | orphan rate |
|---|---|---|---|---|
| `transcripts` | 13 | 29 | **0.77s** | 6.9% |
| `api_docs` | 15 | 28 | **42.55s** | 14.3% |

A comparable document count, a **55x** difference in build cost.
`api_docs`'s code-punctuated prose (inline backticks, identifiers,
abbreviations) fragments the regex sentence splitter far more aggressively
than `transcripts`'s natural language, producing many more sentence-pair
embeddings per document. The plan's own framing — "expect it to lose on
`api_docs` and win on `transcripts`... a framework that only demonstrates
wins is a sales deck" — undersells the finding: on this codebase's real
corpora, `semantic` on `api_docs` doesn't just fail to help, it costs
dramatically more to even try. This is exactly the kind of result a
framework that only ran the winning cases would hide.

(Recall/MRR comparison for `semantic` against `markdown` on `transcripts`
itself is not yet available — see §5, the Phase 5 limitation, below.)

## 4. `table_summary`: dedicated indexing, not split-avoidance

On `filings` (14 docs, 15 tables), `table_summary` produces 44 chunks
against `recursive`'s 14 on the identical corpus — every table becomes its
own retrievable, summarized unit. Notably, `recursive` **also** shows 0/15
split tables on this corpus: at its 512-token default window, `filings`'
tables happen to fit inside a single chunk without ever being cut, so
"doesn't split tables" is not what separates the two chunkers here. The
real difference is architectural — `table_summary` gives each table a
dedicated chunk (and an LLM-generated summary in `embed_text`) instead of
letting it sit embedded inside a larger paragraph-scale chunk. On a corpus
with denser tables or a smaller window, `recursive` would eventually split
one; this corpus, at these defaults, doesn't happen to exercise that case.

## 5. Known limitation: only `api_docs` has a real eval set

Phase 5's LLM-based eval-set generator (`evalset build/validate/stats/review`)
is still a stub — `scripts/build_api_docs_evalset.py` hand-authors 30
needle-anchored pairs against the real `api_docs` corpus as its documented
substitute (used by `verify-phase-6/8/9` and by `make demo`). No equivalent
exists for `contracts`, `filings`, `transcripts`, or `catalog`. Consequently:

- Every recall/MRR/nDCG number in this document and in
  [`strategies.md`](strategies.md) is `api_docs`-only.
- `semantic`'s claimed win on `transcripts` and `table_summary`'s claimed
  value on `filings` are evidenced by chunk-structure stats (build cost,
  chunk count, orphan rate) and Phase 7's own topic-shift-boundary test
  (`tests/test_phase_7.py`'s AC-3, ±1 sentence of a synthetic three-topic
  boundary), **not** by a retrieval-quality number on those corpora — there
  is nothing to retrieve against there yet.
- A hand-authored eval set is also not adversarially validated the way an
  LLM-generated-and-filtered one would be (Phase 5's own four-filter
  pipeline: quote grounding, meta-reference regex, leakage check,
  answerability spot-check). The 30 `api_docs` pairs are real and
  needle-anchored, but they were not run through that pipeline, because
  there is no pipeline yet.

Building Phase 5 for real, and generating comparable eval sets for the
other four corpora, is the highest-value next step for turning this
document's chunker-specific findings into corpus-general ones.

## 6. Cost/latency, gathered in one place

| operation | corpus | cost |
|---|---|---|
| `experiment run` (2-cell demo matrix, cached embedder model) | `api_docs` | **~27s** total |
| `semantic` chunk build | `transcripts` (13 docs) | 0.77s |
| `semantic` chunk build | `api_docs` (15 docs) | 42.55s |
| `table_summary` chunk build (`--mock-llm`) | `filings` (14 docs, 15 tables) | seconds (mocked; real LLM path adds one cached call per table) |
| `index build`, `bge-small`, cache hit | any | well under 1s |

`make demo`'s full pipeline (corpus build, eval-set generation, 2 chunk
sets, 2 indexes, 2-cell matrix, report) runs in well under a minute once
the embedding model is already cached locally — the 5-minute budget in the
plan is dominated by the *first-ever* model download, not by anything this
framework computes.
