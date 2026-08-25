"""Step 9.3 — results dashboard.

Load a run: metrics table with CI error bars, a chunker x embedder heatmap
for a selected metric, a latency-vs-recall scatter (the cost/quality
frontier), and a failure browser filtered by difficulty tier.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from rag_lab.app import data, ui

st.title("📊 Results dashboard")

with ui.guarded("experiment runs"):
    run_ids, run_ids_is_fixture = data.list_run_ids()
ui.fixture_banner(run_ids_is_fixture, "Run list")
if not run_ids:
    st.warning("No experiment runs available.")
    st.stop()

run_id = st.selectbox("Run", run_ids)

with ui.guarded(f"run {run_id!r}"):
    config, results, run_is_fixture = data.load_run(run_id)
ui.fixture_banner(run_is_fixture, f"run {run_id!r}")

cells = data.cells_for_run(config)
metrics = data.headline_metrics()


def _cell_label(cell) -> str:  # noqa: ANN001 - local helper, Cell is a dataclass
    return f"{cell.corpus}/{cell.chunker}/{cell.embedder}/{cell.retriever}"


# --------------------------------------------------------------------------- #
# Metrics table with CI
# --------------------------------------------------------------------------- #

st.subheader("Metrics")
rows = []
for cell in cells:
    result = data.result_for_cell(results, cell)
    if result is None:
        continue
    row: dict[str, object] = {
        "corpus": cell.corpus,
        "chunker": cell.chunker,
        "embedder": cell.embedder,
        "retriever": cell.retriever,
        "n": len(result.per_query),
    }
    for metric in metrics:
        ci = data.metric_ci(result, metric)
        row[metric] = f"{ci[0]:.3f} [{ci[1]:.3f}, {ci[2]:.3f}]" if ci else "-"
    rows.append(row)

if not rows:
    st.warning("This run has no computed cells yet.")
    st.stop()

st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# --------------------------------------------------------------------------- #
# Heatmap: chunker x embedder for a selected metric, one retriever at a time
# --------------------------------------------------------------------------- #

st.subheader("Chunker × embedder heatmap")
default_metric_idx = metrics.index("recall@5") if "recall@5" in metrics else 0
metric_for_heatmap = st.selectbox("Metric", metrics, index=default_metric_idx)
retrievers_present = sorted(
    {cell.retriever for cell in cells if data.result_for_cell(results, cell)}
)
if retrievers_present:
    retriever_for_heatmap = st.selectbox("Retriever", retrievers_present)
    heat_rows = []
    for cell in cells:
        if cell.retriever != retriever_for_heatmap:
            continue
        result = data.result_for_cell(results, cell)
        if result is None:
            continue
        ci = data.metric_ci(result, metric_for_heatmap)
        heat_rows.append({"chunker": cell.chunker, "embedder": cell.embedder, "value": ci[0] if ci else None})
    if heat_rows:
        pivot = pd.DataFrame(heat_rows).pivot_table(index="chunker", columns="embedder", values="value")
        fig = px.imshow(pivot, text_auto=".3f", aspect="auto", color_continuous_scale="Viridis")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.caption("No cells for this retriever.")

# --------------------------------------------------------------------------- #
# Latency vs. recall scatter (cost/quality frontier)
# --------------------------------------------------------------------------- #

st.subheader("Latency vs. recall (cost/quality frontier)")
scatter_rows = []
for cell in cells:
    result = data.result_for_cell(results, cell)
    if result is None:
        continue
    scatter_rows.append(
        {
            "label": _cell_label(cell),
            "chunker": cell.chunker,
            "latency_p50": result.metrics.get("latency_p50"),
            "recall@5": result.metrics.get("recall@5"),
        }
    )
scatter_df = pd.DataFrame([r for r in scatter_rows if r["latency_p50"] is not None and r["recall@5"] is not None])
if not scatter_df.empty:
    fig = px.scatter(
        scatter_df, x="latency_p50", y="recall@5", color="chunker", hover_name="label",
        labels={"latency_p50": "latency p50 (ms)", "recall@5": "recall@5"},
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.caption("No cells report both latency and recall@5.")

# --------------------------------------------------------------------------- #
# Failure browser, filtered by difficulty tier
# --------------------------------------------------------------------------- #

st.subheader("Failure browser")
cell_options = {_cell_label(cell): cell for cell in cells if data.result_for_cell(results, cell)}
if cell_options:
    cell_choice = st.selectbox("Cell", list(cell_options))
    chosen_cell = cell_options[cell_choice]
    result = data.result_for_cell(results, chosen_cell)
    with ui.guarded(f"eval set for corpus {chosen_cell.corpus!r}"):
        tagged = data.worst_failures(result, chosen_cell.corpus, n=None)
    tiers = sorted({tier for _, tier in tagged})
    picked_tiers = st.multiselect("Difficulty", tiers, default=tiers)
    show_n = st.number_input("Show top N", min_value=1, max_value=max(1, len(tagged)), value=min(20, len(tagged)))
    filtered = [(t, tier) for t, tier in tagged if tier in picked_tiers][: int(show_n)]
    table_rows = [
        {
            "difficulty": tier,
            "query": t.query,
            "status": "excluded" if t.excluded else f"recall@5={t.metrics.get('recall@5', 0.0):.2f}",
            "gold_chunks": ", ".join(t.gold_chunk_ids) or "(none)",
            "retrieved_top5": ", ".join(t.retrieved_chunk_ids[:5]) or "(none)",
        }
        for t, tier in filtered
    ]
    st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)
else:
    st.caption("No cells with results to browse.")
