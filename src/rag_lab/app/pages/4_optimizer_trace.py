"""Step 9.4 — optimizer trace viewer.

Iteration timeline with the metric trajectory, and per-iteration an
expandable card: hypothesis -> config diff -> metrics delta -> diagnosis.
Reads directly from ``optimizer_trace.jsonl`` (via ``app.data``, which adds
the fixture fallback ``agents.optimizer.load_trace`` doesn't have on its own).
"""

from __future__ import annotations

import plotly.express as px
import streamlit as st

from rag_lab.app import data, optimizer_diff, ui


def _config_str(config: dict) -> str:
    return f"{config.get('chunker')}/{config.get('embedder')}/{config.get('retriever')}"


st.title("🧪 Optimizer trace viewer")

with ui.guarded("optimizer runs"):
    run_ids, run_ids_is_fixture = data.list_optimizer_run_ids()
ui.fixture_banner(run_ids_is_fixture, "Optimizer run list")
if not run_ids:
    st.warning("No optimizer runs available.")
    st.stop()

run_id = st.selectbox("Optimizer run", run_ids)

with ui.guarded(f"optimizer trace {run_id!r}"):
    trace, trace_is_fixture = data.load_optimizer_trace(run_id)
ui.fixture_banner(trace_is_fixture, f"optimizer trace {run_id!r}")

if not trace:
    st.warning("This optimizer run has no recorded iterations (budget exhausted immediately).")
    st.stop()

dev_entries = [e for e in trace if e.split == "dev"]
test_entries = [e for e in trace if e.split == "test"]

all_metrics = sorted({m for e in trace for m in e.metrics})
if "recall@5" in all_metrics:
    default_metric = "recall@5"
else:
    default_metric = all_metrics[0] if all_metrics else None

st.subheader("Iteration timeline")
if default_metric is None:
    st.caption("No metrics recorded on this trace.")
else:
    metric = st.selectbox("Metric", all_metrics, index=all_metrics.index(default_metric))
    timeline_rows = [
        {
            "iteration": e.iteration,
            "value": e.metrics.get(metric),
            "split": e.split,
            "config": _config_str(e.config),
        }
        for e in trace
        if metric in e.metrics
    ]
    if timeline_rows:
        fig = px.line(
            timeline_rows,
            x="iteration",
            y="value",
            color="split",
            markers=True,
            hover_data=["config"],
            labels={"value": metric},
        )
        st.plotly_chart(fig, use_container_width=True)

    if dev_entries and test_entries:
        best_dev = max(dev_entries, key=lambda e: e.metrics.get(metric, float("-inf")))
        winner_cfg = _config_str(best_dev.config)
        st.success(
            f"**Winner**: iteration {best_dev.iteration} ({winner_cfg}) — "
            f"dev {metric}={best_dev.metrics.get(metric, 0.0):.3f}, "
            f"test {metric}={test_entries[-1].metrics.get(metric, 0.0):.3f}"
        )

st.subheader("Iterations")
prev_dev = None
for entry in trace:
    label = f"iteration {entry.iteration} ({entry.split}) — {entry.hypothesis[:80]}"
    with st.expander(label):
        st.markdown(f"**Hypothesis:** {entry.hypothesis}")

        prev_config = prev_dev.config if prev_dev else None
        config_diff = optimizer_diff.diff_config(prev_config, entry.config)
        if config_diff:
            st.markdown("**Config diff vs. previous dev iteration:**")
            st.table(
                {
                    "key": list(config_diff),
                    "old": [str(v[0]) for v in config_diff.values()],
                    "new": [str(v[1]) for v in config_diff.values()],
                }
            )
        else:
            st.caption("No config change from the previous dev iteration.")

        prev_metrics = prev_dev.metrics if prev_dev else None
        metrics_delta = optimizer_diff.diff_metrics(prev_metrics, entry.metrics)
        if metrics_delta:
            st.markdown("**Metrics delta:**")
            st.table(
                {
                    "metric": list(metrics_delta),
                    "delta": [f"{v:+.3f}" for v in metrics_delta.values()],
                }
            )

        if entry.diagnosis:
            st.markdown(f"**Diagnosis:** {entry.diagnosis}")
        if entry.mutation:
            st.markdown(f"**Mutation:** {entry.mutation}")
        st.caption(
            f"tokens: {entry.input_tokens} in / {entry.output_tokens} out · "
            f"run_id: {entry.run_id}"
        )

    if entry.split == "dev":
        prev_dev = entry
