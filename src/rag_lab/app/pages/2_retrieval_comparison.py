"""Step 9.2 — retrieval comparison.

Query box, multi-select of retriever configurations, N result columns. Each
result shows rank, score, chunk text with query terms highlighted, and a
gold-match badge when the query came from the corpus's eval set. Hybrid
results expand ``ScoredChunk.debug`` for the dense rank, sparse rank, and
fused score.
"""

from __future__ import annotations

import html
import re

import streamlit as st

from rag_lab.app import data, ui
from rag_lab.retrievers import build_retriever

st.title("🔍 Retrieval comparison")

with ui.guarded("indexes"):
    manifests, indexes_is_fixture = data.list_indexes()
ui.fixture_banner(indexes_is_fixture, "Index list")
if not manifests:
    st.warning("No indexes available.")
    st.stop()

manifest_label = {f"{m.index_id}  ({m.corpus}, {m.embedder})": m for m in manifests}
choice = st.selectbox("Index", list(manifest_label))
selected_manifest = manifest_label[choice]

query_source = st.radio("Query source", ["Free text", "From eval set"], horizontal=True)
query = ""
if query_source == "From eval set":
    with ui.guarded(f"eval set for corpus {selected_manifest.corpus!r}"):
        pairs, evalset_is_fixture = data.load_evalset(selected_manifest.corpus)
    ui.fixture_banner(evalset_is_fixture, f"{selected_manifest.corpus!r} eval set")
    if not pairs:
        st.warning("This corpus's eval set is empty.")
        st.stop()
    pair_label = {f"[{p.difficulty}] {p.query}": p for p in pairs}
    picked = st.selectbox("Eval-set query", list(pair_label))
    query = pair_label[picked].query
else:
    query = st.text_input("Query", value="")

k = st.slider("k", min_value=1, max_value=20, value=5)

retriever_names = data.list_retriever_names()
default_selection = [n for n in ("dense", "bm25", "hybrid") if n in retriever_names] or retriever_names[:1]
selected_retrievers = st.multiselect("Retrievers to compare", retriever_names, default=default_selection)

if not query.strip():
    st.info("Enter or pick a query to run retrieval.")
    st.stop()
if not selected_retrievers:
    st.info("Pick at least one retriever.")
    st.stop()

with ui.guarded(f"index {selected_manifest.index_id!r}"):
    manifest, store, index_is_fixture = data.open_index(selected_manifest.index_id)
ui.fixture_banner(index_is_fixture, f"index {manifest.index_id!r}")

with ui.guarded(f"chunk set {manifest.chunk_set_id!r}"):
    chunk_set, _ = data.load_chunks(manifest.chunk_set_id)
gold_ids = data.gold_chunk_ids_for_query(query, manifest.corpus, chunk_set)

_WORD_RE = re.compile(r"\w+")


def _highlight(text: str, query: str) -> str:
    terms = {t.lower() for t in _WORD_RE.findall(query) if len(t) > 2}
    if not terms:
        return html.escape(text)
    pattern = re.compile(r"\b(" + "|".join(re.escape(t) for t in terms) + r")\b", re.IGNORECASE)

    def _wrap(m: re.Match[str]) -> str:
        return f"<mark>{html.escape(m.group(0))}</mark>"

    out = []
    last = 0
    for m in pattern.finditer(text):
        out.append(html.escape(text[last : m.start()]))
        out.append(_wrap(m))
        last = m.end()
    out.append(html.escape(text[last:]))
    return "".join(out)


columns = st.columns(len(selected_retrievers))
for col, name in zip(columns, selected_retrievers):
    with col:
        st.markdown(f"**{name}**")
        try:
            retriever = build_retriever(name, None, manifest=manifest, store=store)
            results = retriever.retrieve(query, k)
        except Exception as exc:  # noqa: BLE001 - isolate one bad retriever from the rest of the page
            st.error(f"{name} failed: {exc}")
            continue
        if not results:
            st.caption("(no results)")
            continue
        for r in results:
            is_gold = r.chunk.chunk_id in gold_ids
            badge = " 🥇 gold" if is_gold else ""
            st.markdown(f"`#{r.rank}` score={r.score:.4f}{badge}")
            preview = " ".join(r.chunk.text.split())
            preview = preview if len(preview) <= 300 else preview[:299] + "…"
            st.markdown(
                f'<div style="font-size:0.85em; line-height:1.4;">{_highlight(preview, query)}</div>',
                unsafe_allow_html=True,
            )
            if name == "hybrid" and "components" in r.debug:
                with st.expander("dense / sparse / fused"):
                    st.write(
                        {
                            "rrf_score": r.debug.get("rrf_score"),
                            **{
                                comp: info
                                for comp, info in r.debug.get("components", {}).items()
                            },
                        }
                    )
            st.divider()
