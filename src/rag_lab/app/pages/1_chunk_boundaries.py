"""Step 9.1 — chunk boundary viewer.

Pick corpus -> document -> one or two chunk sets. A single chunk set renders
as one shaded pane; two render as scroll-synced side-by-side panes -- the
spec's "highest-value screen in the app."
"""

from __future__ import annotations

import streamlit as st

from rag_lab.app import boundaries, data, ui

st.title("🧩 Chunk boundary viewer")
st.caption(
    "Colored spans are chunk boundaries; alternating shades distinguish adjacent chunks, "
    "hatched spans are covered by more than one chunk. Hover a span for its chunk id, "
    "token count, and heading path."
)

with ui.guarded("corpora"):
    corpora, corpora_is_fixture = data.list_corpora()
ui.fixture_banner(corpora_is_fixture, "Corpus list")
if not corpora:
    st.warning("No corpora available.")
    st.stop()

corpus = st.selectbox("Corpus", corpora)

with ui.guarded(f"documents for corpus {corpus!r}"):
    docs, docs_is_fixture = data.load_documents(corpus)
ui.fixture_banner(docs_is_fixture, f"{corpus!r} documents")

doc_label = {f"{d.title} ({d.doc_id})": d for d in docs}
doc_choice = st.selectbox("Document", list(doc_label))
doc = doc_label[doc_choice]

with ui.guarded("chunk sets"):
    chunk_set_ids, chunk_sets_is_fixture = data.list_chunk_sets()
ui.fixture_banner(chunk_sets_is_fixture, "Chunk-set list")
if not chunk_set_ids:
    st.warning("No chunk sets available.")
    st.stop()

selected = st.multiselect(
    "Chunk set(s) — pick one to view, two to compare side by side",
    chunk_set_ids,
    default=chunk_set_ids[:1],
    max_selections=2,
)
if not selected:
    st.info("Pick at least one chunk set.")
    st.stop()

panes = []
for chunk_set_id in selected:
    with ui.guarded(f"chunk set {chunk_set_id!r}"):
        chunks, is_fixture = data.load_chunks(chunk_set_id)
    ui.fixture_banner(is_fixture, f"chunk set {chunk_set_id!r}")
    doc_chunks = [c for c in chunks if c.doc_id == doc.doc_id]
    if not doc_chunks:
        st.warning(f"Chunk set {chunk_set_id!r} has no chunks for this document.")
        st.stop()
    segments = boundaries.segment_document(len(doc.text), doc_chunks)
    inner = boundaries.render_segments_html(doc.text, segments)
    panes.append((chunk_set_id, inner))

st.caption(f"Document: {doc.doc_id} · {len(doc.text):,} characters")

if len(panes) == 1:
    label, inner = panes[0]
    st.markdown(f"**{label}**")
    st.markdown(boundaries.wrap_single_pane_html(inner), unsafe_allow_html=True)
else:
    (label_a, inner_a), (label_b, inner_b) = panes
    st.components.v1.html(
        boundaries.wrap_dual_pane_html(inner_a, inner_b, label_a, label_b),
        height=650,
        scrolling=False,
    )
