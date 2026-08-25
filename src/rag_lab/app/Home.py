"""rag-lab Visual explorer -- entry point (plan §Phase 9).

Launch with:

    streamlit run src/rag_lab/app/Home.py

Wires the four pages via ``st.navigation``/``st.Page`` (stable on the pinned
``streamlit>=1.38``). Pages stay file-based -- not inline functions -- so
``streamlit.testing.v1.AppTest.from_file(...)`` can drive each one standalone
in tests, independent of this navigation shell.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

PAGES_DIR = Path(__file__).parent / "pages"

st.set_page_config(page_title="rag-lab explorer", layout="wide")

pages = [
    st.Page(
        str(PAGES_DIR / "1_chunk_boundaries.py"),
        title="Chunk boundaries",
        icon="🧩",
        default=True,
    ),
    st.Page(str(PAGES_DIR / "2_retrieval_comparison.py"), title="Retrieval comparison", icon="🔍"),
    st.Page(str(PAGES_DIR / "3_results_dashboard.py"), title="Results dashboard", icon="📊"),
    st.Page(str(PAGES_DIR / "4_optimizer_trace.py"), title="Optimizer trace", icon="🧪"),
]

nav = st.navigation(pages)
nav.run()
