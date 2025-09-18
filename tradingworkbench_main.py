"""
tradingworkbench_main.py

Entry point for the TradingWorkbench multi-page Streamlit app.
Creates pages by importing modules from the local `pages` directory.
Does NOT modify any of your existing project files.
"""
from __future__ import annotations
import streamlit as st
from importlib import import_module
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
PAGES_DIR = os.path.join(ROOT, "pages")
if PAGES_DIR not in sys.path:
    sys.path.insert(0, PAGES_DIR)

st.set_page_config(page_title="TradingWorkbench", layout="wide")

st.sidebar.title("TradingWorkbench")
st.sidebar.markdown("Stepwise pipeline. Each page implements a single logical operation.")

# Add pages here as tuples: (label shown in sidebar, module_name in pages/)
PAGES = [
    ("1 — Select OHLC CSV", "step1_select_input"),
    # Add further pages as new modules when ready:
    # ("2 — Fundamental Filter", "step2_fundamental_filter"),
    # ("3 — Technical Indicators", "step3_technical_indicators"),
]

page_label = st.sidebar.radio("Choose page", [p[0] for p in PAGES], index=0)

module_to_load = None
for label, mod in PAGES:
    if label == page_label:
        module_to_load = mod
        break

if module_to_load is None:
    st.error("Page not found.")
else:
    try:
        page = import_module(module_to_load)
        if hasattr(page, "render"):
            page.render()
        else:
            st.error(f"Page module `{module_to_load}` does not expose a render() function.")
    except Exception as exc:
        st.exception(exc)
