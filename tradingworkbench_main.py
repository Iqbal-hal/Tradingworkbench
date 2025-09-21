"""
tradingworkbench_main.py

Enhanced entry point for the TradingWorkbench multi-page Streamlit app.
Features dynamic page discovery, better error handling, and session state management.

Module Overview:
----------------
This module serves as the main entry point for the TradingWorkbench application.
It handles:
- Dynamic discovery of page modules
- Session state initialization and management
- Navigation between different steps of the trading analysis pipeline

Key Components:
---------------
- discover_pages(): Automatically finds and registers page modules
- Session state management for data persistence between pages
- Error handling for robust application operation

Example:
--------
>>> import tradingworkbench_main
>>> # The application is typically run with: streamlit run tradingworkbench_main.py
"""

from __future__ import annotations
import streamlit as st
from importlib import import_module
import os
import sys
import glob

ROOT = os.path.dirname(os.path.abspath(__file__))
PAGES_DIR = os.path.join(ROOT, "pages")
if PAGES_DIR not in sys.path:
    sys.path.insert(0, PAGES_DIR)

st.set_page_config(page_title="TradingWorkbench", layout="wide")

# Initialize session state
if 'current_data' not in st.session_state:
    st.session_state.current_data = None
if 'selected_file' not in st.session_state:
    st.session_state.selected_file = None

def discover_pages():
    """Automatically discover pages in the pages directory"""
    page_files = glob.glob(os.path.join(PAGES_DIR, "step*.py"))
    pages = []
    
    for file_path in page_files:
        file_name = os.path.basename(file_path)
        if file_name == "__init__.py":
            continue
            
        module_name = file_name.replace('.py', '')
        
        # Extract page number and title from filename
        if file_name.startswith('step') and '_' in file_name:
            parts = module_name.split('_', 1)
            try:
                page_num = parts[0].replace('step', '')
                page_title = parts[1].replace('_', ' ').title()
                pages.append((f"{page_num} — {page_title}", module_name))
            except (IndexError, ValueError):
                continue
    
    return sorted(pages, key=lambda x: int(x[0].split('—')[0].strip()))

# Discover pages or use fallback
try:
    PAGES = discover_pages()
    if not PAGES:
        raise Exception("No pages found")
except Exception as e:
    st.sidebar.warning(f"Dynamic page discovery failed: {e}")
    PAGES = [("1 — Select OHLC CSV", "step1_select_input")]

# Sidebar layout
st.sidebar.title("📊 TradingWorkbench")
st.sidebar.markdown("Stepwise pipeline for trading analysis")

page_label = st.sidebar.radio("Navigate to:", [p[0] for p in PAGES], index=0)

st.sidebar.markdown("---")
st.sidebar.caption(f"Pages available: {len(PAGES)}")

# Page loading
module_to_load = None
for label, mod in PAGES:
    if label == page_label:
        module_to_load = mod
        break

if module_to_load is None:
    st.error("Selected page not found in available pages.")
else:
    try:
        page = import_module(module_to_load)
        if hasattr(page, "render"):
            page.render()
        else:
            st.error(f"Page module `{module_to_load}` is missing the required render() function.")
    except ImportError as e:
        st.error(f"Failed to import page module '{module_to_load}': {e}")
    except Exception as e:
        st.error(f"Unexpected error loading page '{module_to_load}': {e}")
        if st.secrets.get("debug", False):  # Only show full traceback in debug mode
            st.exception(e)