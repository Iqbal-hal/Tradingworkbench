"""
tradingworkbench_main.py

Streamlit app for TradingWorkbench with predefined column structure.
Optimized for fixed CSV format without column mapping or upload features.
"""

from __future__ import annotations
import streamlit as st
from importlib import import_module
import os
import sys
import glob

# --- PATH CONFIGURATION FOR YOUR SYSTEM ---
ROOT = os.path.dirname(os.path.abspath(__file__))
# ROOT = "E:\Sync To GD\GIT\Tradingworkbench"

PAGES_DIR = os.path.join(ROOT, "pages")
# PAGES_DIR = "E:\Sync To GD\GIT\Tradingworkbench\pages"

if PAGES_DIR not in sys.path:
    sys.path.insert(0, PAGES_DIR)

# Configure Streamlit page
st.set_page_config(page_title="TradingWorkbench", layout="wide")

# --- SESSION STATE INITIALIZATION ---
# Persistent storage for user data across page reloads
if 'current_data' not in st.session_state:
    st.session_state.current_data = None
if 'selected_file' not in st.session_state:
    st.session_state.selected_file = None

def discover_pages():
    """
    Discover page modules in pages directory matching step*.py pattern.
    
    HOW IT WORKS:
    1. Searches for all .py files starting with 'step' in pages folder
    2. Extracts page number and title from filename
    3. Creates navigation labels like "1 — Select Input"
    """
    # Find all step page files
    page_files = glob.glob(os.path.join(PAGES_DIR, "step*.py"))
    pages = []
    
    for file_path in page_files:
        file_name = os.path.basename(file_path)
        if file_name == "__init__.py":
            continue
            
        # Remove .py extension to get module name
        module_name = file_name.replace('.py', '')
        
        # Extract page number and title from filename pattern: stepX_description.py
        if file_name.startswith('step') and '_' in file_name:
            parts = module_name.split('_', 1)  # Split at first underscore only
            try:
                page_num = parts[0].replace('step', '')  # Extract number from "step1"
                page_title = parts[1].replace('_', ' ').title()  # Convert to title case
                pages.append((f"{page_num} — {page_title}", module_name))
            except (IndexError, ValueError):
                continue
    
    # Sort by page number (convert string numbers to integers for proper sorting)
    return sorted(pages, key=lambda x: int(x[0].split('—')[0].strip()))

# --- PAGE DISCOVERY WITH ERROR HANDLING ---
try:
    PAGES = discover_pages()
    if not PAGES:
        raise Exception("No pages found")
except Exception as e:
    st.sidebar.warning(f"Dynamic page discovery failed: {e}")
    # Fallback to default page if automatic discovery fails
    PAGES = [("1 — Select OHLC CSV", "step1_select_input")]

# --- SIDEBAR NAVIGATION INTERFACE ---
st.sidebar.title("📊 TradingWorkbench")
st.sidebar.markdown("Stepwise pipeline for trading analysis")

# Create radio buttons for page navigation
page_labels = [p[0] for p in PAGES]  # Extract just the display labels
page_label = st.sidebar.radio("Navigate to:", page_labels, index=0)

st.sidebar.markdown("---")
st.sidebar.caption(f"Pages available: {len(PAGES)}")

# --- DYNAMIC PAGE LOADING ---
module_to_load = None
# Find which module corresponds to the selected page label
for label, mod in PAGES:
    if label == page_label:
        module_to_load = mod
        break

if module_to_load is None:
    st.error("Selected page not found in available pages.")
else:
    try:
        # Dynamically import the selected page module
        page = import_module(module_to_load)
        
        # Check if the module has the required render function
        if hasattr(page, "render"):
            # Execute the page's render function to display its content
            page.render()
        else:
            st.error(f"Page module `{module_to_load}` is missing the required render() function.")
    except ImportError as e:
        st.error(f"Failed to import page module '{module_to_load}': {e}")
    except Exception as e:
        st.error(f"Unexpected error loading page '{module_to_load}': {e}")