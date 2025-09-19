"""
pages/step1_select_input.py

Enhanced Step 1 page for TradingWorkbench: select OHLC CSV from `input_data/`.
Features improved validation, better UI/UX, and enhanced error handling.
"""
from __future__ import annotations
import streamlit as st
import pandas as pd
import os
import glob
import re
from datetime import datetime
from typing import Tuple, Optional, Dict, Any

# ---------- Configuration ----------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # file belongs to TradingWorkbench pages folder.Two folder up is root
INPUT_DIR = os.path.join(ROOT, "input_data")
UPLOADS_DIR = os.path.join(INPUT_DIR, "uploads")
LARGE_FILE_MB_WARN = 50  # if larger than this skip full validation and show warning
REQUIRED_COLUMNS = {"date", "open", "high", "low", "close", "volume"}
TICKER_CANDIDATES = {"ticker", "symbol", "scrip", "security", "name"}

# ---------- Helpers ----------
def ensure_dirs():
    """Ensure required directories exist"""
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(UPLOADS_DIR, exist_ok=True)

def list_csv_files():
    """List CSV files in input directory, sorted by modification time"""
    ensure_dirs()
    pattern = os.path.join(INPUT_DIR, "*.csv")
    files = glob.glob(pattern)
    upload_pattern = os.path.join(UPLOADS_DIR, "*.csv")
    upload_files = glob.glob(upload_pattern)
    
    # Combine and sort by modification time
    all_files = files + upload_files
    files_sorted = sorted(all_files, key=os.path.getmtime, reverse=True)
    return files_sorted

def human_size(nbytes: int) -> str:
    """Convert bytes to human readable format"""
    for unit in ("B", "KB", "MB", "GB"):
        if nbytes < 1024.0:
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024.0
    return f"{nbytes:.1f} TB"

def sanitize_filename(filename: str) -> str:
    """Sanitize filename by removing special characters"""
    # Keep alphanumeric, spaces, hyphens, underscores, and dots
    name = re.sub(r'[^\w\s\-_.]', '', filename)
    # Replace spaces with underscores
    name = name.replace(' ', '_')
    return name

def save_upload(uploaded_file) -> Tuple[Optional[str], Optional[str]]:
    """Save uploaded file and return (saved_path, error_message)"""
    ensure_dirs()
    name = sanitize_filename(uploaded_file.name)
    target = os.path.join(UPLOADS_DIR, name)
    
    # Handle existing files
    if os.path.exists(target):
        ts = datetime.now().strftime("%Y%m%dT%H%M%S")
        base, ext = os.path.splitext(name)
        name = f"{base}_{ts}{ext}"
        target = os.path.join(UPLOADS_DIR, name)
    
    try:
        with open(target, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return target, None
    except Exception as e:
        return None, f"Failed to save file: {e}"

def detect_mapping(cols: list[str]) -> dict:
    """Enhanced column mapping detection with more flexible matching"""
    lm = {c.lower().strip(): c for c in cols}
    mapping = {}
    
    # Define column aliases for more flexible matching
    column_aliases = {
        "date": ["date", "timestamp", "datetime", "time", "dt"],
        "open": ["open", "opening", "open price", "op", "open_value"],
        "high": ["high", "high price", "maximum", "max", "hi"],
        "low": ["low", "low price", "minimum", "min", "lo"],
        "close": ["close", "closing", "close price", "last", "cp", "close_value"],
        "volume": ["volume", "vol", "quantity", "qty", "shares", "turnover"]
    }
    
    # Map required columns
    for standard_name, aliases in column_aliases.items():
        for alias in aliases:
            if alias in lm:
                mapping[standard_name] = lm[alias]
                break
    
    # Map ticker column
    for cand in TICKER_CANDIDATES:
        if cand in lm:
            mapping["ticker"] = lm[cand]
            break
    
    return mapping

def header_check(path: str) -> dict:
    """Validate CSV header and detect column mapping"""
    try:
        df0 = pd.read_csv(path, nrows=0)
        cols = list(df0.columns)
    except Exception as e:
        return {"status": "FAIL", "messages": [f"Header read error: {e}"], "cols": [], "mapping": {}}
    
    mapping = detect_mapping(cols)
    missing = sorted(list(REQUIRED_COLUMNS - set(mapping.keys())))
    
    if missing:
        return {"status": "FAIL", "messages": [f"Missing required columns: {', '.join(missing)}"], 
                "cols": cols, "mapping": mapping}
    
    return {"status": "PASS", "messages": ["Required columns present."], "cols": cols, "mapping": mapping}

def enhanced_deep_validation(path: str, mapping: dict) -> dict:
    """Comprehensive data validation with sampling for large files"""
    size_mb = os.path.getsize(path) / 1_000_000
    msgs = []
    
    # Handle large files with sampling
    if size_mb > LARGE_FILE_MB_WARN:
        try:
            # Sample data for large files
            date_col = mapping["date"]
            df = pd.read_csv(path, parse_dates=[date_col], nrows=1000, low_memory=False)
            msgs.append(f"Large file ({size_mb:.1f} MB): Validated sample of 1000 rows")
        except Exception as e:
            return {"status": "FAIL", "messages": [f"Parse error: {e}"], "size_mb": size_mb}
    else:
        try:
            date_col = mapping["date"]
            df = pd.read_csv(path, parse_dates=[date_col], infer_datetime_format=True, low_memory=False)
            msgs.append(f"Full validation completed. Rows: {len(df):,}")
        except Exception as e:
            return {"status": "FAIL", "messages": [f"Parse error: {e}"], "size_mb": size_mb}
    
    # Check date parsing
    date_col_name = mapping["date"]
    if pd.api.types.is_datetime64_any_dtype(df[date_col_name]):
        msgs.append(f"Parsed '{date_col_name}' as datetime")
    else:
        msgs.append(f"Warning: '{date_col_name}' not fully parsed as datetime")
    
    # Check for missing values
    for col in ("open", "high", "low", "close", "volume"):
        if col in mapping:
            col_name = mapping[col]
            missing = df[col_name].isna().sum()
            if missing > 0:
                pct = 100.0 * missing / max(1, len(df))
                msgs.append(f"'{col_name}': {missing} missing values ({pct:.1f}%)")
    
    # Check for negative values where inappropriate
    for col in ("open", "high", "low", "close"):
        if col in mapping:
            col_name = mapping[col]
            negative = (df[col_name] < 0).sum()
            if negative > 0:
                msgs.append(f"'{col_name}': {negative} negative values")
    
    if "volume" in mapping:
        vol_name = mapping["volume"]
        negative_vol = (df[vol_name] < 0).sum()
        if negative_vol > 0:
            msgs.append(f"'{vol_name}': {negative_vol} negative volumes")
    
    # Check numeric conversion
    for col in ("open", "high", "low", "close", "volume"):
        if col in mapping:
            col_name = mapping[col]
            coerced = pd.to_numeric(df[col_name], errors="coerce")
            n_bad = int(coerced.isna().sum())
            if n_bad > 0:
                pct = 100.0 * n_bad / max(1, len(coerced))
                msgs.append(f"'{col_name}': {n_bad} non-numeric cells ({pct:.1f}%)")
    
    # Check for duplicates
    if "ticker" in mapping:
        dup = int(df.duplicated(subset=[mapping["date"], mapping["ticker"]]).sum())
        if dup:
            msgs.append(f"Found {dup} duplicate (date,ticker) rows.")
    else:
        dup = int(df.duplicated(subset=[mapping["date"]]).sum())
        if dup:
            msgs.append(f"Found {dup} duplicate date rows.")
    
    # Check date range and sorting
    try:
        mn = df[mapping["date"]].min()
        mx = df[mapping["date"]].max()
        msgs.append(f"Date range: {mn} → {mx}")
        
        # Check if dates are sorted
        if not df[mapping["date"]].is_monotonic_increasing:
            msgs.append("Warning: Dates are not sorted chronologically")
    except Exception:
        pass
    
    # Determine status
    status = "PASS"
    for m in msgs:
        ml = m.lower()
        if "missing" in ml or "error" in ml or "parse" in ml:
            status = "FAIL"
        if ("duplicate" in ml or "invalid" in ml or "warning" in ml or 
            "negative" in ml or "not sorted" in ml):
            if status != "FAIL":
                status = "WARN"
    
    return {"status": status, "messages": msgs, "size_mb": size_mb}

def get_file_info(path: str) -> Dict[str, Any]:
    """Get comprehensive file information"""
    try:
        stat = os.stat(path)
        # Count rows (excluding header)
        with open(path, 'r') as f:
            row_count = sum(1 for line in f) - 1
        
        return {
            "size": human_size(stat.st_size),
            "modified": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
            "created": datetime.fromtimestamp(stat.st_ctime).strftime("%Y-%m-%d %H:%M:%S"),
            "rows": row_count
        }
    except Exception:
        return {}

# ---------- Render ----------
def render():
    st.title("📊 1 — Select OHLC CSV Data")
    
    # Information expander
    with st.expander("📋 Expected CSV Format & Instructions", expanded=False):
        st.markdown("""
        Your CSV should include these columns (case insensitive):
        - **Date**: Date/time of each observation
        - **Open**: Opening price
        - **High**: Highest price during the period  
        - **Low**: Lowest price during the period
        - **Close**: Closing price
        - **Volume**: Trading volume
        
        Optional columns:
        - **Ticker/Symbol**: Security identifier (if multiple securities in file)
        
        Files are read from the `input_data/` directory. Uploaded files are saved to `input_data/uploads/`.
        """)
        
        # Show example data
        st.markdown("**Example Data Format:**")
        example_data = {
            "date": ["2023-01-01", "2023-01-02", "2023-01-03"],
            "open": [100.0, 101.5, 102.3],
            "high": [102.0, 103.0, 104.5],
            "low": [99.5, 100.5, 101.0],
            "close": [101.5, 102.0, 103.5],
            "volume": [1000000, 1200000, 950000],
            "ticker": ["AAPL", "AAPL", "AAPL"]
        }
        st.dataframe(pd.DataFrame(example_data))
    
    ensure_dirs()
    files = list_csv_files()
    basenames = [os.path.basename(p) for p in files]
    file_paths = {os.path.basename(p): p for p in files}

    # File selection UI
    col1, col2, col3 = st.columns([6, 2, 3])
    
    with col1:
        if basenames:
            default_idx = 0
            prev = st.session_state.get("twb_selected_csv_name")
            if prev in basenames:
                default_idx = basenames.index(prev)
            
            selected_basename = st.selectbox(
                "Choose CSV file", 
                options=basenames, 
                index=default_idx, 
                key="twb_select_dropdown",
                help="Select a CSV file from the input_data directory"
            )
            selected_path = file_paths[selected_basename]
        else:
            st.info("No CSV files found. Upload a file or place CSV files in the input_data/ folder.")
            selected_basename = None
            selected_path = None

    with col2:
        if st.button("🔄 Refresh List", help="Refresh the list of available files"):
            st.experimental_rerun()

    with col3:
        uploaded = st.file_uploader(
            "Upload CSV", 
            type=["csv"], 
            key="twb_uploader",
            help="Upload a CSV file to be saved in input_data/uploads/"
        )
        if uploaded is not None:
            saved_path, error = save_upload(uploaded)
            if error:
                st.error(f"Upload failed: {error}")
            else:
                st.success(f"Uploaded and saved to: {os.path.basename(saved_path)}")
                # Add to session state to select the new file
                st.session_state.twb_select_dropdown = os.path.basename(saved_path)
                st.experimental_rerun()

    st.markdown("---")

    if selected_path:
        # File information
        try:
            st.subheader("📄 File Details")
            file_info = get_file_info(selected_path)
            
            col_info1, col_info2 = st.columns(2)
            with col_info1:
                st.write("**File:**", selected_basename)
                st.write("**Size:**", file_info.get("size", "Unknown"))
                st.write("**Rows:**", file_info.get("rows", "Unknown"))
            with col_info2:
                st.write("**Path:**", selected_path)
                st.write("**Modified:**", file_info.get("modified", "Unknown"))
                st.write("**Created:**", file_info.get("created", "Unknown"))
        except Exception as e:
            st.error(f"Error getting file info: {e}")

        # Data preview
        st.subheader("👀 Preview (first 5 rows)")
        try:
            preview = pd.read_csv(selected_path, nrows=5)
            st.dataframe(preview)
        except Exception as e:
            st.error(f"Preview failed: {e}")

        # Validation sections
        st.subheader("✅ Validation Results")
        
        # Header validation
        with st.expander("Header Validation", expanded=True):
            header = header_check(selected_path)
            if header["status"] == "PASS":
                st.success("✓ " + header["messages"][0])
                if header.get("mapping"):
                    st.write("**Detected column mapping:**")
                    for std_name, orig_name in header["mapping"].items():
                        st.write(f"- {std_name}: {orig_name}")
            else:
                st.error("✗ " + "; ".join(header.get("messages", ["Header check failed"])))

        # Deep validation
        with st.expander("Data Quality Validation", expanded=True):
            if header["status"] == "PASS":
                mapping = header.get("mapping", {})
                deep = enhanced_deep_validation(selected_path, mapping)
            else:
                deep = {"status": "FAIL", "messages": ["Skipping deep validation because header check failed."]}
            
            if deep["status"] == "PASS":
                st.success("✓ Deep validation: PASS")
            elif deep["status"] == "WARN":
                st.warning("⚠ Deep validation: WARN")
            else:
                st.error("✗ Deep validation: FAIL")
            
            # Show validation messages
            for m in deep.get("messages", []):
                st.write("-", m)

        # Action buttons
        st.markdown("---")
        col_act1, col_act2 = st.columns([3, 1])
        
        with col_act1:
            if deep["status"] in ["PASS", "WARN"]:
                if st.button("✅ Proceed with this file", type="primary"):
                    st.session_state["twb_selected_csv_name"] = selected_basename
                    st.session_state["twb_selected_csv_path"] = selected_path
                    st.session_state["twb_selected_csv_validation"] = {
                        "header": header, 
                        "deep": deep
                    }
                    
                    # Add to file history
                    if 'twb_file_history' not in st.session_state:
                        st.session_state.twb_file_history = []
                    if selected_basename not in st.session_state.twb_file_history:
                        st.session_state.twb_file_history.append(selected_basename)
                    
                    st.success(f"Selection saved: {selected_basename}")
                    st.info("You can now proceed to the next step in the sidebar.")
            else:
                force = st.checkbox("I understand the issues and want to proceed anyway", key="twb_force_proceed")
                if force and st.button("⛔ Proceed despite errors"):
                    st.session_state["twb_selected_csv_name"] = selected_basename
                    st.session_state["twb_selected_csv_path"] = selected_path
                    st.session_state["twb_selected_csv_validation"] = {
                        "header": header, 
                        "deep": deep
                    }
                    st.warning(f"Selection saved with issues: {selected_basename}")
        
        with col_act2:
            if st.button("🗑️ Clear selection"):
                keys_to_remove = [
                    "twb_selected_csv_name", 
                    "twb_selected_csv_path", 
                    "twb_selected_csv_validation",
                    "twb_force_proceed"
                ]
                for key in keys_to_remove:
                    if key in st.session_state:
                        del st.session_state[key]
                st.experimental_rerun()

    # Show current selection status
    st.markdown("---")
    if "twb_selected_csv_name" in st.session_state:
        st.info(f"Current selection: `{st.session_state['twb_selected_csv_name']}`")
        
        # Show file history if available
        if 'twb_file_history' in st.session_state and st.session_state.twb_file_history:
            with st.expander("📋 Recent Files", expanded=False):
                for i, file in enumerate(st.session_state.twb_file_history[-5:]):  # Show last 5 files
                    st.write(f"{i+1}. {file}")