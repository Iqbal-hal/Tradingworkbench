"""
pages/step1_select_input.py

Step 1 page for TradingWorkbench: select OHLC CSV from input_data/.
Optimized for predefined column structure without upload features.
"""

from __future__ import annotations
import streamlit as st
import pandas as pd
import os
import glob
from datetime import datetime
from typing import Dict, Any

# --- CONFIGURATION FOR YOUR SYSTEM ---
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# ROOT = "E:\Sync To GD\GIT\Tradingworkbench"

INPUT_DIR = os.path.join(ROOT, "input_data")
# INPUT_DIR = "E:\Sync To GD\GIT\Tradingworkbench\input_data"

LARGE_FILE_MB_WARN = 50  # Skip full validation for files larger than 50MB

# --- PREDEFINED COLUMN NAMES (EXACTLY AS IN YOUR CSV) ---
PREDEFINED_COLUMNS = [
    "Date", "Stock", "Open", "High", "Low", "Close", "Volume", 
    "P/E", "EPS", "Earning Growth", "MCap", "P/B", "D/E", "PEG"
]

# Columns that will be renamed due to Python syntax restrictions
COLUMN_RENAME_MAP = {
    "P/E": "PE",
    "P/B": "PB", 
    "D/E": "DE",
    "Earning Growth": "Earning_Growth"
}

# --- HELPER FUNCTIONS ---

def ensure_dirs():
    """Create required directories if they don't exist"""
    os.makedirs(INPUT_DIR, exist_ok=True)

def list_csv_files():
    """
    List CSV files in input directory, sorted by modification time (newest first)
    
    HOW IT WORKS:
    1. Uses glob pattern matching to find all .csv files
    2. Sorts files by modification time using os.path.getmtime()
    3. Returns list with newest files first
    """
    ensure_dirs()
    pattern = os.path.join(INPUT_DIR, "*.csv")
    files = glob.glob(pattern)
    
    # Sort by modification time (newest first)
    files_sorted = sorted(files, key=os.path.getmtime, reverse=True)
    return files_sorted

def human_size(nbytes: int) -> str:
    """
    Convert bytes to human readable format (KB, MB, GB)
    
    EXAMPLE:
    human_size(1500000) returns "1.4 MB"
    human_size(500) returns "500.0 B"
    """
    for unit in ("B", "KB", "MB", "GB"):
        if nbytes < 1024.0:
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024.0
    return f"{nbytes:.1f} TB"

def validate_column_structure(path: str) -> dict:
    """
    Validate that CSV has the exact predefined column structure
    
    HOW IT WORKS:
    1. Reads just the header row of the CSV file
    2. Compares against PREDEFINED_COLUMNS list
    3. Returns validation status and messages
    """
    try:
        # Read only the header row
        df_header = pd.read_csv(path, nrows=0)
        actual_columns = list(df_header.columns)
        
        # Check if columns match exactly (order doesn't matter)
        missing_columns = set(PREDEFINED_COLUMNS) - set(actual_columns)
        extra_columns = set(actual_columns) - set(PREDEFINED_COLUMNS)
        
        messages = []
        status = "PASS"
        
        if missing_columns:
            messages.append(f"Missing required columns: {', '.join(sorted(missing_columns))}")
            status = "FAIL"
        
        if extra_columns:
            messages.append(f"Extra columns found: {', '.join(sorted(extra_columns))}")
            status = "WARN" if status != "FAIL" else "FAIL"
        
        if status == "PASS":
            messages.append("Column structure validation passed")
        
        return {
            "status": status, 
            "messages": messages, 
            "actual_columns": actual_columns,
            "missing_columns": list(missing_columns),
            "extra_columns": list(extra_columns)
        }
        
    except Exception as e:
        return {
            "status": "FAIL", 
            "messages": [f"Error reading file: {e}"], 
            "actual_columns": [],
            "missing_columns": [],
            "extra_columns": []
        }

def enhanced_deep_validation(path: str) -> dict:
    """
    Comprehensive data validation with sampling for large files
    
    HOW IT WORKS:
    1. Checks file size to determine validation strategy
    2. For large files (>50MB), samples first 1000 rows
    3. Validates data quality, missing values, numeric conversion
    """
    size_mb = os.path.getsize(path) / 1_000_000
    msgs = []
    
    # Handle large files with sampling
    if size_mb > LARGE_FILE_MB_WARN:
        try:
            # Sample data for large files
            df = pd.read_csv(path, nrows=1000, low_memory=False)
            msgs.append(f"Large file ({size_mb:.1f} MB): Validated sample of 1000 rows")
        except Exception as e:
            return {"status": "FAIL", "messages": [f"Parse error: {e}"], "size_mb": size_mb}
    else:
        try:
            # Full validation for smaller files
            df = pd.read_csv(path, low_memory=False)
            msgs.append(f"Full validation completed. Rows: {len(df):,}")
        except Exception as e:
            return {"status": "FAIL", "messages": [f"Parse error: {e}"], "size_mb": size_mb}
    
    # Apply column renaming for problematic column names
    df_renamed = df.rename(columns=COLUMN_RENAME_MAP)
    
    # Check for missing values in critical columns
    critical_columns = ["Date", "Stock", "Open", "High", "Low", "Close", "Volume"]
    for col in critical_columns:
        if col in df.columns:
            missing = df[col].isna().sum()
            if missing > 0:
                pct = 100.0 * missing / max(1, len(df))
                msgs.append(f"'{col}': {missing} missing values ({pct:.1f}%)")
    
    # Check for negative values where inappropriate
    price_columns = ["Open", "High", "Low", "Close"]
    for col in price_columns:
        if col in df.columns:
            negative = (df[col] < 0).sum()
            if negative > 0:
                msgs.append(f"'{col}': {negative} negative values")
    
    # Check numeric conversion for numeric columns
    numeric_columns = ["Open", "High", "Low", "Close", "Volume", "P/E", "EPS", "MCap", "P/B", "D/E", "PEG"]
    for col in numeric_columns:
        if col in df.columns:
            coerced = pd.to_numeric(df[col], errors="coerce")
            n_bad = int(coerced.isna().sum())
            if n_bad > 0:
                pct = 100.0 * n_bad / max(1, len(coerced))
                msgs.append(f"'{col}': {n_bad} non-numeric cells ({pct:.1f}%)")
    
    # Check date parsing
    if "Date" in df.columns:
        try:
            df_date = pd.to_datetime(df["Date"], errors="coerce")
            bad_dates = df_date.isna().sum()
            if bad_dates > 0:
                msgs.append(f"Date: {bad_dates} invalid date formats")
            else:
                msgs.append("Date column parsed successfully")
        except Exception as e:
            msgs.append(f"Date parsing error: {e}")
    
    # Check for duplicates by Date and Stock
    if "Date" in df.columns and "Stock" in df.columns:
        duplicates = df.duplicated(subset=["Date", "Stock"]).sum()
        if duplicates > 0:
            msgs.append(f"Found {duplicates} duplicate (Date,Stock) combinations")
    
    # Determine final validation status
    status = "PASS"
    for m in msgs:
        ml = m.lower()
        if "missing" in ml or "error" in ml or "parse" in ml:
            status = "FAIL"
        elif ("duplicate" in ml or "invalid" in ml or "warning" in ml or 
              "negative" in ml or "not sorted" in ml):
            if status != "FAIL":
                status = "WARN"
    
    return {"status": status, "messages": msgs, "size_mb": size_mb}

def get_file_info(path: str) -> Dict[str, Any]:
    """Get comprehensive file information including size and row count"""
    try:
        stat = os.stat(path)
        
        # Count rows (excluding header)
        with open(path, 'r', encoding='utf-8') as f:
            row_count = sum(1 for line in f) - 1
        
        return {
            "size": human_size(stat.st_size),
            "modified": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
            "created": datetime.fromtimestamp(stat.st_ctime).strftime("%Y-%m-%d %H:%M:%S"),
            "rows": row_count
        }
    except Exception as e:
        return {"error": str(e)}

# --- MAIN RENDER FUNCTION ---
def render():
    """Main function that renders the Step 1 interface"""
    st.title("📊 1 — Select OHLC CSV Data")
    
    # Information expander with predefined column requirements
    with st.expander("📋 Expected CSV Format", expanded=False):
        st.markdown(f"""
        Your CSV must have **exactly these columns** (case-sensitive):
        
        **Required Columns:**
        - `Date`, `Stock`, `Open`, `High`, `Low`, `Close`, `Volume`
        
        **Additional Analysis Columns:**
        - `P/E`, `EPS`, `Earning Growth`, `MCap`, `P/B`, `D/E`, `PEG`
        
        **Note:** Files are automatically read from the `input_data/` directory.
        Place your CSV files in `E:\\Sync To GD\\GIT\\Tradingworkbench\\input_data\\`
        """)
        
        # Show expected data format
        st.markdown("**Expected Data Format:**")
        example_data = {
            "Date": ["2023-01-01", "2023-01-02"],
            "Stock": ["AAPL", "AAPL"],
            "Open": [150.0, 152.5],
            "High": [155.0, 154.0],
            "Low": [149.5, 151.0],
            "Close": [153.0, 152.8],
            "Volume": [1000000, 1200000],
            "P/E": [25.0, 24.8],
            "EPS": [6.0, 6.1],
            "Earning Growth": [0.15, 0.12],
            "MCap": [2500000000, 2550000000],
            "P/B": [5.0, 5.1],
            "D/E": [0.3, 0.29],
            "PEG": [1.2, 1.18]
        }
        st.dataframe(pd.DataFrame(example_data))
    
    # Ensure input directory exists
    ensure_dirs()
    
    # Get list of CSV files
    files = list_csv_files()
    if not files:
        st.info("📁 No CSV files found in input_data folder.")
        st.info("Please place your CSV files in: E:\\Sync To GD\\GIT\\Tradingworkbench\\input_data\\")
        return
    
    # Create mapping of display names to full paths
    basenames = [os.path.basename(p) for p in files]
    file_paths = {os.path.basename(p): p for p in files}
    
    # --- FILE SELECTION INTERFACE ---
    st.subheader("📂 Select CSV File")
    
    col1, col2 = st.columns([4, 1])
    
    with col1:
        # Set default selection (previous selection or first file)
        default_idx = 0
        prev_selection = st.session_state.get("twb_selected_csv_name")
        if prev_selection in basenames:
            default_idx = basenames.index(prev_selection)
        
        selected_basename = st.selectbox(
            "Choose CSV file from input_data folder:", 
            options=basenames, 
            index=default_idx, 
            key="twb_select_dropdown"
        )
        selected_path = file_paths[selected_basename]
    
    with col2:
        # Refresh button to update file list
        if st.button("🔄 Refresh", help="Refresh the list of available files"):
            st.experimental_rerun()
    
    st.markdown("---")
    
    # --- FILE INFORMATION DISPLAY ---
    st.subheader("📄 File Details")
    
    try:
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
    
    # --- DATA PREVIEW ---
    st.subheader("👀 Data Preview (first 5 rows)")
    
    try:
        preview_df = pd.read_csv(selected_path, nrows=5)
        
        # Apply column renaming for display
        preview_renamed = preview_df.rename(columns=COLUMN_RENAME_MAP)
        st.dataframe(preview_renamed)
        
        # Show original column names
        with st.expander("🔍 Original Column Names"):
            st.write("Original columns:", list(preview_df.columns))
            st.write("Renamed columns:", list(preview_renamed.columns))
            
    except Exception as e:
        st.error(f"Preview failed: {e}")
    
    # --- VALIDATION SECTIONS ---
    st.subheader("✅ Validation Results")
    
    # Column structure validation
    with st.expander("Column Structure Validation", expanded=True):
        col_validation = validate_column_structure(selected_path)
        
        if col_validation["status"] == "PASS":
            st.success("✓ Column structure validation passed")
        elif col_validation["status"] == "WARN":
            st.warning("⚠ Column structure validation: WARN")
        else:
            st.error("✗ Column structure validation: FAIL")
        
        for message in col_validation.get("messages", []):
            st.write("-", message)
    
    # Data quality validation
    with st.expander("Data Quality Validation", expanded=True):
        if col_validation["status"] == "FAIL":
            st.warning("Skipping data quality validation due to column structure issues")
            deep_validation = {"status": "SKIP", "messages": ["Validation skipped"]}
        else:
            deep_validation = enhanced_deep_validation(selected_path)
        
        if deep_validation["status"] == "PASS":
            st.success("✓ Data quality validation: PASS")
        elif deep_validation["status"] == "WARN":
            st.warning("⚠ Data quality validation: WARN")
        elif deep_validation["status"] == "FAIL":
            st.error("✗ Data quality validation: FAIL")
        else:
            st.info("Data quality validation: SKIPPED")
        
        for message in deep_validation.get("messages", []):
            st.write("-", message)
    
    # --- ACTION BUTTONS ---
    st.markdown("---")
    col_act1, col_act2 = st.columns([3, 1])
    
    with col_act1:
        # Proceed button (only enabled if validation passes or user forces)
        if deep_validation["status"] in ["PASS", "WARN"] and col_validation["status"] in ["PASS", "WARN"]:
            if st.button("✅ Proceed with this file", type="primary", key="proceed_btn"):
                # Save selection to session state for other pages
                st.session_state["twb_selected_csv_name"] = selected_basename
                st.session_state["twb_selected_csv_path"] = selected_path
                st.session_state["twb_selected_csv_validation"] = {
                    "column_structure": col_validation, 
                    "data_quality": deep_validation
                }
                
                # Add to file history
                if 'twb_file_history' not in st.session_state:
                    st.session_state.twb_file_history = []
                if selected_basename not in st.session_state.twb_file_history:
                    st.session_state.twb_file_history.append(selected_basename)
                
                st.success(f"✅ Selection saved: {selected_basename}")
                st.info("You can now proceed to the next step in the sidebar.")
        else:
            # Allow forced proceed despite validation issues
            force_proceed = st.checkbox(
                "I understand the data quality issues and want to proceed anyway", 
                key="twb_force_proceed"
            )
            if force_proceed and st.button("⛔ Proceed despite errors", key="force_btn"):
                st.session_state["twb_selected_csv_name"] = selected_basename
                st.session_state["twb_selected_csv_path"] = selected_path
                st.session_state["twb_selected_csv_validation"] = {
                    "column_structure": col_validation, 
                    "data_quality": deep_validation
                }
                st.warning(f"⚠ Selection saved with issues: {selected_basename}")
    
    with col_act2:
        # Clear selection button
        if st.button("🗑️ Clear selection", key="clear_btn"):
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
    
    # --- CURRENT SELECTION STATUS ---
    st.markdown("---")
    if "twb_selected_csv_name" in st.session_state:
        st.info(f"**Current selection:** `{st.session_state['twb_selected_csv_name']}`")
        
        # Show recent files history
        if 'twb_file_history' in st.session_state and st.session_state.twb_file_history:
            with st.expander("📋 Recent Files", expanded=False):
                # Show last 5 files used
                recent_files = st.session_state.twb_file_history[-5:]
                for i, file in enumerate(reversed(recent_files), 1):
                    st.write(f"{i}. {file}")