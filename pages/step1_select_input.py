"""
pages/step1_select_input.py

Step 1 page for TradingWorkbench: select OHLC CSV from `input_data/`.
Features:
 - Dropdown listing CSV files in input_data/
 - Refresh list control
 - Upload CSV (saved to input_data/uploads/)
 - Preview first 5 rows
 - Header + deep validation
 - On Proceed stores selection in st.session_state using twb_ prefixes:
     twb_selected_csv_name
     twb_selected_csv_path
     twb_selected_csv_validation
"""
from __future__ import annotations
import streamlit as st
import pandas as pd
import os
import glob
from datetime import datetime

# ---------- Configuration ----------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR = os.path.join(ROOT, "input_data")
UPLOADS_DIR = os.path.join(INPUT_DIR, "uploads")
LARGE_FILE_MB_WARN = 50  # if larger than this skip full validation and show warning
REQUIRED_COLUMNS = {"date", "open", "high", "low", "close", "volume"}
TICKER_CANDIDATES = {"ticker", "symbol", "scrip"}

# ---------- Helpers ----------
def ensure_dirs():
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(UPLOADS_DIR, exist_ok=True)

def list_csv_files():
    ensure_dirs()
    pattern = os.path.join(INPUT_DIR, "*.csv")
    files = glob.glob(pattern)
    files_sorted = sorted(files, key=os.path.getmtime, reverse=True)
    return files_sorted

def human_size(nbytes: int) -> str:
    for unit in ("B","KB","MB","GB"):
        if nbytes < 1024.0:
            return f"{nbytes:.1f}{unit}"
        nbytes /= 1024.0
    return f"{nbytes:.1f}TB"

def save_upload(uploaded):
    ensure_dirs()
    name = uploaded.name
    target = os.path.join(UPLOADS_DIR, name)
    if os.path.exists(target):
        ts = datetime.now().strftime("%Y%m%dT%H%M%S")
        base, ext = os.path.splitext(name)
        name = f"{base}_{ts}{ext}"
        target = os.path.join(UPLOADS_DIR, name)
    with open(target, "wb") as f:
        f.write(uploaded.getbuffer())
    return target

def detect_mapping(cols: list[str]) -> dict:
    lm = {c.lower(): c for c in cols}
    mapping = {}
    for req in REQUIRED_COLUMNS:
        if req in lm:
            mapping[req] = lm[req]
    for cand in TICKER_CANDIDATES:
        if cand in lm:
            mapping["ticker"] = lm[cand]
            break
    if "date" not in mapping:
        for cand in ("date","timestamp","datetime","time"):
            if cand in lm:
                mapping["date"] = lm[cand]
                break
    return mapping

def header_check(path: str) -> dict:
    try:
        df0 = pd.read_csv(path, nrows=0)
        cols = list(df0.columns)
    except Exception as e:
        return {"status":"FAIL","messages":[f"Header read error: {e}"], "cols": [], "mapping": {}}
    mapping = detect_mapping(cols)
    missing = sorted(list(REQUIRED_COLUMNS - set(mapping.keys())))
    if missing:
        return {"status":"FAIL","messages":[f"Missing required columns: {', '.join(missing)}"], "cols": cols, "mapping": mapping}
    return {"status":"PASS","messages":["Required columns present."], "cols": cols, "mapping": mapping}

def deep_validation(path: str, mapping: dict) -> dict:
    size_mb = os.path.getsize(path) / 1_000_000
    if size_mb > LARGE_FILE_MB_WARN:
        return {"status":"WARN", "messages":[f"File is {size_mb:.1f} MB — deep validation skipped. Consider using a sample or allow full validation."], "size_mb": size_mb}
    try:
        date_col = mapping["date"]
        df = pd.read_csv(path, parse_dates=[date_col], infer_datetime_format=True, low_memory=False)
    except Exception as e:
        return {"status":"FAIL", "messages":[f"Parse error: {e}"], "size_mb": size_mb}
    msgs = []
    if pd.api.types.is_datetime64_any_dtype(df[date_col]):
        msgs.append(f"Parsed '{date_col}' as datetime. Rows: {len(df):,}")
    else:
        msgs.append(f"Warning: '{date_col}' not fully parsed as datetime.")
    for col in ("open","high","low","close","volume"):
        act = mapping.get(col)
        if not act:
            msgs.append(f"Mapping missing for '{col}'")
            continue
        coerced = pd.to_numeric(df[act], errors="coerce")
        n_bad = int(coerced.isna().sum())
        pct = 100.0 * n_bad / max(1, len(coerced))
        msgs.append(f"{act}: {n_bad} invalid numeric cells ({pct:.1f}%)")
    if "ticker" in mapping:
        dup = int(df.duplicated(subset=[mapping["date"], mapping["ticker"]]).sum())
        if dup:
            msgs.append(f"Found {dup} duplicate (date,ticker) rows.")
    else:
        dup = int(df.duplicated(subset=[mapping["date"]]).sum())
        if dup:
            msgs.append(f"Found {dup} duplicate date rows.")
    try:
        mn = df[mapping["date"]].min(); mx = df[mapping["date"]].max()
        msgs.append(f"Date range: {mn} -> {mx}")
    except Exception:
        pass
    status = "PASS"
    for m in msgs:
        ml = m.lower()
        if "missing" in ml or "error" in ml or "parse" in ml:
            status = "FAIL"
        if "duplicate" in ml or "invalid" in ml or "warning" in ml:
            if status != "FAIL":
                status = "WARN"
    return {"status": status, "messages": msgs, "size_mb": size_mb}

# ---------- Render ----------
def render():
    st.title("1 — Select OHLC CSV")
    st.write("CSV files are read from `input_data/` relative to the project root.")
    ensure_dirs()

    files = list_csv_files()
    basenames = [os.path.basename(p) for p in files]

    c1, c2, c3 = st.columns([6,2,3])
    with c1:
        if basenames:
            default_idx = 0
            prev = st.session_state.get("twb_selected_csv_name")
            if prev in basenames:
                default_idx = basenames.index(prev)
            selected = st.selectbox("Choose CSV from input_data/", options=basenames, index=default_idx, key="twb_select_dropdown")
            selected_path = os.path.join(INPUT_DIR, selected)
        else:
            st.info("No CSV files in input_data/. Use the Upload control or place CSV files into input_data/ folder.")
            selected = None
            selected_path = None

    with c2:
        if st.button("Refresh list"):
            st.experimental_rerun()

    with c3:
        uploaded = st.file_uploader("Upload CSV (saved to input_data/uploads/)", type=["csv"], key="twb_uploader")
        if uploaded is not None:
            saved = save_upload(uploaded)
            st.success(f"Uploaded and saved to {saved}")
            st.experimental_rerun()

    st.markdown("---")

    if selected_path:
        try:
            st.write("**File:**", selected)
            st.write("**Path:**", selected_path)
            stat = os.stat(selected_path)
            st.write("**Size:**", human_size(stat.st_size), " • **Modified:**", datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"))
        except Exception:
            pass

        st.subheader("Preview (first 5 rows)")
        try:
            preview = pd.read_csv(selected_path, nrows=5)
            st.dataframe(preview)
        except Exception as e:
            st.error(f"Preview failed: {e}")

        st.subheader("Header validation")
        header = header_check(selected_path)
        if header["status"] == "PASS":
            st.success(header["messages"][0])
        else:
            st.error("; ".join(header.get("messages", ["Header check failed"])))

        st.subheader("Deep validation")
        mapping = header.get("mapping", {}) or {}
        deep = deep_validation(selected_path, mapping) if header["status"] == "PASS" else {"status":"FAIL","messages":["Skipping deep validation because header check failed."]}
        if deep["status"] == "PASS":
            st.success("Deep validation: PASS")
            for m in deep.get("messages", []):
                st.write("-", m)
        elif deep["status"] == "WARN":
            st.warning("Deep validation: WARN")
            for m in deep.get("messages", []):
                st.write("-", m)
        else:
            st.error("Deep validation: FAIL")
            for m in deep.get("messages", []):
                st.write("-", m)

        st.markdown("---")
        colp1, colp2 = st.columns([3,1])
        with colp1:
            if deep["status"] == "PASS":
                if st.button("Proceed with this file"):
                    st.session_state["twb_selected_csv_name"] = selected
                    st.session_state["twb_selected_csv_path"] = selected_path
                    st.session_state["twb_selected_csv_validation"] = {"header": header, "deep": deep}
                    st.success(f"Selection saved to session: {selected}")
            else:
                force = st.checkbox("I accept the warnings/errors and want to proceed", key="twb_force_proceed")
                if force and st.button("Proceed despite warnings"):
                    st.session_state["twb_selected_csv_name"] = selected
                    st.session_state["twb_selected_csv_path"] = selected_path
                    st.session_state["twb_selected_csv_validation"] = {"header": header, "deep": deep}
                    st.success(f"Selection saved to session (forced): {selected}")
        with colp2:
            if st.button("Clear selection"):
                for k in ("twb_selected_csv_name","twb_selected_csv_path","twb_selected_csv_validation","twb_select_dropdown"):
                    if k in st.session_state:
                        del st.session_state[k]
                st.experimental_rerun()

    st.markdown("---")
    if "twb_selected_csv_name" in st.session_state:
        st.info(f"Current session selection: `{st.session_state['twb_selected_csv_name']}`")
