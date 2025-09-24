import sys
import os
from pathlib import Path
from pathlib import Path as _Path

# Canonical export directory used by the backend and Streamlit UI
OUTPUT_EXPORT_DIR = os.path.join("output_data", "dashboard_exports")
_Path(OUTPUT_EXPORT_DIR).mkdir(parents=True, exist_ok=True)

# ensure project root is on sys.path (DO NOT add the Class_Implimentation folder itself)
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

# quick import check & diagnostic (safe to keep)
try:
    import importlib
    importlib.import_module('Class_Implimentation.support_files')
except Exception:
    # Silenced non-critical import diagnostics
    pass

import streamlit as st
import subprocess
import pathlib
import glob
import json
import pandas as pd
import plotly.express as px
import importlib
import re

# CONFIG_PATH is used to locate the config file
CONFIG_PATH = Path(__file__).parent / 'support_files' / 'updated_config.py'

def execute_backtest(csv_path: str | None = None):
    """Run Enhanced_stock_trading_V8.py using the venv Python if present; capture UTF-8 logs."""
    try:
        venv_python = pathlib.Path('.venv/Scripts/python.exe')  # Correct Windows venv path
        if not venv_python.exists():
            venv_python = pathlib.Path(sys.executable)

        env = os.environ.copy()
        if csv_path:
            env['BACKTEST_INPUT_CSV'] = csv_path

        result = subprocess.run(
            [str(venv_python), 'Enhanced_stock_trading_V8.py'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            errors='replace',
            env=env
        )

        if result.returncode != 0:
            st.error('❌ Backtest failed. Check logs below.')
            with st.expander('Show backtest error logs'):
                st.code(result.stderr or 'No stderr captured', language='bash')
            return False
        else:
            st.success('✅ Backtest completed successfully.')
            with st.expander('Show backtest logs'):
                st.code(result.stdout or 'No stdout captured', language='bash')
            return True
    except Exception as e:
        st.error(f"⚠️ Failed to run backtest: {e}")
        return False

def run_backtest_command(selected_csv: str):
    """Persist last CSV selection and run the backtest using robust runner."""
    global cfg
    # Try to persist last selection in memory (and config if possible)
    try:
        if isinstance(cfg, dict):
            cfg["last_csv"] = selected_csv
        else:
            setattr(cfg, "last_csv", selected_csv)
    except Exception:
        pass

    ok = execute_backtest(selected_csv)
    if ok:
        st.success("✅ Backtest completed successfully!")
    else:
        st.error("❌ Backtest failed. See logs above.")

# Centralized export directory for all dashboard artifacts
EXPORT_DIR = OUTPUT_EXPORT_DIR
scrips_file = os.path.join(EXPORT_DIR, "backtested_scrips.xlsx")
transactions_file = os.path.join(EXPORT_DIR, "backtested_transactions.xlsx")
json_file = os.path.join(EXPORT_DIR, "trading_data.json")

def read_config_lines():
    return CONFIG_PATH.read_text(encoding='utf-8').splitlines()

def write_config_lines(lines):
    CONFIG_PATH.write_text("\n".join(lines), encoding='utf-8')

def update_config_values(lines, updates):
    """Update simple assignment lines in the config file for keys in updates."""
    pattern = re.compile(r'^(?P<key>\w+)\s*=.*$')
    out = []
    for line in lines:
        m = pattern.match(line)
        if m and (m.group('key') in updates):
            key = m.group('key')
            val = updates[key]
            if val is None:
                out.append(f"{key} = None")
            elif isinstance(val, str):
                # Trim surrounding whitespace first
                trimmed = val.strip()
                # If the trimmed value is quoted with a matching pair of single or double quotes,
                # accept it as-is (preserve existing quoting). Only accept if len>=2 and first==last and both quotes
                if len(trimmed) >= 2 and ((trimmed[0] == trimmed[-1]) and trimmed[0] in ("'", '"')):
                    out.append(f"{key} = {trimmed}")
                else:
                    # Wrap in single quotes and escape any internal single quotes
                    escaped = trimmed.replace("'", "\\'")
                    out.append(f"{key} = '{escaped}'")
            else:
                out.append(f"{key} = {repr(val)}")
        else:
            out.append(line)
    return out

def save_configuration():
    """Persist current sidebar settings to support_files/updated_config.py."""
    global cfg
    # Determine active filter based on selection/market condition
    if selected_filter != "<use market condition>":
        active = selected_filter
    else:
        rf = getattr(cfg, 'get_recommended_filters', None)
        try:
            if callable(rf) and market_condition != "none":
                recs = rf(market_condition) or []
            else:
                recs = []
        except Exception:
            recs = []
        active = recs[0] if recs else getattr(cfg, 'ACTIVE_FILTER', 'filter_ensemble_weighted')

    updates = {
        'ACTIVE_FILTER': active,
        'MIN_HOLDING_PERIOD': int(min_holding),
        'MIN_PROFIT_PERCENTAGE': float(min_profit),
        'MIN_FILTER_SELL_PROFIT': float(min_filter_sell),
        'TRAILING_STOP_PERCENT': float(trailing_stop),
        'SUPPORT_TYPE': support_type,
        'SR_LOOKBACK': int(sr_lookback),
        'SR_TOLERANCE': float(sr_tolerance),
        'ENABLE_DETAILED_LOGGING': bool(enable_detailed),
        'POSITION_SIZING_ENABLED': bool(position_sizing),
        'PORTFOLIO_STRATEGY': portfolio_strategy
    }
    lines = read_config_lines()
    new_lines = update_config_values(lines, updates)
    write_config_lines(new_lines)
    # refresh cfg for display
    cfg = reload_config()
    st.success("Configuration saved to support_files/updated_config.py")

def reload_latest_results():
    """Load latest Excel outputs from output_data/dashboard_exports and preview."""
    # Always read from the canonical export directory
    results_path = OUTPUT_EXPORT_DIR
    scrips_file = f"{results_path}/backtested_scrips.xlsx"
    txns_file   = f"{results_path}/backtested_transactions.xlsx"
    perf_file   = f"{results_path}/portfolio_performance_report.xlsx"

    try:
        scrips_df = pd.read_excel(scrips_file)
        txns_df   = pd.read_excel(txns_file)
        perf_df   = pd.read_excel(perf_file)
        st.success("✅ Latest backtest results loaded successfully!")
        st.dataframe(scrips_df.head())
        st.dataframe(txns_df.head())
        st.dataframe(perf_df.head())
    except FileNotFoundError:
        st.warning("⚠️ Backtest files not found. Running backtest automatically...")
        # Use execute_backtest which is the new safe runner and avoids name collisions
        if execute_backtest():
            st.info("Reloading results after backtest...")
            try:
                scrips_df = pd.read_excel(scrips_file)
                txns_df   = pd.read_excel(txns_file)
                perf_df   = pd.read_excel(perf_file)
                st.success("✅ Latest backtest results loaded successfully!")
                st.dataframe(scrips_df.head())
                st.dataframe(txns_df.head())
                st.dataframe(perf_df.head())
            except Exception as e:
                st.error(f"Failed to load results after backtest: {e}")
        else:
            st.error("Backtest did not complete; cannot load results.")
    except Exception as e:
        st.error(f"Failed to load results: {e}")
def reload_config():
    """
    Load CONFIG_PATH robustly:
     - try JSON
     - else try import of support_files.updated_config
     - else parse simple KEY = VALUE text lines
    Returns a module-like object (module or SimpleNamespace) with attributes for UI use.
    """
    import json, importlib, ast
    from types import SimpleNamespace
    cfg_path = CONFIG_PATH

    # 1) Try JSON
    try:
        with open(cfg_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return SimpleNamespace(**data)
    except Exception:
        pass

    # 2) Try importing as Python module
    try:
        mod_name = 'support_files.updated_config'
        mod = importlib.import_module(mod_name)
        importlib.reload(mod)
        return mod
    except Exception:
        pass

    # 3) Fallback: parse simple KEY = VALUE lines
    try:
        text = Path(cfg_path).read_text(encoding='utf-8')
        cfg = {}
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            k, v = line.split('=', 1)
            k = k.strip()
            v = v.strip()
            try:
                cfg[k] = ast.literal_eval(v)
            except Exception:
                cfg[k] = v.strip('\'"')
        return SimpleNamespace(**cfg)
    except Exception as e:
        raise RuntimeError(f"Unable to load config (tried JSON/module/line-parse): {e}")

def reload_config_ui(with_message: bool = True):
    """UI wrapper: refresh global cfg from disk and optionally inform the user."""
    global cfg
    cfg = reload_config()
    if with_message:
        st.info("♻️ Config reloaded from file (unsaved changes discarded).")

try:
    # preferred arg name for modern Streamlit
    st.set_page_config(page_title="Portfolio Dashboard", layout="wide")
except TypeError:
    # fallback for older Streamlit builds that used 'title'
    st.set_page_config(title="Portfolio Dashboard", layout="wide")

st.title("\U0001F4C1 Portfolio Dashboard \u2014 Filters & Risk Settings")
st.write("Use this UI to choose filters, market condition and risk/exit parameters. Descriptions are shown with each control. Saving updates support_files/updated_config.py (minimal change).")

# Load latest config values
cfg = reload_config()

    # Try loading recently exported Excel artifacts (if present)
try:
    if os.path.exists(scrips_file) and os.path.exists(transactions_file):
        _scrips_df = pd.read_excel(scrips_file)
        _tx_df = pd.read_excel(transactions_file)
        st.sidebar.success(f"Data loaded from {EXPORT_DIR}")
        # Optionally preview small shapes in the sidebar
        st.sidebar.caption(f"Scrips: {_scrips_df.shape}; Tx: {_tx_df.shape}")
except Exception:
    # Silent best-effort; JSON viewer below is the primary path
    pass

with st.sidebar:
    st.header("Mode")
    mode = st.radio("Action", ["Edit & Save Config", "Edit, Save & Run Backtest"], index=0)
    st.markdown("Choose whether to only update config or also run the backtest after saving.")

    st.header("Filter Selection")
    st.markdown("Pick a single active filter or use Market Condition to auto-select a recommended filter.")
    market_condition = st.selectbox("Market Condition (auto-select filter)", 
                                    options=["none", "trending_market", "sideways_market", "volatile_market", "balanced_approach", "conservative", "aggressive"],
                                    index=0,
                                    help="Selecting a market condition will pick recommended filter(s) from the config recommendations.")
    available_filters = sorted(list(cfg.AVAILABLE_FILTERS.keys()))
    selected_filter = st.selectbox("Active Filter (overrides market condition)", options=["<use market condition>"] + available_filters, index=0, help="Choose a specific filter. Leave as '<use market condition>' to follow the Market Condition selection.")
    st.write("")

    st.header("Risk & Exit Parameters")
    min_holding = st.number_input("Minimum Holding Period (days)", value=getattr(cfg, 'MIN_HOLDING_PERIOD', 45), min_value=1, help="Minimum number of days to hold a position before considering exit.")
    min_profit = st.number_input("Minimum Profit Percentage (%)", value=getattr(cfg, 'MIN_PROFIT_PERCENTAGE', 35.0), min_value=0.0, max_value=100.0, step=0.1, help="Minimum percent gain required before selling.")
    min_filter_sell = st.number_input("Min Filter Sell Profit (%)", value=getattr(cfg, 'MIN_FILTER_SELL_PROFIT', 15.0), min_value=0.0, max_value=100.0, step=0.1, help="Profit % required for filter-based sells.")
    trailing_stop = st.number_input("Trailing Stop Percent (%)", value=getattr(cfg, 'TRAILING_STOP_PERCENT', 15.0), min_value=0.0, max_value=100.0, step=0.1, help="Drop from peak to trigger trailing stop exit.")
    st.write("")

    st.header("Support / Resistance Settings")
    support_type = st.selectbox("Support Type", options=["min","single","pivot","fractal"], index=["min","single","pivot","fractal"].index(getattr(cfg,'SUPPORT_TYPE','min')), help="Method used to compute support levels (affects trailing exit).")
    sr_lookback = st.number_input("Support/Resistance Lookback (days)", value=int(getattr(cfg,'SR_LOOKBACK', 60)), min_value=5, step=1, help="Lookback window to compute support/resistance pivots or rolling highs/lows.")
    sr_tolerance = st.number_input("SR Tolerance (%)", value=float(getattr(cfg,'SR_TOLERANCE', 1.5)), min_value=0.0, step=0.1, help="Tolerance band around SR levels used for exit/signals.")

    st.write("")
    st.header("Other Toggles")
    enable_detailed = st.checkbox("Enable Detailed Logging", value=getattr(cfg,'ENABLE_DETAILED_LOGGING', True), help="When enabled, backtest logs include step-by-step arithmetic for the first runs.")
    position_sizing = st.checkbox("Enable Position Sizing", value=getattr(cfg,'POSITION_SIZING_ENABLED', True), help="Enable dynamic position sizing based on volatility / signal strength.")

    # Portfolio Optimisation strategy selector
    st.header("Portfolio Optimisation")
    _opts = ["score_rank_claude", "legacy_risk_weighted"]
    _default_strategy = getattr(cfg, 'PORTFOLIO_STRATEGY', 'score_rank_claude')
    _idx = _opts.index(_default_strategy) if _default_strategy in _opts else 0
    portfolio_strategy = st.selectbox(
        "Allocation Strategy",
        options=_opts,
        index=_idx,
        help="Choose how to allocate funds among filtered stocks."
    )

    if st.button("Save Configuration"):
        # Determine active filter to write
        if selected_filter != "<use market condition>":
            active = selected_filter
        else:
            # Safely obtain recommended filters from cfg (cfg may be SimpleNamespace without the method)
            rf = getattr(cfg, 'get_recommended_filters', None)
            try:
                if callable(rf) and market_condition != "none":
                    recs = rf(market_condition) or []
                else:
                    recs = []
            except Exception:
                recs = []
            if recs:
                active = recs[0]
            else:
                active = getattr(cfg, 'ACTIVE_FILTER', 'filter_ensemble_weighted')

        updates = {
            'ACTIVE_FILTER': active,
            'MIN_HOLDING_PERIOD': int(min_holding),
            'MIN_PROFIT_PERCENTAGE': float(min_profit),
            'MIN_FILTER_SELL_PROFIT': float(min_filter_sell),
            'TRAILING_STOP_PERCENT': float(trailing_stop),
            'SUPPORT_TYPE': support_type,
            'SR_LOOKBACK': int(sr_lookback),
            'SR_TOLERANCE': float(sr_tolerance),
            'ENABLE_DETAILED_LOGGING': bool(enable_detailed),
            'POSITION_SIZING_ENABLED': bool(position_sizing),
            'PORTFOLIO_STRATEGY': portfolio_strategy
        }
        lines = read_config_lines()
        new_lines = update_config_values(lines, updates)
        write_config_lines(new_lines)
        st.success("Configuration saved to support_files/updated_config.py")
        cfg = reload_config()

    if st.button("Reload Config (without saving)"):
        cfg = reload_config()
        st.success("Configuration reloaded")

    # Show current optimiser strategy from config
    current_strategy = getattr(cfg, 'PORTFOLIO_STRATEGY', 'score_rank_claude')
    st.sidebar.markdown(f"**Current Portfolio Strategy:** `{current_strategy}`")

st.header("Overview & Descriptions")
st.markdown("""
- Active Filter: The selected filter defines signals (Buy / Sell) used by the backtester.
- Market Condition: quick presets mapping to recommended filters (see config recommendations).
- Support Type: method to compute support/resistance. 'min' = local minima, 'pivot' = pivot points, 'fractal' = fractal detection.
- SR Lookback/Tolerance: tune sensitivity of SR levels.
- All changes are written to support_files/updated_config.py to keep minimal alterations to core code.
""")

st.subheader("Current Effective Configuration")
cfg = reload_config()
col1, col2 = st.columns(2)
with col1:
    st.write("Active Filter")
    _active = getattr(cfg, 'ACTIVE_FILTER', None)
    # If the active value is JSON-serializable (dict/list), show as JSON; otherwise show as text/code
    if isinstance(_active, (dict, list)):
        st.json(_active)
    else:
        st.code(str(_active) if _active is not None else "N/A")
    st.write("Market Filter Recommendations")
    st.json(getattr(cfg, 'FILTER_RECOMMENDATIONS', {}) )
with col2:
    st.write("Key Risk Params")
    st.write({
        'MIN_HOLDING_PERIOD': getattr(cfg,'MIN_HOLDING_PERIOD',None),
        'MIN_PROFIT_PERCENTAGE': getattr(cfg,'MIN_PROFIT_PERCENTAGE',None),
        'TRAILING_STOP_PERCENT': getattr(cfg,'TRAILING_STOP_PERCENT',None),
        'SUPPORT_TYPE': getattr(cfg,'SUPPORT_TYPE',None)
    })

run_backtest = False
if mode == "Edit, Save & Run Backtest":
    run_backtest = st.checkbox("Also run backtest now (may take time)", value=False)

if st.button("Run Backtest Now") or (mode=="Edit, Save & Run Backtest" and run_backtest):
    st.info("Running backtest using current config. Output will be printed to console and log file as before.")
    try:
        # minimal integration: import the class and run
        from Enhanced_stock_trading_V8 import FilteringAndBacktesting
        import support_files.File_IO as fio
        # read inputs from input_data inside this package directory (do not hardcode repo root)
        repo_dir = Path(__file__).resolve().parent
        # ensure subsequent relative calls that expect 'input_data' refer to this folder
        os.chdir(repo_dir)
        input_dir = repo_dir / 'input_data'
        if not input_dir.exists():
            st.error(f"Input folder not found: {input_dir}. Create and place your CSVs there.")
            raise FileNotFoundError(f"Input folder not found: {input_dir}")
        master_df = fio.read_csv_to_df('Nif50_5y_1w.csv', 'A', 'input_data')
        # Pass initial cash from config (default 100000 if not set)
        fb = FilteringAndBacktesting(initial_cash=getattr(cfg, 'INITIAL_CASH', 100000.0))
        # run with dashboard creation disabled to keep it headless here
        backtested_scrips_df, backtested_tx_df, dashboard = fb.run(master_df, create_dashboard=False)
        st.success("Backtest finished (check log files and gain_details).")
        st.write("Backtested scrips (preview):")
        st.dataframe(backtested_scrips_df.head())
        st.write("Transactions (preview):")
        st.dataframe(backtested_tx_df.head())
    except Exception as e:
        st.error(f"Backtest failed: {e}")

st.markdown("---")
st.caption("This dashboard writes minimal changes to support_files/updated_config.py and does not alter core backtest logic. If you prefer not to run long backtests from the UI, uncheck the run option.")

# =============================
# Choose Input CSV & Run Backtest
# =============================
def list_csv_files(input_dir="input_data"):
    """Return list of available CSV files in input_data folder."""
    return [os.path.basename(f) for f in glob.glob(os.path.join(input_dir, "*.csv"))]

st.header("\U0001F5C3 Choose Input CSV")
csv_files = list_csv_files()
if not csv_files:
    st.error("⚠️ No CSV files found in input_data/ folder")
else:
    if "last_csv" not in st.session_state:
        st.session_state.last_csv = csv_files[0]
    selected_csv = st.selectbox(
        "Choose Input CSV File",
        csv_files,
        index=csv_files.index(st.session_state.last_csv) if st.session_state.last_csv in csv_files else 0
    )
    st.session_state.last_csv = selected_csv
    # Button to trigger backtest safely (avoid shadowing function name)
    clicked = st.button("⚡ Run Backtest for Selected CSV")
    if clicked:
        csv_path = os.path.join("input_data", selected_csv)
        st.info(f"🔄 Running backtest for: {selected_csv}")
        try:
            run_backtest_command(csv_path)
        except Exception as e:
            st.error(f"❌ Backtest failed: {e}")

# =============================
# Reload Latest Results (Excel)
# =============================
st.header("\U0001F504 Reload Latest Results")
st.write("Reload the most recent backtest outputs from output_data/dashboard_exports.")
reload_btn = st.button("Reload Latest Results")
if reload_btn:
    try:
        reload_latest_results()
    except Exception as e:
        st.error(f"❌ Reload failed: {e}")

# ==========================
# Trading Data Viewer (JSON)
# ==========================
st.header("\U0001F4C8 Trading Data Viewer")
st.write("Loads OHLC data exported to output_data/dashboard_exports/trading_data.json and plots per-stock series.")

def _to_df_from_stock_rows(obj: dict) -> pd.DataFrame:
    """Schema A: { stock: [ {date, open, high, low, close, volume, ...}, ... ], ... }"""
    records = []
    for stock, rows in (obj or {}).items():
        if not isinstance(rows, list):
            continue
        for r in rows:
            if not isinstance(r, dict):
                continue
            rec = {
                'Stock': stock,
                'Date': r.get('Date') or r.get('date'),
                'Open': r.get('Open', r.get('open')),
                'High': r.get('High', r.get('high')),
                'Low': r.get('Low', r.get('low')),
                'Close': r.get('Close', r.get('close')),
                'Volume': r.get('Volume', r.get('volume')),
            }
            records.append(rec)
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame.from_records(records)
    # Normalize dates to string YYYY-MM-DD
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.strftime('%Y-%m-%d')
    return df

def _to_df_from_plotly(obj: dict) -> pd.DataFrame:
    """Schema B: { stock: { data: [ {type:candlestick, x, open, high, low, close, ...} ], layout: {...} }, ... }"""
    frames = []
    for stock, chart in (obj or {}).items():
        try:
            traces = (chart or {}).get('data') or []
            trace = traces[0] if traces else None
            if not trace:
                continue
            x = trace.get('x') or []
            o = trace.get('open') or []
            h = trace.get('high') or []
            l = trace.get('low') or []
            c = trace.get('close') or []
            n = min(len(x), len(o), len(h), len(l), len(c))
            if n == 0:
                continue
            df = pd.DataFrame({
                'Stock': [stock]*n,
                'Date': pd.to_datetime(x[:n], errors='coerce').strftime('%Y-%m-%d'),
                'Open': o[:n],
                'High': h[:n],
                'Low': l[:n],
                'Close': c[:n],
            })
            frames.append(df)
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)

json_path = json_file
if not os.path.exists(json_path):
    st.info("No trading_data.json found yet. Run the backtest to export it (see console instructions).")
else:
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            trading_obj = json.load(f)
        # Try schema A first, then schema B
        data_df = _to_df_from_stock_rows(trading_obj)
        if data_df.empty:
            data_df = _to_df_from_plotly(trading_obj)

        if data_df.empty:
            st.warning("Could not parse trading_data.json into a table. Check file format.")
        else:
            stocks = sorted(data_df['Stock'].dropna().unique().tolist())
            if not stocks:
                st.warning("No stocks found in trading data.")
            else:
                colA, colB = st.columns([1, 3])
                with colA:
                    sel = st.selectbox("Select Stock", stocks)
                    st.metric("Total Records", len(data_df))
                    st.metric("Unique Stocks", len(stocks))
                with colB:
                    st.subheader(f"\U0001F4C8 {sel} Close Price")
                    sub = data_df[data_df['Stock'] == sel].copy()
                    sub = sub.sort_values('Date')
                    fig = px.line(sub, x='Date', y='Close', title=f"{sel} - Close Price")
                    st.plotly_chart(fig, use_container_width=True)

                # Optional indicators view if present
                indicator_cols = [c for c in ['RSI', 'MACD', 'Signal'] if c in sub.columns]
                if indicator_cols:
                    st.subheader(f"Indicators - {sel}")
                    try:
                        fig2 = px.line(sub, x='Date', y=indicator_cols, title=f"{sel} - Indicators ({', '.join(indicator_cols)})")
                        st.plotly_chart(fig2, use_container_width=True)
                    except Exception:
                        # Fallback to simple line charts
                        st.line_chart(sub.set_index('Date')[indicator_cols])

            with st.expander("Preview Data", expanded=False):
                st.dataframe(data_df.head(200), use_container_width=True)
    except Exception as e:
        st.error(f"Failed to load trading_data.json: {e}")
