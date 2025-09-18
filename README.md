# TradingWorkbench — Stepwise App (Step 1)

These new files implement **Step 1** (Select OHLC CSV) for your TradingWorkbench project,
without changing any existing project files.

Files to create:
- `tradingworkbench_main.py` — app entry (place at project root)
- `pages/step1_select_input.py` — Step 1 page (place inside `pages/` folder)
- `input_data/` will be created automatically (place CSVs here)
- `input_data/uploads/` receives uploaded CSVs

## How to run

1. Place `tradingworkbench_main.py` in your project root.
2. Create a `pages/` folder and save `step1_select_input.py` inside it.
3. (Optional) Put CSV files into `input_data/` (same level as `tradingworkbench_main.py`).
4. Install dependencies: `pip install streamlit pandas`
5. Run: `streamlit run tradingworkbench_main.py`

## What this does
- Shows dropdown listing `*.csv` in `input_data/`.
- Allows uploading CSV to `input_data/uploads/`.
- Shows a 5-row preview and performs header & deep validation.
- On Proceed stores selection in `st.session_state` keys:
  - `twb_selected_csv_name`
  - `twb_selected_csv_path`
  - `twb_selected_csv_validation`

## Next steps (we will add these as new pages/files)
- Step 2: Fundamental filter (consume `twb_selected_csv_path`)
- Step 3: Technical indicators and caching
- Step 4: Signal generation & allocation
- Step 5: Backtest & reports (single centralized exporter)

If you want, I can now create **Step 2** as a new page that consumes the selected CSV and
calculates a simple indicator (e.g., 20-period SMA) to demonstrate downstream usage.
