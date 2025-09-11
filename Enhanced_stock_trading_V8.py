import sys, io, os
try:
    # Preferred (Python 3.7+): ensure stdout/stderr use UTF-8
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    # Fallback for older Python builds
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# ensure subprocesses/child processes inherit UTF-8 I/O where possible
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
os.environ.setdefault('PYTHONUTF8', '1')

import os
from pathlib import Path
import support_files.updated_config as config
import pandas as pd
import numpy as np
from support_files.scrip_extractor import scrip_extractor
import support_files.compute_indicators_helper as cmp  # For computing technical indicators
from dashboard_integration import TradingDashboard
from portfolio_optimiser import optimize_portfolio


# Trade constraint constants
MIN_PROFIT_PERCENTAGE = config.MIN_PROFIT_PERCENTAGE
MIN_HOLDING_PERIOD = config.MIN_HOLDING_PERIOD


class FilteringAndBacktesting:
    def __init__(self, initial_cash=100000.0):
        self.initial_cash = initial_cash
        self.backtested_scrip_df_list = []
        self.backtested_transactions_df_list = []
        self.portfolio_value = 0.0
        self.stock_allocations = {}  # Store allocation amounts for each stock
        self.stock_scores = {}  # Store scoring data for each stock
        # Control verbose detailed prints: show full calculation details once (at start)
        self._detailed_print_shown = False

    # --------------------- PORTFOLIO ALLOCATION METHODS ---------------------     
    
    def allocate_portfolio(self, prices_df, signals_df=None, fund_df=None,
                        method="legacy_risk_weighted", constraints=None, config=None):
        """
        Unified allocator: call any strategy in portfolio_optimiser (including the new legacy_risk_weighted).
        Expects prices_df (and optional signals/fund_df) already sliced to filtered tickers.
        Returns a weights Series (sum≈1).
        """
        constraints = constraints or {"long_only": True, "max_weight": 0.15, "budget": 1.0}
        config = config or {"lookback_days": 252, "cov_method": "ledoit_wolf_simple"}

        return optimize_portfolio(
            prices=prices_df,
            signals=signals_df,
            fundamentals=fund_df,
            method=method,
            constraints=constraints,
            config=config,
        )


    # --------------------- FILTERING METHODS ---------------------
    def apply_filter(self, master_df):
        """
        Computes technical indicators and applies the configured filter on the master OHLC
        dataframe. Then performs intelligent portfolio allocation.
        Returns the filtered dataframe (filtered_scrips_df).
        """
        scrips_list = []
        filtered_ta_df_list = []
        master_scrips_list = master_df['Stock'].unique()

        print("\n" + "="*60)
        print("STOCK SCREENING & FILTERING".center(60))
        print("="*60)
        print(f"Universe Size: {len(master_scrips_list)} securities")
        print(f"Screening Filter: {config.ACTIVE_FILTER}")

        for scrip, scrip_df in scrip_extractor(master_df):
            scrip_ta_df = cmp.compute_indicators(scrip_df)
            # Ensure the 'Stock' column is present and correct for this slice
            try:
                scrip_ta_df['Stock'] = scrip
            except Exception:
                scrip_ta_df = scrip_ta_df.copy()
                scrip_ta_df['Stock'] = scrip
            # Ensure 'Date' is a column (reset from index if needed) before concatenation
            if 'Date' not in scrip_ta_df.columns:
                scrip_ta_df = scrip_ta_df.reset_index()
                if 'Date' not in scrip_ta_df.columns and 'index' in scrip_ta_df.columns:
                    scrip_ta_df = scrip_ta_df.rename(columns={'index': 'Date'})

            if config.FILTER_ENABLED:
                filter_func = config.AVAILABLE_FILTERS.get(config.ACTIVE_FILTER)
                buy_signal, sell_signal = filter_func(scrip_ta_df)
                scrip_ta_df['Buy'] = buy_signal
                scrip_ta_df['Sell'] = sell_signal

                if (buy_signal.sum() >= 2) and (sell_signal.sum() >= 2):
                    filtered_ta_df_list.append(scrip_ta_df)
                    scrips_list.append(scrip)
                else:
                    continue
            else:
                filtered_ta_df_list.append(scrip_ta_df)
                scrips_list.append(scrip)

        # ✅ Safer: attach Stock as column before concat
        try:
            if filtered_ta_df_list:
                for i, (scrip, df) in enumerate(zip(scrips_list, filtered_ta_df_list)):
                    _df = df.copy()
                    _df["Stock"] = scrip
                    filtered_ta_df_list[i] = _df
                # Safer concatenation with error handling
                try:
                    filtered_ta_df = pd.concat(filtered_ta_df_list, ignore_index=True)
                except Exception as concat_error:
                    print(f"Concatenation error: {concat_error}")
                    # Create minimal DataFrame with required columns to prevent downstream errors
                    filtered_ta_df = pd.DataFrame(columns=['Stock', 'Date', 'Open', 'High', 'Low', 'Close', 'Buy', 'Sell'])
            else:
                # Create empty DataFrame with required columns instead of completely empty
                filtered_ta_df = pd.DataFrame(columns=['Stock', 'Date', 'Open', 'High', 'Low', 'Close', 'Buy', 'Sell'])
        except Exception as e:
            print(f"Error in screening process: {e}")
            # Ensure fallback DataFrame has required columns
            filtered_ta_df = pd.DataFrame(columns=['Stock', 'Date', 'Open', 'High', 'Low', 'Close', 'Buy', 'Sell'])

        print(f"Securities Passed Screening: {len(scrips_list)}")
        print(f"Filter Success Rate: {len(scrips_list)/len(master_scrips_list)*100:.1f}%")

        # Additional safety check before returning
        if filtered_ta_df.empty:
            print("⚠️ Warning: No securities passed the screening filter.")
            return pd.DataFrame(columns=['Stock', 'Date', 'Open', 'High', 'Low', 'Close', 'Buy', 'Sell'])

        if not filtered_ta_df.empty:
            # --- NEW MODE: optimiser does all scoring (score_rank_claude) ---

            # Normalize column names to avoid hidden whitespace or duplicates
            filtered_ta_df.columns = [str(c).strip() for c in filtered_ta_df.columns]
            fdf = filtered_ta_df.copy()
            fdf.columns = [str(c).strip() for c in fdf.columns]

            # Debug print removed for cleaner logs
            # Normalize date/ticker column names to avoid KeyError when sources differ
            date_candidates = [
                "Date", "date", "DATE", "Datetime", "datetime", "Timestamp", "timestamp"
            ]
            stock_candidates = ["Stock", "Ticker", "Symbol", "SYMBOL", "ticker", "stock"]
            # Also detect by normalized names (strip/lower)
            _date_syns = {"date", "datetime", "timestamp"}
            _stock_syns = {"stock", "ticker", "symbol"}
            # Prefer existing exact matches first
            if not any(c in fdf.columns for c in stock_candidates):
                _alt = [c for c in fdf.columns if isinstance(c, str) and c.strip().lower() in _stock_syns]
                if _alt:
                    fdf = fdf.rename(columns={_alt[0]: "Stock"})

            date_col = next((c for c in date_candidates if c in fdf.columns), None)
            stock_col = next((c for c in stock_candidates if c in fdf.columns), None)

            # If no explicit date column, try using the index if it looks like dates
            if date_col is None:
                if isinstance(fdf.index, pd.DatetimeIndex) or pd.api.types.is_datetime64_any_dtype(fdf.index):
                    fdf = fdf.reset_index().rename(columns={fdf.columns[0]: "Date"})
                    date_col = "Date"
                else:
                    # As a last resort, leave date_col as None and we will raise a clearer error below
                    pass

            if date_col is None:
                raise KeyError("Input data is missing a date column. Expected one of: "
                               f"{date_candidates} or a DatetimeIndex.")
            if stock_col is None:
                raise KeyError("Input data is missing a stock/ticker column. Expected one of: "
                               f"{stock_candidates}.")

            # Ensure canonical column names
            if date_col != "Date":
                fdf = fdf.rename(columns={date_col: "Date"})
            if stock_col != "Stock":
                fdf = fdf.rename(columns={stock_col: "Stock"})

            # If duplicates of 'Date' or 'Stock' exist (e.g., both 'DATE' and 'Date' renamed), coalesce to one safely
            def _coalesce_same_named(df: pd.DataFrame, col_name: str) -> pd.DataFrame:
                try:
                    mask = (df.columns == col_name)
                    if mask.sum() > 1:
                        subset = df.loc[:, mask]
                        try:
                            merged = subset.bfill(axis=1).iloc[:, 0]
                        except Exception:
                            merged = subset.iloc[:, 0]
                        # Only drop after successful merge
                        new_df = df.drop(columns=list(subset.columns))
                        new_df[col_name] = merged
                        return new_df
                except Exception:
                    return df
                return df
            fdf = _coalesce_same_named(fdf, 'Date')
            fdf = _coalesce_same_named(fdf, 'Stock')

            # Ensure 'Stock' is a proper column (not hidden in the index)
            if 'Stock' not in fdf.columns:
                # If index name looks like a stock identifier, reset it to a column
                idx_name = fdf.index.name
                cand_set = {s.lower() for s in (stock_candidates + ["Stock"]) }
                if idx_name and idx_name.lower() in cand_set:
                    fdf = fdf.reset_index().rename(columns={idx_name: 'Stock'})
                elif isinstance(fdf.index, pd.MultiIndex):
                    for lvl_name in fdf.index.names:
                        if lvl_name and lvl_name.lower() in cand_set:
                            fdf = fdf.reset_index(level=lvl_name).rename(columns={lvl_name: 'Stock'})
                            break

            # Clean and sort
            # Handle cases where 'Date' is provided as a dict-like mapping (e.g., {'year','month','day'})
            if isinstance(fdf.get("Date"), (dict,)):
                # Not expected in our data; skip and let error surface if needed
                pass
            try:
                fdf["Date"] = pd.to_datetime(fdf["Date"], errors="coerce")
            except Exception:
                # If 'Date' is a DataFrame-like (multiple date parts columns), try to reduce
                if hasattr(fdf["Date"], 'columns'):
                    parts = fdf["Date"]
                    for keys in (("year","month","day"), ("Year","Month","Day")):
                        if all(k in parts.columns for k in keys):
                            fdf["Date"] = pd.to_datetime(dict(year=parts[keys[0]], month=parts[keys[1]], day=parts[keys[2]]), errors="coerce")
                            break
                else:
                    raise
            # Drop any rows lacking Date or Stock, then de-duplicate and sort
            missing_cols = [c for c in ["Date", "Stock"] if c not in fdf.columns]
            if missing_cols:
                # Diagnostic context for easier debugging
                raise KeyError(
                    "Required columns missing for sorting: "
                    f"{missing_cols}. Available columns: {list(fdf.columns)}; "
                    f"index names: {getattr(fdf.index, 'names', fdf.index.name)}"
                )
            fdf = (
                fdf.dropna(subset=["Date", "Stock"])\
                   .sort_values(["Date", "Stock"])\
                   .drop_duplicates(subset=["Date","Stock"], keep="last")
            )
            tickers = sorted(fdf["Stock"].unique())

            # 1) Prices (wide: dates × tickers)
            # Use pivot_table with aggfunc='last' to tolerate accidental duplicate rows
            prices_df = fdf.pivot_table(index="Date", columns="Stock", values="Close", aggfunc="last").sort_index()

            # 2) Optional feature frames (avoid recomputing inside the strategy)
            def pivot_or_none(col):
                return (fdf.pivot_table(index="Date", columns="Stock", values=col, aggfunc="last").sort_index()
                        if col in fdf.columns else None)

            feature_frames = {"Close": prices_df}
            rsi   = pivot_or_none("RSI")
            macd  = pivot_or_none("MACD")
            sma20 = pivot_or_none("SMA_20")
            buy   = pivot_or_none("Buy")
            sell  = pivot_or_none("Sell")
            if rsi   is not None: feature_frames["RSI"]    = rsi
            if macd  is not None: feature_frames["MACD"]   = macd
            if sma20 is not None: feature_frames["SMA_20"] = sma20
            if buy   is not None: feature_frames["Buy"]    = buy.fillna(0)
            if sell  is not None: feature_frames["Sell"]   = sell.fillna(0)

            # 3) Optional fundamentals (latest per ticker)
            fund_df = None
            fund_cols = [c for c in ["P/E","EPS","ROE","OperatingMargin","ReturnOnCapital",
                                    "EarningsStability","EBIT_EV","FCF_Yield","Leverage"] if c in fdf.columns]
            if fund_cols:
                fund_df = (
                    fdf.sort_values(["Stock","Date"]).groupby("Stock")[fund_cols].last().reindex(tickers)
                )
                if "P/E" in fund_df.columns:
                    fund_df = fund_df.rename(columns={"P/E": "PE"})  # optimiser expects "PE"

            # 4) Strategy + knobs (read from config; default to score_rank_claude)
            method = getattr(config, 'PORTFOLIO_STRATEGY', 'score_rank_claude')
            constraints = {"long_only": True, "max_weight": 0.15, "budget": 1.0}
            optimiser_config = {
                "lookback_days": 252,
                "cov_method": "ledoit_wolf_simple",
                "decay_rate": 0.92,               # lower → more weight to top ranks
                "feature_frames": feature_frames, # give RSI/MACD/SMA/Buy/Sell/Close to strategy
                "use_log_returns": True
            }

            # 5) Run optimiser → weights; store ₹ allocations for your backtester
            print(f"🔧 Using optimiser strategy: {method}")
            self.active_strategy = method  # store for reporting
            weights = self.allocate_portfolio(
                prices_df=prices_df[tickers],
                signals_df=None,
                fund_df=fund_df,                  # can be None
                method=method,
                constraints=constraints,
                config=optimiser_config,
            )

            # Convert weights to rupee allocations (what your backtester uses)
            self.stock_allocations = {
                t: float(weights.get(t, 0.0)) * float(self.initial_cash) for t in tickers
            }

        return filtered_ta_df


    # --------------------- BACKTESTING METHODS ---------------------
    def calculate_fee(self, trade_value):
        """Returns the broker fee: ₹20 or 2.5% of trade value (whichever is lower)."""
        fee_percent = 0.025 * trade_value
        fixed_fee = 20.0
        return min(fixed_fee, fee_percent)

    def apply_backtest_strategy(self, filtered_scrip_df, scrip, buy_signal, sell_signal):
        """
        Backtests a trading strategy on a filtered scrip dataframe using allocated cash.
        Returns a tuple: (backtested scrip dataframe, transactions dataframe)
        Uses standard trading terminology in logs.
        """
        df_bt = filtered_scrip_df.copy()
        allocated_capital = self.stock_allocations.get(
            scrip,
            self.initial_cash / len(self.stock_allocations) if self.stock_allocations else self.initial_cash,
        )
        available_cash = float(allocated_capital)

        print("\n" + "=" * 70)
        print(f"BACKTESTING: {scrip}".center(70))
        print("=" * 70)

        # Print allocation info concisely if detailed already shown, otherwise verbose
        if not self._detailed_print_shown:
            print("Capital Allocation Details:")
            if scrip in self.stock_allocations:
                print(f"  Allocated Capital: ₹{self.stock_allocations[scrip]:,.2f}")
            else:
                fallback = self.initial_cash / len(self.stock_allocations) if self.stock_allocations else self.initial_cash
                print(f"  Default Allocation: ₹{fallback:,.2f} (equal weight fallback)")
            print(f"  Starting Cash Available: ₹{allocated_capital:,.2f}")
        else:
            allocation_pct = (allocated_capital / self.initial_cash) * 100.0 if self.initial_cash else 0.0
            print(f"Allocated Capital: ₹{allocated_capital:,.2f} ({allocation_pct:.1f}% of portfolio)")

        position_qty = 0
        portfolio_values: list[float] = []
        positions: list[int] = []
        entry_date = None
        entry_price = None
        trade_status = 'NO POSITION'
        transactions: list[dict] = []

        if 'P/E' in df_bt.columns and 'EPS' in df_bt.columns:
            print(f"Fundamental Data - P/E: {df_bt['P/E'].iloc[-1]:.2f} | EPS: ₹{df_bt['EPS'].iloc[-1]:.2f}")

        if scrip in self.stock_scores:
            score_info = self.stock_scores[scrip]
            print(f"Investment Score: {score_info['composite_score']:.1f}/100")
            print(f"  • Technical Analysis: {score_info['technical_score']:.1f}/100")
            print(f"  • Signal Quality: {score_info['signal_score']:.1f}/100")
            print(f"  • Price Momentum: {score_info['momentum_score']:.1f}/100")
            print(f"  • Risk Assessment: {score_info['volatility_score']:.1f}/100")
            allocation_pct = (allocated_capital / self.initial_cash) * 100.0 if self.initial_cash else 0.0
            print(f"Portfolio Weight: {allocation_pct:.1f}%")

        print("\nTRADING ACTIVITY LOG:")
        print("-" * 70)

        def _fmt_date(val):
            """Format index or value safely as dd-MMM-YYYY string."""
            if hasattr(val, "strftime"):
                return val.strftime("%d-%b-%Y")
            return str(val)

        for idx, row in df_bt.iterrows():
            market_price = float(row['Close'])
            stock_name = row['Stock']
            trade_status = 'NO ACTION'

            # Ensure we always use datetime for trade dates
            trade_date = None
            if 'Date' in row and pd.notna(row['Date']):
                try:
                    trade_date = pd.to_datetime(row['Date'])
                except Exception:
                    trade_date = None

            # fallback: if idx is Timestamp already
            if trade_date is None and isinstance(idx, pd.Timestamp):
                trade_date = idx

            # final fallback: try to coerce idx to datetime; else keep as-is
            if trade_date is None:
                try:
                    trade_date = pd.to_datetime(idx)
                except Exception:
                    trade_date = idx

            trade_date_str = _fmt_date(trade_date)
            # ISO date for data outputs
            def _fmt_iso(val, fallback_idx):
                try:
                    ts = pd.to_datetime(val, errors='coerce')
                    if pd.notna(ts):
                        return ts.strftime('%Y-%m-%d')
                except Exception:
                    pass
                # Fallback to index if provided
                try:
                    ts2 = pd.to_datetime(fallback_idx, errors='coerce')
                    if pd.notna(ts2):
                        return ts2.strftime('%Y-%m-%d')
                except Exception:
                    pass
                return str(val) if val is not None else str(fallback_idx)

            trade_date_iso = _fmt_iso(trade_date, idx)

            # BUY LOGIC
            if buy_signal.loc[idx] and position_qty == 0:
                shares_affordable = int(available_cash // market_price)
                if shares_affordable > 0:
                    gross_cost = shares_affordable * market_price
                    brokerage = self.calculate_fee(gross_cost)
                    total_investment = gross_cost + brokerage

                    if available_cash >= total_investment:
                        position_qty = shares_affordable
                        available_cash -= total_investment
                        entry_date = trade_date
                        entry_price = market_price
                        trade_status = 'LONG ENTRY'

                        # Record BUY transaction (include Revenue as NaN for schema consistency)
                        transactions.append({
                            'Date': trade_date_iso,
                            'Event': 'BUY',
                            'Stock': stock_name,
                            'Price': round(market_price, 2),
                            'Shares': position_qty,
                            'Cost': round(gross_cost, 2),   # for capital deployed calc
                            'Revenue': np.nan,
                            'Fee': round(brokerage, 2),
                            'Cash_After': round(available_cash, 2),
                            'Position_After': position_qty,
                            'Holding_Period': 0,
                            'Profit_%': 0.0,
                            'Allocated_Cash': round(allocated_capital, 2),
                        })

                        if not self._detailed_print_shown:
                            print(f"\n📈 LONG ENTRY - {trade_date_str}")
                            print(f"   Security: {stock_name}")
                            print(f"   Entry Price: ₹{market_price:.2f}")
                            print(f"   Quantity: {position_qty:,} shares")
                            print(f"   Gross Investment: ₹{gross_cost:,.2f}")
                            print(f"   Brokerage: ₹{brokerage:.2f}")
                            print(f"   Total Investment: ₹{total_investment:,.2f}")
                            print(f"   Cash Remaining: ₹{available_cash:,.2f}")
                        else:
                            print(
                                f"📈 {trade_date_str} | LONG ENTRY | {position_qty:,} shares @ ₹{market_price:.2f} | Cash: ₹{available_cash:,.0f}"
                            )
                else:
                    print(f"⚠️ {trade_date_str} | Insufficient funds for {stock_name} @ ₹{market_price:.2f}")

            # SELL LOGIC
            elif sell_signal.loc[idx] and position_qty > 0:
                if entry_date is not None and entry_price is not None:
                    holding_period = (trade_date - entry_date).days
                    unrealized_pnl_pct = ((market_price - entry_price) / entry_price) * 100.0
                else:
                    holding_period = 0
                    unrealized_pnl_pct = 0.0

                # Check exit conditions
                if holding_period >= MIN_HOLDING_PERIOD and unrealized_pnl_pct >= MIN_PROFIT_PERCENTAGE:
                    gross_proceeds = position_qty * market_price
                    brokerage = self.calculate_fee(gross_proceeds)
                    net_proceeds = gross_proceeds - brokerage
                    available_cash += net_proceeds

                    # Calculate realized P&L
                    total_cost = position_qty * entry_price + self.calculate_fee(position_qty * entry_price)
                    realized_pnl = net_proceeds - total_cost
                    realized_pnl_pct = (realized_pnl / total_cost) * 100.0

                    trade_status = 'LONG EXIT'

                    if not self._detailed_print_shown:
                        print(f"\n📉 LONG EXIT - {trade_date_str}")
                        print(f"   Security: {stock_name}")
                        print(f"   Exit Price: ₹{market_price:.2f}")
                        print(f"   Quantity: {position_qty:,} shares")
                        print(f"   Holding Period: {holding_period} days")
                        print(f"   Gross Proceeds: ₹{gross_proceeds:,.2f}")
                        print(f"   Brokerage: ₹{brokerage:.2f}")
                        print(f"   Net Proceeds: ₹{net_proceeds:,.2f}")
                        print(f"   Realized P&L: ₹{realized_pnl:,.2f} ({realized_pnl_pct:+.2f}%)")
                        print(f"   Total Cash: ₹{available_cash:,.2f}")
                    else:
                        print(
                            f"📉 {trade_date_str} | LONG EXIT | {position_qty:,} shares @ ₹{market_price:.2f} | P&L: ₹{realized_pnl:+.0f} ({realized_pnl_pct:+.1f}%) | Cash: ₹{available_cash:,.0f}"
                        )

                    transactions.append({
                        'Date': trade_date_iso,
                        'Event': 'SELL',
                        'Stock': stock_name,
                        'Price': round(market_price, 2),
                        'Shares': position_qty,
                        'Cost': np.nan,
                        'Revenue': round(gross_proceeds, 2),   # for capital deployed calc
                        'Fee': round(brokerage, 2),
                        'Cash_After': round(available_cash, 2),
                        'Position_After': 0,
                        'Holding_Period': holding_period,
                        'Profit_%': round(unrealized_pnl_pct, 2),
                        'Allocated_Cash': round(allocated_capital, 2),
                    })

                    position_qty = 0
                    entry_date = None
                    entry_price = None
                else:
                    print(
                        f"⏳ {trade_date_str} | Position held | Days: {holding_period} | Unrealized P&L: {unrealized_pnl_pct:+.1f}% | Conditions not met"
                    )

            # Calculate current portfolio value for this stock
            current_position_value = position_qty * market_price
            total_portfolio_value = available_cash + current_position_value
            portfolio_values.append(round(total_portfolio_value, 2))
            positions.append(position_qty)

            df_bt.loc[idx, 'current_value'] = round(total_portfolio_value, 2)
            df_bt.loc[idx, 'Position'] = round(position_qty, 2)
            df_bt.loc[idx, 'balance_cash'] = round(available_cash, 2)
            df_bt.loc[idx, 'trade_position'] = trade_status
            df_bt.loc[idx, 'Trade_Date'] = trade_date

        df_bt['Portfolio_Value'] = portfolio_values
        df_bt['Position'] = positions

        # Final position summary
        last_idx = df_bt.index[-1]
        final_cash = float(df_bt.loc[last_idx, 'balance_cash'])
        final_position = int(df_bt.loc[last_idx, 'Position'])
        current_market_price = float(df_bt.loc[last_idx, 'Close'])

        final_position_value = final_position * current_market_price
        final_portfolio_value = final_cash + final_position_value

        total_return_amount = final_portfolio_value - allocated_capital
        total_return_pct = (total_return_amount / allocated_capital) * 100.0 if allocated_capital else 0.0

        print("\n" + "-" * 70)
        last_date = df_bt.loc[last_idx, 'Trade_Date'] if 'Trade_Date' in df_bt.columns else last_idx
        print(f"POSITION SUMMARY as of {_fmt_date(last_date)}")
        print("-" * 70)
        print(f"Current Market Price: ₹{current_market_price:.2f}")
        print(f"Position: {final_position:,} shares")
        print(f"Position Value: ₹{final_position_value:,.2f}")
        print(f"Available Cash: ₹{final_cash:,.2f}")
        print(f"Total Portfolio Value: ₹{final_portfolio_value:,.2f}")
        print(f"Total Return: ₹{total_return_amount:+,.2f} ({total_return_pct:+.2f}%)")

        df_bt['Final_Value'] = np.nan
        df_bt['Total_Return'] = np.nan
        df_bt['Allocated_Cash'] = allocated_capital
        df_bt.at[df_bt.index[-1], 'Final_Value'] = final_portfolio_value
        df_bt.at[df_bt.index[-1], 'Total_Return'] = round(total_return_pct, 2)

        trade_return = [None] * len(df_bt)
        for i, idx_val in enumerate(df_bt.index):
            if sell_signal.loc[idx_val]:
                pv = df_bt.loc[idx_val, 'Portfolio_Value']
                trade_return[i] = round((pv - allocated_capital) / allocated_capital * 100.0, 2)
        df_bt['Trade_Return'] = trade_return

        # Update global portfolio value
        prev_portfolio = self.portfolio_value
        print("\nGLOBAL PORTFOLIO UPDATE:")
        print(f"Previous Portfolio Value: ₹{prev_portfolio:,.2f}")
        self.portfolio_value += final_portfolio_value
        print(f"Added from {scrip}: ₹{final_portfolio_value:,.2f}")
        print(f"New Portfolio Value: ₹{self.portfolio_value:,.2f}")
        global_return = (self.portfolio_value - self.initial_cash) * 100.0 / self.initial_cash if self.initial_cash else 0.0
        print(f"Overall Portfolio Return: {global_return:+.2f}%")

        # Transaction log
        transactions_df = pd.DataFrame(transactions)
        if not transactions_df.empty:
            # Ensure schema consistency: always have Cost + Revenue columns
            if 'Cost' not in transactions_df.columns:
                transactions_df['Cost'] = np.nan
            if 'Revenue' not in transactions_df.columns:
                transactions_df['Revenue'] = np.nan

            # Normalize Date column to consistent ISO format
            try:
                transactions_df['Date'] = pd.to_datetime(transactions_df['Date'], errors='coerce')
                # Always export in ISO format for consistency
                transactions_df['Date'] = transactions_df['Date'].dt.strftime('%Y-%m-%d')
            except Exception as e:
                print(f"[WARNING] Could not normalize transaction dates: {e}")

            transactions_df.sort_values(by='Date', inplace=True)
            print("\n" + "=" * 60)
            print("TRANSACTION HISTORY".center(60))
            print("=" * 60)
            with pd.option_context('display.float_format', '{:.2f}'.format):
                print(transactions_df.to_string(index=False))

        return df_bt, transactions_df

    def backtest_strategy(self, filtered_scrips_df):
        """
        Iterates over each scrip in filtered_scrips_df,
        applies the backtest strategy, and writes results to Excel.
        """
        print(f"\n" + "="*80)
        print("PORTFOLIO BACKTESTING STARTED".center(80))
        print("="*80)

        # Ensure Stock column is preserved
        if 'Stock' not in filtered_scrips_df.columns:
            print("⚠️ WARNING: 'Stock' column missing in filtered_scrips_df. Attempting to restore...")
            try:
                # Try to restore from index
                if getattr(filtered_scrips_df.index, 'name', None) == 'Stock':
                    filtered_scrips_df = filtered_scrips_df.reset_index()
                # Try to restore from MultiIndex
                elif hasattr(filtered_scrips_df.index, 'names') and 'Stock' in (filtered_scrips_df.index.names or []):
                    filtered_scrips_df = filtered_scrips_df.reset_index()
                else:
                    # If DataFrame is empty, return early to avoid KeyError
                    if filtered_scrips_df.empty:
                        print("⚠️ No data to backtest - filtered DataFrame is empty.")
                        return pd.DataFrame(), pd.DataFrame()
                    else:
                        raise KeyError("'Stock' column not found and cannot be restored from index")
            except Exception as _e:
                if filtered_scrips_df.empty:
                    print("⚠️ No data to backtest - filtered DataFrame is empty.")
                    return pd.DataFrame(), pd.DataFrame()
                raise
        export_dir = os.path.join(pkg_dir, "output_data")
        dashboard_dir = os.path.join(export_dir, "dashboard_exports")
        os.makedirs(export_dir, exist_ok=True)
        os.makedirs(dashboard_dir, exist_ok=True)

        for scrip, filtered_scrip_df in scrip_extractor(filtered_scrips_df):
            buy_signal = filtered_scrip_df['Buy']
            sell_signal = filtered_scrip_df['Sell']

            bs_df, bt_df = self.apply_backtest_strategy(filtered_scrip_df, scrip, buy_signal, sell_signal)
            self.backtested_scrip_df_list.append(bs_df)
            self.backtested_transactions_df_list.append(bt_df)

        backtested_scrip_df = pd.concat(self.backtested_scrip_df_list) if self.backtested_scrip_df_list else pd.DataFrame()
        backtested_transactions_df = pd.concat(self.backtested_transactions_df_list) if self.backtested_transactions_df_list else pd.DataFrame()

        if not backtested_scrip_df.empty:
            # Ensure single Date column without duplicates
            backtested_scrip_df = backtested_scrip_df.reset_index(drop=True)
            if 'Date' in backtested_scrip_df.columns:
                backtested_scrip_df['Date'] = pd.to_datetime(
                    backtested_scrip_df['Date'], errors="coerce", dayfirst=True
                )
                backtested_scrip_df['Date'] = backtested_scrip_df['Date'].dt.strftime('%d-%b-%Y')
            backtested_scrip_df.to_excel(
                "backtested_scrips.xlsx", sheet_name=f"{config.ACTIVE_FILTER}", index=False
            )

        if not backtested_transactions_df.empty:
            backtested_transactions_df.reset_index(inplace=True)
            
            # Handle duplicate Date columns properly
            if 'Date' in backtested_transactions_df.columns:
                date_columns = [col for col in backtested_transactions_df.columns if col == 'Date']
                if len(date_columns) > 1:
                    # Combine all Date columns (take first non-null value)
                    date_data = backtested_transactions_df[date_columns].bfill(axis=1).iloc[:, 0]
                    # Remove all Date columns
                    backtested_transactions_df = backtested_transactions_df.loc[:, ~(backtested_transactions_df.columns == 'Date')]
                    # Add back single Date column
                    backtested_transactions_df['Date'] = date_data
            
            # Now safely format dates
            try:
                # Let pandas infer (ISO-safe) without forcing dayfirst to avoid warnings
                backtested_transactions_df['Date'] = pd.to_datetime(backtested_transactions_df['Date'], errors='coerce')
                backtested_transactions_df['Date'] = backtested_transactions_df['Date'].dt.strftime('%d-%m-%Y')
            except Exception as e:
                print(f"Warning: Date formatting issue: {e}")
            
            # Handle index column naming
            if 'index' in backtested_transactions_df.columns:
                backtested_transactions_df.drop('index', axis=1, inplace=True)
            
            backtested_transactions_df.to_excel("backtested_transactions.xlsx", sheet_name=f"{config.ACTIVE_FILTER}", index=False)

        return backtested_scrip_df, backtested_transactions_df

    def backtested_global_summary(self, backtested_scrips_df, backtested_transactions_df, master_df=None):
        """
        Aggregates global summary from backtested scrips and writes a summary Excel file.
        Uses professional trading terminology and metrics.
        """
        import os, shutil
        import support_files.File_IO as fio

        # Normalize inputs
        backtested_scrips_df = backtested_scrips_df.copy() if isinstance(backtested_scrips_df, pd.DataFrame) and not backtested_scrips_df.empty else pd.DataFrame()
        backtested_transactions_df = backtested_transactions_df.copy() if isinstance(backtested_transactions_df, pd.DataFrame) and not backtested_transactions_df.empty else pd.DataFrame()

        # Safely compute scrip list if Stock column exists
        scrips = backtested_scrips_df['Stock'].unique().tolist() if (not backtested_scrips_df.empty and 'Stock' in backtested_scrips_df.columns) else []
        num_positions = len(scrips)
        initial_capital = self.initial_cash

        print(f"\n" + "="*80)
        print("PORTFOLIO PERFORMANCE ANALYSIS".center(80))
        print("="*80)

        # Build map of final cash and positions per stock
        final_cash_map: dict[str, float] = {}
        final_position_map: dict[str, int] = {}
        if scrips and 'Stock' in backtested_scrips_df.columns:
            for scrip, df_s in scrip_extractor(backtested_scrips_df):
                if df_s.empty:
                    final_cash_map[scrip] = 0.0
                    final_position_map[scrip] = 0
                    continue
                try:
                    last_row = df_s.loc[df_s.index.max()]
                except Exception:
                    last_row = df_s.iloc[-1] if not df_s.empty else None
                if last_row is None:
                    final_cash_map[scrip] = 0.0
                    final_position_map[scrip] = 0
                else:
                    if hasattr(last_row, "get"):
                        final_cash_map[scrip] = float(last_row.get('balance_cash', 0.0) or 0.0)
                        final_position_map[scrip] = int(last_row.get('Position', 0) or 0)
                    else:
                        final_cash_map[scrip] = float(last_row['balance_cash']) if 'balance_cash' in last_row.index else 0.0
                        final_position_map[scrip] = int(last_row['Position']) if 'Position' in last_row.index else 0

        # Get latest market prices
        current_market_prices: dict[str, float] = {}
        if isinstance(master_df, pd.DataFrame) and not master_df.empty and 'Date' in master_df.columns and 'Stock' in master_df.columns:
            try:
                latest_data = master_df.sort_values(by=['Stock', 'Date']).groupby('Stock').last()
                for scrip in scrips:
                    current_market_prices[scrip] = float(latest_data.loc[scrip]['Close']) if scrip in latest_data.index else None
            except Exception:
                current_market_prices = {}
        
        # Fallback to backtested data if master_df not available
        for scrip in scrips:
            if scrip not in current_market_prices or current_market_prices.get(scrip) is None:
                try:
                    df_s = backtested_scrips_df[backtested_scrips_df['Stock'] == scrip]
                    current_market_prices[scrip] = float(df_s['Close'].iloc[-1]) if not df_s.empty else 0.0
                except Exception:
                    current_market_prices[scrip] = 0.0

        # Calculate portfolio components
        total_cash = sum(final_cash_map.values()) if final_cash_map else 0.0
        unrealized_market_value = 0.0

        print("CURRENT PORTFOLIO HOLDINGS:")
        print("-" * 80)
        print(f"{'Security':<15} {'Shares':<10} {'Market Price':<12} {'Market Value':<15} {'Cash':<12}")
        print("-" * 80)

        for scrip in scrips:
            shares = final_position_map.get(scrip, 0)
            market_price = current_market_prices.get(scrip, 0.0) or 0.0
            market_value = shares * market_price
            cash = final_cash_map.get(scrip, 0.0)
            unrealized_market_value += market_value
            print(f"{scrip:<15} {shares:<10,} ₹{market_price:<11.2f} ₹{market_value:<14,.0f} ₹{cash:<11,.0f}")

        print("-" * 80)
        print(f"{'TOTALS':<15} {'':<10} {'':<12} ₹{unrealized_market_value:<14,.0f} ₹{total_cash:<11,.0f}")

        total_portfolio_value = total_cash + unrealized_market_value
        print(f"\nPORTFOLIO VALUATION SUMMARY:")
        print(f"Cash Balance: ₹{total_cash:,.2f}")
        print(f"Unrealized Market Value: ₹{unrealized_market_value:,.2f}")
        print(f"Total Portfolio Value: ₹{total_portfolio_value:,.2f}")

        # Calculate Realized P&L using transaction history (FIFO method)
        realized_pnl = 0.0
        print(f"\n" + "="*60)
        print("REALIZED P&L CALCULATION (FIFO Method)".center(60))
        print("="*60)
        
        if not backtested_transactions_df.empty:
            tx_df = backtested_transactions_df.copy()
            
            # Handle duplicate Date columns properly
            if 'Date' in tx_df.columns:
                date_columns = [col for col in tx_df.columns if col == 'Date']
                if len(date_columns) > 1:
                    # Combine all Date columns (take first non-null value)
                    date_data = tx_df[date_columns].bfill(axis=1).iloc[:, 0]
                    # Remove all Date columns
                    tx_df = tx_df.loc[:, ~(tx_df.columns == 'Date')]
                    # Add back single Date column
                    tx_df['Date'] = date_data
                
            # Convert to datetime safely
            try:
                tx_df['Date'] = pd.to_datetime(tx_df['Date'], dayfirst=True, errors='coerce')
            except Exception as e:
                print(f"Warning: Date conversion error in P&L calculation: {e}")
                tx_df['Date'] = pd.to_datetime('today')  # Fallback date
            
            tx_df = tx_df.sort_values(by=['Stock', 'Date'])
            
            print(f"{'Security':<15} {'Trade Type':<12} {'Quantity':<10} {'Price':<10} {'P&L':<15}")
            print("-" * 70)
            
            for stock, group in tx_df.groupby('Stock'):
                buy_positions = []  # FIFO queue for buy positions
                stock_realized_pnl = 0.0
                
                for _, transaction in group.iterrows():
                    event_type = str(transaction.get('Event', '')).upper()
                    quantity = int(transaction.get('Shares', 0) or 0)
                    price = float(transaction.get('Price', 0.0) or 0.0)
                    
                    if event_type == 'BUY' and quantity > 0:
                        total_cost = float(transaction.get('Cost', 0.0) or 0.0)  # Actually cost for BUY
                        brokerage = float(transaction.get('Fee', 0.0) or 0.0)
                        buy_positions.append({
                            'quantity': quantity,
                            'avg_price': price,
                            'total_cost': total_cost + brokerage
                        })
                        print(f"{stock:<15} {'LONG ENTRY':<12} {quantity:<10,} ₹{price:<9.2f} {'-':<15}")
                    
                    elif event_type == 'SELL' and quantity > 0:
                        gross_proceeds = float(transaction.get('Revenue', 0.0) or 0.0)
                        brokerage = float(transaction.get('Fee', 0.0) or 0.0)
                        net_proceeds = gross_proceeds - brokerage
                        
                        remaining_to_sell = quantity
                        trade_pnl = 0.0
                        
                        while remaining_to_sell > 0 and buy_positions:
                            buy_lot = buy_positions[0]
                            buy_qty = buy_lot['quantity']
                            
                            if buy_qty <= remaining_to_sell:
                                # Sell entire buy lot
                                sell_proportion = buy_qty / quantity
                                allocated_proceeds = net_proceeds * sell_proportion
                                lot_pnl = allocated_proceeds - buy_lot['total_cost']
                                trade_pnl += lot_pnl
                                
                                remaining_to_sell -= buy_qty
                                buy_positions.pop(0)  # Remove fully sold lot
                            else:
                                # Partially sell buy lot
                                sell_proportion = remaining_to_sell / quantity
                                allocated_proceeds = net_proceeds * sell_proportion
                                cost_proportion = (remaining_to_sell / buy_qty) * buy_lot['total_cost']
                                lot_pnl = allocated_proceeds - cost_proportion
                                trade_pnl += lot_pnl
                                
                                # Update remaining buy lot
                                buy_lot['quantity'] -= remaining_to_sell
                                buy_lot['total_cost'] -= cost_proportion
                                remaining_to_sell = 0
                        
                        stock_realized_pnl += trade_pnl
                        realized_pnl += trade_pnl
                        
                        print(f"{stock:<15} {'LONG EXIT':<12} {quantity:<10,} ₹{price:<9.2f} ₹{trade_pnl:<14,.0f}")
                
                if stock_realized_pnl != 0:
                    print(f"{stock + ' TOTAL':<15} {'':<12} {'':<10} {'':<10} ₹{stock_realized_pnl:<14,.0f}")
                    print("-" * 70)
        else:
            print("No completed transactions found.")
        
        print(f"TOTAL REALIZED P&L: ₹{realized_pnl:,.2f}")

        # Calculate CAGR and other metrics
        print(f"\n" + "="*60)
        print("PERFORMANCE METRICS".center(60))
        print("="*60)
        
        # Date range calculation
        start_date = None
        end_date = None
        if not backtested_transactions_df.empty and 'Date' in backtested_transactions_df.columns:
            try:
                # Handle potential duplicate Date columns
                tx_temp = backtested_transactions_df.copy()
                if 'Date' in tx_temp.columns:
                    date_columns = [col for col in tx_temp.columns if col == 'Date']
                    if len(date_columns) > 1:
                        date_data = tx_temp[date_columns].bfill(axis=1).iloc[:, 0]
                        start_date = pd.to_datetime(date_data, dayfirst=True, errors='coerce').min()
                    else:
                        start_date = pd.to_datetime(tx_temp['Date'], dayfirst=True, errors='coerce').min()
            except Exception as e:
                print(f"Warning: Start date calculation error: {e}")
                start_date = None
                
        if start_date is None and master_df is not None and 'Date' in master_df.columns:
            try:
                start_date = pd.to_datetime(master_df['Date'], dayfirst=True, errors='coerce').min()
            except Exception:
                start_date = None
        
        if master_df is not None and 'Date' in master_df.columns:
            try:
                end_date = pd.to_datetime(master_df['Date'], dayfirst=True, errors='coerce').max()
            except Exception:
                end_date = None

        # Calculate CAGR
        cagr = None
        if start_date and end_date and not pd.isna(start_date) and not pd.isna(end_date):
            days = (end_date - start_date).days
            years = max(days / 365.25, 1.0 / 365.25)  # Minimum 1 day = 1/365.25 years
            
            try:
                cagr = (total_portfolio_value / initial_capital) ** (1.0 / years) - 1.0
                print(f"Investment Period: {start_date.date()} to {end_date.date()} ({days} days, {years:.2f} years)")
            except Exception as e:
                cagr = None
                print(f"CAGR calculation error: {e}")

        # Portfolio performance summary
        total_return_amount = total_portfolio_value - initial_capital
        total_return_pct = (total_return_amount / initial_capital) * 100.0 if initial_capital else 0.0
        unrealized_pnl = total_return_amount - realized_pnl

        # Transaction statistics
        if not backtested_transactions_df.empty:
            buy_transactions = backtested_transactions_df[backtested_transactions_df.get('Event') == 'BUY'].shape[0]
            sell_transactions = backtested_transactions_df[backtested_transactions_df.get('Event') == 'SELL'].shape[0]

            # Calculate total brokerage
            total_brokerage = 0.0
            if 'Fee' in backtested_transactions_df.columns:
                total_brokerage = pd.to_numeric(backtested_transactions_df['Fee'], errors='coerce').fillna(0.0).sum()

            # Calculate capital deployment
            capital_deployed = 0.0
            buy_txns = backtested_transactions_df[backtested_transactions_df['Event'] == 'BUY']
            if not buy_txns.empty and 'Revenue' in buy_txns.columns:
                capital_deployed = pd.to_numeric(buy_txns['Revenue'], errors='coerce').fillna(0.0).sum()
            elif not buy_txns.empty and 'Price' in buy_txns.columns and 'Shares' in buy_txns.columns:
                prices = pd.to_numeric(buy_txns['Price'], errors='coerce').fillna(0.0)
                shares = pd.to_numeric(buy_txns['Shares'], errors='coerce').fillna(0.0)
                capital_deployed = (prices * shares).sum()
        else:
            buy_transactions = sell_transactions = total_brokerage = capital_deployed = 0

        print(f"\nPERFORMANCE SUMMARY:")
        print(f"{'Metric':<30} {'Value':<20}")
        print("-" * 50)
        print(f"{'Initial Capital':<30} ₹{initial_capital:>15,.2f}")
        print(f"{'Final Portfolio Value':<30} ₹{total_portfolio_value:>15,.2f}")
        print(f"{'Total Return (Amount)':<30} ₹{total_return_amount:>15,.2f}")
        print(f"{'Total Return (%)':<30} {total_return_pct:>15.2f}%")
        print(f"{'Realized P&L':<30} ₹{realized_pnl:>15,.2f}")
        print(f"{'Unrealized P&L':<30} ₹{unrealized_pnl:>15,.2f}")
        if cagr is not None:
            print(f"{'CAGR':<30} {cagr*100:>15.2f}%")
        print(f"{'Capital Deployed':<30} ₹{capital_deployed:>15,.2f}")
        print(f"{'Total Brokerage':<30} ₹{total_brokerage:>15,.2f}")
        print(f"{'Number of Positions':<30} {num_positions:>15}")
        print(f"{'Buy Transactions':<30} {buy_transactions:>15}")
        print(f"{'Sell Transactions':<30} {sell_transactions:>15}")

        # Create summary dataframe for Excel export
        global_summary_df = pd.DataFrame({
            "Metric": [
                "STRATEGY_NAME", "PORTFOLIO_STRATEGY", "MIN_HOLDING_PERIOD_DAYS", "MIN_PROFIT_PERCENTAGE",
                "NUMBER_OF_POSITIONS", "INVESTMENT_APPROACH", "INITIAL_CAPITAL",
                "Final Portfolio Value", "Capital Deployed", "Total Return (Amount)", "Total Return (%)",
                "Total Brokerage Paid", "Buy Transactions", "Sell Transactions",
                "Realized P&L", "Unrealized P&L", "CAGR (%)"
            ],
            "Value": [
                config.ACTIVE_FILTER,
                getattr(self, 'active_strategy', 'unknown'),
                MIN_HOLDING_PERIOD,
                MIN_PROFIT_PERCENTAGE,
                num_positions,
                "Risk-Weighted Allocation",
                initial_capital,
                round(total_portfolio_value, 2),
                round(capital_deployed, 2),
                round(total_return_amount, 2),
                round(total_return_pct, 2),
                round(total_brokerage, 2),
                buy_transactions,
                sell_transactions,
                round(realized_pnl, 2),
                round(unrealized_pnl, 2),
                round((cagr or 0.0) * 100.0, 2)
            ]
        })

        # Portfolio allocation summary
        allocation_summary = []
        for scrip in scrips:
            allocation_amount = self.stock_allocations.get(scrip, 0.0)
            allocation_pct = (allocation_amount / self.initial_cash) * 100.0 if self.initial_cash else 0.0
            score = self.stock_scores.get(scrip, {}).get('composite_score', 0.0)
            current_price = current_market_prices.get(scrip, 0.0)
            position = final_position_map.get(scrip, 0)
            market_value = position * current_price
            cash = final_cash_map.get(scrip, 0.0)
            position_value = market_value + cash
            position_return = ((position_value - allocation_amount) / allocation_amount * 100.0) if allocation_amount else 0.0
            
            allocation_summary.append({
                'Security': scrip,
                'Initial_Allocation': round(allocation_amount, 2),
                'Allocation_%': round(allocation_pct, 2),                
                'Current_Price': round(current_price, 2),
                'Position_Qty': position,
                'Market_Value': round(market_value, 2),
                'Cash_Balance': round(cash, 2),
                'Total_Value': round(position_value, 2),
                'Return_%': round(position_return, 2)
            })
        
        allocation_df = pd.DataFrame(allocation_summary)

        # Export to Excel with professional formatting
        import support_files.File_IO as fio
        fio.change_cwd('../output_data')

        print(f"\n" + "="*60)
        print("EXPORTING PORTFOLIO ANALYSIS".center(60))
        print("="*60)

        # Create comprehensive Excel report
        with pd.ExcelWriter("portfolio_performance_report.xlsx", engine="openpyxl") as writer:
            # Performance Summary Sheet
            title_df = pd.DataFrame({"PORTFOLIO PERFORMANCE REPORT": [""]})
            title_df.to_excel(writer, sheet_name="Performance_Summary", startrow=0, index=False, header=False)
            
            summary_start = 2
            global_summary_df.to_excel(writer, sheet_name="Performance_Summary", startrow=summary_start, index=False)
            
            # Holdings Analysis Sheet
            allocation_df.to_excel(writer, sheet_name="Holdings_Analysis", index=False)
            
            # Transaction History Sheet
            if not backtested_transactions_df.empty:
                tx_export = backtested_transactions_df.copy()

                # Ensure schema consistency: always have Cost + Revenue columns
                if 'Cost' not in tx_export.columns:
                    tx_export['Cost'] = np.nan
                if 'Revenue' not in tx_export.columns:
                    tx_export['Revenue'] = np.nan

                # Handle duplicate Date columns before datetime conversion
                if 'Date' in tx_export.columns:
                    date_columns = [col for col in tx_export.columns if col == 'Date']
                    if len(date_columns) > 1:
                        # Get all Date columns and combine them (take first non-null value)
                        date_data = tx_export[date_columns].bfill(axis=1).iloc[:, 0]
                        # Drop all Date columns
                        tx_export = tx_export.loc[:, ~(tx_export.columns == 'Date')]
                        # Add single Date column back
                        tx_export['Date'] = date_data

                # Now safely convert to datetime and format to ISO
                try:
                    # Original transaction dates were exported as '%d-%m-%Y'; parse explicitly to avoid warnings
                    tx_export['Date'] = pd.to_datetime(tx_export['Date'], format='%d-%m-%Y', errors='coerce')
                    tx_export['Date'] = tx_export['Date'].dt.strftime('%Y-%m-%d')
                except Exception as e:
                    print(f"Warning: Date formatting error in transactions export: {e}")
                    # Fallback: keep original date format
                    pass

                tx_export.to_excel(writer, sheet_name="Transaction_History", index=False)
            
            # Summary only for quick reference
            global_summary_df.to_excel(writer, sheet_name="Quick_Summary", index=False)

        print("✅ Portfolio Performance Report exported to 'portfolio_performance_report.xlsx'")

        if not backtested_transactions_df.empty:
            backtested_transactions_df.to_excel('detailed_transactions.xlsx', index=False)
            print("✅ Detailed transactions exported to 'detailed_transactions.xlsx'")

        # Also place copies under dashboard_exports for co-located dashboard assets
        try:
            import shutil, os
            # At this point cwd is ../output_data; avoid nesting another 'output_data'
            export_dir = os.path.join('dashboard_exports')
            os.makedirs(export_dir, exist_ok=True)
            for fname in [
                'portfolio_performance_report.xlsx',
                'detailed_transactions.xlsx',
                'backtested_scrips.xlsx',
                'backtested_transactions.xlsx',
            ]:
                if os.path.exists(fname):
                    shutil.copy2(fname, os.path.join(export_dir, fname))
            print(f"[OK] Reports copied to: {export_dir}")
        except Exception as _copy_err:
            print(f"[WARNING] Could not copy reports to dashboard_exports: {_copy_err}")

    print(f"\n" + "="*80)
    print("FINAL PORTFOLIO RESULTS".center(80))
    print("="*80)

    # --------------------- RUN METHOD ---------------------
    def run(self, master_df,create_dashboard=True):
        """
        Complete portfolio management workflow:
        1. Screen and filter securities from universe
        2. Allocate capital using risk-weighted approach  
        3. Execute backtesting with realistic trading constraints
        4. Generate comprehensive performance analysis
        """
        print("🚀 STARTING PORTFOLIO MANAGEMENT SYSTEM")
        print(f"💰 Total Investment Capital: ₹{self.initial_cash:,.2f}")
        print(f"📊 Strategy: {config.ACTIVE_FILTER}")
        print(f"⏱️ Min Holding Period: {MIN_HOLDING_PERIOD} days")
        print(f"🎯 Min Profit Target: {MIN_PROFIT_PERCENTAGE}%")

        # Execute the complete workflow
        filtered_scrips_df = self.apply_filter(master_df)
        backtested_scrips_df, backtested_transactions_df = self.backtest_strategy(filtered_scrips_df)
        self.backtested_global_summary(backtested_scrips_df, backtested_transactions_df, master_df)
        # Add dashboard creation
        dashboard = None
        if create_dashboard and not backtested_scrips_df.empty:
            try:
                from dashboard_integration import TradingDashboard
                combined_scrips_df = pd.concat(self.backtested_scrip_df_list, ignore_index=True)
                combined_transactions_df = pd.concat(self.backtested_transactions_df_list, ignore_index=True)
                # 🔧 Ensure Date is normalized for dashboard JSON
                if 'Date' in combined_scrips_df.columns:
                    try:
                        combined_scrips_df['Date'] = pd.to_datetime(combined_scrips_df['Date'], errors='coerce')
                        combined_scrips_df['Date'] = combined_scrips_df['Date'].dt.strftime('%Y-%m-%d')
                    except Exception as e:
                        print(f"[WARNING] Could not normalize scrip dates for dashboard: {e}")

                if 'Date' in combined_transactions_df.columns:
                    try:
                        combined_transactions_df['Date'] = pd.to_datetime(combined_transactions_df['Date'], errors='coerce')
                        combined_transactions_df['Date'] = combined_transactions_df['Date'].dt.strftime('%Y-%m-%d')
                    except Exception as e:
                        print(f"[WARNING] Could not normalize transaction dates for dashboard: {e}")
                # Use config strategy if available, fallback to active_strategy, then default
                try:
                    from support_files import updated_config as cfg
                    strategy_name = getattr(cfg, 'PORTFOLIO_STRATEGY', None)
                except Exception:
                    strategy_name = None
                if not strategy_name:
                    strategy_name = getattr(self, 'active_strategy', None) or 'score_rank_claude'
                dashboard = TradingDashboard(
                    combined_scrips_df,
                    combined_transactions_df,
                    strategy_name=strategy_name
                )
                # Export JSON for Streamlit dashboard and provide launch instructions
                # cwd is output_data at runtime; write to a sibling 'dashboard_exports'
                export_dir = "dashboard_exports"
                dashboard.export_dashboard_data(export_dir=export_dir)
                json_path = os.path.join(export_dir, "trading_data.json")
                print(f"[OK] Dashboard data exported: {json_path}")
                print("🚀 To launch interactive dashboard, run:")
                print("   streamlit run streamlit_dashboard.py")
                print("📊 Interactive Streamlit dashboard ready!")
            except Exception as e:
                print(f"⚠️ Dashboard creation failed: {e}")      

        return backtested_scrips_df, backtested_transactions_df,dashboard


# --------------------- MAIN EXECUTION ---------------------
if __name__ == '__main__':
    import sys
    from support_files.dual_logger import DualLogger
    import support_files.File_IO as fio

    print("=" * 80)
    print("ALGORITHMIC TRADING PORTFOLIO MANAGEMENT SYSTEM".center(80))
    print("=" * 80)
    
    # Ensure working directory is this package folder so relative folders like 'input_data'/'output_data' work
    pkg_dir = Path(__file__).resolve().parent
    try:
        os.chdir(pkg_dir)
    except Exception:
        pass

    # Load market data (from input_data); allow override from env BACKTEST_INPUT_CSV
    _csv_override = os.environ.get('BACKTEST_INPUT_CSV')
    if _csv_override and os.path.exists(_csv_override):
        csv_name = os.path.basename(_csv_override)
        csv_folder = os.path.dirname(_csv_override) or 'input_data'
        master_df = fio.read_csv_to_df(csv_name, 'A', csv_folder, dayfirst=True)
        print(f"[INFO] Using input CSV from override: {_csv_override}")
    else:
        master_df = fio.read_csv_to_df('Nif50_5y_1w.csv', 'A', 'input_data', dayfirst=True)

    # Validate required columns before continuing
    required_cols = {"Open", "High", "Low", "Close", "Volume", "Stock"}
    available_cols = set(master_df.columns)

    # Ensure 'Date' exists either as a column or index named 'Date'
    try:
        has_date_col = "Date" in master_df.columns
        has_date_index = getattr(master_df.index, 'name', None) == "Date"
    except Exception:
        has_date_col = "Date" in master_df.columns
        has_date_index = False
    if not (has_date_col or has_date_index):
        print("❌ ERROR: Input data is missing 'Date' column or index.")
        try:
            print(f"   Available columns: {master_df.columns.tolist()}")
            print(f"   Index name: {master_df.index.name}")
        except Exception:
            pass
        sys.exit(1)

    # Handle missing OHLCV/Stock columns
    if not required_cols.issubset(available_cols):
        missing_cols = sorted(list(required_cols - available_cols))
        print("❌ ERROR: Input data is missing required columns for backtest.")
        try:
            print(f"   Missing columns: {missing_cols}")
            print(f"   Available columns: {master_df.columns.tolist()}")
        except Exception:
            pass
        sys.exit(1)
    
    # Initialize portfolio management system
    portfolio_manager = FilteringAndBacktesting(initial_cash=100000.0)
    
    # Setup logging
    fio.change_cwd('../output_data')
    sys.stdout = DualLogger("portfolio_trading_log.txt")
    
    print(f"📁 Data Source: Nif50_5y_1w.csv")
    print(f"💼 Portfolio Manager Initialized")
    print(f"📄 Logging to: portfolio_trading_log.txt")
    
    # Execute portfolio strategy
    portfolio_manager.run(master_df)
    
    # Finalize
    fio.get_cwd()
    sys.stdout.flush()
    
    print("\n🎉 PORTFOLIO ANALYSIS COMPLETED SUCCESSFULLY!")
    print("📊 Check 'output_data' folder for detailed reports")
    print("📄 Review 'portfolio_trading_log.txt' for complete trading history")