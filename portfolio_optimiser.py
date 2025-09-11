
"""
portfolio_optimiser.py
----------------------
A modular, strategy-pluggable portfolio optimisation engine designed to integrate cleanly
with Enhanced_stock_trading_V8.py without requiring any changes to the latter's encoding or structure.

Usage (imported):
    from portfolio_optimiser import optimize_portfolio, list_strategies, get_strategy

    weights = optimize_portfolio(
        prices=prices_df,                      # wide DF: dates x tickers (adjusted close)
        signals=signals_df,                    # optional DF: dates x tickers (e.g., momentum/vol/alpha)
        fundamentals=fundamentals_df,          # optional DF: tickers x features (value/quality/etc.)
        method="risk_parity",                  # see list_strategies()
        constraints={
            "long_only": True,
            "min_weight": 0.0,
            "max_weight": 0.15,
            "budget": 1.0,
            "group_limits": {                  # optional: {'SECTOR:IT': ('TCS','INFY'), 'SECTOR:FIN': ('HDFCBANK','ICICIBANK')}
                # "SECTOR:IT": {"tickers": ["TCS","INFY"], "max": 0.4, "min": 0.0}
            },
        },
        config={
            "lookback_days": 252,
            "cov_method": "ledoit_wolf_simple",   # 'sample' or 'ledoit_wolf_simple'
            "rebalance": "M",                     # 'D','W','M'; caller may downsample before calling
            "risk_aversion": 3.0,                 # for mean-variance / BL-like
            "target_vol": None,                   # optional scaling
            "kelly_cap": 0.25,                    # cap Kelly fraction
            "use_log_returns": True,
            "nan_policy": "drop"                   # 'drop' or 'fill_zero'
        }
    )

Design:
- Strategy Registry pattern to add/extend strategies without touching core logic.
- Returns a pandas.Series of weights indexed by ticker (sum to 1 within tolerance).
- No external heavy solvers; uses numerically stable closed-form + projections and simple iterative methods.
- Fully deterministic given inputs.

Author: ChatGPT (GPT-5 Thinking)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Iterable, Callable
import numpy as np
import pandas as pd

# ---------- Utilities ----------

def _to_returns(prices: pd.DataFrame, use_log: bool = True) -> pd.DataFrame:
    if prices is None or prices.empty:
        raise ValueError("prices DataFrame is required and cannot be empty.")
    prices = prices.sort_index()
    if use_log:
        rets = np.log(prices / prices.shift(1))
    else:
        rets = prices.pct_change()
    return rets.dropna(how="all")

def _sample_cov(returns: pd.DataFrame) -> pd.DataFrame:
    return returns.cov()

def _ledoit_wolf_simple(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Lightweight Ledoit-Wolf shrinkage to identity scaled by average variance.
    This is a simple, robust approximation that avoids sklearn dependency.
    """
    X = returns.dropna().values
    if X.shape[0] < 2:
        return returns.cov()
    S = np.cov(X, rowvar=False)
    var = np.diag(S).mean()
    F = var * np.eye(S.shape[0])
    diff = S - F
    num = np.sum(diff**2)
    den = np.sum((S - np.diag(np.diag(S)))**2) + 1e-12
    shrink = float(np.clip(num / (num + den), 0.05, 0.95))
    Sigma = shrink * F + (1 - shrink) * S
    return pd.DataFrame(Sigma, index=returns.columns, columns=returns.columns)

def _covariance(returns: pd.DataFrame, method: str = "ledoit_wolf_simple") -> pd.DataFrame:
    method = (method or "ledoit_wolf_simple").lower()
    if method == "sample":
        return _sample_cov(returns)
    return _ledoit_wolf_simple(returns)

def _project_to_simplex(w: np.ndarray, z: float = 1.0) -> np.ndarray:
    if (w >= 0).all() and abs(w.sum() - z) < 1e-12:
        return w
    u = np.sort(w)[::-1]
    cssv = np.cumsum(u)
    rho = np.where(u > (cssv - z) / (np.arange(1, len(w)+1)))[0]
    if len(rho) == 0:
        theta = (cssv[-1] - z) / len(w)
    else:
        rho = rho[-1]
        theta = (cssv[rho] - z) / float(rho + 1)
    return np.maximum(w - theta, 0.0)

def _clip_and_renorm(w: np.ndarray, min_w: float, max_w: float, budget: float) -> np.ndarray:
    w = np.clip(w, min_w, max_w)
    return _project_to_boxed_simplex(w, min_w, max_w, budget)

def _project_to_boxed_simplex(w: np.ndarray, lb: float, ub: float, z: float) -> np.ndarray:
    n = w.size
    if n == 0:
        return w
    def clip_sum(lmbda: float) -> float:
        x = np.clip(w - lmbda, lb, ub)
        return x.sum() - z
    l, r = -1e4, 1e4
    for _ in range(100):
        m = 0.5 * (l + r)
        val = clip_sum(m)
        if abs(val) < 1e-12: break
        if val > 0: l = m
        else: r = m
    x = np.clip(w - m, lb, ub)
    if x.sum() != 0:
        x *= z / x.sum()
    return x

def _enforce_group_limits(w: pd.Series, groups: Dict[str, Dict]) -> pd.Series:
    if not groups:
        return w
    w = w.copy()
    budget = float(w.sum())
    for g, meta in groups.items():
        tickers = meta.get("tickers", [])
        gmin = float(meta.get("min", 0.0)) * budget
        gmax = float(meta.get("max", 1.0)) * budget
        idx = [t for t in tickers if t in w.index]
        if not idx:
            continue
        s = w.loc[idx].sum()
        if s > gmax + 1e-12:
            scale = gmax / s if s > 0 else 0.0
            w.loc[idx] *= scale
        elif s < gmin - 1e-12:
            scale = gmin / s if s > 0 else 0.0
            w.loc[idx] *= scale
        w *= budget / w.sum()
    return w

# ---------- Strategy Base & Registry ----------

_STRATEGY_REGISTRY: Dict[str, "BaseStrategy"] = {}

def register_strategy(name: str):
    def deco(cls):
        key = name.lower().strip()
        _STRATEGY_REGISTRY[key] = cls
        cls._registry_name = key
        return cls
    return deco

def list_strategies() -> Dict[str, str]:
    return {k: v.__doc__.strip().splitlines()[0] if v.__doc__ else "" for k, v in _STRATEGY_REGISTRY.items()}

def get_strategy(name: str) -> "BaseStrategy":
    key = (name or "").lower().strip()
    if key not in _STRATEGY_REGISTRY:
        raise KeyError(f"Unknown strategy '{name}'. Known: {list(_STRATEGY_REGISTRY.keys())}")
    return _STRATEGY_REGISTRY[key]()

@dataclass
class Constraints:
    long_only: bool = True
    min_weight: float = 0.0
    max_weight: float = 1.0
    budget: float = 1.0
    group_limits: Optional[Dict[str, Dict]] = None

@dataclass
class Config:
    lookback_days: int = 252
    cov_method: str = "ledoit_wolf_simple"
    risk_aversion: float = 3.0
    target_vol: Optional[float] = None
    kelly_cap: float = 0.25
    use_log_returns: bool = True
    nan_policy: str = "drop"
    feature_frames: Optional[dict] = None  # NEW: optional dict of indicator frames
    # NEW: decay rate used by score_rank_claude strategy (weight decay by rank)
    decay_rate: float = 1.0

class BaseStrategy:
    """Base class for portfolio strategies."""
    def allocate(self,
                 returns: pd.DataFrame,
                 cov: pd.DataFrame,
                 signals: Optional[pd.DataFrame],
                 fundamentals: Optional[pd.DataFrame],
                 constraints: Constraints,
                 config: Config) -> pd.Series:
        raise NotImplementedError

# ---------- Strategies ----------

@register_strategy("equal_weight")
class EqualWeight(BaseStrategy):
    """Equal Weight (EW): w_i = 1/N for available tickers (post-constraints)."""
    def allocate(self, returns, cov, signals, fundamentals, constraints, config):
        tickers = list(returns.columns)
        N = len(tickers)
        w = np.ones(N) / N
        w = _clip_and_renorm(w, constraints.min_weight if constraints.long_only else -constraints.max_weight,
                             constraints.max_weight,
                             constraints.budget)
        s = pd.Series(w, index=tickers, name="weight")
        return _enforce_group_limits(s, constraints.group_limits or {})

@register_strategy("min_variance")
class MinimumVariance(BaseStrategy):
    """Minimum Variance: argmin_w w' Σ w subject to sum w = 1 and box constraints."""
    def allocate(self, returns, cov, signals, fundamentals, constraints, config):
        Sigma = cov.values + 1e-8 * np.eye(cov.shape[0])
        invS = np.linalg.pinv(Sigma)
        ones = np.ones((cov.shape[0], 1))
        w = invS @ ones
        w = (w / (ones.T @ invS @ ones)).ravel()
        lb = constraints.min_weight if constraints.long_only else -constraints.max_weight
        w = _clip_and_renorm(w, lb, constraints.max_weight, constraints.budget)
        s = pd.Series(w, index=cov.index, name="weight")
        return _enforce_group_limits(s, constraints.group_limits or {})

@register_strategy("mean_variance")
class MeanVariance(BaseStrategy):
    """Mean-Variance (Markowitz): maximize μ' w - (λ/2) w' Σ w with box + budget constraints."""
    def allocate(self, returns, cov, signals, fundamentals, constraints, config):
        mu = returns.mean().values
        Sigma = cov.values + 1e-8 * np.eye(cov.shape[0])
        lam = float(config.risk_aversion or 3.0)
        try:
            w = np.linalg.pinv(lam * Sigma) @ mu
        except np.linalg.LinAlgError:
            w = np.ones(len(mu)) / len(mu)
        lb = constraints.min_weight if constraints.long_only else -constraints.max_weight
        w = _clip_and_renorm(w, lb, constraints.max_weight, constraints.budget)
        s = pd.Series(w, index=cov.index, name="weight")
        return _enforce_group_limits(s, constraints.group_limits or {})

@register_strategy("risk_parity")
class RiskParity(BaseStrategy):
    """Equal Risk Contribution (Risk Parity): each asset contributes equal marginal risk."""
    def allocate(self, returns, cov, signals, fundamentals, constraints, config):
        Sigma = cov.values + 1e-10 * np.eye(cov.shape[0])
        n = Sigma.shape[0]
        w = np.ones(n) / n
        lb = constraints.min_weight if constraints.long_only else -constraints.max_weight
        ub = constraints.max_weight
        budget = constraints.budget
        for _ in range(1000):
            port_var = float(w @ Sigma @ w)
            if port_var <= 0:
                break
            mrc = (Sigma @ w)
            rc = w * mrc
            target = port_var / n
            grad = rc - target
            step = 0.1 / (np.linalg.norm(grad) + 1e-12)
            w = w - step * (grad / (mrc + 1e-12))
            w = _project_to_boxed_simplex(w, lb, ub, budget)
        s = pd.Series(w, index=cov.index, name="weight")
        return _enforce_group_limits(s, constraints.group_limits or {})

@register_strategy("momentum_inverse_vol")
class MomentumInverseVol(BaseStrategy):
    """Momentum + Inverse-Vol: long assets with positive momentum, weight ∝ momentum / volatility."""
    def allocate(self, returns, cov, signals, fundamentals, constraints, config):
        look = min(config.lookback_days, max(returns.shape[0], 1))
        rets = returns.tail(look)
        cum = (1 + rets).prod() - 1 if not config.use_log_returns else np.exp(rets.sum()) - 1
        vol = rets.std() * np.sqrt(252)
        score = cum / (vol + 1e-12)
        score = score.clip(lower=0.0)
        if (score <= 0).all():
            score = pd.Series(1.0, index=score.index)
        w = score.values
        lb = constraints.min_weight if constraints.long_only else -constraints.max_weight
        w = _clip_and_renorm(w, lb, constraints.max_weight, constraints.budget)
        s = pd.Series(w, index=returns.columns, name="weight")
        return _enforce_group_limits(s, constraints.group_limits or {})

@register_strategy("fundamental_quality_value_tilt")
class FundamentalQualityValueTilt(BaseStrategy):
    """
    Fundamental Tilt (Quality + Value):
    - Build composite score = z(ROE) + z(OperatingMargin) + z(ReturnOnCapital) + z(EarningsStability)
      + z(EBIT/EV) + z(FreeCashFlowYield) - z(Leverage)
    - Allocate ∝ max(score, 0); fallback to EW if all non-positive.
    """
    def allocate(self, returns, cov, signals, fundamentals, constraints, config):
        if fundamentals is None or fundamentals.empty:
            return EqualWeight().allocate(returns, cov, signals, fundamentals, constraints, config)
        feats = ["ROE","OperatingMargin","ReturnOnCapital","EarningsStability","EBIT_EV","FCF_Yield","Leverage"]
        cols = [c for c in feats if c in fundamentals.columns]
        if not cols:
            return EqualWeight().allocate(returns, cov, signals, fundamentals, constraints, config)
        F = fundamentals.reindex(returns.columns).copy()
        Z = (F[cols] - F[cols].mean()) / (F[cols].std(ddof=0) + 1e-12)
        comp = (
            Z.get("ROE", 0) +
            Z.get("OperatingMargin", 0) +
            Z.get("ReturnOnCapital", 0) +
            Z.get("EarningsStability", 0) +
            Z.get("EBIT_EV", 0) +
            Z.get("FCF_Yield", 0) -
            Z.get("Leverage", 0)
        )
        score = comp.clip(lower=0.0)
        if (score <= 0).all():
            score = pd.Series(1.0, index=score.index)
        w = score.values
        lb = constraints.min_weight if constraints.long_only else -constraints.max_weight
        w = _clip_and_renorm(w, lb, constraints.max_weight, constraints.budget)
        s = pd.Series(w, index=returns.columns, name="weight")
        return _enforce_group_limits(s, constraints.group_limits or {})

@register_strategy("kelly_capped")
class KellyCapped(BaseStrategy):
    """Kelly (fractional, capped): w ∝ Σ^{-1} μ, then scale by kelly_cap to control leverage."""
    def allocate(self, returns, cov, signals, fundamentals, constraints, config):
        mu = returns.mean().values
        Sigma = cov.values + 1e-8 * np.eye(cov.shape[0])
        w = np.linalg.pinv(Sigma) @ mu
        w = w / (np.abs(w).sum() + 1e-12)
        cap = float(config.kelly_cap or 0.25)
        w = np.clip(w, -cap, cap)
        lb = constraints.min_weight if constraints.long_only else -constraints.max_weight
        w = _clip_and_renorm(w, lb, constraints.max_weight, constraints.budget)
        s = pd.Series(w, index=cov.index, name="weight")
        return _enforce_group_limits(s, constraints.group_limits or {})


@register_strategy("score_rank_claude")
class ScoreRankClaude(BaseStrategy):
    """
    Score & Rank Allocator (Claude-style wrapper):
    - Builds a composite score per ticker from technicals, signals, and fundamentals.
    - Sorts by score and applies exponential rank decay + box constraints.
    - Emits **weights** (not currency amounts), compatible with the core API.

    Expected inputs (best effort, all optional):
    - If `config.feature_frames` is provided, it should be a dict of DataFrames (dates x tickers)
      with any of the following keys: 'RSI', 'MACD', 'SMA_20', 'Buy', 'Sell', 'Volume', 'Close'.
      Missing keys are computed from `prices` when feasible.
    - If `fundamentals` includes 'PE' and/or 'EPS', they will be incorporated.

    Config keys (optional):
      feature_frames: dict[str, pd.DataFrame]
      decay_rate: float in (0,1], default 0.92 (stronger decay → top ranks get more)
      min_weight: override per-asset min bound for this strategy (else use constraints.min_weight)
      max_weight: override per-asset max bound for this strategy (else use constraints.max_weight)
    """
    def allocate(self, returns, cov, signals, fundamentals, constraints, config):
        import numpy as np
        import pandas as pd

        def last_row(df: pd.DataFrame) -> pd.Series:
            return df.iloc[-1].reindex(cols)

        # Recreate a price level from returns (scale-invariant). If Close is provided in feature_frames, we will use it.
        prices_level = returns.cumsum().pipe(np.exp) if config.use_log_returns else (1+returns).cumprod()
        prices_level = prices_level / prices_level.iloc[0].replace(0, np.nan)

        ff = getattr(config, "feature_frames", None) or {}
        cols = list(returns.columns)

        # RSI(14)
        if "RSI" in ff:
            rsi_df = ff["RSI"].reindex(returns.index, columns=cols)
        else:
            delta = returns.diff().fillna(0.0)
            up = delta.clip(lower=0.0).rolling(14).mean()
            dn = (-delta.clip(upper=0.0)).rolling(14).mean()
            rs = up / (dn + 1e-12)
            rsi_df = 100 - (100 / (1 + rs))

        # MACD(12,26,9)
        if "MACD" in ff:
            macd_df = ff["MACD"].reindex(returns.index, columns=cols)
        else:
            price_level = prices_level
            ema12 = price_level.ewm(span=12, adjust=False).mean()
            ema26 = price_level.ewm(span=26, adjust=False).mean()
            macd_df = ema12 - ema26
            macd_signal = macd_df.ewm(span=9, adjust=False).mean()
            macd_df = macd_df - macd_signal

        # SMA20
        if "SMA_20" in ff:
            sma20 = ff["SMA_20"].reindex(returns.index, columns=cols)
        else:
            sma20 = prices_level.rolling(20).mean()

        buy_df = ff.get("Buy", pd.DataFrame(index=returns.index, columns=cols)).fillna(0)
        sell_df = ff.get("Sell", pd.DataFrame(index=returns.index, columns=cols)).fillna(0)
        close_df = ff.get("Close", prices_level)

        look = min(getattr(config, "lookback_days", 252), max(returns.shape[0], 1))
        R = returns.tail(look)
        C = close_df.tail(look)
        RSI = rsi_df.tail(look)
        MACD = macd_df.tail(look)
        SMA20 = sma20.tail(look)
        BUY = buy_df.tail(look)
        SELL = sell_df.tail(look)

        last_rsi = last_row(RSI).fillna(50.0)
        last_macd = last_row(MACD).fillna(0.0)
        last_close = last_row(C).replace(0, np.nan).fillna(1.0)
        last_sma = last_row(SMA20).replace(0, np.nan).fillna(last_close)
        sma_ratio = (last_close / last_sma).clip(lower=0, upper=np.inf)

        rsi_score = (100.0 - (last_rsi - 40.0).abs()).clip(lower=0, upper=100)
        macd_score = (50.0 + (last_macd * 10.0)).clip(lower=0, upper=100)
        sma_score = ((sma_ratio - 0.95) * 200.0).clip(lower=0, upper=100)
        technical_score = (0.4 * rsi_score + 0.3 * macd_score + 0.3 * sma_score)

        total_signals = (BUY.sum() + SELL.sum()).reindex(cols).fillna(0.0)
        signal_score = total_signals.copy()
        mask = total_signals <= 20
        signal_score[mask] = (total_signals[mask] * 5.0).clip(0, 100)
        signal_score[~mask] = (100.0 - (total_signals[~mask] - 20.0) * 2.0).clip(0, 100)

        cumret = (np.exp(R.sum()) - 1.0) if getattr(config, "use_log_returns", True) else (1+R).prod()-1
        mom_score = ((cumret * 400.0) + 50.0).clip(0, 100)

        vol = R.std() * np.sqrt(252)
        inv_vol_score = (100.0 / (1.0 + 100.0 * vol)).clip(0, 100)

        fund_score = pd.Series(50.0, index=cols)
        if fundamentals is not None and not fundamentals.empty:
            F = fundamentals.reindex(index=cols)
            if "PE" in F.columns:
                pe = F["PE"].replace(0, np.nan)
                inv_pe = (1/pe).fillna(0)
                inv_pe = (inv_pe - inv_pe.mean()) / (inv_pe.std(ddof=0) + 1e-12)
                fund_score = fund_score + 20.0 * inv_pe.clip(-2, 2)
            if "EPS" in F.columns:
                eps = F["EPS"]
                eps_z = (eps - eps.mean()) / (eps.std(ddof=0) + 1e-12)
                fund_score = fund_score + 20.0 * eps_z.clip(-2, 2)
            fund_score = fund_score.clip(0, 100)

        score = (0.35 * technical_score +
                 0.20 * signal_score +
                 0.25 * mom_score +
                 0.10 * inv_vol_score +
                 0.10 * fund_score).fillna(50.0)

        decay = float(getattr(config, "decay_rate", 0.92))
        rank = score.rank(ascending=False, method="first")
        base = (score / 100.0) * (decay ** (rank - 1))

        lb = float(getattr(config, "min_weight", constraints.min_weight if constraints.long_only else -constraints.max_weight))
        ub = float(getattr(config, "max_weight", constraints.max_weight))
        w = base.values
        s = w.sum()
        if s > 0: w = w / s
        w = _project_to_boxed_simplex(np.clip(w, lb, ub), lb, ub, constraints.budget)
        weights = pd.Series(w, index=cols, name="weight")
        return _enforce_group_limits(weights, constraints.group_limits or {})
@register_strategy("legacy_risk_weighted")
class LegacyRiskWeighted(BaseStrategy):
    """
    Legacy Risk-Weighted (ported from Enhanced_stock_trading_V8.allocate_portfolio):
    - Uses your legacy per-ticker scoring + exponential rank decay.
    - Applies per-asset min/max caps and projects to the budget simplex.
    How to feed scores:
      Option A (preferred for exact parity):
        Pass a precomputed score Series via config["feature_frames"]["LEGACY_SCORE"].
      Option B (full port):
        Paste your old scoring snippet inside this strategy (marked below).
    """
    def allocate(self, returns, cov, signals, fundamentals, constraints, config):
        import numpy as np, pandas as pd

        cols = list(returns.columns)
        ff = getattr(config, "feature_frames", None) or {}

        # ---------- Option A: use precomputed legacy scores ----------
        legacy_score = None
        if "LEGACY_SCORE" in ff:
            obj = ff["LEGACY_SCORE"]
            if isinstance(obj, pd.DataFrame):
                # If a DF is provided, take the last row or squeeze to Series
                legacy_score = obj.iloc[-1].reindex(cols)
            elif isinstance(obj, pd.Series):
                legacy_score = obj.reindex(cols)

        # ---------- Option B: (if you prefer to keep all logic here) ----------
        # If legacy_score is still None, you can paste your original scoring code here.
        # Inputs you can rely on:
        #   - 'returns' (lookback window controlled by config.lookback_days)
        #   - 'fundamentals' (index=tickers) if you used PE/EPS etc.
        #   - 'signals' or config.feature_frames (RSI/MACD/SMA/Buy/Sell/Close) if needed
        #
        # Example fallback (keeps behavior sane if you haven't pasted your snippet yet):
        if legacy_score is None:
            # Simple momentum as a harmless default
            L = min(getattr(config, "lookback_days", 252), max(returns.shape[0], 1))
            R = returns.tail(L)
            legacy_score = (np.exp(R.sum()) - 1.0)  # log return cum (if use_log_returns True)

        # ---------- Rank + exponential decay ----------
        decay = float(getattr(config, "decay_rate", 0.92))
        rank = legacy_score.rank(ascending=False, method="first")
        base = (legacy_score.clip(lower=0) + 1e-12)  # ensure positive
        base = (base / base.sum()) * (decay ** (rank - 1))

        # ---------- Box constraints + budget projection ----------
        lb = constraints.min_weight if constraints.long_only else -constraints.max_weight
        ub = constraints.max_weight
        w = base.values.astype(float)
        s = w.sum()
        if s > 0:
            w = w / s
        w = _project_to_boxed_simplex(np.clip(w, lb, ub), lb, ub, constraints.budget)
        weights = pd.Series(w, index=cols, name="weight")
        return _enforce_group_limits(weights, constraints.group_limits or {})

# ---------- Core API ----------

def optimize_portfolio(prices: pd.DataFrame,
                       signals: Optional[pd.DataFrame] = None,
                       fundamentals: Optional[pd.DataFrame] = None,
                       method: str = "risk_parity",
                       constraints: Optional[Dict] = None,
                       config: Optional[Dict] = None) -> pd.Series:
    constraints = Constraints(**(constraints or {}))
    config = Config(**(config or {}))

    returns = _to_returns(prices, use_log=config.use_log_returns)
    if config.nan_policy == "drop":
        returns = returns.dropna(axis=1, how="any")
    else:
        returns = returns.fillna(0.0)

    if signals is not None:
        signals = signals.reindex(columns=returns.columns).reindex(returns.index).dropna(how="all")
    if fundamentals is not None:
        fundamentals = fundamentals.reindex(index=returns.columns)

    cov = _covariance(returns.tail(config.lookback_days), method=config.cov_method)

    strategy = get_strategy(method)
    weights = strategy.allocate(returns, cov, signals, fundamentals, constraints, config)
    if weights.sum() != 0:
        weights = weights * (constraints.budget / weights.sum())
    return weights.round(10)
    
if __name__ == "__main__":
    import argparse, json, sys
    parser = argparse.ArgumentParser(description="Portfolio Optimiser CLI")
    parser.add_argument("--method", default="risk_parity", help="Strategy name")
    parser.add_argument("--constraints", default="{}", help="JSON dict")
    parser.add_argument("--config", default="{}", help="JSON dict")
    parser.add_argument("--prices_csv", required=True, help="CSV path: index=date, columns=tickers")
    parser.add_argument("--signals_csv", default=None, help="Optional CSV")
    parser.add_argument("--fundamentals_csv", default=None, help="Optional CSV (index=tickers)")
    args = parser.parse_args()

    prices = pd.read_csv(args.prices_csv, index_col=0, parse_dates=True)
    signals = pd.read_csv(args.signals_csv, index_col=0, parse_dates=True) if args.signals_csv else None
    fundamentals = pd.read_csv(args.fundamentals_csv, index_col=0) if args.fundamentals_csv else None
    w = optimize_portfolio(prices, signals, fundamentals,
                           method=args.method,
                           constraints=json.loads(args.constraints),
                           config=json.loads(args.config))
    w.to_csv(sys.stdout)
