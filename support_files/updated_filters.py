#================================================================
# enhanced_filters.py - Profit Optimized Trading Filters
#================================================================

import sys
from pathlib import Path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import pandas as pd
import numpy as np

def filter_ensemble_weighted(scrip_df, weights=None, return_scores=False):
    """
    Ensemble filter that combines multiple strategies with weighted voting.
    Higher confidence when multiple filters agree.
    """
    if weights is None:
        weights = {
            'basic': 0.15,
            'aggressive': 0.10,
            'momentum': 0.15,
            'breakout': 0.12,
            'mean_reversion': 0.08,
            'trend_following': 0.20,
            'volume_surge': 0.10,
            'adx': 0.10
        }
    
    # Get signals from individual filters with error handling
    filters = {}
    try:
        filters['basic'] = filter_basic_original(scrip_df)
        filters['aggressive'] = filter_aggressive(scrip_df)
        filters['momentum'] = filter_momentum(scrip_df)
        filters['breakout'] = filter_breakout(scrip_df)
        filters['mean_reversion'] = filter_mean_reversion(scrip_df)
        filters['trend_following'] = filter_trend_following(scrip_df)
        filters['volume_surge'] = filter_volume_surge(scrip_df)
        filters['adx'] = filter_adx(scrip_df)
    except Exception as e:
        print(f"Error in ensemble filter: {e}")
        empty_signal = pd.Series(False, index=scrip_df.index)
        if return_scores:
            return empty_signal, empty_signal, empty_signal, empty_signal
        return empty_signal, empty_signal
    
    # Calculate weighted scores
    buy_score = pd.Series(0.0, index=scrip_df.index)
    sell_score = pd.Series(0.0, index=scrip_df.index)
    
    for name, (buy_sig, sell_sig) in filters.items():
        if name in weights:
            buy_score += buy_sig * weights[name]
            sell_score += sell_sig * weights[name]
    
    # Generate signals with confidence thresholds
    buy_signal = buy_score >= 0.35  # Require 35% weighted agreement
    sell_signal = sell_score >= 0.35
    
    if return_scores:
        return buy_signal, sell_signal, buy_score, sell_score
    return buy_signal, sell_signal


def filter_adaptive_rsi(scrip_df, lookback_period=20):
    """
    Adaptive RSI filter that adjusts thresholds based on market volatility.
    More conservative in volatile markets, more aggressive in stable markets.
    """
    # Calculate volatility (rolling std of returns)
    returns = scrip_df['Close'].pct_change()
    volatility = returns.rolling(window=lookback_period).std()
    vol_percentile = volatility.rolling(window=60).rank(pct=True)
    
    # Adaptive RSI thresholds
    rsi_buy_threshold = 30 + (vol_percentile * 15)  # 30-45 range
    rsi_sell_threshold = 70 - (vol_percentile * 15)  # 55-70 range
    
    buy_signal = (scrip_df['RSI'] < rsi_buy_threshold) & \
                 (scrip_df['MACD'] > scrip_df['Signal']) & \
                 (scrip_df['Close'] > scrip_df['ema_60'].shift(1))  # Trend confirmation
    
    sell_signal = (scrip_df['RSI'] > rsi_sell_threshold) & \
                  (scrip_df['MACD'] < scrip_df['Signal'])
    
    return buy_signal, sell_signal


def filter_multi_timeframe_confluence(scrip_df):
    """
    Multi-timeframe confluence filter using different EMA periods
    to simulate higher timeframe analysis.
    """
    # Short-term (simulated daily from intraday)
    short_trend_up = (scrip_df['ema_20'] > scrip_df['ema_50']) & \
                     (scrip_df['Close'] > scrip_df['ema_20'])
    
    # Medium-term (simulated weekly)
    medium_trend_up = (scrip_df['ema_60'] > scrip_df['ema_100']) & \
                      (scrip_df['ema_100'] > scrip_df['ema_200'])
    
    # Long-term momentum
    long_momentum_up = scrip_df['ROC'] > 0
    
    # Confluence buy: All timeframes aligned + oversold RSI
    buy_signal = short_trend_up & medium_trend_up & long_momentum_up & \
                 (scrip_df['RSI'] < 50) & \
                 (scrip_df['Volume'] > scrip_df['vol_ma_60'])
    
    # Confluence sell: Trend weakening + overbought
    sell_signal = (~short_trend_up | ~medium_trend_up) & \
                  (scrip_df['RSI'] > 60)
    
    return buy_signal, sell_signal


def filter_risk_adjusted(scrip_df, max_drawdown_threshold=0.15):
    """
    Risk-adjusted filter that considers recent drawdown and position sizing.
    Reduces position size or avoids trades during high drawdown periods.
    """
    # Calculate rolling drawdown
    rolling_high = scrip_df['Close'].rolling(window=60).max()
    drawdown = (scrip_df['Close'] - rolling_high) / rolling_high
    
    # Base signals from trend following
    base_buy, base_sell = filter_trend_following(scrip_df)
    
    # Risk adjustment: reduce exposure during high drawdown
    low_risk_period = drawdown > -max_drawdown_threshold
    
    # Enhanced buy conditions during low risk
    buy_signal = base_buy & low_risk_period & \
                 (scrip_df['Volume'] > scrip_df['vol_ma_60'] * 1.2) & \
                 (scrip_df['RSI'] < 55)
    
    # Faster sell during high risk
    sell_signal = base_sell | (~low_risk_period & (scrip_df['RSI'] > 50))
    
    return buy_signal, sell_signal


def filter_volatility_breakout(scrip_df, volatility_threshold=1.5):
    """
    Enhanced breakout filter that uses volatility expansion
    to identify high-probability breakouts.
    """
    # Calculate volatility measures
    returns = scrip_df['Close'].pct_change()
    current_vol = returns.rolling(window=20).std()
    avg_vol = returns.rolling(window=60).std()
    vol_expansion = current_vol / avg_vol
    
    # Price breakout conditions
    price_breakout_up = scrip_df['Close'] > scrip_df['60_day_high'].shift(1)
    price_breakout_down = scrip_df['Close'] < scrip_df['60_day_low'].shift(1)
    
    # Volume confirmation
    volume_surge = scrip_df['Volume'] > scrip_df['vol_ma_60'] * 1.5
    
    buy_signal = price_breakout_up & \
                 (vol_expansion > volatility_threshold) & \
                 volume_surge & \
                 (scrip_df['RSI'] > 45)  # Avoid oversold breakouts
    
    sell_signal = price_breakout_down & \
                  (vol_expansion > volatility_threshold) & \
                  volume_surge
    
    return buy_signal, sell_signal


def filter_momentum_with_support_resistance(scrip_df):
    """
    Momentum filter enhanced with dynamic support/resistance levels.
    """
    # Dynamic support/resistance using rolling highs/lows
    support = scrip_df['Close'].rolling(window=30).min()
    resistance = scrip_df['Close'].rolling(window=30).max()
    
    # Price position within range
    range_position = (scrip_df['Close'] - support) / (resistance - support)
    
    # Momentum conditions
    momentum_up = (scrip_df['MACD'] > scrip_df['Signal']) & \
                  (scrip_df['ROC'] > 0)
    
    # Buy near support with momentum
    buy_signal = (range_position < 0.3) & momentum_up & \
                 (scrip_df['RSI'] < 45) & \
                 (scrip_df['Volume'] > scrip_df['vol_ma_60'])
    
    # Sell near resistance or momentum failure
    sell_signal = (range_position > 0.8) | \
                  ((scrip_df['MACD'] < scrip_df['Signal']) & (scrip_df['RSI'] > 55))
    
    return buy_signal, sell_signal


def calculate_position_size(scrip_df, signal_strength, max_position_pct=0.05):
    """
    Dynamic position sizing based on signal strength and volatility.
    """
    # Base position size
    base_size = max_position_pct
    
    # Adjust based on volatility (ATR-based)
    returns = scrip_df['Close'].pct_change()
    volatility = returns.rolling(window=20).std()
    vol_adjustment = 1 / (1 + volatility * 100)  # Reduce size for high vol
    
    # Adjust based on signal strength (0-1 scale)
    position_size = base_size * signal_strength * vol_adjustment
    
    return np.clip(position_size, 0.01, max_position_pct)


# Check required indicators function
def check_required_indicators(scrip_df, filter_name):
    """
    Check if all required indicators are present in the dataframe
    for the specified filter. Returns True if all required indicators exist.
    """
    required_indicators = {
        'filter_basic': ['RSI', 'Close', 'ema_60', 'MACD', 'Signal', 'Bollinger_Lower', 'Bollinger_Upper', 'vol_ma_60'],
        'filter_adaptive_rsi': ['RSI', 'Close', 'MACD', 'Signal', 'ema_60'],
        'filter_multi_timeframe_confluence': ['ema_20', 'ema_50', 'ema_60', 'ema_100', 'ema_200', 'ROC', 'RSI', 'Volume', 'vol_ma_60'],
        'filter_ensemble_weighted': ['RSI', 'Close', 'ema_60', 'ema_90', 'ema_100', 'ema_200', 'MACD', 'Signal', 'Bollinger_Lower', 'Bollinger_Upper', 'Volume', 'vol_ma_60', 'volume_ema_60', 'ROC', '60_day_high', '60_day_low', 'ADX', 'plus_DI', 'minus_DI'],
        # Add other filters as needed
    }
    
    if filter_name not in required_indicators:
        return True  # Assume compatible if not listed
    
    required = required_indicators[filter_name]
    missing = [col for col in required if col not in scrip_df.columns]
    
    if missing:
        print(f"Warning: {filter_name} requires missing indicators: {missing}")
        return False
    return True

# Safe wrapper for filters
def safe_filter_wrapper(filter_func):
    """
    Wrapper that handles errors gracefully and returns empty signals if indicators are missing
    """
    def wrapper(scrip_df):
        try:
            return filter_func(scrip_df)
        except KeyError as e:
            print(f"Missing indicator for {filter_func.__name__}: {e}")
            # Return empty signals to maintain compatibility
            empty_signal = pd.Series(False, index=scrip_df.index)
            return empty_signal, empty_signal
        except Exception as e:
            print(f"Error in {filter_func.__name__}: {e}")
            empty_signal = pd.Series(False, index=scrip_df.index)
            return empty_signal, empty_signal
    return wrapper

# Apply safe wrapper to all filters (moved after definitions to avoid NameError during import)

# Original filters with minor enhancements
# Rename original basic filter to avoid recursion
def filter_basic_original(scrip_df):
    """Original basic filter"""
    buy_signal = ((scrip_df['RSI'] < 40) &
                  (scrip_df['Close'] <= scrip_df['ema_60']) &
                  (scrip_df['MACD'] > scrip_df['Signal']) &
                  (scrip_df['Close'] > scrip_df['Bollinger_Lower'] * 1.02))
    sell_signal = ((scrip_df['RSI'] > 60) &
                   (scrip_df['Close'] >= scrip_df['ema_60']) &
                   (scrip_df['MACD'] < scrip_df['Signal']) &
                   (scrip_df['Close'] < scrip_df['Bollinger_Upper'] * 0.98))
    return buy_signal, sell_signal

def filter_basic(scrip_df):
    """Enhanced basic filter with tighter conditions"""
    try:
        # Check for volume column availability
        vol_condition = (scrip_df['Volume'] > scrip_df['vol_ma_60'] * 0.8) if 'vol_ma_60' in scrip_df.columns else True
        
        buy_signal = ((scrip_df['RSI'] < 40) &
                      (scrip_df['Close'] <= scrip_df['ema_60']) &
                      (scrip_df['MACD'] > scrip_df['Signal']) &
                      (scrip_df['Close'] > scrip_df['Bollinger_Lower'] * 1.02) &
                      vol_condition)  # Volume filter if available

        sell_signal = ((scrip_df['RSI'] > 60) &
                       (scrip_df['Close'] >= scrip_df['ema_60']) &
                       (scrip_df['MACD'] < scrip_df['Signal']) &
                       (scrip_df['Close'] < scrip_df['Bollinger_Upper'] * 0.98))
        return buy_signal, sell_signal
    except KeyError as e:
        print(f"Missing indicator in filter_basic: {e}")
        return filter_basic_original(scrip_df)  # Fallback to original

def filter_aggressive(scrip_df):
    buy_signal = ((scrip_df['RSI'] < 35) &
                  (scrip_df['Close'] <= scrip_df['ema_90']) &
                  (scrip_df['MACD'] > scrip_df['Signal']) &
                  (scrip_df['Volume'] > scrip_df['volume_ema_60']))
    sell_signal = ((scrip_df['RSI'] > 65) &
                   (scrip_df['Close'] >= scrip_df['ema_90']) &
                   (scrip_df['MACD'] < scrip_df['Signal']) &
                   (scrip_df['Volume'] < scrip_df['volume_ema_60']))
    return buy_signal, sell_signal

def filter_momentum(scrip_df):
    buy_signal = (scrip_df['ROC'] > 0) & (scrip_df['RSI'] < 45)
    sell_signal = (scrip_df['ROC'] < 0) & (scrip_df['RSI'] > 55)
    return buy_signal, sell_signal

def filter_breakout(scrip_df):
    buy_signal = (scrip_df['Close'] > scrip_df['60_day_high']) & (scrip_df['Volume'] > scrip_df['vol_ma_60'])
    sell_signal = (scrip_df['Close'] < scrip_df['60_day_low']) & (scrip_df['Volume'] > scrip_df['vol_ma_60'])
    return buy_signal, sell_signal

def filter_mean_reversion(scrip_df):
    buy_signal = (scrip_df['Close'] < scrip_df['Bollinger_Lower']) & (scrip_df['RSI'] < 30)
    sell_signal = (scrip_df['Close'] > scrip_df['Bollinger_Upper']) & (scrip_df['RSI'] > 70)
    return buy_signal, sell_signal

def filter_trend_following(scrip_df):
    buy_signal = (
        (scrip_df['Close'] > scrip_df['ema_60']) &
        (scrip_df['ema_60'] > scrip_df['ema_100']) &
        (scrip_df['ema_100'] > scrip_df['ema_200'])
    )
    sell_signal = (
        (scrip_df['Close'] < scrip_df['ema_60']) &
        (scrip_df['ema_60'] < scrip_df['ema_100']) &
        (scrip_df['ema_100'] < scrip_df['ema_200'])
    )
    return buy_signal, sell_signal

def filter_volume_surge(scrip_df):
    buy_signal = (scrip_df['Volume'] > 2 * scrip_df['vol_ma_60']) & (scrip_df['MACD'] > scrip_df['Signal'])
    sell_signal = (scrip_df['Volume'] > 2 * scrip_df['vol_ma_60']) & (scrip_df['MACD'] < scrip_df['Signal'])
    return buy_signal, sell_signal

def filter_adx(scrip_df):
    buy_signal = (scrip_df['ADX'] > 25) & (scrip_df['plus_DI'] > scrip_df['minus_DI'])
    sell_signal = (scrip_df['ADX'] > 25) & (scrip_df['minus_DI'] > scrip_df['plus_DI'])
    return buy_signal, sell_signal

def filter_vwap(scrip_df, return_signals=False):
    """
    VWAP-based filter stub.
    Returns (buy_series, sell_series) or signals depending on caller expectations.
    Implement properly later; stub returns empty boolean series of same index.
    """
    import pandas as pd
    if scrip_df is None or scrip_df.empty:
        return pd.Series(dtype=bool), pd.Series(dtype=bool)
    buy = pd.Series(False, index=scrip_df.index)
    sell = pd.Series(False, index=scrip_df.index)
    if return_signals:
        return buy, sell
    return buy, sell

import pandas as pd

# If some enhanced filters are not yet implemented, create safe fallback stubs
# so imports like `from Class_Implimentation.support_files.updated_filters import filter_divergence`
# succeed and Streamlit won't fall back repeatedly.
_expected_filters = [
    "filter_divergence",
    "filter_vwap",
    "filter_golden_death_cross",
    "filter_breakout",
    "filter_trend_following",
    "filter_supertrend",
    # add any other expected filter function names here
]

def _make_stub(name):
    def stub(df, return_signals=False):
        """
        Auto-generated stub for {name}. Returns empty boolean buy/sell Series
        with the same index as df (or empty Series if df is None).
        """
        if df is None:
            return pd.Series(dtype=bool), pd.Series(dtype=bool)
        buy = pd.Series(False, index=df.index)
        sell = pd.Series(False, index=df.index)
        return (buy, sell) if return_signals else (buy, sell)
    stub.__name__ = name
    stub.__doc__ = f"Auto-generated stub for {name}"
    return stub

for _fname in _expected_filters:
    if _fname not in globals():
        globals()[_fname] = _make_stub(_fname)

# export filter_ names
__all__ = [n for n in globals().keys() if n.startswith("filter_")]
