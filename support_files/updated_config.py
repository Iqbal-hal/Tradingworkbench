# =================================
# config.py - Enhanced Optimized Version
# =================================
from importlib import import_module
from support_files.utils import sanitize_signals

# Import and decorate all filters dynamically
filters_module = import_module('support_files.updated_filters')  # Updated to use enhanced_filters

# List of all available filters (original + enhanced)
FILTER_NAMES = [
    # Original filters
    'filter_basic',
    'filter_aggressive',
    'filter_momentum',
    'filter_breakout',
    'filter_mean_reversion',
    'filter_trend_following',
    'filter_volume_surge',
    'filter_vwap',
    'filter_golden_death_cross',
    'filter_divergence',
    'filter_adx',
    'filter_supertrend',
    
    # Enhanced filters for profit optimization
    'filter_ensemble_weighted',           # Combines multiple filters with weighted voting
    'filter_adaptive_rsi',               # RSI with dynamic thresholds based on volatility
    'filter_multi_timeframe_confluence', # Multi-timeframe trend alignment
    'filter_risk_adjusted',              # Risk-aware filter with drawdown consideration
    'filter_volatility_breakout',        # Enhanced breakout with volatility confirmation
    'filter_momentum_with_support_resistance'  # Momentum with dynamic S/R levels
]

# Create decorated filter dictionary
AVAILABLE_FILTERS = {
    name: sanitize_signals(getattr(filters_module, name))
    for name in FILTER_NAMES
    if hasattr(filters_module, name)  # Safety check for missing filters
}

# Enhanced filter recommendations by market condition
FILTER_RECOMMENDATIONS = {
    'trending_market': ['filter_trend_following', 'filter_breakout', 'filter_multi_timeframe_confluence'],
    'sideways_market': ['filter_mean_reversion', 'filter_momentum_with_support_resistance'],
    'volatile_market': ['filter_adaptive_rsi', 'filter_risk_adjusted'],
    'balanced_approach': ['filter_ensemble_weighted', 'filter_multi_timeframe_confluence'],
    'conservative': ['filter_risk_adjusted', 'filter_adaptive_rsi'],
    'aggressive': ['filter_volatility_breakout', 'filter_volume_surge']
}

# Active configuration parameters - Updated with enhanced filter
ACTIVE_FILTER = 'filter_ensemble_weighted'
FILTER_ENABLED = True
LOGGING_ENABLED = True
ENABLE_DETAILED_LOGGING = True

# Enhanced risk management parameters
USE_SUPPORT_TRAILING_EXITS = True
TRAILING_STOP_PERCENT = 15.0
SUPPORT_TYPE = 'min'
MIN_HOLDING_PERIOD = 120
MIN_PROFIT_PERCENTAGE = 25.0
MIN_FILTER_SELL_PROFIT = 15.0

# New profit optimization parameters
POSITION_SIZING_ENABLED = True
MAX_POSITION_SIZE = 0.05  # 5% max per position
VOLATILITY_ADJUSTMENT = True
CORRELATION_LIMIT = 0.7  # Avoid highly correlated positions

# Enhanced filter-specific parameters
ENSEMBLE_WEIGHTS = {
    'basic': 0.15,
    'aggressive': 0.10,
    'momentum': 0.15,
    'breakout': 0.12,
    'mean_reversion': 0.08,
    'trend_following': 0.20,
    'volume_surge': 0.10,
    'adx': 0.10
}

ADAPTIVE_RSI_LOOKBACK = 20
VOLATILITY_THRESHOLD = 1.5
MAX_DRAWDOWN_THRESHOLD = 0.12  # 12% max drawdown before defensive mode

# Signal quality filters
MIN_SIGNAL_STRENGTH = 0.35  # Minimum ensemble score for trades
MIN_VOLUME_MULTIPLE = 1.2   # Require volume > 1.2x average
REQUIRE_TREND_CONFIRMATION = True

# Portfolio optimization settings
ENABLE_SECTOR_DIVERSIFICATION = True
MAX_POSITIONS_PER_SECTOR = 3
REBALANCING_FREQUENCY = 'weekly'  # 'daily', 'weekly', 'monthly'
PORTFOLIO_STRATEGY = 'score_rank_claude'

# Explicit exports for clean namespace
__all__ = [
    'AVAILABLE_FILTERS',
    'FILTER_RECOMMENDATIONS',
    'ACTIVE_FILTER',
    'FILTER_ENABLED',
    'LOGGING_ENABLED',
    'ENABLE_DETAILED_LOGGING',
    'USE_SUPPORT_TRAILING_EXITS',
    'TRAILING_STOP_PERCENT',
    'SUPPORT_TYPE',
    'MIN_HOLDING_PERIOD',
    'MIN_PROFIT_PERCENTAGE',
    'MIN_FILTER_SELL_PROFIT',
    'POSITION_SIZING_ENABLED',
    'MAX_POSITION_SIZE',
    'ENSEMBLE_WEIGHTS',
    'MIN_SIGNAL_STRENGTH',
    'PORTFOLIO_STRATEGY'
]

# Maintain original import structure for compatibility
try:
    from support_files.updated_filters import (
        # Original filters
        filter_basic,
        filter_aggressive,
        filter_momentum,
        filter_breakout,
        filter_mean_reversion,
        filter_trend_following,
        filter_volume_surge,
        filter_vwap,
        filter_golden_death_cross,
        filter_divergence,
        filter_adx,
        filter_supertrend,
        
        # Enhanced filters
        filter_ensemble_weighted,
        filter_adaptive_rsi,
        filter_multi_timeframe_confluence,
        filter_risk_adjusted,
        filter_volatility_breakout,
        filter_momentum_with_support_resistance
    )
    print("✓ Enhanced filters loaded successfully")
except ImportError as e:
    print(f"⚠  Warning: Could not import enhanced filters: {e}")
    print("   Falling back to original filters...")
    try:
        from support_files.filters import (
            filter_basic,
            filter_aggressive,
            filter_momentum,
            filter_breakout,
            filter_mean_reversion,
            filter_trend_following,
            filter_volume_surge,
            filter_vwap,
            filter_golden_death_cross,
            filter_divergence,
            filter_adx
        )
        ACTIVE_FILTER = 'filter_adx'  # Fallback to original
        print("✓ Original filters loaded as fallback")
    except ImportError:
        print("✖ Error: Could not load any filters!")

# GUI settings:
main_plot_enabled = True     # Main chart: close price, EMAs, Bollinger, etc.
bollinger_enabled = True
ema_9_enabled = True
ema_20_enabled = True
ema_50_enabled = True
ema_100_enabled = True
pe_enabled = True
annotation_text_enabled = True
backtest_annotation_enabled = False
portfolio_enabled = True
close_price_enabled = True
axv_line_enabled = True
candlestick_enabled = True
# candlestick_tf can be "Daily", "Weekly", or "Monthly". Default is "Daily".
candlestick_tf = "Daily"

# Font sizes and layout settings:
annotation_fontsize = 7     
annotation_orient = 7
limt_multiplier = 1.1       
xaxis_label_fontsize = 8    

# Extra indicator toggles:
macd_enabled = False        
rsi_enabled = False         
atr_enabled = False   

# Load portfolio details globally.
portfolio_details = None

volume_twin_enabled = True  # For twin volume in main plot
volume_independent_enabled = False  # For separate volume plot

# Plot height ratios for multiple panels:
plot_ratios = {
    "main": 1.5,
    "volume": 0.7,  # Add volume ratio
    "macd": 1.0,
    "rsi": 1.0,
    "atr": 1.0
}

legend_visible = True  # Initially, legend is shown.
counter = 0  # used by backtester.py
folio_final_value = 0

# Helper function to get recommended filters
def get_recommended_filters(market_condition='balanced_approach'):
    """
    Get recommended filters based on market condition
    
    Args:
        market_condition: 'trending_market', 'sideways_market', 'volatile_market', 
                         'balanced_approach', 'conservative', 'aggressive'
    
    Returns:
        List of recommended filter names
    """
    return FILTER_RECOMMENDATIONS.get(market_condition, ['filter_ensemble_weighted'])

def switch_filter_by_market_condition(condition):
    """
    Dynamically switch active filter based on market condition
    """
    global ACTIVE_FILTER
    recommended = get_recommended_filters(condition)
    if recommended and recommended[0] in AVAILABLE_FILTERS:
        ACTIVE_FILTER = recommended[0]
        print(f"Switched to {ACTIVE_FILTER} for {condition}")
        return ACTIVE_FILTER
    else:
        print(f"⚠  No suitable filter found for {condition}, keeping {ACTIVE_FILTER}")
        return ACTIVE_FILTER

# Performance tracking for filter optimization
FILTER_PERFORMANCE_TRACKING = True
PERFORMANCE_LOG_FILE = "logs/filter_performance.json"