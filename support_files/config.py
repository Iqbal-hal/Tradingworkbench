# =================================
# config.py - Full Optimized Version
# =================================
from importlib import import_module
from support_files.utils import sanitize_signals

# Import and decorate all filters dynamically
filters_module = import_module('support_files.filters')

# List of all available filters
FILTER_NAMES = [
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
    'filter_adx'    
]

# Create decorated filter dictionary
AVAILABLE_FILTERS = {
    name: sanitize_signals(getattr(filters_module, name))
    for name in FILTER_NAMES
}

# Active configuration parameters
ACTIVE_FILTER = 'filter_adx'
FILTER_ENABLED = True
LOGGING_ENABLED = True
ENABLE_DETAILED_LOGGING = True

# Risk management parameters
USE_SUPPORT_TRAILING_EXITS = True
TRAILING_STOP_PERCENT = 20.0
SUPPORT_TYPE = 'min'
MIN_HOLDING_PERIOD = 60
MIN_PROFIT_PERCENTAGE = 40.0
MIN_FILTER_SELL_PROFIT = 20.0

# Explicit exports for clean namespace
__all__ = [
    'AVAILABLE_FILTERS',
    'ACTIVE_FILTER',
    'FILTER_ENABLED',
    'LOGGING_ENABLED',
    'ENABLE_DETAILED_LOGGING',
    'USE_SUPPORT_TRAILING_EXITS',
    'TRAILING_STOP_PERCENT',
    'SUPPORT_TYPE',
    'MIN_HOLDING_PERIOD',
    'MIN_PROFIT_PERCENTAGE'
]

# Maintain original import structure for compatibility
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

# GUI settings:
main_plot_enabled = True     # Main chart: close price, EMAs, Bollinger, etc.
bollinger_enabled = True
ema_9_enabled = True
ema_20_enabled = True
ema_50_enabled = True
ema_100_enabled = True
pe_enabled = True
annotation_text_enabled = True
backtest_annotation_enabled=False
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
portfolio_details =None

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
counter=0 # used by backtester.py
folio_final_value=0
