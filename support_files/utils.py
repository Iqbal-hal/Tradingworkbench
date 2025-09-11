# utils.py
import functools
import pandas as pd

def sanitize_signals(func):
    """Decorator to ensure clean boolean signals"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        buy_signal, sell_signal = func(*args, **kwargs)
        
        # Convert to boolean and handle NaNs
        buy_signal = buy_signal.fillna(False).astype(bool)
        sell_signal = sell_signal.fillna(False).astype(bool)
        
        return buy_signal, sell_signal
    return wrapper
