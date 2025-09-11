#================================================================
# filters.py
#================================================================

def filter_basic(scrip_df):
    """
    Basic long-term filter:
    Buy when RSI < 40, Close is below EMA(60), MACD > Signal, and 
    Close is above Bollinger Lower band (with a slight premium).
    Sell when RSI > 60, Close is above EMA(60), MACD < Signal, and 
    Close is below Bollinger Upper band.
    """
    buy_signal = ((scrip_df['RSI'] < 40) &
                  (scrip_df['Close'] <= scrip_df['ema_60']) &
                  (scrip_df['MACD'] > scrip_df['Signal']) &
                  (scrip_df['Close'] > scrip_df['Bollinger_Lower'] * 1.02))

    sell_signal = ((scrip_df['RSI'] > 60) &
                   (scrip_df['Close'] >= scrip_df['ema_60']) &
                   (scrip_df['MACD'] < scrip_df['Signal']) &
                   (scrip_df['Close'] < scrip_df['Bollinger_Upper'] * 0.98))
    return buy_signal, sell_signal


def filter_aggressive(scrip_df):
    """
    Aggressive long-term filter:
    Buy when RSI < 35, Close is below EMA(90), MACD > Signal, and 
    Volume is above the 60-day EMA of volume.
    Sell when RSI > 65, Close is above EMA(90), MACD < Signal, and 
    Volume is below the 60-day EMA of volume.
    """
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
    """
    Momentum filter:
    Buy when Rate of Change (ROC) > 0 and RSI < 45.
    Sell when ROC < 0 and RSI > 55.
    (Here, ROC and RSI are computed using longer-term periods.)
    """
    buy_signal = (scrip_df['ROC'] > 0) & (scrip_df['RSI'] < 45)
    sell_signal = (scrip_df['ROC'] < 0) & (scrip_df['RSI'] > 55)
    return buy_signal, sell_signal


def filter_breakout(scrip_df):
    """
    Breakout filter:
    Buy when Close > 60-day High and Volume > 60-day average volume.
    Sell when Close < 60-day Low and Volume > 60-day average volume.
    """
    buy_signal = (scrip_df['Close'] > scrip_df['60_day_high']) & (scrip_df['Volume'] > scrip_df['vol_ma_60'])
    sell_signal = (scrip_df['Close'] < scrip_df['60_day_low']) & (scrip_df['Volume'] > scrip_df['vol_ma_60'])
    return buy_signal, sell_signal


def filter_mean_reversion(scrip_df):
    """
    Mean reversion filter:
    Buy when Close < Bollinger Lower band and RSI < 30.
    Sell when Close > Bollinger Upper band and RSI > 70.
    (Bollinger Bands are now computed over a 60-day period.)
    """
    buy_signal = (scrip_df['Close'] < scrip_df['Bollinger_Lower']) & (scrip_df['RSI'] < 30)
    sell_signal = (scrip_df['Close'] > scrip_df['Bollinger_Upper']) & (scrip_df['RSI'] > 70)
    return buy_signal, sell_signal


def filter_trend_following(scrip_df):
    """
    Trend following filter:
    Buy when Close > EMA(60) > EMA(100) > EMA(200).
    Sell when Close < EMA(60) < EMA(100) < EMA(200).
    """
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
    """
    Volume surge filter:
    Buy when Volume > 2x the 60-day average volume and MACD > Signal.
    Sell when Volume > 2x the 60-day average volume and MACD < Signal.
    """
    buy_signal = (scrip_df['Volume'] > 2 * scrip_df['vol_ma_60']) & (scrip_df['MACD'] > scrip_df['Signal'])
    sell_signal = (scrip_df['Volume'] > 2 * scrip_df['vol_ma_60']) & (scrip_df['MACD'] < scrip_df['Signal'])
    return buy_signal, sell_signal


def filter_vwap(scrip_df):
    """
    VWAP-based filter:
    Buy when Close > VWAP and RSI < 50.
    Sell when Close < VWAP and RSI > 50.
    (VWAP is typically an intraday metric, but may still be useful in some long-term contexts.)
    """
    buy_signal = (scrip_df['Close'] > scrip_df['VWAP']) & (scrip_df['RSI'] < 50)
    sell_signal = (scrip_df['Close'] < scrip_df['VWAP']) & (scrip_df['RSI'] > 50)
    return buy_signal, sell_signal


def filter_golden_death_cross(scrip_df):
    """
    Golden Cross / Death Cross filter:
    Buy when EMA(60) is above EMA(200) (indicative of a Golden Cross).
    Sell when EMA(60) is below EMA(200) (indicative of a Death Cross).
    """
    buy_signal = scrip_df['ema_60'] > scrip_df['ema_200']
    sell_signal = scrip_df['ema_60'] < scrip_df['ema_200']
    return buy_signal, sell_signal


def filter_divergence(scrip_df):
    """
    Divergence filter:
    Simplified version:
    - Buy when RSI is oversold (RSI < 30) and MACD > Signal (potential bullish divergence).
    - Sell when RSI is overbought (RSI > 70) and MACD < Signal (potential bearish divergence).
    Note: True divergence detection may require more complex historical comparisons.
    """
    buy_signal = (scrip_df['RSI'] < 30) & (scrip_df['MACD'] > scrip_df['Signal'])
    sell_signal = (scrip_df['RSI'] > 70) & (scrip_df['MACD'] < scrip_df['Signal'])
    return buy_signal, sell_signal


def filter_adx(scrip_df):
    """
    ADX trend strength filter:
    Buy when ADX > 25 and the positive directional indicator (+DI) is greater than the negative (-DI).
    Sell when ADX > 25 and -DI is greater than +DI.
    """
    buy_signal = (scrip_df['ADX'] > 25) & (scrip_df['plus_DI'] > scrip_df['minus_DI'])
    sell_signal = (scrip_df['ADX'] > 25) & (scrip_df['minus_DI'] > scrip_df['plus_DI'])
    return buy_signal, sell_signal


def filter_supertrend(scrip_df):
    """
    Supertrend-based filter:
    Buy when Close > Supertrend indicator.
    Sell when Close < Supertrend indicator.
    """
    buy_signal = scrip_df['Close'] > scrip_df['Supertrend']
    sell_signal = scrip_df['Close'] < scrip_df['Supertrend']
    return buy_signal, sell_signal
