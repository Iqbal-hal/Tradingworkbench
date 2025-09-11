
import talib
import numpy as np
def compute_indicators(scrip_df):
    """
    Compute all required technical indicators once and store them in the DataFrame.
    This function is called once per scrip to avoid re-calculation during interactive events.
    """
    # Calculate various EMAs for closing prices
    scrip_df['ema_200'] = talib.EMA(scrip_df['Close'], timeperiod=200)
    scrip_df['ema_100'] = talib.EMA(scrip_df['Close'], timeperiod=100)
    scrip_df['ema_90']  = talib.EMA(scrip_df['Close'], timeperiod=90)
    scrip_df['ema_60']  = talib.EMA(scrip_df['Close'], timeperiod=60)
    scrip_df['ema_50']  = talib.EMA(scrip_df['Close'], timeperiod=50)
    scrip_df['ema_30']  = talib.EMA(scrip_df['Close'], timeperiod=30)
    scrip_df['ema_20']  = talib.EMA(scrip_df['Close'], timeperiod=20)
    scrip_df['ema_9']   = talib.EMA(scrip_df['Close'], timeperiod=9)
    
    # EMAs for Volume
    scrip_df['ema_vol'] = talib.EMA(scrip_df['Volume'], timeperiod=9)
    scrip_df['volume_ema_20'] = talib.EMA(scrip_df['Volume'], timeperiod=20)
    scrip_df['volume_ema_60'] = talib.EMA(scrip_df['Volume'], timeperiod=60)
    
    # Simple 60-day moving average for Volume
    scrip_df['vol_ma_60'] = scrip_df['Volume'].rolling(window=60).mean()
    
    # RSI Calculation
    scrip_df['RSI'] = talib.RSI(scrip_df['Close'], timeperiod=14)
    
    # MACD Calculation (MACD, Signal, Histogram)
    scrip_df['MACD'], scrip_df['Signal'], scrip_df['Hist'] = talib.MACD(
        scrip_df['Close'], fastperiod=12, slowperiod=26, signalperiod=9
    )
    
    # Bollinger Bands (using period=20 and 2 standard deviations)
    n = 20
    std_dev = 2
    upper, middle, lower = talib.BBANDS(scrip_df['Close'],
                                        timeperiod=n,
                                        nbdevup=std_dev,
                                        nbdevdn=std_dev,
                                        matype=0)
    scrip_df['Bollinger_Upper'] = upper
    scrip_df['Bollinger_Middle'] = middle
    scrip_df['Bollinger_Lower'] = lower
    
    # Rate of Change (ROC) with a period of 10 (adjust as needed)
    scrip_df['ROC'] = talib.ROC(scrip_df['Close'], timeperiod=10)
    
    # 60-day High and Low for breakout filters
    scrip_df['60_day_high'] = scrip_df['Close'].rolling(window=60).max()
    scrip_df['60_day_low'] = scrip_df['Close'].rolling(window=60).min()
    
    # VWAP Calculation:
    # Typical price = (High + Low + Close) / 3
    # VWAP = cumulative (Typical Price * Volume) / cumulative Volume
    tp = (scrip_df['High'] + scrip_df['Low'] + scrip_df['Close']) / 3
    scrip_df['VWAP'] = (tp * scrip_df['Volume']).cumsum() / scrip_df['Volume'].cumsum()
    
    # ADX and Directional Indicators (using a period of 14)
    scrip_df['ADX'] = talib.ADX(scrip_df['High'], scrip_df['Low'], scrip_df['Close'], timeperiod=14)
    scrip_df['plus_DI'] = talib.PLUS_DI(scrip_df['High'], scrip_df['Low'], scrip_df['Close'], timeperiod=14)
    scrip_df['minus_DI'] = talib.MINUS_DI(scrip_df['High'], scrip_df['Low'], scrip_df['Close'], timeperiod=14)
    
    # Supertrend Calculation (custom implementation)
    # Parameters for Supertrend
    period = 10
    multiplier = 3
    
    # ATR calculation for Supertrend
    scrip_df['ATR'] = talib.ATR(scrip_df['High'], scrip_df['Low'], scrip_df['Close'], timeperiod=period)
    
    # Basic Upper and Lower Bands
    scrip_df['Basic_Upper'] = ((scrip_df['High'] + scrip_df['Low']) / 2) + (multiplier * scrip_df['ATR'])
    scrip_df['Basic_Lower'] = ((scrip_df['High'] + scrip_df['Low']) / 2) - (multiplier * scrip_df['ATR'])
    
    # Initialize Final Bands with the basic values
    scrip_df['Final_Upper'] = scrip_df['Basic_Upper']
    scrip_df['Final_Lower'] = scrip_df['Basic_Lower']
    
    # Compute Final Bands iteratively
    for i in range(1, len(scrip_df)):
        # Final Upper Band
        if (scrip_df['Basic_Upper'].iloc[i] < scrip_df['Final_Upper'].iloc[i-1]) or (scrip_df['Close'].iloc[i-1] > scrip_df['Final_Upper'].iloc[i-1]):
            scrip_df.at[scrip_df.index[i], 'Final_Upper'] = scrip_df['Basic_Upper'].iloc[i]
        else:
            scrip_df.at[scrip_df.index[i], 'Final_Upper'] = scrip_df['Final_Upper'].iloc[i-1]
        
        # Final Lower Band
        if (scrip_df['Basic_Lower'].iloc[i] > scrip_df['Final_Lower'].iloc[i-1]) or (scrip_df['Close'].iloc[i-1] < scrip_df['Final_Lower'].iloc[i-1]):
            scrip_df.at[scrip_df.index[i], 'Final_Lower'] = scrip_df['Basic_Lower'].iloc[i]
        else:
            scrip_df.at[scrip_df.index[i], 'Final_Lower'] = scrip_df['Final_Lower'].iloc[i-1]
    
    # Determine Supertrend: if Close <= Final_Upper, then Supertrend equals Final_Upper; otherwise, it equals Final_Lower.
    supertrend = [np.nan] * len(scrip_df)
    for i in range(len(scrip_df)):
        if i == 0:
            supertrend[i] = np.nan
        else:
            if scrip_df['Close'].iloc[i] <= scrip_df['Final_Upper'].iloc[i]:
                supertrend[i] = scrip_df['Final_Upper'].iloc[i]
            else:
                supertrend[i] = scrip_df['Final_Lower'].iloc[i]
    scrip_df['Supertrend'] = supertrend
    scrip_ta_df=scrip_df

    return scrip_ta_df
