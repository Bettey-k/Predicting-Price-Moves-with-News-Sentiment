import pandas as pd
import numpy as np
import talib

def add_all_indicators(df):
    """Add technical indicators to the DataFrame."""
    if df.empty:
        # Return empty DataFrame with indicator columns
        return pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume', 
                                   'SMA_20', 'SMA_50', 'RSI_14', 
                                   'MACD', 'MACD_signal', 'MACD_hist'])

    # Ensure we have all required columns
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Missing required columns. Need: {required_cols}")

    # Convert to numpy arrays with float64 dtype for TA-Lib
    close = np.asarray(df['Close'], dtype=np.float64)
    high = np.asarray(df['High'], dtype=np.float64)
    low = np.asarray(df['Low'], dtype=np.float64)
    volume = np.asarray(df['Volume'], dtype=np.float64)

    # Calculate indicators
    df = df.copy()
    df['SMA_20'] = talib.SMA(close, timeperiod=20)
    df['SMA_50'] = talib.SMA(close, timeperiod=50)
    df['RSI_14'] = talib.RSI(close, timeperiod=14)
    
    # MACD
    macd, signal, hist = talib.MACD(close)
    df['MACD'] = macd
    df['MACD_signal'] = signal
    df['MACD_hist'] = hist
    
    return df