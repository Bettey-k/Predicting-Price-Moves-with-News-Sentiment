
import pandas as pd
import talib

def add_sma(df: pd.DataFrame, close_col="Close") -> pd.DataFrame:
    df = df.copy()
    close = df[close_col].values
    df["SMA_20"] = talib.SMA(close, 20)
    df["SMA_50"] = talib.SMA(close, 50)
    return df

def add_rsi(df: pd.DataFrame, close_col="Close") -> pd.DataFrame:
    df = df.copy()
    close = df[close_col].values
    df["RSI_14"] = talib.RSI(close, 14)
    return df

def add_macd(df: pd.DataFrame, close_col="Close") -> pd.DataFrame:
    df = df.copy()
    close = df[close_col].values
    macd, macd_signal, macd_hist = talib.MACD(close)
    df["MACD"] = macd
    df["MACD_signal"] = macd_signal
    df["MACD_hist"] = macd_hist
    return df

def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = add_sma(df)
    df = add_rsi(df)
    df = add_macd(df)
    return df
