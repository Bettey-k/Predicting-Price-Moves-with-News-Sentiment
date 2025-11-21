# src/metrics.py
import pandas as pd
import numpy as np

def add_daily_returns(df, price_col='Close'):
    """Add daily returns to the DataFrame."""
    df = df.copy()
    df['daily_return'] = df[price_col].pct_change()
    return df

def estimate_basic_metrics(df):
    """Calculate basic financial metrics."""
    if df.empty or 'daily_return' not in df.columns:
        return {'mean_return': np.nan, 'volatility': np.nan}
    
    returns = df['daily_return'].dropna()
    if len(returns) == 0:
        return {'mean_return': np.nan, 'volatility': np.nan}
    
    return {
        'mean_return': float(returns.mean()),
        'volatility': float(returns.std())
    }