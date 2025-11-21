# src/metrics.py

import pandas as pd

def add_daily_returns(df: pd.DataFrame, close_col="Close") -> pd.DataFrame:
    df = df.copy()
    df["daily_return"] = df[close_col].pct_change()
    return df

def estimate_basic_metrics(df: pd.DataFrame) -> dict:
    """
    Compute average daily return and volatility.
    """
    ret = df["daily_return"].dropna()

    return {
        "mean_return": ret.mean(),
        "volatility": ret.std()
    }
