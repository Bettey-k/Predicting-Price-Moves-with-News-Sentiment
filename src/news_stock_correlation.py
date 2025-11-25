# src/news_stock_correlation.py

from __future__ import annotations

import pandas as pd


def align_daily_sentiment_and_returns(
    daily_sentiment: pd.DataFrame,
    daily_returns: pd.DataFrame,
    sentiment_date_col: str = "date_only",
    returns_date_col: str = "Date",
) -> pd.DataFrame:
    """
    Merge daily sentiment and daily returns on date.

    daily_sentiment: must have columns [date_only, avg_sentiment]
    daily_returns: must have columns [Date, daily_return]

    Returns:
    - DataFrame with columns:
      date_only, avg_sentiment, daily_return
    """
    ds = daily_sentiment.copy()
    dr = daily_returns.copy()

    # ensure date-only in both
    ds[sentiment_date_col] = pd.to_datetime(ds[sentiment_date_col]).dt.date
    dr["date_only"] = pd.to_datetime(dr[returns_date_col]).dt.date

    merged = pd.merge(ds, dr, on="date_only", how="inner")

    # keep only needed columns
    merged = merged[["date_only", "avg_sentiment", "headline_count", "daily_return"]]

    # drop rows where return or sentiment is missing
    merged = merged.dropna(subset=["avg_sentiment", "daily_return"])

    return merged


def compute_sentiment_return_correlation(
    df: pd.DataFrame,
    sentiment_col: str = "avg_sentiment",
    return_col: str = "daily_return",
) -> float:
    """
    Compute Pearson correlation between daily sentiment and daily returns.
    """
    return df[sentiment_col].corr(df[return_col])
