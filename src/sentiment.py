# src/sentiment.py

from __future__ import annotations

from typing import Tuple

import pandas as pd
from textblob import TextBlob


def add_textblob_sentiment(
    df: pd.DataFrame,
    text_col: str = "headline",
) -> pd.DataFrame:
    """
    Add TextBlob sentiment scores (polarity, subjectivity) for each row.

    polarity   ∈ [-1, 1]  (negative → positive)
    subjectivity ∈ [0, 1] (objective → subjective)
    """
    df_out = df.copy()
    texts = df_out[text_col].astype(str)

    polarities = []
    subjectivities = []

    for t in texts:
        blob = TextBlob(t)
        s = blob.sentiment
        polarities.append(s.polarity)
        subjectivities.append(s.subjectivity)

    df_out["sentiment_polarity"] = polarities
    df_out["sentiment_subjectivity"] = subjectivities

    return df_out


def aggregate_daily_sentiment(
    df: pd.DataFrame,
    date_col: str = "date",
    sentiment_col: str = "sentiment_polarity",
) -> pd.DataFrame:
    """
    Aggregate headline-level sentiment into daily averages.

    Returns a DataFrame with:
    - date_only
    - avg_sentiment
    - headline_count
    """
    df_out = df.copy()

    # ensure datetime
    df_out[date_col] = pd.to_datetime(df_out[date_col], errors="coerce")
    df_out["date_only"] = df_out[date_col].dt.date

    daily = (
        df_out
        .groupby("date_only")[sentiment_col]
        .agg(["mean", "count"])
        .reset_index()
    )

    daily = daily.rename(
        columns={
            "mean": "avg_sentiment",
            "count": "headline_count",
        }
    )

    return daily
