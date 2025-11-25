# tests/test_sentiment.py

import pandas as pd

from src.sentiment import add_textblob_sentiment, aggregate_daily_sentiment


def test_add_textblob_sentiment_adds_columns_and_values():
    # Create a tiny DataFrame with example headlines and dates
    df = pd.DataFrame(
        {
            "headline": [
                "Stock jumps after strong earnings",
                "Company faces regulatory investigation",
            ],
            "date": ["2024-01-01", "2024-01-02"],
        }
    )

    # Apply sentiment function
    out = add_textblob_sentiment(df, text_col="headline")

    # New columns should exist
    assert "sentiment_polarity" in out.columns
    assert "sentiment_subjectivity" in out.columns

    # Polarity values should be between -1 and 1
    assert out["sentiment_polarity"].between(-1, 1).all()


def test_aggregate_daily_sentiment_computes_daily_mean():
    # Example with two headlines on the same day
    df = pd.DataFrame(
        {
            "headline": [
                "Good news for investors",
                "Bad news for investors",
            ],
            "date": ["2024-01-01", "2024-01-01"],
        }
    )

    # First add sentiment scores
    df_sent = add_textblob_sentiment(df, text_col="headline")

    # Then aggregate by day
    daily = aggregate_daily_sentiment(
        df_sent,
        date_col="date",
        sentiment_col="sentiment_polarity",
    )

    # We expect exactly one row (one day)
    assert len(daily) == 1

    # Columns should include date_only, avg_sentiment and headline_count
    assert "date_only" in daily.columns
    assert "avg_sentiment" in daily.columns
    assert "headline_count" in daily.columns

    # headline_count should be 2 for that day
    assert daily.loc[0, "headline_count"] == 2
