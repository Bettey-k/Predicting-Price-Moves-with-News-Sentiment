
import pandas as pd
import numpy as np

from src.news_features import (
    add_headline_lengths,
    add_datetime_parts,
    add_publisher_domain,
    compute_publisher_counts,
)


def test_add_headline_lengths_adds_columns_and_values():
    df = pd.DataFrame(
        {"headline": ["Hello world", "Test headline here"]}
    )

    out = add_headline_lengths(df)

    # Columns exist
    assert "headline_length_chars" in out.columns
    assert "headline_length_words" in out.columns

    # Values are correct for first row
    assert out.loc[0, "headline_length_words"] == 2
    assert out.loc[1, "headline_length_words"] == 3

    # chars roughly correct (we don't check exact count for spaces)
    assert out.loc[0, "headline_length_chars"] == len("Hello world")


def test_add_datetime_parts_creates_expected_columns():
    df = pd.DataFrame(
        {"date": ["2024-01-01 10:30:00", "2024-01-02 15:45:00"]}
    )

    out = add_datetime_parts(df)

    # New columns exist
    for col in ["date_only", "weekday", "hour"]:
        assert col in out.columns

    # Type of date_only is date, hour matches
    assert out.loc[0, "hour"] == 10
    assert out.loc[1, "hour"] == 15
    assert hasattr(out.loc[0, "date_only"], "year")


def test_add_publisher_domain_extracts_email_domain():
    df = pd.DataFrame(
        {
            "publisher": [
                "newsdesk@reuters.com",
                "Bloomberg",
                "editor@nytimes.com",
            ]
        }
    )

    out = add_publisher_domain(df)

    assert out.loc[0, "publisher_domain"] == "reuters.com"
    assert pd.isna(out.loc[1, "publisher_domain"])
    assert out.loc[2, "publisher_domain"] == "nytimes.com"


def test_compute_publisher_counts_counts_correctly():
    df = pd.DataFrame(
        {"publisher": ["A", "B", "A", "C", "A", "B"]}
    )

    counts = compute_publisher_counts(df)

    # A appears 3 times, B 2 times, C 1 time
    assert counts["A"] == 3
    assert counts["B"] == 2
    assert counts["C"] == 1

    # Series index is sorted by count descending
    assert list(counts.index) == ["A", "B", "C"]
