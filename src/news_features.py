# src/news_features.py
"""
Utility functions for news EDA:
- headline lengths (chars, words)
- datetime parts (date_only, weekday, hour)
- publisher counts
- publisher email domains
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_headline_lengths(
    df: pd.DataFrame,
    headline_col: str = "headline",
) -> pd.DataFrame:
    """
    Add headline_length_chars and headline_length_words columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with a headline column.
    headline_col : str
        Name of the column containing the headline text.

    Returns
    -------
    pd.DataFrame
        Copy of df with two extra columns:
        - headline_length_chars
        - headline_length_words
    """
    df_out = df.copy()
    series = df_out[headline_col].astype(str)

    df_out["headline_length_chars"] = series.str.len()
    df_out["headline_length_words"] = series.str.split().str.len()
    return df_out


def add_datetime_parts(
    df: pd.DataFrame,
    date_col: str = "date",
) -> pd.DataFrame:
    """
    Ensure `date_col` is datetime and add:
    - date_only  (date only, without time)
    - weekday    (e.g. 'Monday')
    - hour       (0–23)

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with a date column.
    date_col : str
        Name of the datetime column.

    Returns
    -------
    pd.DataFrame
        Copy of df with extra columns: date_only, weekday, hour.
    """
    df_out = df.copy()

    # Convert to datetime if needed
    if not np.issubdtype(df_out[date_col].dtype, np.datetime64):
        df_out[date_col] = pd.to_datetime(df_out[date_col], errors="coerce")

    df_out["date_only"] = df_out[date_col].dt.date
    df_out["weekday"] = df_out[date_col].dt.day_name()
    df_out["hour"] = df_out[date_col].dt.hour

    return df_out


def compute_publisher_counts(
    df: pd.DataFrame,
    publisher_col: str = "publisher",
    normalize: bool = False,
) -> pd.Series:
    """
    Count articles per publisher.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with a publisher column.
    publisher_col : str
        Name of the publisher column.
    normalize : bool
        If True, return proportions instead of counts.

    Returns
    -------
    pd.Series
        Publisher counts (or proportions), sorted descending.
    """
    return df[publisher_col].value_counts(normalize=normalize)


def add_publisher_domain(
    df: pd.DataFrame,
    publisher_col: str = "publisher",
    new_col: str = "publisher_domain",
) -> pd.DataFrame:
    """
    If publisher looks like an email (contains '@'),
    extract the domain part into a new column.

    Example:
    publisher = 'newsdesk@reuters.com' -> publisher_domain = 'reuters.com'

    Non-email publishers get NaN.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with a publisher column.
    publisher_col : str
        Name of the publisher column.
    new_col : str
        Name of the new column for domains.

    Returns
    -------
    pd.DataFrame
        Copy of df with the new domain column added.
    """
    df_out = df.copy()
    publisher_str = df_out[publisher_col].astype(str)

    is_email = publisher_str.str.contains("@")
    df_out[new_col] = np.where(
        is_email,
        publisher_str.str.split("@").str[-1],
        np.nan,
    )
    return df_out
