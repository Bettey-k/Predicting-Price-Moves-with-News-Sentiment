"""
Data loading and preprocessing utilities for stock price data.

This module provides functions to load, validate, and preprocess
financial time series data from various sources.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd
from pandas import DataFrame

# Define expected columns and their data types
EXPECTED_COLUMNS = {
    'Date': 'datetime64[ns]',
    'Open': 'float64',
    'High': 'float64',
    'Low': 'float64',
    'Close': 'float64',
    'Volume': 'float64'  # Using float to handle potential null values
}


def validate_price_data(df: DataFrame) -> None:
    """
    Validate the structure and content of price data.
    
    Args:
        df: DataFrame to validate
        
    Raises:
        ValueError: If data doesn't meet validation criteria
    """
    # Check for required columns
    missing_columns = set(EXPECTED_COLUMNS.keys()) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Check for missing values
    if df[list(EXPECTED_COLUMNS.keys())].isnull().any().any():
        raise ValueError("Data contains missing values")
    
    # Validate price relationships
    if (df['High'] < df['Low']).any():
        raise ValueError("High price cannot be less than Low price")
    if (df['Close'] > df['High']).any() or (df['Close'] < df['Low']).any():
        raise ValueError("Close price must be between High and Low")
    if (df['Volume'] < 0).any():
        raise ValueError("Volume cannot be negative")


def load_price_data(filepath: Union[str, Path]) -> DataFrame:
    """
    Load and validate stock price data from a CSV file.

    Args:
        filepath: Path to the CSV file containing price data.
                 Expected columns: 'Date', 'Open', 'High', 'Low', 'Close', 'Volume'

    Returns:
        DataFrame: Processed DataFrame with proper data types and index.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If required columns are missing or data is invalid.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    try:
        # Read the CSV file
        df = pd.read_csv(
            filepath,
            parse_dates=['Date'],
            date_parser=pd.to_datetime
        )

        # Convert data types
        for col, dtype in EXPECTED_COLUMNS.items():
            if col in df.columns:
                df[col] = df[col].astype(dtype)

        # Validate the data
        validate_price_data(df)

        # Sort by date to ensure chronological order
        df = df.sort_values('Date').reset_index(drop=True)

        return df

    except Exception as e:
        raise ValueError(f"Error loading price data: {str(e)}")


def save_processed_data(
    df: DataFrame, 
    filepath: Union[str, Path], 
    format: str = 'parquet',
    **kwargs
) -> None:
    """
    Save processed data to disk in the specified format.

    Args:
        df: DataFrame to save.
        filepath: Output file path (without extension).
        format: Output format ('parquet' or 'csv').
        **kwargs: Additional arguments passed to pandas to_* functions.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    if format == 'parquet':
        df.to_parquet(f"{filepath}.parquet", index=False, **kwargs)
    elif format == 'csv':
        df.to_csv(f"{filepath}.csv", index=False, **kwargs)
    else:
        raise ValueError(f"Unsupported format: {format}")


def load_processed_data(
    filepath: Union[str, Path], 
    format: str = 'parquet',
    **kwargs
) -> DataFrame:
    """
    Load processed data from disk.

    Args:
        filepath: Input file path (without extension).
        format: Input format ('parquet' or 'csv').
        **kwargs: Additional arguments passed to pandas read_* functions.

    Returns:
        DataFrame: The loaded data.
    """
    filepath = Path(filepath)
    
    if format == 'parquet':
        df = pd.read_parquet(f"{filepath}.parquet", **kwargs)
    elif format == 'csv':
        df = pd.read_csv(
            f"{filepath}.csv", 
            parse_dates=['Date'],
            **kwargs
        )
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    # Ensure proper data types
    for col, dtype in EXPECTED_COLUMNS.items():
        if col in df.columns:
            df[col] = df[col].astype(dtype)
    
    return df
