# tests/test_technical_indicators.py
import pandas as pd
import numpy as np
import pytest
from src.technical_indicators import add_all_indicators

def test_add_all_indicators():
    # Create test data
    test_data = {
        'Close': [100, 101, 102, 101, 103, 104, 103, 105, 106, 107] * 10,
        'High': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110] * 10,
        'Low': [99, 100, 101, 100, 102, 103, 102, 104, 105, 106] * 10,
        'Volume': [1000000] * 100
    }
    test_df = pd.DataFrame(test_data)
    
    # Add indicators
    result_df = add_all_indicators(test_df)
    
    # Check if all expected columns are added
    expected_columns = ['SMA_20', 'SMA_50', 'RSI_14', 'MACD', 'MACD_signal', 'MACD_hist']
    for col in expected_columns:
        assert col in result_df.columns
    
    # Check SMA calculations
    assert not result_df['SMA_20'].isnull().all()  # Some SMAs should be calculated
    assert result_df['SMA_20'].iloc[19] == test_df['Close'].iloc[0:20].mean()  # First valid SMA20
    
    # Check RSI range (should be between 0 and 100)
    assert 0 <= result_df['RSI_14'].dropna().iloc[0] <= 100

def test_add_all_indicators_empty():
    # Test with empty DataFrame
    empty_df = pd.DataFrame()
    with pytest.raises(ValueError):
        add_all_indicators(empty_df)