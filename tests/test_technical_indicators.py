# tests/test_technical_indicators.py
import pandas as pd
import pytest
from src.technical_indicators import add_all_indicators

def test_add_all_indicators_empty():
    # Test with empty DataFrame with required columns
    empty_df = pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume'])
    result_df = add_all_indicators(empty_df)
    
    # Should return empty DataFrame with all columns
    assert result_df.empty
    expected_columns = ['Open', 'High', 'Low', 'Close', 'Volume', 
                       'SMA_20', 'SMA_50', 'RSI_14', 
                       'MACD', 'MACD_signal', 'MACD_hist']
    assert all(col in result_df.columns for col in expected_columns)

def test_add_all_indicators_missing_columns():
    # Test with missing required columns
    df = pd.DataFrame({'Close': [100, 101, 102]})
    with pytest.raises(ValueError):
        add_all_indicators(df)