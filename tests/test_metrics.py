# tests/test_metrics.py
import pandas as pd
import numpy as np
import pytest
from src.metrics import add_daily_returns, estimate_basic_metrics

def test_add_daily_returns():
    # Create test data
    test_data = {
        'Close': [100, 101, 102.01, 100.98, 103.03]
    }
    test_df = pd.DataFrame(test_data)
    
    # Add daily returns
    result_df = add_daily_returns(test_df)
    
    # Check if 'daily_return' column is added
    assert 'daily_return' in result_df.columns
    
    # Check calculations
    expected_returns = [np.nan, 0.01, 0.01, -0.0101, 0.0203]
    for i in range(1, len(test_df)):
        if not np.isnan(expected_returns[i]):
            assert abs(result_df['daily_return'].iloc[i] - expected_returns[i]) < 1e-6

def test_estimate_basic_metrics():
    # Create test data with known statistics
    test_data = {
        'Close': [100, 101, 102, 103, 104, 105]
    }
    test_df = pd.DataFrame(test_data)
    test_df = add_daily_returns(test_df)
    
    # Calculate metrics
    metrics = estimate_basic_metrics(test_df)
    
    # Check if expected metrics are present
    assert 'mean_return' in metrics
    assert 'volatility' in metrics
    
    # Check calculations (approximate due to floating point)
    assert abs(metrics['mean_return'] - 0.01) < 1e-6
    assert metrics['volatility'] > 0  # Should be positive

def test_estimate_basic_metrics_empty():
    # Test with empty DataFrame
    empty_df = pd.DataFrame()
    with pytest.raises(KeyError):
        estimate_basic_metrics(empty_df)