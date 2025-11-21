# tests/test_metrics.py
import pandas as pd
import numpy as np
import pytest
from src.metrics import add_daily_returns, estimate_basic_metrics

def test_add_daily_returns():
    test_data = {
        'Close': [100, 101, 102.01, 100.98, 103.03]
    }
    test_df = pd.DataFrame(test_data)
    
    result_df = add_daily_returns(test_df)
    
    assert 'daily_return' in result_df.columns
    assert result_df['daily_return'].iloc[1] == pytest.approx(0.01, abs=1e-4)
    assert result_df['daily_return'].iloc[2] == pytest.approx(0.01, abs=1e-4)
    assert result_df['daily_return'].iloc[3] == pytest.approx(-0.0101, abs=1e-4)
    assert result_df['daily_return'].iloc[4] == pytest.approx(0.0203, abs=1e-4)

def test_estimate_basic_metrics():
    test_data = {
        'Close': [100, 101, 102, 103, 104, 105]
    }
    test_df = pd.DataFrame(test_data)
    test_df = add_daily_returns(test_df)
    
    metrics = estimate_basic_metrics(test_df)
    
    assert 'mean_return' in metrics
    assert 'volatility' in metrics
    assert metrics['mean_return'] == pytest.approx(0.0098058, abs=1e-6)
    assert metrics['volatility'] > 0