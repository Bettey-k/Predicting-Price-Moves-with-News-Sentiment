import pandas as pd
import pytest
import os
from src.data_loading import load_price_data

def test_load_price_data():
    # Create a sample CSV file for testing
    test_data = {
        'Date': ['2023-01-01', '2023-01-02'],
        'Open': [100.0, 101.0],
        'High': [102.0, 103.0],
        'Low': [99.0, 100.0],
        'Close': [101.0, 102.0],
        'Volume': [1000000, 1500000]
    }
    test_df = pd.DataFrame(test_data)
    test_file = 'test_data.csv'
    test_df.to_csv(test_file, index=False)
    
    # Test loading the data
    try:
        loaded_df = load_price_data(test_file)
        
        # Basic checks
        assert isinstance(loaded_df, pd.DataFrame)
        assert not loaded_df.empty
        assert 'Date' in loaded_df.columns
        assert len(loaded_df) == 2
        
        # Check data types
        assert pd.api.types.is_datetime64_any_dtype(loaded_df['Date'])
        
    finally:
        # Clean up
        if os.path.exists(test_file):
            os.remove(test_file)

def test_load_price_data_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_price_data('non_existent_file.csv')