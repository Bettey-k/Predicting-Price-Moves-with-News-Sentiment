"""
Date Utilities for Time Series Analysis
"""
import pandas as pd
from typing import Optional

def align_dates(s1: pd.Series, 
               s2: pd.Series, 
               method: str = 'inner') -> tuple:
    """
    Align two time series by their indices.
    
    Args:
        s1: First time series
        s2: Second time series
        method: How to align the series ('inner' or 'outer')
        
    Returns:
        Tuple of aligned series (s1_aligned, s2_aligned)
    """
    return s1.align(s2, join=method)