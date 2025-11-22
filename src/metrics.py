"""
Financial metrics calculation module for time series analysis.

This module provides classes and functions for calculating various financial
metrics such as returns, volatility, and risk-adjusted returns.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
from pandas import DataFrame, Series


@dataclass
class ReturnMetrics:
    """Container for return-related metrics."""
    daily_returns: Series
    cumulative_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    
    def to_dict(self) -> Dict[str, float]:
        """Convert metrics to a dictionary."""
        return {
            'cumulative_return': self.cumulative_return,
            'annualized_return': self.annualized_return,
            'annualized_volatility': self.annualized_volatility,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown
        }


class FinancialMetricsCalculator:
    """
    A class for calculating various financial metrics from price or return data.
    
    This class provides methods to calculate common financial metrics such as
    returns, volatility, Sharpe ratio, and drawdowns.
    """
    
    def __init__(self, risk_free_rate: float = 0.0, trading_days: int = 252):
        """
        Initialize the FinancialMetricsCalculator.
        
        Args:
            risk_free_rate: Annual risk-free rate for risk-adjusted metrics.
            trading_days: Number of trading days in a year for annualization.
        """
        self.risk_free_rate = risk_free_rate
        self.trading_days = trading_days
    
    def calculate_daily_returns(self, prices: Union[DataFrame, Series], 
                              price_col: Optional[str] = None) -> Series:
        """
        Calculate daily returns from price data.
        
        Args:
            prices: DataFrame with price data or Series of prices.
            price_col: Column name if prices is a DataFrame. If None, assumes prices is a Series.
            
        Returns:
            Series: Daily returns.
        """
        if isinstance(prices, DataFrame):
            if price_col is None:
                raise ValueError("price_col must be specified when prices is a DataFrame")
            return prices[price_col].pct_change().dropna()
        return prices.pct_change().dropna()
    
    def calculate_cumulative_return(self, returns: Series) -> float:
        """Calculate cumulative return from daily returns."""
        return float((1 + returns).prod() - 1)
    
    def calculate_annualized_return(self, returns: Series) -> float:
        """Calculate annualized return from daily returns."""
        return float((1 + returns).mean() ** self.trading_days - 1)
    
    def calculate_annualized_volatility(self, returns: Series) -> float:
        """Calculate annualized volatility from daily returns."""
        return float(returns.std() * np.sqrt(self.trading_days))
    
    def calculate_sharpe_ratio(self, returns: Series) -> float:
        """
        Calculate the annualized Sharpe ratio.
        
        The Sharpe ratio is a measure of risk-adjusted return, calculated as:
        (portfolio_return - risk_free_rate) / volatility
        """
        excess_returns = returns - (self.risk_free_rate / self.trading_days)
        return float(np.sqrt(self.trading_days) * excess_returns.mean() / returns.std())
    
    def calculate_max_drawdown(self, returns: Series) -> float:
        """
        Calculate the maximum drawdown from cumulative returns.
        
        Returns:
            Maximum drawdown as a decimal (e.g., 0.20 for 20% drawdown).
        """
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdowns = (cumulative - rolling_max) / rolling_max
        return float(drawdowns.min())
    
    def calculate_all_metrics(self, prices: Union[DataFrame, Series], 
                            price_col: Optional[str] = 'Close') -> ReturnMetrics:
        """
        Calculate all available return metrics.
        
        Args:
            prices: DataFrame with price data or Series of prices.
            price_col: Column name if prices is a DataFrame.
            
        Returns:
            ReturnMetrics: Object containing all calculated metrics.
        """
        returns = self.calculate_daily_returns(prices, price_col)
        
        if len(returns) == 0:
            return ReturnMetrics(
                daily_returns=returns,
                cumulative_return=0.0,
                annualized_return=0.0,
                annualized_volatility=0.0,
                sharpe_ratio=None,
                max_drawdown=None
            )
        
        cumulative_return = self.calculate_cumulative_return(returns)
        annualized_return = self.calculate_annualized_return(returns)
        annualized_volatility = self.calculate_annualized_volatility(returns)
        
        try:
            sharpe_ratio = self.calculate_sharpe_ratio(returns)
            max_drawdown = self.calculate_max_drawdown(returns)
        except (ValueError, ZeroDivisionError):
            # Handle cases where calculation is not possible
            sharpe_ratio = None
            max_drawdown = None
        
        return ReturnMetrics(
            daily_returns=returns,
            cumulative_return=cumulative_return,
            annualized_return=annualized_return,
            annualized_volatility=annualized_volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown
        )


# Package-level convenience functions
def add_daily_returns(df: DataFrame, price_col: str = 'Close') -> DataFrame:
    """
    Add a column of daily returns to a DataFrame.
    
    Args:
        df: Input DataFrame with price data.
        price_col: Name of the column containing price data.
        
    Returns:
        DataFrame: A new DataFrame with an added 'daily_return' column.
    """
    calculator = FinancialMetricsCalculator()
    df = df.copy()
    df['daily_return'] = calculator.calculate_daily_returns(df, price_col)
    return df


def estimate_basic_metrics(df: DataFrame, price_col: str = 'Close') -> Dict[str, float]:
    """
    Calculate basic financial metrics from price data.
    
    Args:
        df: Input DataFrame with price data.
        price_col: Name of the column containing price data.
        
    Returns:
        dict: Dictionary containing basic financial metrics.
    """
    calculator = FinancialMetricsCalculator()
    
    if df.empty:
        return {
            'mean_return': np.nan,
            'volatility': np.nan,
            'cumulative_return': np.nan
        }
    
    returns = calculator.calculate_daily_returns(df, price_col)
    
    if len(returns) == 0:
        return {
            'mean_return': np.nan,
            'volatility': np.nan,
            'cumulative_return': np.nan
        }
    
    return {
        'mean_return': float(returns.mean()),
        'volatility': float(returns.std()),
        'cumulative_return': calculator.calculate_cumulative_return(returns)
    }