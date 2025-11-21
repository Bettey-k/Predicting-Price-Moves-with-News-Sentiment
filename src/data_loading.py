
import pandas as pd
from pathlib import Path

def load_price_data(filepath: str) -> pd.DataFrame:
    """
    Load price data from CSV (e.g., NVDA.csv).
    Returns a DataFrame with proper datetime sorting.
    """
    df = pd.read_csv(filepath)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.sort_values("Date").reset_index(drop=True)
    return df
