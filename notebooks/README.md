

This directory contains Jupyter notebooks for analyzing stock price movements and news sentiment.

## Notebooks Overview

### 01_news_eda.ipynb
- **Purpose**: Analyze financial news and analyst ratings
- **Features**:
  - Loads and preprocesses news data
  - Performs sentiment analysis on news articles
  - Visualizes news trends and publisher impact
  - Analyzes topic distribution and trends over time
  - Examines publisher influence on market sentiment

### 02_technical_analysis.ipynb
- **Purpose**: Technical analysis of stock price data
- **Features**:
  - Loads and processes historical price data
  - Calculates technical indicators:
    - Simple Moving Averages (SMA 20 & 50)
    - Relative Strength Index (RSI 14)
    - Moving Average Convergence Divergence (MACD)
  - Visualizes price action and indicators
  - Computes and analyzes return metrics

## Dependencies

- Python 3.7+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- nltk
- ta-lib (for technical indicators)

## Usage

1. Ensure all dependencies are installed (see main README for setup instructions)
2. Place your data files in the appropriate directories:
   - News data: `../data/raw/`
   - Price data: `../data/raw/`
3. Run the notebooks in order for complete analysis

## Data Sources

- **News Data**: 
  - Raw financial news and analyst ratings
  - Expected format: CSV with columns for date, headline, content, publisher, etc.
  
- **Price Data**:
  - Historical stock price data for NVDA
  - Expected format: CSV with OHLCV (Open, High, Low, Close, Volume) data

## Notes

- The notebooks are designed to work with the project structure defined in the main README
- All data paths are relative to the project root directory
- Ensure you have sufficient memory for processing large news datasets
- Some visualizations may take time to render with large datasets

## Future Work

- Implement real-time news sentiment analysis
- Add more advanced technical indicators
- Develop predictive models combining news sentiment and technical indicators
- Create interactive dashboards for visualization
- Expand to multiple stocks and comparative analysis