"""
Main script to analyze correlation between news sentiment and stock prices
"""
import pandas as pd
from src.sentiment_analysis import SentimentAnalyzer
from src.correlation_analysis import CorrelationAnalyzer

def load_stock_data(ticker: str) -> pd.DataFrame:
    """Load stock price data for a given ticker."""
    filepath = f'data/{ticker}.csv'
    df = pd.read_csv(filepath, parse_dates=['Date'], index_col='Date')
    return df[['Close']]  # We only need the closing prices

def analyze_stock_news_correlation(ticker: str):
    """Analyze correlation between news sentiment and stock returns for a given ticker."""
    print(f"\nAnalyzing {ticker}...")
    
    # 1. Load stock data
    try:
        prices = load_stock_data(ticker)
        print(f"Loaded price data from {prices.index.min()} to {prices.index.max()}")
    except Exception as e:
        print(f"Error loading stock data for {ticker}: {e}")
        return

    # 2. Load and analyze news data
    try:
        # Load news data (assuming it has 'date' and 'headline' columns)
        news_df = pd.read_csv('data/raw_analyst_ratings.csv', parse_dates=['date'])
        print(f"Loaded {len(news_df)} news articles")
        
        # Filter news for this ticker (if there's a ticker column)
        if 'ticker' in news_df.columns:
            news_df = news_df[news_df['ticker'] == ticker]
            print(f"Found {len(news_df)} news articles for {ticker}")
        
        # Analyze sentiment
        analyzer = SentimentAnalyzer()
        news_with_sentiment = analyzer.analyze_news(news_df)
        daily_sentiment = analyzer.get_daily_sentiment(news_with_sentiment)
        
        # 3. Calculate correlation
        if not daily_sentiment.empty:
            corr_analyzer = CorrelationAnalyzer()
            correlation, combined_data = corr_analyzer.analyze_correlation(
                daily_sentiment,
                prices['Close']
            )
            
            print(f"\nCorrelation between news sentiment and {ticker} returns: {correlation:.3f}")
            print(f"Analysis period: {combined_data.index.min().date()} to {combined_data.index.max().date()}")
            print(f"Number of matching days with both news and price data: {len(combined_data)}")
            
            return combined_data
        else:
            print("No sentiment data available for correlation analysis")
            
    except Exception as e:
        print(f"Error analyzing news data: {e}")

def main():
    # List of tickers to analyze
    tickers = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'META', 'NVDA']
    
    # Analyze each ticker
    for ticker in tickers:
        analyze_stock_news_correlation(ticker)

if __name__ == "__main__":
    main()