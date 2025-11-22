# 📈 Predicting Price Moves with News Sentiment

A comprehensive analysis platform that combines financial news sentiment with technical indicators to predict stock price movements, using NVIDIA (NVDA) as a case study.

## 🎯 Features

### 📰 News Analysis Module
- **Sentiment Analysis**: Quantify market sentiment from financial news articles
- **Topic Modeling**: Extract and track key discussion topics over time
- **Publisher Analytics**: Measure impact and bias of different news sources
- **Trend Visualization**: Interactive charts showing sentiment trends

### 📊 Technical Analysis Module
- **Price Action**: OHLCV charts with customizable timeframes
- **Indicators**:
  - Moving Averages (SMA, EMA)
  - Oscillators (RSI, MACD, Bollinger Bands)
  - Volume Analysis
- **Performance Metrics**:
  - Daily/Weekly Returns
  - Volatility Measures
  - Risk-Adjusted Returns (Sharpe Ratio, etc.)

## 🏗️ Project Structure

```
Predicting-Price-Moves-with-News-Sentiment/
├── data/                       # Data storage
│   ├── raw/                   # Raw data files
│   └── processed/             # Processed datasets
├── notebooks/                 # Jupyter notebooks
│   ├── 01_news_eda.ipynb      # News analysis and exploration
│   └── 02_technical_analysis.ipynb  # Stock price analysis
├── src/                       # Source code
│   ├── data_loading.py        # Data loading utilities
│   ├── technical_indicators.py # Technical analysis functions
│   ├── metrics.py            # Financial metrics calculation
│   ├── news_features.py      # News processing features
│   └── nlp_topics.py         # NLP and topic modeling
└── tests/                    # Test files
    └── ...
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- TA-Lib (see installation notes below)
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/Predicting-Price-Moves-with-News-Sentiment.git
   cd Predicting-Price-Moves-with-News-Sentiment
   ```

2. **Create and activate virtual environment**
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate
   
   # Mac/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install TA-Lib** (Required for technical indicators)
   - Windows: Download pre-built binary from [here](https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib)
   - Mac: `brew install ta-lib`
   - Linux: `sudo apt-get install ta-lib`

### Example: Running the Analysis

```python
# Example of loading and analyzing data
from src.data_loading import load_price_data
from src.technical_indicators import add_all_indicators

# Load and prepare data
df = load_price_data('data/raw/NVDA.csv')
df = add_all_indicators(df)

# View the first few rows with indicators
print(df[['Date', 'Close', 'SMA_20', 'RSI_14']].head())
```

## 📊 Notebooks Overview

### 01_news_eda.ipynb
- **Purpose**: Comprehensive analysis of financial news impact
- **Key Features**:
  - Sentiment scoring of news articles
  - Topic modeling using LDA
  - Publisher influence analysis
  - Time-series visualization of sentiment trends

### 02_technical_analysis.ipynb
- **Purpose**: Technical analysis of stock price movements
- **Key Features**:
  - Interactive price charts with indicators
  - Strategy backtesting
  - Performance metrics calculation
  - Risk analysis

## 🛠 Dependencies

Core dependencies are listed in `requirements.txt`:
```
pandas>=1.3.0
numpy>=1.20.0
matplotlib>=3.4.0
seaborn>=0.11.0
talib>=0.4.24
scikit-learn>=0.24.0
nltk>=3.6.0
jupyter>=1.0.0
python-dotenv>=0.19.0
```

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with ❤️ using Python's data science stack
- Special thanks to the open-source community for their invaluable tools and libraries
