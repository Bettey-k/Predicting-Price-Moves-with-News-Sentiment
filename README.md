# Predicting Price Moves with News Sentiment

This project analyzes stock price movements using technical indicators and news sentiment analysis, with NVIDIA (NVDA) stock as a case study.

## Features

### 1. News Analysis
- Web scraping of financial news and analyst ratings
- Sentiment analysis of news articles
- Topic modeling and trend analysis
- Publisher impact assessment

### 2. Technical Analysis
- Price action visualization
- Technical indicators:
  - Simple Moving Averages (SMA 20 & 50)
  - Relative Strength Index (RSI 14)
  - Moving Average Convergence Divergence (MACD)
- Return metrics calculation

## Project Structure
Predicting-Price-Moves-with-News-Sentiment/ ├── data/ # Data storage │ ├── raw/ # Raw data files │ └── processed/ # Processed datasets ├── notebooks/ # Jupyter notebooks │ ├── 01_news_eda.ipynb # News analysis and exploration │ └── 02_technical_analysis.ipynb # Stock price analysis ├── src/ # Source code │ ├── data_loading.py # Data loading utilities │ ├── technical_indicators.py # Technical analysis functions │ ├── metrics.py # Financial metrics calculation │ ├── news_features.py # News processing features │ └── nlp_topics.py # NLP and topic modeling └── tests/ # Test files


## Getting Started

1. Clone the repository
   ```bash
   git clone [repository-url]
   cd Predicting-Price-Moves-with-News-Sentiment
2.Create and activate a virtual environment (recommended)  
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`

3. Install dependencies:
   ```bash
   pip install -r requirements.txt

4. Run the Jupyter notebook:
   ```bash
   jupyter notebook notebooks/noteboks_name.ipynb

Notebooks Overview
01_news_eda.ipynb
Analyzes financial news and analyst ratings
Performs sentiment analysis
Visualizes news trends and publisher impact

02_technical_analysis.ipynb
Loads and processes stock price data
Calculates technical indicators
Visualizes price action and indicators
Computes performance metrics


Dependencies

Python 3.7+
pandas
numpy
matplotlib
scikit-learn
nltk
ta-lib (for technical indicators)
jupyter

