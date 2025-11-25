import pytest
import pandas as pd
import numpy as np
from src.sentiment_analysis import SentimentAnalyzer

@pytest.fixture
def sample_news_data():
    return pd.DataFrame({
        'date': pd.date_range('2023-01-01', periods=3),
        'headline': [
            'Great news for the company!',
            'Terrible quarter results',
            'Neutral market conditions'
        ]
    })

def test_get_sentiment():
    analyzer = SentimentAnalyzer()
    assert analyzer.get_sentiment("Great news!") > 0
    assert analyzer.get_sentiment("Terrible news!") < 0
    assert analyzer.get_sentiment("") == 0

def test_analyze_news(sample_news_data):
    analyzer = SentimentAnalyzer()
    result = analyzer.analyze_news(sample_news_data)
    assert 'sentiment' in result.columns
    assert len(result) == 3