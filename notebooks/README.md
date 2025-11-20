# News Headlines EDA

## Overview
This notebook performs an exploratory data analysis (EDA) on financial news headlines data, focusing on understanding the dataset's structure, content, and patterns.

## Dataset
- **Source**: `data/raw_analyst_ratings.csv`
- **Size**: ~1.4 million entries
- **Columns**:
  - `headline`: News headline text
  - `url`: Source URL
  - `publisher`: News publisher/organization
  - `date`: Publication timestamp
  - `stock`: Stock ticker symbol

## Key Analyses
1. **Data Loading & Cleaning**:
   - Handles missing values and date parsing
   - Removes unnecessary columns
   - Converts date strings to datetime objects

2. **Basic Statistics**:
   - Dataset dimensions and data types
   - Temporal distribution of news
   - Most active publishers
   - Most covered stocks

3. **Text Analysis**:
   - Headline length analysis (characters and words)
   - Common words and phrases
   - Publisher-specific patterns

4. **Visualizations**:
   - Top publishers by article count
   - Distribution of headline lengths
   - Publication frequency over time
   - Stock coverage distribution

## Dependencies
- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn

## Usage
1. Ensure dependencies are installed
2. Update the input file path if needed
3. Run the notebook cells sequentially

## Notes
- The dataset contains financial news headlines from various sources
- Date range: [Start Date] to [End Date] (to be filled based on your data)
- Focuses on English-language financial news

## Future Work
- Sentiment analysis of headlines
- Correlation with stock price movements
- Topic modeling of news content
- Advanced time series analysis