"""
Sentiment analysis utilities for MarketHacker.

This module provides functions to analyze sentiment from various sources
including news articles, social media posts, and financial data.
"""

import pandas as pd
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def fetch_social_sentiment(
    symbols: List[str],
    days_back: int = 7
) -> pd.DataFrame:
    """
    Fetch and analyze sentiment from social media sources.
    
    Args:
        symbols: List of stock symbols to analyze
        days_back: Number of days to look back
    
    Returns:
        DataFrame with sentiment analysis results
    """
    try:
        logger.info(f"Fetching social sentiment for symbols: {symbols}")
        
        # Import here to avoid circular imports
        from .fetch_news import fetch_reddit_posts, fetch_twitter_posts
        
        all_sentiment = []
        
        for symbol in symbols:
            # Fetch Reddit posts
            reddit_df = fetch_reddit_posts('investing', symbol, limit=50)
            if not reddit_df.empty:
                reddit_df['source'] = 'reddit'
                reddit_df['symbol'] = symbol
                all_sentiment.append(reddit_df)
            
            # Fetch Twitter posts
            twitter_df = fetch_twitter_posts(symbol, count=50)
            if not twitter_df.empty:
                twitter_df['source'] = 'twitter'
                twitter_df['symbol'] = symbol
                all_sentiment.append(twitter_df)
        
        if all_sentiment:
            combined_df = pd.concat(all_sentiment, ignore_index=True)
            # Analyze sentiment
            combined_df = analyze_sentiment(combined_df)
            logger.info(f"Analyzed sentiment for {len(combined_df)} social media posts")
            return combined_df
        else:
            return pd.DataFrame()
            
    except Exception as e:
        logger.error(f"Error fetching social sentiment: {str(e)}")
        return pd.DataFrame()


def analyze_sentiment(df: pd.DataFrame, text_column: Optional[str] = None) -> pd.DataFrame:
    """
    Analyze sentiment of text data using VADER sentiment analysis.
    
    Args:
        df: DataFrame containing text data
        text_column: Column name containing text to analyze
    
    Returns:
        DataFrame with sentiment scores added
    """
    try:
        if df.empty:
            return df
        
        # Initialize VADER sentiment analyzer
        analyzer = SentimentIntensityAnalyzer()
        
        # Determine text column
        if text_column is None:
            if 'text' in df.columns:
                text_column = 'text'
            elif 'title' in df.columns:
                text_column = 'title'
            elif 'body' in df.columns:
                text_column = 'body'
            elif 'content' in df.columns:
                text_column = 'content'
            else:
                logger.warning("No suitable text column found for sentiment analysis")
                return df
        
        # Analyze sentiment for each row
        sentiment_scores = []
        for text in df[text_column]:
            if pd.isna(text) or text == '':
                sentiment_scores.append({'compound': 0, 'pos': 0, 'neg': 0, 'neu': 0})
            else:
                # Clean text for better sentiment analysis
                cleaned_text = clean_text_for_sentiment(text)
                scores = analyzer.polarity_scores(cleaned_text)
                sentiment_scores.append(scores)
        
        # Add sentiment scores to DataFrame
        sentiment_df = pd.DataFrame(sentiment_scores)
        df = pd.concat([df, sentiment_df], axis=1)
        
        # Add sentiment classification
        df['sentiment_label'] = df['compound'].apply(classify_sentiment)
        
        logger.info(f"Analyzed sentiment for {len(df)} texts")
        return df
        
    except Exception as e:
        logger.error(f"Error analyzing sentiment: {str(e)}")
        return df


def clean_text_for_sentiment(text: str) -> str:
    """
    Clean text for better sentiment analysis.
    
    Args:
        text: Raw text to clean
    
    Returns:
        Cleaned text
    """
    if not isinstance(text, str):
        return ""
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove special characters but keep important ones for sentiment
    text = re.sub(r'[^\w\s!?.,;:()]', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text


def classify_sentiment(compound_score: float) -> str:
    """
    Classify sentiment based on compound score.
    
    Args:
        compound_score: VADER compound sentiment score
    
    Returns:
        Sentiment classification
    """
    if compound_score >= 0.05:
        return 'positive'
    elif compound_score <= -0.05:
        return 'negative'
    else:
        return 'neutral'


def calculate_aggregate_sentiment(
    df: pd.DataFrame,
    group_by: str = 'symbol',
    weight_by: Optional[str] = None
) -> pd.DataFrame:
    """
    Calculate aggregate sentiment scores grouped by a column.
    
    Args:
        df: DataFrame with sentiment scores
        group_by: Column to group by (e.g., 'symbol', 'date')
        weight_by: Column to weight by (e.g., 'score', 'followers_count')
    
    Returns:
        DataFrame with aggregate sentiment metrics
    """
    try:
        if df.empty or group_by not in df.columns:
            return pd.DataFrame()
        
        # Calculate weights if specified
        if weight_by and weight_by in df.columns:
            weights = df[weight_by].fillna(1)
        else:
            weights = pd.Series(1, index=df.index)
        
        # Group and calculate weighted averages
        agg_sentiment = df.groupby(group_by).agg({
            'compound': lambda x: (x * weights.loc[x.index]).sum() / weights.loc[x.index].sum(),
            'pos': lambda x: (x * weights.loc[x.index]).sum() / weights.loc[x.index].sum(),
            'neg': lambda x: (x * weights.loc[x.index]).sum() / weights.loc[x.index].sum(),
            'neu': lambda x: (x * weights.loc[x.index]).sum() / weights.loc[x.index].sum(),
        }).reset_index()
        
        # Add sentiment classification
        agg_sentiment['sentiment_label'] = agg_sentiment['compound'].apply(classify_sentiment)
        
        # Add count of posts
        post_counts = df.groupby(group_by).size().reset_index()
        post_counts.columns = [group_by, 'post_count']
        agg_sentiment = agg_sentiment.merge(post_counts, on=group_by)
        
        logger.info(f"Calculated aggregate sentiment for {len(agg_sentiment)} groups")
        return agg_sentiment
        
    except Exception as e:
        logger.error(f"Error calculating aggregate sentiment: {str(e)}")
        return pd.DataFrame()


def analyze_sentiment_trends(
    df: pd.DataFrame,
    date_column: str = 'created_at',
    symbol_column: str = 'symbol',
    window: int = 7
) -> pd.DataFrame:
    """
    Analyze sentiment trends over time.
    
    Args:
        df: DataFrame with sentiment data
        date_column: Column containing dates
        symbol_column: Column containing symbols
        window: Rolling window size in days
    
    Returns:
        DataFrame with sentiment trends
    """
    try:
        if df.empty:
            return pd.DataFrame()
        
        # Ensure date column is datetime
        df[date_column] = pd.to_datetime(df[date_column])
        
        # Sort by date
        df = df.sort_values([symbol_column, date_column])
        
        # Calculate rolling sentiment averages
        trends = []
        
        for symbol in df[symbol_column].unique():
            symbol_df = df[df[symbol_column] == symbol].copy()
            
            # Resample to daily frequency and calculate rolling averages
            daily_sentiment = symbol_df.set_index(date_column)['compound'].resample('D').mean()
            rolling_sentiment = daily_sentiment.rolling(window=window, min_periods=1).mean()
            
            # Create trend DataFrame
            trend_df = pd.DataFrame({
                'symbol': symbol,
                'date': rolling_sentiment.index,
                'rolling_sentiment': rolling_sentiment.values,
                'daily_sentiment': daily_sentiment.values
            })
            
            trends.append(trend_df)
        
        if trends:
            combined_trends = pd.concat(trends, ignore_index=True)
            logger.info(f"Calculated sentiment trends for {len(combined_trends)} data points")
            return combined_trends
        else:
            return pd.DataFrame()
            
    except Exception as e:
        logger.error(f"Error analyzing sentiment trends: {str(e)}")
        return pd.DataFrame() 