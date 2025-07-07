"""
Data ingestion module for MarketHacker.

This module handles fetching financial data from various sources including:
- Yahoo Finance for stock data
- News APIs for market news
- Social media platforms for sentiment analysis
"""

from .fetch_market_data import download_stock_data, download_market_data_batch
from .fetch_news import fetch_news_data
from .fetch_sentiment import fetch_social_sentiment

__all__ = [
    'download_stock_data',
    'download_market_data_batch', 
    'fetch_news_data',
    'fetch_social_sentiment'
] 