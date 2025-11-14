"""
Market data fetching utilities for MarketHacker.

This module provides functions to download stock data from Yahoo Finance
and other financial data sources.
"""

import yfinance as yf
import pandas as pd
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_stock_data(
    symbol: str,
    start_date: str,
    end_date: str,
    interval: str = "1d"
) -> pd.DataFrame:
    """
    Download historical stock data for a given symbol.
    
    Args:
        symbol: Stock symbol (e.g., 'AAPL', 'MSFT')
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        interval: Data interval ('1d', '1h', '5m', etc.)
    
    Returns:
        DataFrame with OHLCV data and additional indicators
    """
    try:
        logger.info(f"Downloading data for {symbol} from {start_date} to {end_date}")
        
        # Download data from Yahoo Finance
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date, interval=interval)
        
        if df.empty:
            logger.warning(f"No data found for {symbol}")
            return pd.DataFrame()
        
        # Add symbol column
        df['Symbol'] = symbol
        
        # Add technical indicators
        df = add_technical_indicators(df)
        
        logger.info(f"Successfully downloaded {len(df)} records for {symbol}")
        return df
        
    except Exception as e:
        logger.error(f"Error downloading data for {symbol}: {str(e)}")
        return pd.DataFrame()


def download_market_data_batch(
    symbols: List[str],
    start_date: str,
    end_date: str,
    interval: str = "1d"
) -> Dict[str, pd.DataFrame]:
    """
    Download historical stock data for multiple symbols.
    
    Args:
        symbols: List of stock symbols
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        interval: Data interval ('1d', '1h', '5m', etc.)
    
    Returns:
        Dictionary mapping symbols to their respective DataFrames
    """
    results = {}
    
    for symbol in symbols:
        df = download_stock_data(symbol, start_date, end_date, interval)
        if not df.empty:
            results[symbol] = df
    
    logger.info(f"Downloaded data for {len(results)} out of {len(symbols)} symbols")
    return results


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators to the stock data.
    
    Args:
        df: DataFrame with OHLCV data
    
    Returns:
        DataFrame with additional technical indicators
    """
    if df.empty:
        return df
    
    # Calculate moving averages
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['EMA_12'] = df['Close'].ewm(span=12).mean()
    df['EMA_26'] = df['Close'].ewm(span=26).mean()
    
    # Calculate MACD
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
    df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
    
    # Calculate RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Calculate Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
    
    # Calculate volatility
    df['Volatility'] = df['Close'].pct_change().rolling(window=20).std()
    
    # Calculate volume indicators
    df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
    
    return df


def fetch_stock_data(symbol: str, days_back: int = 365) -> pd.DataFrame:
    """
    Fetch stock data for the last N days (convenience function).

    Args:
        symbol: Stock symbol (e.g., 'AAPL', 'MSFT')
        days_back: Number of days of historical data to fetch

    Returns:
        DataFrame with OHLCV data and technical indicators
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)

    return download_stock_data(
        symbol,
        start_date.strftime('%Y-%m-%d'),
        end_date.strftime('%Y-%m-%d')
    )


def get_stock_info(symbol: str) -> Dict[str, Any]:
    """
    Get basic information about a stock.

    Args:
        symbol: Stock symbol

    Returns:
        Dictionary with stock information
    """
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info

        return {
            'symbol': symbol,
            'name': info.get('longName', ''),
            'sector': info.get('sector', ''),
            'industry': info.get('industry', ''),
            'market_cap': info.get('marketCap', 0),
            'pe_ratio': info.get('trailingPE', 0),
            'dividend_yield': info.get('dividendYield', 0),
            'beta': info.get('beta', 0)
        }
    except Exception as e:
        logger.error(f"Error getting info for {symbol}: {str(e)}")
        return {} 