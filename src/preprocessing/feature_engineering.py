"""
Feature engineering utilities for MarketHacker.

This module provides functions to create technical indicators and features
for machine learning models.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create comprehensive features for stock prediction.
    
    Args:
        df: DataFrame with OHLCV data
    
    Returns:
        DataFrame with additional features
    """
    try:
        if df.empty:
            return df
        
        logger.info("Creating features for stock prediction")
        
        # Make a copy
        feature_df = df.copy()
        
        # Price-based features
        feature_df = add_price_features(feature_df)
        
        # Volume-based features
        feature_df = add_volume_features(feature_df)
        
        # Technical indicators
        feature_df = add_technical_indicators(feature_df)
        
        # Time-based features
        feature_df = add_time_features(feature_df)
        
        # Volatility features
        feature_df = add_volatility_features(feature_df)
        
        # Momentum features
        feature_df = add_momentum_features(feature_df)
        
        # Target variables
        feature_df = add_target_variables(feature_df)
        
        logger.info(f"Created {len(feature_df.columns)} features")
        return feature_df
        
    except Exception as e:
        logger.error(f"Error creating features: {str(e)}")
        return df


def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add price-based features.
    
    Args:
        df: DataFrame with price data
    
    Returns:
        DataFrame with price features
    """
    try:
        if df.empty:
            return df
        
        # Price changes
        df['price_change'] = df['Close'].pct_change()
        df['price_change_abs'] = df['price_change'].abs()
        
        # High-Low spread
        df['hl_spread'] = (df['High'] - df['Low']) / df['Close']
        df['hl_spread_pct'] = df['hl_spread'] * 100
        
        # Open-Close spread
        df['oc_spread'] = (df['Close'] - df['Open']) / df['Open']
        df['oc_spread_abs'] = df['oc_spread'].abs()
        
        # Price position within day's range
        df['price_position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])
        
        # Price relative to previous day
        df['price_vs_prev_close'] = df['Close'] / df['Close'].shift(1) - 1
        
        # Price relative to previous high/low
        df['price_vs_prev_high'] = df['Close'] / df['High'].shift(1) - 1
        df['price_vs_prev_low'] = df['Close'] / df['Low'].shift(1) - 1
        
        return df
        
    except Exception as e:
        logger.error(f"Error adding price features: {str(e)}")
        return df


def add_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add volume-based features.
    
    Args:
        df: DataFrame with volume data
    
    Returns:
        DataFrame with volume features
    """
    try:
        if df.empty:
            return df
        
        # Volume changes
        df['volume_change'] = df['Volume'].pct_change()
        df['volume_change_abs'] = df['volume_change'].abs()
        
        # Volume moving averages
        df['volume_sma_5'] = df['Volume'].rolling(window=5).mean()
        df['volume_sma_20'] = df['Volume'].rolling(window=20).mean()
        df['volume_ema_12'] = df['Volume'].ewm(span=12).mean()
        
        # Volume ratios
        df['volume_ratio_5'] = df['Volume'] / df['volume_sma_5']
        df['volume_ratio_20'] = df['Volume'] / df['volume_sma_20']
        
        # Volume-price relationship
        df['volume_price_trend'] = df['Volume'] * df['price_change']
        df['volume_price_trend_ma'] = df['volume_price_trend'].rolling(window=20).mean()
        
        # Volume volatility
        df['volume_volatility'] = df['Volume'].rolling(window=20).std() / df['volume_sma_20']
        
        return df
        
    except Exception as e:
        logger.error(f"Error adding volume features: {str(e)}")
        return df


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators.
    
    Args:
        df: DataFrame with price data
    
    Returns:
        DataFrame with technical indicators
    """
    try:
        if df.empty:
            return df
        
        # Moving averages
        df['sma_5'] = df['Close'].rolling(window=5).mean()
        df['sma_10'] = df['Close'].rolling(window=10).mean()
        df['sma_20'] = df['Close'].rolling(window=20).mean()
        df['sma_50'] = df['Close'].rolling(window=50).mean()
        df['sma_200'] = df['Close'].rolling(window=200).mean()
        
        df['ema_12'] = df['Close'].ewm(span=12).mean()
        df['ema_26'] = df['Close'].ewm(span=26).mean()
        
        # Price vs moving averages
        df['price_vs_sma_20'] = df['Close'] / df['sma_20'] - 1
        df['price_vs_sma_50'] = df['Close'] / df['sma_50'] - 1
        df['price_vs_sma_200'] = df['Close'] / df['sma_200'] - 1
        
        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['bb_middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Stochastic Oscillator
        df['stoch_k'] = ((df['Close'] - df['Low'].rolling(window=14).min()) / 
                        (df['High'].rolling(window=14).max() - df['Low'].rolling(window=14).min())) * 100
        df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
        
        return df
        
    except Exception as e:
        logger.error(f"Error adding technical indicators: {str(e)}")
        return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add time-based features.
    
    Args:
        df: DataFrame with datetime index
    
    Returns:
        DataFrame with time features
    """
    try:
        if df.empty:
            return df
        
        # Ensure index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            logger.warning("Index is not datetime, skipping time features")
            return df
        
        # Day of week
        df['day_of_week'] = df.index.dayofweek
        
        # Day of month
        df['day_of_month'] = df.index.day
        
        # Month
        df['month'] = df.index.month
        
        # Quarter
        df['quarter'] = df.index.quarter
        
        # Year
        df['year'] = df.index.year
        
        # Day of year
        df['day_of_year'] = df.index.dayofyear
        
        # Week of year
        df['week_of_year'] = df.index.isocalendar().week
        
        # Is weekend
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Is month end
        df['is_month_end'] = df.index.is_month_end.astype(int)
        
        # Is quarter end
        df['is_quarter_end'] = df.index.is_quarter_end.astype(int)
        
        # Is year end
        df['is_year_end'] = df.index.is_year_end.astype(int)
        
        return df
        
    except Exception as e:
        logger.error(f"Error adding time features: {str(e)}")
        return df


def add_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add volatility-based features.
    
    Args:
        df: DataFrame with price data
    
    Returns:
        DataFrame with volatility features
    """
    try:
        if df.empty:
            return df
        
        # Price volatility
        df['volatility_5'] = df['price_change'].rolling(window=5).std()
        df['volatility_20'] = df['price_change'].rolling(window=20).std()
        df['volatility_60'] = df['price_change'].rolling(window=60).std()
        
        # Annualized volatility
        df['volatility_annualized'] = df['volatility_20'] * np.sqrt(252)
        
        # Volatility ratio
        df['volatility_ratio'] = df['volatility_5'] / df['volatility_20']
        
        # True Range
        df['tr1'] = df['High'] - df['Low']
        df['tr2'] = abs(df['High'] - df['Close'].shift(1))
        df['tr3'] = abs(df['Low'] - df['Close'].shift(1))
        df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        df['atr_14'] = df['true_range'].rolling(window=14).mean()
        
        # Average True Range percentage
        df['atr_pct'] = df['atr_14'] / df['Close'] * 100
        
        # Drop temporary columns
        df = df.drop(['tr1', 'tr2', 'tr3'], axis=1, errors='ignore')
        
        return df
        
    except Exception as e:
        logger.error(f"Error adding volatility features: {str(e)}")
        return df


def add_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add momentum-based features.
    
    Args:
        df: DataFrame with price data
    
    Returns:
        DataFrame with momentum features
    """
    try:
        if df.empty:
            return df
        
        # Price momentum
        df['momentum_1'] = df['Close'] / df['Close'].shift(1) - 1
        df['momentum_5'] = df['Close'] / df['Close'].shift(5) - 1
        df['momentum_10'] = df['Close'] / df['Close'].shift(10) - 1
        df['momentum_20'] = df['Close'] / df['Close'].shift(20) - 1
        
        # Rate of change
        df['roc_5'] = (df['Close'] - df['Close'].shift(5)) / df['Close'].shift(5) * 100
        df['roc_10'] = (df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10) * 100
        df['roc_20'] = (df['Close'] - df['Close'].shift(20)) / df['Close'].shift(20) * 100
        
        # Williams %R
        df['williams_r'] = ((df['High'].rolling(window=14).max() - df['Close']) / 
                           (df['High'].rolling(window=14).max() - df['Low'].rolling(window=14).min())) * -100
        
        # Commodity Channel Index
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3
        sma_tp = typical_price.rolling(window=20).mean()
        mad = typical_price.rolling(window=20).apply(lambda x: np.mean(np.abs(x - x.mean())))
        df['cci'] = (typical_price - sma_tp) / (0.015 * mad)
        
        return df
        
    except Exception as e:
        logger.error(f"Error adding momentum features: {str(e)}")
        return df


def add_target_variables(df: pd.DataFrame, horizons: List[int] = [1, 5, 10, 20]) -> pd.DataFrame:
    """
    Add target variables for prediction.
    
    Args:
        df: DataFrame with price data
        horizons: List of prediction horizons in days
    
    Returns:
        DataFrame with target variables
    """
    try:
        if df.empty:
            return df
        
        # Future returns
        for horizon in horizons:
            df[f'return_{horizon}d'] = df['Close'].shift(-horizon) / df['Close'] - 1
            df[f'return_{horizon}d_abs'] = df[f'return_{horizon}d'].abs()
            
            # Binary classification targets
            df[f'target_{horizon}d_up'] = (df[f'return_{horizon}d'] > 0).astype(int)
            df[f'target_{horizon}d_up_1pct'] = (df[f'return_{horizon}d'] > 0.01).astype(int)
            df[f'target_{horizon}d_up_3pct'] = (df[f'return_{horizon}d'] > 0.03).astype(int)
            
            # Volatility targets
            df[f'volatility_{horizon}d'] = df['price_change'].rolling(window=horizon).std()
        
        # Maximum drawdown
        df['max_drawdown_20d'] = df['Close'].rolling(window=20).apply(
            lambda x: (x / x.expanding().max() - 1).min()
        )
        
        return df
        
    except Exception as e:
        logger.error(f"Error adding target variables: {str(e)}")
        return df


def create_sentiment_features(sentiment_df: pd.DataFrame, stock_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create sentiment-based features by merging sentiment data with stock data.
    
    Args:
        sentiment_df: DataFrame with sentiment data
        stock_df: DataFrame with stock data
    
    Returns:
        DataFrame with sentiment features merged
    """
    try:
        if sentiment_df.empty or stock_df.empty:
            return stock_df
        
        logger.info("Creating sentiment features")
        
        # Ensure both DataFrames have datetime index
        if not isinstance(sentiment_df.index, pd.DatetimeIndex):
            if 'created_at' in sentiment_df.columns:
                sentiment_df['created_at'] = pd.to_datetime(sentiment_df['created_at'])
                sentiment_df.set_index('created_at', inplace=True)
        
        if not isinstance(stock_df.index, pd.DatetimeIndex):
            if 'Date' in stock_df.columns:
                stock_df['Date'] = pd.to_datetime(stock_df['Date'])
                stock_df.set_index('Date', inplace=True)
        
        # Aggregate sentiment by date
        daily_sentiment = sentiment_df.groupby(sentiment_df.index.date).agg({
            'compound': ['mean', 'std', 'count'],
            'pos': 'mean',
            'neg': 'mean',
            'neu': 'mean'
        }).reset_index()
        
        # Flatten column names
        daily_sentiment.columns = ['date', 'sentiment_mean', 'sentiment_std', 'sentiment_count', 
                                 'sentiment_pos', 'sentiment_neg', 'sentiment_neu']
        
        # Convert date to datetime
        daily_sentiment['date'] = pd.to_datetime(daily_sentiment['date'])
        daily_sentiment.set_index('date', inplace=True)
        
        # Merge with stock data
        merged_df = stock_df.merge(daily_sentiment, left_index=True, right_index=True, how='left')
        
        # Forward fill sentiment data
        sentiment_columns = ['sentiment_mean', 'sentiment_std', 'sentiment_count', 
                           'sentiment_pos', 'sentiment_neg', 'sentiment_neu']
        merged_df[sentiment_columns] = merged_df[sentiment_columns].ffill()
        
        # Create sentiment momentum features
        merged_df['sentiment_momentum_5d'] = merged_df['sentiment_mean'].rolling(window=5).mean()
        merged_df['sentiment_momentum_20d'] = merged_df['sentiment_mean'].rolling(window=20).mean()
        merged_df['sentiment_change'] = merged_df['sentiment_mean'].pct_change()
        
        logger.info(f"Created sentiment features for {len(merged_df)} rows")
        return merged_df
        
    except Exception as e:
        logger.error(f"Error creating sentiment features: {str(e)}")
        return stock_df 