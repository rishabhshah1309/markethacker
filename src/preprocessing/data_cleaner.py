"""
Data cleaning utilities for MarketHacker.

This module provides functions to clean and preprocess financial data
before feature engineering and modeling.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def clean_stock_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean stock data by removing outliers, handling missing values, and standardizing formats.
    
    Args:
        df: Raw stock data DataFrame
    
    Returns:
        Cleaned DataFrame
    """
    try:
        if df.empty:
            return df
        
        logger.info(f"Cleaning stock data with {len(df)} rows")
        
        # Make a copy to avoid modifying original data
        cleaned_df = df.copy()
        
        # Convert date index to datetime if it's not already
        if isinstance(cleaned_df.index, pd.DatetimeIndex):
            pass
        elif 'Date' in cleaned_df.columns:
            cleaned_df['Date'] = pd.to_datetime(cleaned_df['Date'])
            cleaned_df.set_index('Date', inplace=True)
        elif cleaned_df.index.name == 'Date':
            cleaned_df.index = pd.to_datetime(cleaned_df.index)
        
        # Handle missing values
        cleaned_df = handle_missing_values(cleaned_df)
        
        # Remove outliers from price data
        cleaned_df = remove_price_outliers(cleaned_df)
        
        # Ensure all numeric columns are float
        numeric_columns = cleaned_df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
        
        # Remove rows with all NaN values
        cleaned_df = cleaned_df.dropna(how='all')
        
        logger.info(f"Cleaned data: {len(cleaned_df)} rows remaining")
        return cleaned_df
        
    except Exception as e:
        logger.error(f"Error cleaning stock data: {str(e)}")
        return df


def handle_missing_values(df: pd.DataFrame, method: str = 'forward_fill') -> pd.DataFrame:
    """
    Handle missing values in the DataFrame.
    
    Args:
        df: DataFrame with missing values
        method: Method to handle missing values ('forward_fill', 'interpolate', 'drop')
    
    Returns:
        DataFrame with missing values handled
    """
    try:
        if df.empty:
            return df
        
        logger.info(f"Handling missing values using method: {method}")
        
        # Make a copy
        cleaned_df = df.copy()
        
        if method == 'forward_fill':
            # Forward fill for time series data
            cleaned_df = cleaned_df.ffill()
            # Backward fill for any remaining NaNs at the beginning
            cleaned_df = cleaned_df.bfill()
            
        elif method == 'interpolate':
            # Linear interpolation for time series data
            cleaned_df = cleaned_df.interpolate(method='linear')
            # Forward/backward fill for any remaining NaNs
            cleaned_df = cleaned_df.ffill().bfill()
            
        elif method == 'drop':
            # Drop rows with any missing values
            cleaned_df = cleaned_df.dropna()
            
        else:
            logger.warning(f"Unknown method '{method}', using forward_fill")
            cleaned_df = cleaned_df.ffill().bfill()
        
        # Count remaining missing values
        missing_count = cleaned_df.isnull().sum().sum()
        if missing_count > 0:
            logger.warning(f"Still have {missing_count} missing values after cleaning")
        
        return cleaned_df
        
    except Exception as e:
        logger.error(f"Error handling missing values: {str(e)}")
        return df


def remove_price_outliers(df: pd.DataFrame, method: str = 'iqr', threshold: float = 3.0) -> pd.DataFrame:
    """
    Remove outliers from price data.
    
    Args:
        df: DataFrame with price data
        method: Method to detect outliers ('iqr', 'zscore')
        threshold: Threshold for outlier detection
    
    Returns:
        DataFrame with outliers removed
    """
    try:
        if df.empty:
            return df
        
        logger.info(f"Removing price outliers using method: {method}")
        
        # Make a copy
        cleaned_df = df.copy()
        
        # Price columns to check for outliers
        price_columns = ['Open', 'High', 'Low', 'Close']
        available_price_cols = [col for col in price_columns if col in cleaned_df.columns]
        
        if not available_price_cols:
            logger.warning("No price columns found for outlier removal")
            return cleaned_df
        
        # Create mask for outliers
        outlier_mask = pd.Series(False, index=cleaned_df.index)
        
        for col in available_price_cols:
            if method == 'iqr':
                # IQR method
                Q1 = cleaned_df[col].quantile(0.25)
                Q3 = cleaned_df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                col_outliers = (cleaned_df[col] < lower_bound) | (cleaned_df[col] > upper_bound)
                
            elif method == 'zscore':
                # Z-score method
                z_scores = np.abs((cleaned_df[col] - cleaned_df[col].mean()) / cleaned_df[col].std())
                col_outliers = z_scores > threshold
                
            else:
                logger.warning(f"Unknown outlier method '{method}', skipping")
                continue
            
            outlier_mask = outlier_mask | col_outliers
        
        # Remove outliers
        outliers_removed = outlier_mask.sum()
        if outliers_removed > 0:
            cleaned_df = cleaned_df[~outlier_mask]
            logger.info(f"Removed {outliers_removed} outlier rows")
        
        return cleaned_df
        
    except Exception as e:
        logger.error(f"Error removing price outliers: {str(e)}")
        return df


def validate_stock_data(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate stock data for common issues.
    
    Args:
        df: DataFrame to validate
    
    Returns:
        Dictionary with validation results
    """
    try:
        validation_results = {
            'is_valid': True,
            'issues': [],
            'warnings': []
        }
        
        if df.empty:
            validation_results['is_valid'] = False
            validation_results['issues'].append("DataFrame is empty")
            return validation_results
        
        # Check for required columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            validation_results['is_valid'] = False
            validation_results['issues'].append(f"Missing required columns: {missing_columns}")
        
        # Check for negative prices
        price_columns = ['Open', 'High', 'Low', 'Close']
        for col in price_columns:
            if col in df.columns:
                negative_prices = (df[col] < 0).sum()
                if negative_prices > 0:
                    validation_results['warnings'].append(f"Found {negative_prices} negative prices in {col}")
        
        # Check for zero volumes
        if 'Volume' in df.columns:
            zero_volumes = (df['Volume'] == 0).sum()
            if zero_volumes > 0:
                validation_results['warnings'].append(f"Found {zero_volumes} zero volume records")
        
        # Check for price consistency
        if all(col in df.columns for col in ['High', 'Low', 'Open', 'Close']):
            invalid_highs = (df['High'] < df[['Open', 'Close']].max(axis=1)).sum()
            invalid_lows = (df['Low'] > df[['Open', 'Close']].min(axis=1)).sum()
            
            if invalid_highs > 0:
                validation_results['warnings'].append(f"Found {invalid_highs} records where High < max(Open, Close)")
            if invalid_lows > 0:
                validation_results['warnings'].append(f"Found {invalid_lows} records where Low > min(Open, Close)")
        
        # Check for missing values
        missing_values = df.isnull().sum()
        if missing_values.sum() > 0:
            validation_results['warnings'].append(f"Found missing values: {missing_values.to_dict()}")
        
        logger.info(f"Data validation completed. Valid: {validation_results['is_valid']}")
        return validation_results
        
    except Exception as e:
        logger.error(f"Error validating stock data: {str(e)}")
        return {'is_valid': False, 'issues': [str(e)], 'warnings': []}


def standardize_data_format(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize data format for consistency.
    
    Args:
        df: DataFrame to standardize
    
    Returns:
        Standardized DataFrame
    """
    try:
        if df.empty:
            return df
        
        logger.info("Standardizing data format")
        
        # Make a copy
        standardized_df = df.copy()
        
        # Ensure index is datetime
        if not isinstance(standardized_df.index, pd.DatetimeIndex):
            if standardized_df.index.name == 'Date':
                standardized_df.index = pd.to_datetime(standardized_df.index)
            elif 'Date' in standardized_df.columns:
                standardized_df['Date'] = pd.to_datetime(standardized_df['Date'])
                standardized_df.set_index('Date', inplace=True)
        
        # Standardize column names
        column_mapping = {
            'Open': 'open',
            'High': 'high', 
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume',
            'Adj Close': 'adj_close'
        }
        
        # Only rename columns that exist
        existing_mapping = {k: v for k, v in column_mapping.items() if k in standardized_df.columns}
        standardized_df = standardized_df.rename(columns=existing_mapping)
        
        # Ensure numeric columns are float
        numeric_columns = standardized_df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            standardized_df[col] = pd.to_numeric(standardized_df[col], errors='coerce')
        
        # Sort by date
        standardized_df = standardized_df.sort_index()
        
        logger.info("Data format standardization completed")
        return standardized_df
        
    except Exception as e:
        logger.error(f"Error standardizing data format: {str(e)}")
        return df 