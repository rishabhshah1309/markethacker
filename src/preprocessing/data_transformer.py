"""
Data transformation utilities for MarketHacker.

This module provides functions to normalize features and encode categorical
variables for machine learning models.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def normalize_features(
    df: pd.DataFrame,
    method: str = 'standard',
    exclude_columns: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, Any]:
    """
    Normalize numerical features in the DataFrame.
    
    Args:
        df: DataFrame with features to normalize
        method: Normalization method ('standard', 'minmax', 'robust')
        exclude_columns: Columns to exclude from normalization
    
    Returns:
        Tuple of (normalized DataFrame, scaler object)
    """
    try:
        if df.empty:
            return df, None
        
        logger.info(f"Normalizing features using method: {method}")
        
        # Make a copy
        normalized_df = df.copy()
        
        # Identify numerical columns
        numerical_columns = normalized_df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Exclude specified columns
        if exclude_columns:
            numerical_columns = [col for col in numerical_columns if col not in exclude_columns]
        
        # Exclude target variables and date columns
        target_patterns = ['target_', 'return_', 'volatility_']
        date_patterns = ['date', 'time', 'year', 'month', 'day']
        
        filtered_columns = []
        for col in numerical_columns:
            is_target = any(pattern in col.lower() for pattern in target_patterns)
            is_date = any(pattern in col.lower() for pattern in date_patterns)
            if not is_target and not is_date:
                filtered_columns.append(col)
        
        if not filtered_columns:
            logger.warning("No numerical columns found for normalization")
            return normalized_df, None
        
        # Initialize scaler
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            logger.warning(f"Unknown method '{method}', using standard scaling")
            scaler = StandardScaler()
        
        # Fit and transform
        normalized_df[filtered_columns] = scaler.fit_transform(normalized_df[filtered_columns])
        
        logger.info(f"Normalized {len(filtered_columns)} features")
        return normalized_df, scaler
        
    except Exception as e:
        logger.error(f"Error normalizing features: {str(e)}")
        return df, None


def encode_categorical(
    df: pd.DataFrame,
    method: str = 'label',
    categorical_columns: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Encode categorical variables.
    
    Args:
        df: DataFrame with categorical variables
        method: Encoding method ('label', 'onehot')
        categorical_columns: List of categorical columns to encode
    
    Returns:
        Tuple of (encoded DataFrame, encoders dictionary)
    """
    try:
        if df.empty:
            return df, {}
        
        logger.info(f"Encoding categorical variables using method: {method}")
        
        # Make a copy
        encoded_df = df.copy()
        
        # Identify categorical columns if not provided
        if categorical_columns is None:
            categorical_columns = encoded_df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if not categorical_columns:
            logger.info("No categorical columns found")
            return encoded_df, {}
        
        encoders = {}
        
        if method == 'label':
            # Label encoding
            for col in categorical_columns:
                if col in encoded_df.columns:
                    le = LabelEncoder()
                    encoded_df[col] = le.fit_transform(encoded_df[col].astype(str))
                    encoders[col] = le
                    
        elif method == 'onehot':
            # One-hot encoding
            for col in categorical_columns:
                if col in encoded_df.columns:
                    ohe = OneHotEncoder(sparse_output=False, drop='first')
                    encoded_values = ohe.fit_transform(encoded_df[[col]])
                    
                    # Create new column names
                    feature_names = [f"{col}_{val}" for val in ohe.categories_[0][1:]]
                    
                    # Add encoded columns to DataFrame
                    for i, feature_name in enumerate(feature_names):
                        encoded_df[feature_name] = encoded_values[:, i]
                    
                    # Drop original column
                    encoded_df = encoded_df.drop(col, axis=1)
                    encoders[col] = ohe
        
        else:
            logger.warning(f"Unknown encoding method '{method}', using label encoding")
            for col in categorical_columns:
                if col in encoded_df.columns:
                    le = LabelEncoder()
                    encoded_df[col] = le.fit_transform(encoded_df[col].astype(str))
                    encoders[col] = le
        
        logger.info(f"Encoded {len(categorical_columns)} categorical variables")
        return encoded_df, encoders
        
    except Exception as e:
        logger.error(f"Error encoding categorical variables: {str(e)}")
        return df, {}


def prepare_features_for_model(
    df: pd.DataFrame,
    target_column: str,
    exclude_columns: Optional[List[str]] = None,
    normalize_method: str = 'standard',
    encode_method: str = 'label'
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    """
    Prepare features for machine learning model training.
    
    Args:
        df: DataFrame with features and target
        target_column: Name of the target column
        exclude_columns: Columns to exclude from features
        normalize_method: Method for feature normalization
        encode_method: Method for categorical encoding
    
    Returns:
        Tuple of (feature DataFrame, target Series, preprocessing info)
    """
    try:
        if df.empty:
            return df, pd.Series(), {}
        
        logger.info("Preparing features for model training")
        
        # Make a copy
        prepared_df = df.copy()
        
        # Separate features and target
        if target_column not in prepared_df.columns:
            logger.error(f"Target column '{target_column}' not found")
            return prepared_df, pd.Series(), {}
        
        y = prepared_df[target_column]
        X = prepared_df.drop(target_column, axis=1)
        
        # Exclude specified columns
        if exclude_columns:
            X = X.drop(exclude_columns, axis=1, errors='ignore')
        
        # Encode categorical variables
        X, encoders = encode_categorical(X, method=encode_method)
        
        # Normalize features
        X, scaler = normalize_features(X, method=normalize_method)
        
        # Store preprocessing information
        preprocessing_info = {
            'encoders': encoders,
            'scaler': scaler,
            'feature_columns': X.columns.tolist(),
            'target_column': target_column
        }
        
        logger.info(f"Prepared {len(X.columns)} features for model training")
        return X, y, preprocessing_info
        
    except Exception as e:
        logger.error(f"Error preparing features for model: {str(e)}")
        return df, pd.Series(), {}


def create_lag_features(
    df: pd.DataFrame,
    columns: List[str],
    lags: List[int] = [1, 2, 3, 5, 10]
) -> pd.DataFrame:
    """
    Create lag features for time series data.
    
    Args:
        df: DataFrame with time series data
        columns: Columns to create lags for
        lags: List of lag periods
    
    Returns:
        DataFrame with lag features
    """
    try:
        if df.empty:
            return df
        
        logger.info(f"Creating lag features for {len(columns)} columns")
        
        # Make a copy
        lag_df = df.copy()
        
        # Create lag features
        for col in columns:
            if col in lag_df.columns:
                for lag in lags:
                    lag_df[f'{col}_lag_{lag}'] = lag_df[col].shift(lag)
        
        logger.info(f"Created {len(columns) * len(lags)} lag features")
        return lag_df
        
    except Exception as e:
        logger.error(f"Error creating lag features: {str(e)}")
        return df


def create_rolling_features(
    df: pd.DataFrame,
    columns: List[str],
    windows: List[int] = [5, 10, 20],
    functions: List[str] = ['mean', 'std', 'min', 'max']
) -> pd.DataFrame:
    """
    Create rolling window features.
    
    Args:
        df: DataFrame with time series data
        columns: Columns to create rolling features for
        windows: List of window sizes
        functions: List of aggregation functions
    
    Returns:
        DataFrame with rolling features
    """
    try:
        if df.empty:
            return df
        
        logger.info(f"Creating rolling features for {len(columns)} columns")
        
        # Make a copy
        rolling_df = df.copy()
        
        # Create rolling features
        for col in columns:
            if col in rolling_df.columns:
                for window in windows:
                    for func in functions:
                        if func == 'mean':
                            rolling_df[f'{col}_rolling_mean_{window}'] = rolling_df[col].rolling(window=window).mean()
                        elif func == 'std':
                            rolling_df[f'{col}_rolling_std_{window}'] = rolling_df[col].rolling(window=window).std()
                        elif func == 'min':
                            rolling_df[f'{col}_rolling_min_{window}'] = rolling_df[col].rolling(window=window).min()
                        elif func == 'max':
                            rolling_df[f'{col}_rolling_max_{window}'] = rolling_df[col].rolling(window=window).max()
        
        logger.info(f"Created rolling features")
        return rolling_df
        
    except Exception as e:
        logger.error(f"Error creating rolling features: {str(e)}")
        return df


def handle_missing_features(df: pd.DataFrame, method: str = 'drop') -> pd.DataFrame:
    """
    Handle missing values in features.
    
    Args:
        df: DataFrame with features
        method: Method to handle missing values ('drop', 'fill_mean', 'fill_median')
    
    Returns:
        DataFrame with missing values handled
    """
    try:
        if df.empty:
            return df
        
        logger.info(f"Handling missing features using method: {method}")
        
        # Make a copy
        handled_df = df.copy()
        
        if method == 'drop':
            # Drop rows with any missing values
            initial_rows = len(handled_df)
            handled_df = handled_df.dropna()
            dropped_rows = initial_rows - len(handled_df)
            logger.info(f"Dropped {dropped_rows} rows with missing values")
            
        elif method == 'fill_mean':
            # Fill missing values with mean
            handled_df = handled_df.fillna(handled_df.mean())
            logger.info("Filled missing values with mean")
            
        elif method == 'fill_median':
            # Fill missing values with median
            handled_df = handled_df.fillna(handled_df.median())
            logger.info("Filled missing values with median")
            
        else:
            logger.warning(f"Unknown method '{method}', dropping rows with missing values")
            handled_df = handled_df.dropna()
        
        return handled_df
        
    except Exception as e:
        logger.error(f"Error handling missing features: {str(e)}")
        return df 