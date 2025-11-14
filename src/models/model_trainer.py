"""
Model Trainer Module
====================

Trains multiple machine learning models for stock price prediction.
Supports ensemble learning with model comparison and evaluation.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
from prophet import Prophet
import warnings
warnings.filterwarnings('ignore')


class ModelTrainer:
    """Train and manage multiple ML models for stock prediction."""

    def __init__(self, test_size=0.2, random_state=42):
        """
        Initialize the model trainer.

        Args:
            test_size: Fraction of data to use for testing
            random_state: Random seed for reproducibility
        """
        self.test_size = test_size
        self.random_state = random_state
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_importance = {}

    def prepare_data(self, df, target_column='return_5d', feature_columns=None):
        """
        Prepare data for model training.

        Args:
            df: DataFrame with features and target
            target_column: Name of the target column
            feature_columns: List of feature column names (auto-detect if None)

        Returns:
            X_train, X_test, y_train, y_test, feature_names
        """
        # Validate input data
        if df is None or df.empty:
            raise ValueError("Input DataFrame is empty or None. Cannot train models without data.")

        print(f"Initial data shape: {df.shape}")

        # Remove NaN values
        df = df.dropna()

        # Validate after removing NaN values
        if df.empty:
            raise ValueError(
                "DataFrame is empty after removing NaN values. "
                "This likely means all rows contain at least one NaN value. "
                "Check your data source or feature engineering pipeline."
            )

        print(f"Data shape after dropna: {df.shape}")

        # Auto-detect feature columns if not provided
        if feature_columns is None:
            exclude_cols = [target_column, 'Date', 'Symbol', 'Close', 'Open', 'High', 'Low', 'Volume']
            feature_columns = [col for col in df.columns if col not in exclude_cols]

        # Ensure target exists
        if target_column not in df.columns:
            # Create target if it doesn't exist (5-day forward return)
            if 'Close' not in df.columns:
                raise ValueError("Cannot create target column: 'Close' column not found in DataFrame")
            df[target_column] = df['Close'].pct_change(5).shift(-5) * 100
            df = df.dropna()

            # Validate again after creating target
            if df.empty:
                raise ValueError(
                    f"DataFrame is empty after creating target column '{target_column}'. "
                    "This can happen if the dataset is too small (needs at least 6 rows for 5-day returns)."
                )

        print(f"Final data shape: {df.shape}")

        X = df[feature_columns].values
        y = df[target_column].values

        # Validate we have enough data for splitting
        min_samples = max(10, int(1 / self.test_size) + 1)  # At least 10 samples or enough for split
        if len(X) < min_samples:
            raise ValueError(
                f"Insufficient data for model training. Found {len(X)} samples, "
                f"but need at least {min_samples} samples (with test_size={self.test_size}). "
                "Try collecting more data or reducing the test_size parameter."
            )

        # Split data using time-series aware split
        split_idx = int(len(X) * (1 - self.test_size))

        # Ensure both train and test sets have at least 1 sample
        if split_idx < 1:
            split_idx = 1
        if split_idx >= len(X):
            split_idx = len(X) - 1

        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # Final validation before scaling
        if len(X_train) == 0:
            raise ValueError(
                f"Training set is empty after split. "
                f"Total samples: {len(X)}, split_idx: {split_idx}, test_size: {self.test_size}. "
                "Try adjusting the test_size parameter or collecting more data."
            )

        if len(X_test) == 0:
            raise ValueError(
                f"Test set is empty after split. "
                f"Total samples: {len(X)}, split_idx: {split_idx}, test_size: {self.test_size}. "
                "Try adjusting the test_size parameter or collecting more data."
            )

        print(f"Train/Test split: {len(X_train)}/{len(X_test)} samples")

        # Scale features
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        return X_train, X_test, y_train, y_test, feature_columns

    def train_random_forest(self, X_train, y_train, X_test, y_test):
        """Train Random Forest model."""
        print("Training Random Forest...")

        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=self.random_state,
            n_jobs=-1
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics = self._calculate_metrics(y_test, y_pred)

        return {
            'model': model,
            'predictions': y_pred,
            'metrics': metrics,
            'feature_importance': model.feature_importances_
        }

    def train_xgboost(self, X_train, y_train, X_test, y_test):
        """Train XGBoost model."""
        print("Training XGBoost...")

        model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=7,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            n_jobs=-1
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics = self._calculate_metrics(y_test, y_pred)

        return {
            'model': model,
            'predictions': y_pred,
            'metrics': metrics,
            'feature_importance': model.feature_importances_
        }

    def train_lightgbm(self, X_train, y_train, X_test, y_test):
        """Train LightGBM model."""
        print("Training LightGBM...")

        model = lgb.LGBMRegressor(
            n_estimators=200,
            max_depth=7,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=-1
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics = self._calculate_metrics(y_test, y_pred)

        return {
            'model': model,
            'predictions': y_pred,
            'metrics': metrics,
            'feature_importance': model.feature_importances_
        }

    def train_gradient_boosting(self, X_train, y_train, X_test, y_test):
        """Train Gradient Boosting model."""
        print("Training Gradient Boosting...")

        model = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            random_state=self.random_state
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics = self._calculate_metrics(y_test, y_pred)

        return {
            'model': model,
            'predictions': y_pred,
            'metrics': metrics,
            'feature_importance': model.feature_importances_
        }

    def train_ridge(self, X_train, y_train, X_test, y_test):
        """Train Ridge Regression model."""
        print("Training Ridge Regression...")

        model = Ridge(alpha=1.0, random_state=self.random_state)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics = self._calculate_metrics(y_test, y_pred)

        return {
            'model': model,
            'predictions': y_pred,
            'metrics': metrics,
            'feature_importance': np.abs(model.coef_)
        }

    def train_lasso(self, X_train, y_train, X_test, y_test):
        """Train Lasso Regression model."""
        print("Training Lasso Regression...")

        model = Lasso(alpha=0.1, random_state=self.random_state)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics = self._calculate_metrics(y_test, y_pred)

        return {
            'model': model,
            'predictions': y_pred,
            'metrics': metrics,
            'feature_importance': np.abs(model.coef_)
        }

    def _calculate_metrics(self, y_true, y_pred):
        """Calculate evaluation metrics."""
        return {
            'r2_score': r2_score(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'mape': np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100,
            'direction_accuracy': np.mean((np.sign(y_true) == np.sign(y_pred)).astype(float)) * 100
        }

    def train_all_models(self, X_train, y_train, X_test, y_test):
        """
        Train all available models.

        Returns:
            Dictionary of trained models with their predictions and metrics
        """
        results = {}

        # Train each model
        results['Random Forest'] = self.train_random_forest(X_train, y_train, X_test, y_test)
        results['XGBoost'] = self.train_xgboost(X_train, y_train, X_test, y_test)
        results['LightGBM'] = self.train_lightgbm(X_train, y_train, X_test, y_test)
        results['Gradient Boosting'] = self.train_gradient_boosting(X_train, y_train, X_test, y_test)
        results['Ridge'] = self.train_ridge(X_train, y_train, X_test, y_test)
        results['Lasso'] = self.train_lasso(X_train, y_train, X_test, y_test)

        # Create ensemble prediction
        ensemble_pred = np.mean([results[name]['predictions'] for name in results.keys()], axis=0)
        ensemble_metrics = self._calculate_metrics(y_test, ensemble_pred)

        results['Ensemble'] = {
            'model': None,  # Ensemble doesn't have a single model
            'predictions': ensemble_pred,
            'metrics': ensemble_metrics,
            'feature_importance': None
        }

        self.models = results
        return results


def train_all_models(df, target_column='return_5d', feature_columns=None):
    """
    Convenience function to train all models on a dataset.

    Args:
        df: DataFrame with features and target
        target_column: Name of the target column
        feature_columns: List of feature columns (auto-detect if None)

    Returns:
        Dictionary with trained models and results
    """
    trainer = ModelTrainer()

    # Prepare data
    X_train, X_test, y_train, y_test, features = trainer.prepare_data(
        df, target_column, feature_columns
    )

    # Train all models
    results = trainer.train_all_models(X_train, y_train, X_test, y_test)

    # Add metadata
    results['_metadata'] = {
        'feature_names': features,
        'target_column': target_column,
        'train_size': len(X_train),
        'test_size': len(X_test),
        'scaler': trainer.scaler
    }

    return results


def train_single_model(df, model_name, target_column='return_5d', feature_columns=None):
    """
    Train a single specific model.

    Args:
        df: DataFrame with features and target
        model_name: Name of model to train
        target_column: Name of the target column
        feature_columns: List of feature columns (auto-detect if None)

    Returns:
        Dictionary with trained model and results
    """
    trainer = ModelTrainer()

    # Prepare data
    X_train, X_test, y_train, y_test, features = trainer.prepare_data(
        df, target_column, feature_columns
    )

    # Train specific model
    model_methods = {
        'Random Forest': trainer.train_random_forest,
        'XGBoost': trainer.train_xgboost,
        'LightGBM': trainer.train_lightgbm,
        'Gradient Boosting': trainer.train_gradient_boosting,
        'Ridge': trainer.train_ridge,
        'Lasso': trainer.train_lasso
    }

    if model_name not in model_methods:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(model_methods.keys())}")

    result = model_methods[model_name](X_train, y_train, X_test, y_test)

    # Add metadata
    result['_metadata'] = {
        'feature_names': features,
        'target_column': target_column,
        'train_size': len(X_train),
        'test_size': len(X_test),
        'scaler': trainer.scaler
    }

    return result
