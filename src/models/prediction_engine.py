"""
Prediction Engine Module
========================

Generates predictions from trained models and provides ensemble predictions
with confidence intervals and detailed comparisons.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class PredictionEngine:
    """Generate and manage predictions from multiple models."""

    def __init__(self, models_dict):
        """
        Initialize the prediction engine.

        Args:
            models_dict: Dictionary of trained models from model_trainer
        """
        self.models = models_dict
        self.metadata = models_dict.get('_metadata', {})
        self.scaler = self.metadata.get('scaler')

    def prepare_features(self, df, feature_columns=None):
        """
        Prepare features for prediction.

        Args:
            df: DataFrame with features
            feature_columns: List of feature columns (use training features if None)

        Returns:
            Scaled feature array
        """
        if feature_columns is None:
            feature_columns = self.metadata.get('feature_names', [])

        # Get latest data point
        if isinstance(df, pd.DataFrame):
            X = df[feature_columns].iloc[-1:].values
        else:
            X = df.reshape(1, -1)

        # Scale features
        if self.scaler is not None:
            X = self.scaler.transform(X)

        return X

    def predict_single_model(self, model_name, X):
        """
        Generate prediction from a single model.

        Args:
            model_name: Name of the model
            X: Feature array

        Returns:
            Prediction value
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")

        model_info = self.models[model_name]
        model = model_info.get('model')

        if model is None:
            return None  # Ensemble doesn't have a direct model

        prediction = model.predict(X)[0]
        return prediction

    def predict_all_models(self, X):
        """
        Generate predictions from all models.

        Args:
            X: Feature array

        Returns:
            Dictionary of predictions by model name
        """
        predictions = {}

        for model_name in self.models.keys():
            if model_name == '_metadata' or model_name == 'Ensemble':
                continue

            try:
                pred = self.predict_single_model(model_name, X)
                if pred is not None:
                    predictions[model_name] = pred
            except Exception as e:
                print(f"Error predicting with {model_name}: {e}")
                continue

        return predictions

    def get_ensemble_prediction(self, predictions=None, X=None):
        """
        Calculate ensemble prediction.

        Args:
            predictions: Dictionary of predictions (will generate if None)
            X: Feature array (required if predictions is None)

        Returns:
            Tuple of (ensemble_prediction, confidence_interval)
        """
        if predictions is None:
            if X is None:
                raise ValueError("Either predictions or X must be provided")
            predictions = self.predict_all_models(X)

        # Calculate ensemble as weighted average
        pred_values = list(predictions.values())

        if len(pred_values) == 0:
            return 0.0, (0.0, 0.0)

        # Use mean as ensemble
        ensemble_pred = np.mean(pred_values)

        # Calculate confidence interval (95%)
        std = np.std(pred_values)
        lower_bound = ensemble_pred - 1.96 * std
        upper_bound = ensemble_pred + 1.96 * std

        return ensemble_pred, (lower_bound, upper_bound)

    def get_prediction_statistics(self, predictions):
        """
        Calculate statistics about the predictions.

        Args:
            predictions: Dictionary of predictions by model

        Returns:
            Dictionary with statistics
        """
        pred_values = list(predictions.values())

        if len(pred_values) == 0:
            return {}

        return {
            'mean': np.mean(pred_values),
            'median': np.median(pred_values),
            'std': np.std(pred_values),
            'min': np.min(pred_values),
            'max': np.max(pred_values),
            'range': np.max(pred_values) - np.min(pred_values),
            'agreement_score': self._calculate_agreement(pred_values)
        }

    def _calculate_agreement(self, predictions):
        """
        Calculate how much models agree on direction.

        Returns:
            Agreement score between 0-100
        """
        if len(predictions) == 0:
            return 0.0

        # Check if all predictions have same sign
        signs = np.sign(predictions)
        positive_ratio = np.sum(signs > 0) / len(signs)
        negative_ratio = np.sum(signs < 0) / len(signs)

        # Agreement is the maximum ratio
        agreement = max(positive_ratio, negative_ratio) * 100

        return agreement

    def compare_predictions(self, predictions):
        """
        Compare predictions across models.

        Args:
            predictions: Dictionary of predictions by model

        Returns:
            DataFrame with comparison details
        """
        comparison_data = []

        ensemble_pred, (lower, upper) = self.get_ensemble_prediction(predictions)

        for model_name, pred_value in predictions.items():
            # Calculate deviation from ensemble
            deviation = pred_value - ensemble_pred
            deviation_pct = (deviation / (abs(ensemble_pred) + 1e-10)) * 100

            # Determine if outlier
            is_outlier = (pred_value < lower) or (pred_value > upper)

            comparison_data.append({
                'Model': model_name,
                'Prediction': pred_value,
                'Direction': 'Bullish' if pred_value > 0 else 'Bearish',
                'Deviation from Ensemble': deviation,
                'Deviation %': deviation_pct,
                'Is Outlier': is_outlier,
                'Confidence': self._get_model_confidence(model_name)
            })

        df = pd.DataFrame(comparison_data)
        df = df.sort_values('Prediction', ascending=False)

        return df

    def _get_model_confidence(self, model_name):
        """Get confidence score for a model based on its metrics."""
        if model_name not in self.models:
            return 0.0

        metrics = self.models[model_name].get('metrics', {})
        r2 = metrics.get('r2_score', 0.0)
        direction_acc = metrics.get('direction_accuracy', 50.0) / 100.0

        # Combine R2 and direction accuracy
        confidence = (r2 * 0.6 + direction_acc * 0.4) * 100

        return max(0, min(100, confidence))

    def generate_prediction_report(self, df):
        """
        Generate comprehensive prediction report.

        Args:
            df: DataFrame with features

        Returns:
            Dictionary with full prediction analysis
        """
        # Prepare features
        X = self.prepare_features(df)

        # Get all predictions
        predictions = self.predict_all_models(X)

        # Get ensemble
        ensemble_pred, confidence_interval = self.get_ensemble_prediction(predictions)

        # Get statistics
        stats = self.get_prediction_statistics(predictions)

        # Get comparison
        comparison_df = self.compare_predictions(predictions)

        # Generate recommendation
        recommendation = self._generate_recommendation(
            ensemble_pred, stats.get('agreement_score', 0)
        )

        return {
            'predictions': predictions,
            'ensemble_prediction': ensemble_pred,
            'confidence_interval': confidence_interval,
            'statistics': stats,
            'comparison': comparison_df,
            'recommendation': recommendation,
            'model_count': len(predictions)
        }

    def _generate_recommendation(self, ensemble_pred, agreement_score):
        """Generate trading recommendation."""
        # Determine action
        if ensemble_pred > 2:
            action = "STRONG BUY"
        elif ensemble_pred > 0.5:
            action = "BUY"
        elif ensemble_pred > -0.5:
            action = "HOLD"
        elif ensemble_pred > -2:
            action = "SELL"
        else:
            action = "STRONG SELL"

        # Determine confidence level
        if agreement_score > 80:
            confidence = "HIGH"
        elif agreement_score > 60:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"

        return {
            'action': action,
            'confidence': confidence,
            'agreement_score': agreement_score,
            'predicted_return': ensemble_pred
        }


def generate_predictions(df, models_dict):
    """
    Convenience function to generate predictions.

    Args:
        df: DataFrame with features
        models_dict: Dictionary of trained models

    Returns:
        Dictionary of predictions by model name
    """
    engine = PredictionEngine(models_dict)
    X = engine.prepare_features(df)
    predictions = engine.predict_all_models(X)

    return predictions


def get_ensemble_prediction(df, models_dict):
    """
    Convenience function to get ensemble prediction with confidence interval.

    Args:
        df: DataFrame with features
        models_dict: Dictionary of trained models

    Returns:
        Tuple of (ensemble_prediction, confidence_interval)
    """
    engine = PredictionEngine(models_dict)
    X = engine.prepare_features(df)
    predictions = engine.predict_all_models(X)

    return engine.get_ensemble_prediction(predictions)


def get_prediction_report(df, models_dict):
    """
    Generate comprehensive prediction report.

    Args:
        df: DataFrame with features
        models_dict: Dictionary of trained models

    Returns:
        Dictionary with full prediction analysis
    """
    engine = PredictionEngine(models_dict)
    return engine.generate_prediction_report(df)
