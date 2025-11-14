"""
Model Evaluator Module
======================

Compares and evaluates multiple model performances with detailed metrics,
visualizations, and recommendations.
"""

import pandas as pd
import numpy as np
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')


class ModelEvaluator:
    """Evaluate and compare model performance."""

    def __init__(self, models_dict):
        """
        Initialize the model evaluator.

        Args:
            models_dict: Dictionary of trained models with their results
        """
        self.models = models_dict
        self.metadata = models_dict.get('_metadata', {})

    def get_performance_summary(self):
        """
        Get performance summary for all models.

        Returns:
            DataFrame with performance metrics for each model
        """
        summary_data = []

        for model_name, model_info in self.models.items():
            if model_name == '_metadata':
                continue

            metrics = model_info.get('metrics', {})

            summary_data.append({
                'Model': model_name,
                'R² Score': metrics.get('r2_score', 0.0),
                'RMSE': metrics.get('rmse', 0.0),
                'MAE': metrics.get('mae', 0.0),
                'MAPE': metrics.get('mape', 0.0),
                'Direction Accuracy': metrics.get('direction_accuracy', 0.0),
                'Overall Score': self._calculate_overall_score(metrics)
            })

        df = pd.DataFrame(summary_data)
        df = df.sort_values('Overall Score', ascending=False)

        return df

    def _calculate_overall_score(self, metrics):
        """
        Calculate overall model score.

        Weighted combination of different metrics.
        """
        r2 = max(0, metrics.get('r2_score', 0.0))
        direction_acc = metrics.get('direction_accuracy', 50.0) / 100.0

        # Weight R2 and direction accuracy
        overall = (r2 * 0.6 + direction_acc * 0.4) * 100

        return overall

    def get_best_model(self, metric='Overall Score'):
        """
        Get the best performing model based on a metric.

        Args:
            metric: Metric to use for ranking

        Returns:
            Tuple of (model_name, score)
        """
        summary = self.get_performance_summary()

        if metric not in summary.columns:
            raise ValueError(f"Unknown metric: {metric}")

        # For error metrics (RMSE, MAE, MAPE), lower is better
        if metric in ['RMSE', 'MAE', 'MAPE']:
            best_idx = summary[metric].idxmin()
        else:
            best_idx = summary[metric].idxmax()

        best_model = summary.loc[best_idx, 'Model']
        best_score = summary.loc[best_idx, metric]

        return best_model, best_score

    def compare_models(self, models_to_compare=None):
        """
        Detailed comparison of models.

        Args:
            models_to_compare: List of model names (all if None)

        Returns:
            DataFrame with detailed comparison
        """
        if models_to_compare is None:
            models_to_compare = [name for name in self.models.keys() if name != '_metadata']

        comparison_data = []

        for model_name in models_to_compare:
            if model_name not in self.models:
                continue

            model_info = self.models[model_name]
            metrics = model_info.get('metrics', {})

            comparison_data.append({
                'Model': model_name,
                'R² Score': metrics.get('r2_score', 0.0),
                'RMSE': metrics.get('rmse', 0.0),
                'MAE': metrics.get('mae', 0.0),
                'Direction Accuracy': metrics.get('direction_accuracy', 0.0),
                'Strength': self._assess_strength(metrics),
                'Best For': self._assess_use_case(model_name, metrics)
            })

        df = pd.DataFrame(comparison_data)
        return df

    def _assess_strength(self, metrics):
        """Assess model strength level."""
        score = self._calculate_overall_score(metrics)

        if score >= 70:
            return "Excellent"
        elif score >= 60:
            return "Good"
        elif score >= 50:
            return "Fair"
        else:
            return "Poor"

    def _assess_use_case(self, model_name, metrics):
        """Determine what each model is best for."""
        r2 = metrics.get('r2_score', 0.0)
        direction_acc = metrics.get('direction_accuracy', 0.0)

        if 'Random Forest' in model_name:
            return "Feature importance & stability"
        elif 'XGBoost' in model_name:
            return "High accuracy predictions"
        elif 'LightGBM' in model_name:
            return "Fast training & good accuracy"
        elif 'Gradient Boosting' in model_name:
            return "Robust predictions"
        elif 'Ridge' in model_name:
            return "Linear relationships"
        elif 'Lasso' in model_name:
            return "Feature selection"
        elif 'Ensemble' in model_name:
            return "Overall best predictions"
        else:
            return "General purpose"

    def get_feature_importance_summary(self, top_n=10):
        """
        Get feature importance summary across models.

        Args:
            top_n: Number of top features to include

        Returns:
            DataFrame with aggregated feature importance
        """
        feature_names = self.metadata.get('feature_names', [])

        if not feature_names:
            return pd.DataFrame()

        importance_data = {}

        for model_name, model_info in self.models.items():
            if model_name in ['_metadata', 'Ensemble']:
                continue

            importance = model_info.get('feature_importance')
            if importance is not None and len(importance) == len(feature_names):
                importance_data[model_name] = importance

        if not importance_data:
            return pd.DataFrame()

        # Create DataFrame
        df = pd.DataFrame(importance_data, index=feature_names)

        # Calculate average importance
        df['Average'] = df.mean(axis=1)

        # Sort by average importance
        df = df.sort_values('Average', ascending=False)

        # Return top N
        return df.head(top_n)

    def generate_model_explanations(self):
        """
        Generate detailed explanations for each model's performance.

        Returns:
            Dictionary with explanations for each model
        """
        explanations = {}

        summary = self.get_performance_summary()

        for _, row in summary.iterrows():
            model_name = row['Model']
            r2_score = row['R² Score']
            direction_acc = row['Direction Accuracy']
            overall_score = row['Overall Score']

            explanation = self._generate_single_model_explanation(
                model_name, r2_score, direction_acc, overall_score
            )

            explanations[model_name] = explanation

        return explanations

    def _generate_single_model_explanation(self, model_name, r2, direction_acc, overall_score):
        """Generate explanation for a single model."""
        explanation = {
            'name': model_name,
            'performance_level': '',
            'strengths': [],
            'weaknesses': [],
            'recommendation': ''
        }

        # Assess performance level
        if overall_score >= 70:
            explanation['performance_level'] = "Excellent - This model shows strong predictive power"
        elif overall_score >= 60:
            explanation['performance_level'] = "Good - This model provides reliable predictions"
        elif overall_score >= 50:
            explanation['performance_level'] = "Fair - This model has moderate predictive ability"
        else:
            explanation['performance_level'] = "Poor - This model struggles with predictions"

        # Assess strengths
        if r2 >= 0.7:
            explanation['strengths'].append(f"High R² score ({r2:.2f}) indicates strong price prediction")
        if direction_acc >= 65:
            explanation['strengths'].append(f"Good direction accuracy ({direction_acc:.1f}%) for trading signals")
        if r2 >= 0.6 and direction_acc >= 60:
            explanation['strengths'].append("Balanced performance across metrics")

        # Assess weaknesses
        if r2 < 0.5:
            explanation['weaknesses'].append(f"Low R² score ({r2:.2f}) suggests limited price accuracy")
        if direction_acc < 55:
            explanation['weaknesses'].append(f"Poor direction accuracy ({direction_acc:.1f}%) limits trading value")
        if abs(r2 - (direction_acc/100)) > 0.3:
            explanation['weaknesses'].append("Imbalanced performance across metrics")

        # Generate recommendation
        if overall_score >= 65:
            explanation['recommendation'] = "Strong model - High confidence in predictions"
        elif overall_score >= 55:
            explanation['recommendation'] = "Moderate model - Use with caution, combine with other signals"
        else:
            explanation['recommendation'] = "Weak model - Low weight in ensemble, verify with other models"

        return explanation

    def get_model_comparison_report(self):
        """
        Generate comprehensive model comparison report.

        Returns:
            Dictionary with full comparison analysis
        """
        summary = self.get_performance_summary()
        best_model, best_score = self.get_best_model()
        feature_importance = self.get_feature_importance_summary(top_n=10)
        explanations = self.generate_model_explanations()

        # Calculate consensus
        predictions_agreement = self._calculate_predictions_agreement()

        report = {
            'summary': summary,
            'best_model': {
                'name': best_model,
                'score': best_score
            },
            'feature_importance': feature_importance,
            'explanations': explanations,
            'predictions_agreement': predictions_agreement,
            'recommendation': self._generate_overall_recommendation(summary)
        }

        return report

    def _calculate_predictions_agreement(self):
        """Calculate how much models agree on predictions."""
        predictions = []

        for model_name, model_info in self.models.items():
            if model_name in ['_metadata', 'Ensemble']:
                continue

            preds = model_info.get('predictions')
            if preds is not None:
                predictions.append(preds)

        if len(predictions) == 0:
            return 0.0

        # Calculate agreement as correlation between predictions
        predictions_array = np.array(predictions)

        # Calculate pairwise correlations
        correlations = []
        for i in range(len(predictions)):
            for j in range(i + 1, len(predictions)):
                corr = np.corrcoef(predictions[i], predictions[j])[0, 1]
                correlations.append(corr)

        if len(correlations) == 0:
            return 0.0

        avg_correlation = np.mean(correlations)
        agreement_score = (avg_correlation + 1) / 2 * 100  # Convert to 0-100 scale

        return agreement_score

    def _generate_overall_recommendation(self, summary_df):
        """Generate overall recommendation based on all models."""
        avg_score = summary_df['Overall Score'].mean()

        if avg_score >= 70:
            confidence = "HIGH"
            recommendation = "Models show strong agreement and high accuracy. Predictions are highly reliable."
        elif avg_score >= 60:
            confidence = "MEDIUM-HIGH"
            recommendation = "Models show good performance. Predictions are reliable with moderate confidence."
        elif avg_score >= 50:
            confidence = "MEDIUM"
            recommendation = "Models show fair performance. Use predictions as one of multiple signals."
        else:
            confidence = "LOW"
            recommendation = "Models show weak performance. Exercise caution and verify with additional analysis."

        return {
            'confidence': confidence,
            'recommendation': recommendation,
            'average_score': avg_score
        }


def evaluate_models(models_dict):
    """
    Convenience function to evaluate models.

    Args:
        models_dict: Dictionary of trained models

    Returns:
        DataFrame with performance summary
    """
    evaluator = ModelEvaluator(models_dict)
    return evaluator.get_performance_summary()


def compare_model_performance(models_dict):
    """
    Convenience function to get comprehensive model comparison.

    Args:
        models_dict: Dictionary of trained models

    Returns:
        Dictionary with full comparison report
    """
    evaluator = ModelEvaluator(models_dict)
    return evaluator.get_model_comparison_report()
