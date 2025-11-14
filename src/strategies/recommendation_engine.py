"""
Recommendation Engine Module
============================

Generates detailed buy/sell/hold recommendations with comprehensive
explanations based on model predictions, market data, and risk analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class RecommendationEngine:
    """Generate trading recommendations with detailed explanations."""

    def __init__(self, stock_data, predictions, model_comparison, current_price):
        """
        Initialize the recommendation engine.

        Args:
            stock_data: DataFrame with historical stock data
            predictions: Dictionary of model predictions
            model_comparison: Model comparison results
            current_price: Current stock price
        """
        self.stock_data = stock_data
        self.predictions = predictions
        self.model_comparison = model_comparison
        self.current_price = current_price

    def analyze_market_conditions(self):
        """
        Analyze current market conditions.

        Returns:
            Dictionary with market analysis
        """
        df = self.stock_data

        # Calculate technical indicators
        recent_data = df.tail(30)

        # Price momentum
        price_change_5d = ((df['Close'].iloc[-1] - df['Close'].iloc[-6]) / df['Close'].iloc[-6]) * 100
        price_change_20d = ((df['Close'].iloc[-1] - df['Close'].iloc[-21]) / df['Close'].iloc[-21]) * 100

        # Volatility
        volatility = df['Close'].tail(20).pct_change().std() * np.sqrt(252) * 100

        # Volume analysis
        avg_volume = df['Volume'].tail(20).mean()
        current_volume = df['Volume'].iloc[-1]
        volume_ratio = current_volume / avg_volume

        # Trend analysis
        sma_20 = df['Close'].tail(20).mean()
        sma_50 = df['Close'].tail(50).mean() if len(df) >= 50 else sma_20

        trend = "Uptrend" if sma_20 > sma_50 else "Downtrend"

        return {
            'price_change_5d': price_change_5d,
            'price_change_20d': price_change_20d,
            'volatility': volatility,
            'volume_ratio': volume_ratio,
            'trend': trend,
            'sma_20': sma_20,
            'sma_50': sma_50
        }

    def calculate_risk_metrics(self):
        """
        Calculate risk metrics.

        Returns:
            Dictionary with risk analysis
        """
        df = self.stock_data

        # Historical volatility
        returns = df['Close'].pct_change()
        volatility = returns.std() * np.sqrt(252) * 100

        # Maximum drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min() * 100

        # Value at Risk (95% confidence)
        var_95 = np.percentile(returns.dropna(), 5) * 100

        # Sharpe Ratio (assuming 2% risk-free rate)
        excess_returns = returns.mean() * 252 - 0.02
        sharpe = excess_returns / (returns.std() * np.sqrt(252)) if returns.std() > 0 else 0

        # Risk level assessment
        if volatility < 20:
            risk_level = "LOW"
        elif volatility < 35:
            risk_level = "MEDIUM"
        else:
            risk_level = "HIGH"

        return {
            'volatility': volatility,
            'max_drawdown': max_drawdown,
            'var_95': var_95,
            'sharpe_ratio': sharpe,
            'risk_level': risk_level
        }

    def calculate_confidence_score(self):
        """
        Calculate overall confidence score.

        Returns:
            Confidence score (0-100)
        """
        # Get prediction agreement
        pred_values = list(self.predictions.values())
        pred_std = np.std(pred_values)
        pred_mean = np.mean(pred_values)

        # Agreement score (lower std relative to mean = higher agreement)
        if abs(pred_mean) > 0.1:
            agreement = max(0, 1 - (pred_std / abs(pred_mean)))
        else:
            agreement = 0.5

        # Model performance score
        avg_model_score = self.model_comparison.get('recommendation', {}).get('average_score', 50)

        # Combine scores
        confidence = (agreement * 50) + (avg_model_score * 0.5)

        return min(100, max(0, confidence))

    def generate_recommendation(self):
        """
        Generate comprehensive trading recommendation.

        Returns:
            Dictionary with recommendation and detailed analysis
        """
        # Get prediction ensemble
        pred_values = list(self.predictions.values())
        ensemble_pred = np.mean(pred_values)

        # Analyze market conditions
        market_analysis = self.analyze_market_conditions()

        # Calculate risk metrics
        risk_metrics = self.calculate_risk_metrics()

        # Calculate confidence
        confidence_score = self.calculate_confidence_score()

        # Determine action
        action = self._determine_action(ensemble_pred, market_analysis, risk_metrics, confidence_score)

        # Generate detailed explanation
        explanation = self._generate_explanation(
            action, ensemble_pred, market_analysis, risk_metrics, confidence_score
        )

        # Calculate target prices
        targets = self._calculate_price_targets(ensemble_pred)

        # Generate risk warnings
        warnings = self._generate_risk_warnings(risk_metrics, market_analysis)

        return {
            'action': action['recommendation'],
            'confidence': action['confidence'],
            'confidence_score': confidence_score,
            'predicted_return': ensemble_pred,
            'target_price': targets['target'],
            'stop_loss': targets['stop_loss'],
            'upside_potential': targets['upside'],
            'downside_risk': targets['downside'],
            'explanation': explanation,
            'market_analysis': market_analysis,
            'risk_metrics': risk_metrics,
            'warnings': warnings,
            'timeframe': '5-10 trading days'
        }

    def _determine_action(self, ensemble_pred, market_analysis, risk_metrics, confidence_score):
        """Determine the trading action."""
        # Base recommendation on prediction
        if ensemble_pred > 5:
            base_action = "STRONG BUY"
        elif ensemble_pred > 2:
            base_action = "BUY"
        elif ensemble_pred > -2:
            base_action = "HOLD"
        elif ensemble_pred > -5:
            base_action = "SELL"
        else:
            base_action = "STRONG SELL"

        # Adjust based on confidence
        if confidence_score < 50:
            # Downgrade strong recommendations to regular
            if base_action == "STRONG BUY":
                base_action = "BUY"
            elif base_action == "STRONG SELL":
                base_action = "SELL"

        # Adjust based on risk
        if risk_metrics['risk_level'] == "HIGH" and base_action in ["STRONG BUY", "BUY"]:
            # Consider downgrading in high risk
            if confidence_score < 70:
                base_action = "HOLD" if base_action == "BUY" else "BUY"

        # Determine confidence level
        if confidence_score >= 75:
            confidence = "HIGH"
        elif confidence_score >= 55:
            confidence = "MEDIUM"
        else:
            confidence = "LOW"

        return {
            'recommendation': base_action,
            'confidence': confidence
        }

    def _calculate_price_targets(self, ensemble_pred):
        """Calculate price targets and stop loss."""
        # Convert percentage prediction to price
        predicted_change = ensemble_pred / 100
        target_price = self.current_price * (1 + predicted_change)

        # Calculate upside/downside
        upside = (target_price - self.current_price) / self.current_price * 100
        downside = -abs(upside) * 0.5  # Assume 2:1 reward/risk ratio

        # Calculate stop loss (5% below current or predicted downside)
        stop_loss = self.current_price * (1 + min(downside / 100, -0.05))

        return {
            'target': target_price,
            'stop_loss': stop_loss,
            'upside': upside,
            'downside': downside
        }

    def _generate_explanation(self, action, ensemble_pred, market_analysis, risk_metrics, confidence_score):
        """Generate detailed explanation for the recommendation."""
        explanation = {
            'summary': '',
            'prediction_analysis': '',
            'market_context': '',
            'risk_assessment': '',
            'reasoning': []
        }

        # Summary
        explanation['summary'] = (
            f"Based on analysis of {len(self.predictions)} machine learning models, "
            f"the recommendation is to {action['recommendation']} with {action['confidence']} confidence. "
            f"Models predict a {ensemble_pred:.2f}% return over the next 5-10 trading days."
        )

        # Prediction analysis
        bullish_models = sum(1 for p in self.predictions.values() if p > 0)
        bearish_models = len(self.predictions) - bullish_models

        explanation['prediction_analysis'] = (
            f"{bullish_models} out of {len(self.predictions)} models predict positive returns, "
            f"while {bearish_models} predict negative returns. "
            f"The ensemble prediction is {ensemble_pred:.2f}%, suggesting "
            f"{'bullish' if ensemble_pred > 0 else 'bearish'} sentiment."
        )

        # Market context
        trend_word = market_analysis['trend'].lower()
        explanation['market_context'] = (
            f"The stock is currently in a {trend_word}, with "
            f"{market_analysis['price_change_5d']:.2f}% change over 5 days and "
            f"{market_analysis['price_change_20d']:.2f}% over 20 days. "
            f"Volatility is {risk_metrics['volatility']:.1f}%, indicating "
            f"{risk_metrics['risk_level'].lower()} risk levels."
        )

        # Risk assessment
        explanation['risk_assessment'] = (
            f"Risk level is {risk_metrics['risk_level']} with annualized volatility of "
            f"{risk_metrics['volatility']:.1f}%. Maximum historical drawdown is "
            f"{abs(risk_metrics['max_drawdown']):.1f}%. "
            f"{'This stock has above-average risk.' if risk_metrics['risk_level'] == 'HIGH' else 'Risk levels are manageable.'}"
        )

        # Reasoning points
        if ensemble_pred > 2:
            explanation['reasoning'].append(
                f"Strong positive prediction ({ensemble_pred:.2f}%) suggests good upside potential"
            )
        elif ensemble_pred < -2:
            explanation['reasoning'].append(
                f"Strong negative prediction ({ensemble_pred:.2f}%) suggests downside risk"
            )

        if confidence_score >= 70:
            explanation['reasoning'].append(
                f"High model agreement (confidence: {confidence_score:.1f}%) increases reliability"
            )
        elif confidence_score < 50:
            explanation['reasoning'].append(
                f"Low model agreement (confidence: {confidence_score:.1f}%) suggests uncertainty"
            )

        if market_analysis['trend'] == "Uptrend" and ensemble_pred > 0:
            explanation['reasoning'].append(
                "Current uptrend aligns with positive prediction"
            )
        elif market_analysis['trend'] == "Downtrend" and ensemble_pred < 0:
            explanation['reasoning'].append(
                "Current downtrend aligns with negative prediction"
            )

        if risk_metrics['risk_level'] == "HIGH":
            explanation['reasoning'].append(
                f"High volatility ({risk_metrics['volatility']:.1f}%) requires careful risk management"
            )

        if market_analysis['volume_ratio'] > 1.5:
            explanation['reasoning'].append(
                "Elevated volume suggests strong market interest"
            )
        elif market_analysis['volume_ratio'] < 0.7:
            explanation['reasoning'].append(
                "Low volume may indicate weak conviction"
            )

        return explanation

    def _generate_risk_warnings(self, risk_metrics, market_analysis):
        """Generate risk warnings."""
        warnings_list = []

        if risk_metrics['volatility'] > 40:
            warnings_list.append(
                "HIGH VOLATILITY: This stock has very high volatility. Expect large price swings."
            )

        if abs(risk_metrics['max_drawdown']) > 30:
            warnings_list.append(
                f"SIGNIFICANT DRAWDOWN RISK: Historical max drawdown is {abs(risk_metrics['max_drawdown']):.1f}%"
            )

        if risk_metrics['sharpe_ratio'] < 0:
            warnings_list.append(
                "NEGATIVE RISK-ADJUSTED RETURNS: Sharpe ratio is negative"
            )

        if market_analysis['volume_ratio'] < 0.5:
            warnings_list.append(
                "LOW LIQUIDITY: Trading volume is significantly below average"
            )

        return warnings_list

    def get_model_predictions_detail(self):
        """
        Get detailed breakdown of model predictions.

        Returns:
            DataFrame with individual model predictions and analysis
        """
        detail_data = []

        for model_name, prediction in self.predictions.items():
            # Get model metrics if available
            model_info = self.model_comparison.get('summary', pd.DataFrame())

            if not model_info.empty:
                model_row = model_info[model_info['Model'] == model_name]
                if not model_row.empty:
                    r2_score = model_row['R² Score'].values[0]
                    direction_acc = model_row['Direction Accuracy'].values[0]
                    overall_score = model_row['Overall Score'].values[0]
                else:
                    r2_score = direction_acc = overall_score = 0
            else:
                r2_score = direction_acc = overall_score = 0

            detail_data.append({
                'Model': model_name,
                'Prediction (%)': prediction,
                'Direction': 'Bullish' if prediction > 0 else 'Bearish',
                'Strength': 'Strong' if abs(prediction) > 3 else 'Moderate' if abs(prediction) > 1 else 'Weak',
                'Model Accuracy': overall_score,
                'R² Score': r2_score,
                'Direction Accuracy': direction_acc
            })

        df = pd.DataFrame(detail_data)
        df = df.sort_values('Prediction (%)', ascending=False)

        return df


def generate_recommendation(stock_data, predictions, model_comparison, current_price):
    """
    Convenience function to generate recommendation.

    Args:
        stock_data: DataFrame with historical stock data
        predictions: Dictionary of model predictions
        model_comparison: Model comparison results
        current_price: Current stock price

    Returns:
        Dictionary with recommendation and analysis
    """
    engine = RecommendationEngine(stock_data, predictions, model_comparison, current_price)
    return engine.generate_recommendation()


def get_detailed_analysis(stock_data, predictions, model_comparison, current_price):
    """
    Get comprehensive detailed analysis.

    Args:
        stock_data: DataFrame with historical stock data
        predictions: Dictionary of model predictions
        model_comparison: Model comparison results
        current_price: Current stock price

    Returns:
        Dictionary with full analysis including recommendation and model details
    """
    engine = RecommendationEngine(stock_data, predictions, model_comparison, current_price)

    recommendation = engine.generate_recommendation()
    model_details = engine.get_model_predictions_detail()

    return {
        'recommendation': recommendation,
        'model_predictions': model_details,
        'current_price': current_price
    }
