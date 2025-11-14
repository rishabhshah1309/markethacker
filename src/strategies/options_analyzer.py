"""
Options Analyzer Module
=======================

Analyzes options strategies and provides recommendations based on
stock predictions, volatility, and risk-reward profiles.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')


class OptionsAnalyzer:
    """Analyze options strategies and generate recommendations."""

    def __init__(self, current_price, predicted_return, volatility, days_to_expiration=30):
        """
        Initialize the options analyzer.

        Args:
            current_price: Current stock price
            predicted_return: Predicted return percentage
            volatility: Annualized volatility (as percentage)
            days_to_expiration: Days until options expiration
        """
        self.current_price = current_price
        self.predicted_return = predicted_return
        self.volatility = volatility / 100  # Convert to decimal
        self.days_to_expiration = days_to_expiration
        self.time_to_expiration = days_to_expiration / 365

    def generate_strike_prices(self, num_strikes=10):
        """
        Generate relevant strike prices around current price.

        Args:
            num_strikes: Number of strike prices to generate

        Returns:
            List of strike prices
        """
        # Generate strikes from 10% below to 10% above current price
        lower_bound = self.current_price * 0.90
        upper_bound = self.current_price * 1.10

        strikes = np.linspace(lower_bound, upper_bound, num_strikes)

        # Round to nearest dollar
        strikes = np.round(strikes)

        return strikes

    def calculate_black_scholes_price(self, strike, option_type='call'):
        """
        Calculate option price using Black-Scholes model.

        Args:
            strike: Strike price
            option_type: 'call' or 'put'

        Returns:
            Option price
        """
        S = self.current_price
        K = strike
        T = self.time_to_expiration
        r = 0.05  # Risk-free rate (5%)
        sigma = self.volatility

        # Handle edge cases
        if T <= 0:
            if option_type == 'call':
                return max(0, S - K)
            else:
                return max(0, K - S)

        # Calculate d1 and d2
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if option_type == 'call':
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:  # put
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

        return price

    def calculate_probability_profit(self, strike, option_type='call'):
        """
        Calculate probability of profit for an option.

        Args:
            strike: Strike price
            option_type: 'call' or 'put'

        Returns:
            Probability of profit (0-1)
        """
        # Predicted future price
        future_price = self.current_price * (1 + self.predicted_return / 100)

        # Standard deviation of future price
        std_dev = self.current_price * self.volatility * np.sqrt(self.time_to_expiration)

        if option_type == 'call':
            # For call, profit when future price > strike + premium
            option_price = self.calculate_black_scholes_price(strike, 'call')
            breakeven = strike + option_price
            z_score = (future_price - breakeven) / std_dev
        else:  # put
            # For put, profit when future price < strike - premium
            option_price = self.calculate_black_scholes_price(strike, 'put')
            breakeven = strike - option_price
            z_score = (breakeven - future_price) / std_dev

        probability = norm.cdf(z_score)

        return probability

    def calculate_expected_value(self, strike, option_type='call'):
        """
        Calculate expected value of an option position.

        Args:
            strike: Strike price
            option_type: 'call' or 'put'

        Returns:
            Expected value
        """
        option_price = self.calculate_black_scholes_price(strike, option_type)
        future_price = self.current_price * (1 + self.predicted_return / 100)

        if option_type == 'call':
            intrinsic_value = max(0, future_price - strike)
        else:  # put
            intrinsic_value = max(0, strike - future_price)

        expected_profit = intrinsic_value - option_price

        return expected_profit

    def analyze_options(self):
        """
        Analyze all options strategies.

        Returns:
            DataFrame with options analysis
        """
        strikes = self.generate_strike_prices()

        options_data = []

        for strike in strikes:
            # Analyze calls
            call_price = self.calculate_black_scholes_price(strike, 'call')
            call_prob = self.calculate_probability_profit(strike, 'call')
            call_ev = self.calculate_expected_value(strike, 'call')

            options_data.append({
                'Type': 'CALL',
                'Strike': strike,
                'Premium': call_price,
                'Probability of Profit': call_prob * 100,
                'Expected Value': call_ev,
                'Return Potential': (call_ev / call_price * 100) if call_price > 0 else 0,
                'Risk': call_price,
                'Reward': call_ev if call_ev > 0 else 0,
                'Risk/Reward Ratio': (call_ev / call_price) if call_price > 0 else 0
            })

            # Analyze puts
            put_price = self.calculate_black_scholes_price(strike, 'put')
            put_prob = self.calculate_probability_profit(strike, 'put')
            put_ev = self.calculate_expected_value(strike, 'put')

            options_data.append({
                'Type': 'PUT',
                'Strike': strike,
                'Premium': put_price,
                'Probability of Profit': put_prob * 100,
                'Expected Value': put_ev,
                'Return Potential': (put_ev / put_price * 100) if put_price > 0 else 0,
                'Risk': put_price,
                'Reward': put_ev if put_ev > 0 else 0,
                'Risk/Reward Ratio': (put_ev / put_price) if put_price > 0 else 0
            })

        df = pd.DataFrame(options_data)
        return df

    def recommend_strategies(self):
        """
        Recommend best options strategies.

        Returns:
            Dictionary with strategy recommendations
        """
        options_df = self.analyze_options()

        # Filter for positive expected value
        profitable_options = options_df[options_df['Expected Value'] > 0]

        recommendations = {
            'best_overall': None,
            'best_call': None,
            'best_put': None,
            'strategies': []
        }

        # Best overall option
        if not profitable_options.empty:
            best_idx = profitable_options['Risk/Reward Ratio'].idxmax()
            recommendations['best_overall'] = profitable_options.loc[best_idx].to_dict()

        # Best call option
        profitable_calls = options_df[(options_df['Type'] == 'CALL') & (options_df['Expected Value'] > 0)]
        if not profitable_calls.empty:
            best_call_idx = profitable_calls['Risk/Reward Ratio'].idxmax()
            recommendations['best_call'] = profitable_calls.loc[best_call_idx].to_dict()

        # Best put option
        profitable_puts = options_df[(options_df['Type'] == 'PUT') & (options_df['Expected Value'] > 0)]
        if not profitable_puts.empty:
            best_put_idx = profitable_puts['Risk/Reward Ratio'].idxmax()
            recommendations['best_put'] = profitable_puts.loc[best_put_idx].to_dict()

        # Generate strategy recommendations
        recommendations['strategies'] = self._generate_strategy_recommendations()

        return recommendations

    def _generate_strategy_recommendations(self):
        """Generate specific strategy recommendations."""
        strategies = []

        # Determine market outlook based on prediction
        if self.predicted_return > 5:
            outlook = "STRONGLY BULLISH"
        elif self.predicted_return > 2:
            outlook = "BULLISH"
        elif self.predicted_return > -2:
            outlook = "NEUTRAL"
        elif self.predicted_return > -5:
            outlook = "BEARISH"
        else:
            outlook = "STRONGLY BEARISH"

        # Volatility level
        vol_level = "HIGH" if self.volatility > 0.35 else "MEDIUM" if self.volatility > 0.20 else "LOW"

        # Recommend strategies based on outlook and volatility
        if outlook in ["STRONGLY BULLISH", "BULLISH"]:
            strategies.append({
                'strategy': 'Long Call',
                'description': f'Buy call options with strike near ${self.current_price * 1.02:.2f}',
                'rationale': f'Predicted {self.predicted_return:.2f}% upside with {vol_level.lower()} volatility',
                'risk_level': 'Medium',
                'max_loss': 'Limited to premium paid',
                'max_gain': 'Unlimited'
            })

            if vol_level == "HIGH":
                strategies.append({
                    'strategy': 'Bull Call Spread',
                    'description': f'Buy call at ${self.current_price:.2f}, sell call at ${self.current_price * 1.05:.2f}',
                    'rationale': 'Reduce cost in high volatility environment',
                    'risk_level': 'Low-Medium',
                    'max_loss': 'Limited to net premium paid',
                    'max_gain': 'Limited to spread width minus premium'
                })

        if outlook in ["STRONGLY BEARISH", "BEARISH"]:
            strategies.append({
                'strategy': 'Long Put',
                'description': f'Buy put options with strike near ${self.current_price * 0.98:.2f}',
                'rationale': f'Predicted {abs(self.predicted_return):.2f}% downside with {vol_level.lower()} volatility',
                'risk_level': 'Medium',
                'max_loss': 'Limited to premium paid',
                'max_gain': 'Significant (price can go to zero)'
            })

            if vol_level == "HIGH":
                strategies.append({
                    'strategy': 'Bear Put Spread',
                    'description': f'Buy put at ${self.current_price:.2f}, sell put at ${self.current_price * 0.95:.2f}',
                    'rationale': 'Reduce cost in high volatility environment',
                    'risk_level': 'Low-Medium',
                    'max_loss': 'Limited to net premium paid',
                    'max_gain': 'Limited to spread width minus premium'
                })

        if outlook == "NEUTRAL" or vol_level == "HIGH":
            strategies.append({
                'strategy': 'Iron Condor',
                'description': 'Sell OTM call and put, buy further OTM call and put',
                'rationale': f'Profit from expected low movement with {vol_level.lower()} volatility',
                'risk_level': 'Medium',
                'max_loss': 'Limited to spread width minus credit',
                'max_gain': 'Limited to credit received'
            })

            strategies.append({
                'strategy': 'Short Straddle/Strangle',
                'description': f'Sell call and put at or near ${self.current_price:.2f}',
                'rationale': 'Profit from time decay in range-bound market',
                'risk_level': 'High',
                'max_loss': 'Unlimited',
                'max_gain': 'Limited to premium received'
            })

        if vol_level == "LOW" and abs(self.predicted_return) > 3:
            strategies.append({
                'strategy': 'Long Straddle',
                'description': f'Buy call and put at ${self.current_price:.2f}',
                'rationale': 'Expect significant move but uncertain of direction',
                'risk_level': 'Medium-High',
                'max_loss': 'Limited to total premium paid',
                'max_gain': 'Unlimited upside, significant downside'
            })

        return strategies


def analyze_options(current_price, predicted_return, volatility=30, days_to_expiration=30):
    """
    Convenience function to analyze options.

    Args:
        current_price: Current stock price
        predicted_return: Predicted return percentage
        volatility: Annualized volatility (as percentage)
        days_to_expiration: Days until options expiration

    Returns:
        DataFrame with options analysis
    """
    analyzer = OptionsAnalyzer(current_price, predicted_return, volatility, days_to_expiration)
    return analyzer.analyze_options()


def recommend_options_strategy(current_price, predicted_return, volatility=30, days_to_expiration=30):
    """
    Get options strategy recommendations.

    Args:
        current_price: Current stock price
        predicted_return: Predicted return percentage
        volatility: Annualized volatility (as percentage)
        days_to_expiration: Days until options expiration

    Returns:
        Dictionary with strategy recommendations
    """
    analyzer = OptionsAnalyzer(current_price, predicted_return, volatility, days_to_expiration)
    return analyzer.recommend_strategies()
