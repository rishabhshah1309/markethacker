"""
MarketHacker Strategies Module
===============================

This module contains trading strategies, options analysis, and
recommendation engines for stock trading decisions.
"""

from .recommendation_engine import generate_recommendation, get_detailed_analysis
from .options_analyzer import analyze_options, recommend_options_strategy

__all__ = [
    'generate_recommendation',
    'get_detailed_analysis',
    'analyze_options',
    'recommend_options_strategy'
]
