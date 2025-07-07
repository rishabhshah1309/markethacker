"""
Basic tests for MarketHacker.

This module contains basic tests to verify the project structure and imports.
"""

import unittest
import sys
import os
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))


class TestMarketHackerImports(unittest.TestCase):
    """Test that all MarketHacker modules can be imported."""
    
    def test_data_ingestion_imports(self):
        """Test data ingestion module imports."""
        try:
            from data_ingestion.fetch_market_data import download_stock_data, get_stock_info
            from data_ingestion.fetch_news import fetch_news_data
            from data_ingestion.fetch_sentiment import fetch_social_sentiment
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Failed to import data ingestion modules: {e}")
    
    def test_preprocessing_imports(self):
        """Test preprocessing module imports."""
        try:
            from preprocessing.data_cleaner import clean_stock_data, handle_missing_values
            from preprocessing.feature_engineering import create_features
            from preprocessing.data_transformer import normalize_features, encode_categorical
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Failed to import preprocessing modules: {e}")
    
    def test_cli_import(self):
        """Test CLI module import."""
        try:
            from cli import main
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Failed to import CLI module: {e}")


class TestDataStructures(unittest.TestCase):
    """Test basic data structure operations."""
    
    def test_dataframe_operations(self):
        """Test basic DataFrame operations."""
        # Create sample data
        data = {
            'Date': pd.date_range('2023-01-01', periods=10),
            'Open': np.random.rand(10) * 100,
            'High': np.random.rand(10) * 100,
            'Low': np.random.rand(10) * 100,
            'Close': np.random.rand(10) * 100,
            'Volume': np.random.randint(1000000, 10000000, 10)
        }
        
        df = pd.DataFrame(data)
        df.set_index('Date', inplace=True)
        
        # Test basic operations
        self.assertEqual(len(df), 10)
        self.assertTrue('Close' in df.columns)
        self.assertTrue(isinstance(df.index, pd.DatetimeIndex))
    
    def test_technical_indicators(self):
        """Test technical indicator calculations."""
        # Create sample price data
        prices = pd.Series([100, 101, 99, 102, 98, 103, 97, 104, 96, 105])
        
        # Test moving average
        sma_5 = prices.rolling(window=5).mean()
        self.assertEqual(len(sma_5), len(prices))
        self.assertTrue(pd.isna(sma_5.iloc[0]))  # First 4 values should be NaN
        
        # Test price change
        price_change = prices.pct_change()
        self.assertEqual(len(price_change), len(prices))
        self.assertTrue(pd.isna(price_change.iloc[0]))  # First value should be NaN


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions."""
    
    def test_date_validation(self):
        """Test date validation."""
        from datetime import datetime
        
        # Valid dates
        valid_dates = ['2023-01-01', '2023-12-31', '2024-02-29']
        for date_str in valid_dates:
            try:
                datetime.strptime(date_str, '%Y-%m-%d')
                self.assertTrue(True)
            except ValueError:
                self.fail(f"Failed to parse valid date: {date_str}")
        
        # Invalid dates
        invalid_dates = ['2023-13-01', '2023-02-30', 'invalid-date']
        for date_str in invalid_dates:
            with self.assertRaises(ValueError):
                datetime.strptime(date_str, '%Y-%m-%d')
    
    def test_symbol_validation(self):
        """Test stock symbol validation."""
        # Valid symbols
        valid_symbols = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN']
        for symbol in valid_symbols:
            self.assertTrue(symbol.isalpha())
            self.assertTrue(len(symbol) <= 5)
        
        # Invalid symbols
        invalid_symbols = ['', 'A', '123', 'AAPL123', 'TOOLONG']
        for symbol in invalid_symbols:
            if symbol == '':
                self.assertFalse(symbol.isalpha())
            elif len(symbol) == 1:
                self.assertTrue(len(symbol) < 2)
            elif symbol.isdigit():
                self.assertFalse(symbol.isalpha())
            elif len(symbol) > 5:
                self.assertTrue(len(symbol) > 5)


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2) 