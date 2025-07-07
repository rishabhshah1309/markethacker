"""
Command-line interface for MarketHacker.

This module provides a CLI for running various MarketHacker operations.
"""

import argparse
import sys
import os
from datetime import datetime, timedelta
import logging

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_ingestion.fetch_market_data import download_stock_data, download_market_data_batch
from data_ingestion.fetch_sentiment import fetch_social_sentiment
from preprocessing.data_cleaner import clean_stock_data
from preprocessing.feature_engineering import create_features

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="MarketHacker - Intelligent Stock Prediction & Options Strategy Platform",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download stock data for AAPL
  markethacker download --symbol AAPL --start-date 2023-01-01 --end-date 2023-12-31
  
  # Download data for multiple stocks
  markethacker download --symbols AAPL MSFT GOOGL --start-date 2023-01-01 --end-date 2023-12-31
  
  # Analyze sentiment for stocks
  markethacker sentiment --symbols AAPL MSFT --days-back 7
  
  # Create features for modeling
  markethacker features --symbol AAPL --start-date 2023-01-01 --end-date 2023-12-31 --output features.csv
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Download command
    download_parser = subparsers.add_parser('download', help='Download stock data')
    download_parser.add_argument('--symbol', help='Single stock symbol')
    download_parser.add_argument('--symbols', nargs='+', help='Multiple stock symbols')
    download_parser.add_argument('--start-date', required=True, help='Start date (YYYY-MM-DD)')
    download_parser.add_argument('--end-date', required=True, help='End date (YYYY-MM-DD)')
    download_parser.add_argument('--output', help='Output file path')
    download_parser.add_argument('--interval', default='1d', help='Data interval (1d, 1h, 5m)')
    
    # Sentiment command
    sentiment_parser = subparsers.add_parser('sentiment', help='Analyze sentiment')
    sentiment_parser.add_argument('--symbols', nargs='+', required=True, help='Stock symbols to analyze')
    sentiment_parser.add_argument('--days-back', type=int, default=7, help='Days to look back')
    sentiment_parser.add_argument('--output', help='Output file path')
    
    # Features command
    features_parser = subparsers.add_parser('features', help='Create features for modeling')
    features_parser.add_argument('--symbol', required=True, help='Stock symbol')
    features_parser.add_argument('--start-date', required=True, help='Start date (YYYY-MM-DD)')
    features_parser.add_argument('--end-date', required=True, help='End date (YYYY-MM-DD)')
    features_parser.add_argument('--output', help='Output file path')
    
    # Dashboard command
    dashboard_parser = subparsers.add_parser('dashboard', help='Launch Streamlit dashboard')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'download':
            handle_download(args)
        elif args.command == 'sentiment':
            handle_sentiment(args)
        elif args.command == 'features':
            handle_features(args)
        elif args.command == 'dashboard':
            handle_dashboard(args)
        else:
            parser.print_help()
            
    except Exception as e:
        logger.error(f"Error executing command '{args.command}': {e}")
        sys.exit(1)


def handle_download(args):
    """Handle download command."""
    logger.info("Starting data download...")
    
    symbols = []
    if args.symbol:
        symbols = [args.symbol]
    elif args.symbols:
        symbols = args.symbols
    else:
        logger.error("Please specify either --symbol or --symbols")
        return
    
    if len(symbols) == 1:
        # Single symbol download
        df = download_stock_data(
            symbol=symbols[0],
            start_date=args.start_date,
            end_date=args.end_date,
            interval=args.interval
        )
        
        if not df.empty:
            logger.info(f"Downloaded {len(df)} records for {symbols[0]}")
            
            if args.output:
                df.to_csv(args.output)
                logger.info(f"Data saved to {args.output}")
            else:
                print(df.head())
        else:
            logger.warning(f"No data downloaded for {symbols[0]}")
    
    else:
        # Multiple symbols download
        data_dict = download_market_data_batch(
            symbols=symbols,
            start_date=args.start_date,
            end_date=args.end_date,
            interval=args.interval
        )
        
        logger.info(f"Downloaded data for {len(data_dict)} symbols")
        
        if args.output:
            # Save each symbol to separate file
            base_name = args.output.replace('.csv', '')
            for symbol, df in data_dict.items():
                filename = f"{base_name}_{symbol}.csv"
                df.to_csv(filename)
                logger.info(f"Data for {symbol} saved to {filename}")
        else:
            # Print summary
            for symbol, df in data_dict.items():
                print(f"{symbol}: {len(df)} records")


def handle_sentiment(args):
    """Handle sentiment command."""
    logger.info("Starting sentiment analysis...")
    
    sentiment_df = fetch_social_sentiment(
        symbols=args.symbols,
        days_back=args.days_back
    )
    
    if not sentiment_df.empty:
        logger.info(f"Analyzed sentiment for {len(sentiment_df)} posts")
        
        if args.output:
            sentiment_df.to_csv(args.output, index=False)
            logger.info(f"Sentiment data saved to {args.output}")
        else:
            print(sentiment_df.head())
            
        # Print summary
        summary = sentiment_df.groupby('symbol').agg({
            'compound': ['mean', 'count'],
            'sentiment_label': lambda x: x.value_counts().index[0] if len(x) > 0 else 'neutral'
        }).round(3)
        
        print("\nSentiment Summary:")
        print(summary)
    else:
        logger.warning("No sentiment data available")


def handle_features(args):
    """Handle features command."""
    logger.info("Creating features...")
    
    # Download and clean data
    df = download_stock_data(
        symbol=args.symbol,
        start_date=args.start_date,
        end_date=args.end_date
    )
    
    if df.empty:
        logger.error(f"No data available for {args.symbol}")
        return
    
    # Clean data
    df = clean_stock_data(df)
    
    # Create features
    df = create_features(df)
    
    logger.info(f"Created {len(df.columns)} features for {args.symbol}")
    
    if args.output:
        df.to_csv(args.output)
        logger.info(f"Features saved to {args.output}")
    else:
        print(f"Features shape: {df.shape}")
        print("\nFeature columns:")
        for col in df.columns:
            print(f"  - {col}")


def handle_dashboard(args):
    """Handle dashboard command."""
    logger.info("Launching Streamlit dashboard...")
    
    # Get the path to the dashboard app
    dashboard_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'dashboard',
        'app.py'
    )
    
    if not os.path.exists(dashboard_path):
        logger.error(f"Dashboard not found at {dashboard_path}")
        return
    
    # Launch Streamlit
    import subprocess
    try:
        subprocess.run(['streamlit', 'run', dashboard_path], check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Error launching dashboard: {e}")
    except FileNotFoundError:
        logger.error("Streamlit not found. Please install it with: pip install streamlit")


if __name__ == "__main__":
    main() 