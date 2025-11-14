"""
MarketHacker Dashboard
======================

A comprehensive Streamlit dashboard for stock prediction with multiple ML models,
detailed recommendations, and options analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import MarketHacker modules
try:
    from data_ingestion.fetch_market_data import fetch_stock_data, get_stock_info
    from preprocessing.data_cleaner import clean_stock_data
    from preprocessing.feature_engineering import create_features
    from models.model_trainer import train_all_models
    from models.prediction_engine import PredictionEngine, get_prediction_report
    from models.model_evaluator import ModelEvaluator, compare_model_performance
    from strategies.recommendation_engine import get_detailed_analysis
    from strategies.options_analyzer import recommend_options_strategy, analyze_options
except ImportError as e:
    st.error(f"Error importing modules: {e}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="MarketHacker - AI Stock Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
        padding: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 1rem;
        margin: 0.5rem 0;
    }
    .buy-signal {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        color: white;
        font-size: 1.5rem;
        font-weight: bold;
        text-align: center;
    }
    .sell-signal {
        background: linear-gradient(135deg, #ee0979 0%, #ff6a00 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        color: white;
        font-size: 1.5rem;
        font-weight: bold;
        text-align: center;
    }
    .hold-signal {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        color: white;
        font-size: 1.5rem;
        font-weight: bold;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


def load_and_prepare_data(symbol, days_back=365):
    """Load and prepare stock data."""
    with st.spinner(f'Fetching data for {symbol}...'):
        # Download stock data
        df = fetch_stock_data(symbol, days_back=days_back)

        if df is None or len(df) == 0:
            st.error(f"Could not fetch data for {symbol}")
            return None, None

        # Clean data
        df_clean = clean_stock_data(df)

        # Create features
        df_features = create_features(df_clean)

        return df, df_features


def train_models_cached(df_features):
    """Train all models (cached)."""
    with st.spinner('Training ML models... This may take a minute.'):
        models_dict = train_all_models(df_features, target_column='return_5d')
        return models_dict


def create_price_chart(df, symbol):
    """Create interactive price chart."""
    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Price'
    ))

    # Add volume bars
    fig.add_trace(go.Bar(
        x=df.index,
        y=df['Volume'],
        name='Volume',
        yaxis='y2',
        opacity=0.3
    ))

    fig.update_layout(
        title=f'{symbol} Stock Price & Volume',
        yaxis_title='Price ($)',
        yaxis2=dict(title='Volume', overlaying='y', side='right'),
        xaxis_title='Date',
        height=500,
        hovermode='x unified'
    )

    return fig


def create_model_comparison_radar(model_comparison):
    """Create radar chart comparing model performance."""
    summary = model_comparison.get('summary', pd.DataFrame())

    if summary.empty:
        return None

    # Select top 6 models
    top_models = summary.head(6)

    categories = ['R² Score', 'Direction Accuracy', 'Overall Score']

    fig = go.Figure()

    for _, row in top_models.iterrows():
        fig.add_trace(go.Scatterpolar(
            r=[
                row['R² Score'] * 100,
                row['Direction Accuracy'],
                row['Overall Score']
            ],
            theta=categories,
            fill='toself',
            name=row['Model']
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )
        ),
        showlegend=True,
        title='Model Performance Comparison',
        height=500
    )

    return fig


def create_prediction_comparison_chart(predictions):
    """Create bar chart comparing model predictions."""
    pred_df = pd.DataFrame(list(predictions.items()), columns=['Model', 'Prediction'])
    pred_df = pred_df.sort_values('Prediction', ascending=False)

    colors = ['green' if x > 0 else 'red' for x in pred_df['Prediction']]

    fig = go.Figure(data=[
        go.Bar(
            x=pred_df['Model'],
            y=pred_df['Prediction'],
            marker_color=colors,
            text=pred_df['Prediction'].round(2),
            textposition='auto',
        )
    ])

    fig.update_layout(
        title='Model Predictions Comparison',
        xaxis_title='Model',
        yaxis_title='Predicted Return (%)',
        height=400,
        showlegend=False
    )

    return fig


def create_options_heatmap(options_df):
    """Create heatmap for options analysis."""
    # Pivot for calls
    calls = options_df[options_df['Type'] == 'CALL'].pivot_table(
        values='Probability of Profit',
        index='Strike',
        aggfunc='first'
    )

    # Pivot for puts
    puts = options_df[options_df['Type'] == 'PUT'].pivot_table(
        values='Probability of Profit',
        index='Strike',
        aggfunc='first'
    )

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Call Options', 'Put Options')
    )

    fig.add_trace(
        go.Heatmap(
            z=[calls.values],
            x=calls.index,
            colorscale='Greens',
            showscale=True
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Heatmap(
            z=[puts.values],
            x=puts.index,
            colorscale='Reds',
            showscale=True
        ),
        row=1, col=2
    )

    fig.update_layout(
        title='Options Probability of Profit by Strike Price',
        height=300
    )

    return fig


def main():
    """Main dashboard function."""

    # Header
    st.markdown('<h1 class="main-header">📈 MarketHacker - AI Stock Prediction</h1>', unsafe_allow_html=True)
    st.markdown('---')

    # Sidebar
    with st.sidebar:
        st.header('⚙️ Configuration')

        symbol = st.text_input('Stock Symbol', 'AAPL').upper()
        days_back = st.slider('Historical Data (days)', 90, 730, 365)

        st.markdown('---')

        analyze_button = st.button('🚀 Run Analysis', type='primary', use_container_width=True)

        st.markdown('---')
        st.markdown('### About')
        st.info('''
        **MarketHacker** uses 6+ machine learning models to predict stock prices and generate trading recommendations.

        ✅ Multiple ML Models
        ✅ Real-time Data
        ✅ Options Analysis
        ✅ Risk Assessment
        ''')

    # Main content
    if analyze_button:
        # Load data
        df_raw, df_features = load_and_prepare_data(symbol, days_back)

        if df_raw is None:
            return

        current_price = df_raw['Close'].iloc[-1]

        # Display current price
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric('Current Price', f'${current_price:.2f}')

        with col2:
            day_change = ((df_raw['Close'].iloc[-1] - df_raw['Close'].iloc[-2]) / df_raw['Close'].iloc[-2]) * 100
            st.metric('1-Day Change', f'{day_change:.2f}%', delta=f'{day_change:.2f}%')

        with col3:
            week_change = ((df_raw['Close'].iloc[-1] - df_raw['Close'].iloc[-6]) / df_raw['Close'].iloc[-6]) * 100
            st.metric('5-Day Change', f'{week_change:.2f}%', delta=f'{week_change:.2f}%')

        with col4:
            volume = df_raw['Volume'].iloc[-1]
            st.metric('Volume', f'{volume:,.0f}')

        st.markdown('---')

        # Price chart
        st.subheader('📊 Price Chart')
        price_chart = create_price_chart(df_raw.tail(90), symbol)
        st.plotly_chart(price_chart, use_container_width=True)

        st.markdown('---')

        # Train models
        st.subheader('🤖 Training ML Models')
        models_dict = train_models_cached(df_features)

        # Get model comparison
        model_comparison = compare_model_performance(models_dict)

        # Get predictions
        pred_engine = PredictionEngine(models_dict)
        pred_report = pred_engine.generate_prediction_report(df_features)

        predictions = pred_report['predictions']
        ensemble_pred = pred_report['ensemble_prediction']
        confidence_interval = pred_report['confidence_interval']

        # Display ensemble prediction
        st.subheader('🎯 Ensemble Prediction')

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                'Predicted 5-Day Return',
                f'{ensemble_pred:.2f}%',
                delta=f'{ensemble_pred:.2f}%'
            )

        with col2:
            st.metric(
                'Lower Bound (95% CI)',
                f'{confidence_interval[0]:.2f}%'
            )

        with col3:
            st.metric(
                'Upper Bound (95% CI)',
                f'{confidence_interval[1]:.2f}%'
            )

        # Model comparison
        st.markdown('---')
        st.subheader('📈 Model Performance Comparison')

        col1, col2 = st.columns(2)

        with col1:
            # Radar chart
            radar_chart = create_model_comparison_radar(model_comparison)
            if radar_chart:
                st.plotly_chart(radar_chart, use_container_width=True)

        with col2:
            # Prediction comparison
            pred_chart = create_prediction_comparison_chart(predictions)
            st.plotly_chart(pred_chart, use_container_width=True)

        # Model performance table
        st.dataframe(
            model_comparison['summary'].style.background_gradient(cmap='RdYlGn', subset=['Overall Score']),
            use_container_width=True
        )

        # Recommendation
        st.markdown('---')
        st.subheader('💡 Trading Recommendation')

        # Calculate volatility
        volatility = df_raw['Close'].pct_change().std() * np.sqrt(252) * 100

        # Get detailed analysis
        detailed_analysis = get_detailed_analysis(
            df_raw, predictions, model_comparison, current_price
        )

        recommendation = detailed_analysis['recommendation']

        # Display recommendation signal
        action = recommendation['action']
        if 'BUY' in action:
            st.markdown(f'<div class="buy-signal">🟢 {action}</div>', unsafe_allow_html=True)
        elif 'SELL' in action:
            st.markdown(f'<div class="sell-signal">🔴 {action}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="hold-signal">🟡 {action}</div>', unsafe_allow_html=True)

        # Recommendation details
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric('Confidence Level', recommendation['confidence'])
            st.metric('Confidence Score', f"{recommendation['confidence_score']:.1f}%")

        with col2:
            st.metric('Target Price', f"${recommendation['target_price']:.2f}")
            st.metric('Upside Potential', f"{recommendation['upside_potential']:.2f}%")

        with col3:
            st.metric('Stop Loss', f"${recommendation['stop_loss']:.2f}")
            st.metric('Downside Risk', f"{recommendation['downside_risk']:.2f}%")

        # Explanation
        st.markdown('#### 📝 Detailed Explanation')

        explanation = recommendation['explanation']

        st.markdown(f"**Summary:** {explanation['summary']}")
        st.markdown(f"**Prediction Analysis:** {explanation['prediction_analysis']}")
        st.markdown(f"**Market Context:** {explanation['market_context']}")
        st.markdown(f"**Risk Assessment:** {explanation['risk_assessment']}")

        if explanation['reasoning']:
            st.markdown('**Key Reasons:**')
            for reason in explanation['reasoning']:
                st.markdown(f'- {reason}')

        # Warnings
        if recommendation['warnings']:
            st.warning('**⚠️ Risk Warnings:**')
            for warning in recommendation['warnings']:
                st.markdown(f'- {warning}')

        # Model predictions detail
        st.markdown('---')
        st.subheader('🔍 Individual Model Predictions')

        model_predictions = detailed_analysis['model_predictions']
        st.dataframe(
            model_predictions.style.background_gradient(cmap='RdYlGn', subset=['Prediction (%)']),
            use_container_width=True
        )

        # Options Analysis
        st.markdown('---')
        st.subheader('📊 Options Analysis')

        options_recommendations = recommend_options_strategy(
            current_price, ensemble_pred, volatility, days_to_expiration=30
        )

        # Best options
        col1, col2 = st.columns(2)

        with col1:
            st.markdown('#### Best Call Option')
            best_call = options_recommendations.get('best_call')
            if best_call:
                st.metric('Strike Price', f"${best_call['Strike']:.2f}")
                st.metric('Premium', f"${best_call['Premium']:.2f}")
                st.metric('Probability of Profit', f"{best_call['Probability of Profit']:.1f}%")
                st.metric('Expected Value', f"${best_call['Expected Value']:.2f}")
            else:
                st.info('No profitable call options identified')

        with col2:
            st.markdown('#### Best Put Option')
            best_put = options_recommendations.get('best_put')
            if best_put:
                st.metric('Strike Price', f"${best_put['Strike']:.2f}")
                st.metric('Premium', f"${best_put['Premium']:.2f}")
                st.metric('Probability of Profit', f"{best_put['Probability of Profit']:.1f}%")
                st.metric('Expected Value', f"${best_put['Expected Value']:.2f}")
            else:
                st.info('No profitable put options identified')

        # Strategy recommendations
        st.markdown('#### 🎯 Recommended Options Strategies')

        strategies = options_recommendations.get('strategies', [])

        for strategy in strategies[:3]:  # Show top 3
            with st.expander(f"**{strategy['strategy']}** - Risk: {strategy['risk_level']}"):
                st.markdown(f"**Description:** {strategy['description']}")
                st.markdown(f"**Rationale:** {strategy['rationale']}")
                st.markdown(f"**Max Loss:** {strategy['max_loss']}")
                st.markdown(f"**Max Gain:** {strategy['max_gain']}")

        # Options table
        options_df = analyze_options(current_price, ensemble_pred, volatility)

        st.markdown('#### All Options')
        st.dataframe(
            options_df.style.background_gradient(cmap='RdYlGn', subset=['Probability of Profit']),
            use_container_width=True
        )

        # Risk Disclaimer
        st.markdown('---')
        st.error('''
        **⚠️ DISCLAIMER:** This tool is for educational purposes only.
        Past performance does not guarantee future results. Always do your own research
        and consider consulting with a financial advisor before making investment decisions.
        Never invest more than you can afford to lose.
        ''')

    else:
        # Welcome screen
        st.markdown('''
        ## Welcome to MarketHacker! 🚀

        ### Features:
        - **6+ Machine Learning Models** - Random Forest, XGBoost, LightGBM, Gradient Boosting, Ridge, Lasso
        - **Ensemble Predictions** - Combine multiple models for robust predictions
        - **Model Comparison** - Compare performance across all models
        - **Detailed Recommendations** - Get buy/sell/hold recommendations with explanations
        - **Options Analysis** - Analyze options strategies with probability calculations
        - **Risk Assessment** - Comprehensive risk metrics and warnings
        - **Real-time Data** - Live stock market data

        ### How to Use:
        1. Enter a stock symbol in the sidebar (e.g., AAPL, TSLA, NVDA)
        2. Adjust the historical data period if needed
        3. Click "Run Analysis" to start
        4. Review predictions, recommendations, and options strategies

        **Get started by entering a stock symbol and clicking "Run Analysis"!**
        ''')


if __name__ == '__main__':
    main()
