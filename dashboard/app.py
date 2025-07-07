"""
MarketHacker Dashboard

A Streamlit-based dashboard for stock prediction and options strategy analysis.
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
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import MarketHacker modules
try:
    from data_ingestion.fetch_market_data import download_stock_data, get_stock_info
    from data_ingestion.fetch_sentiment import fetch_social_sentiment, analyze_sentiment
    from preprocessing.data_cleaner import clean_stock_data
    from preprocessing.feature_engineering import create_features
    from preprocessing.data_transformer import prepare_features_for_model
except ImportError as e:
    st.error(f"Error importing MarketHacker modules: {e}")
    st.info("Please ensure all dependencies are installed: pip install -r requirements.txt")

# Page configuration
st.set_page_config(
    page_title="MarketHacker - AI Stock Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    /* Set deep gold background for entire application */
    .stApp {
        background: linear-gradient(135deg, #DAA520 0%, #B8860B 50%, #DAA520 100%);
    }
    
    /* Main content background */
    .main .block-container {
        background-color: rgba(218, 165, 32, 0.9) !important;
    }
    
    /* Sidebar background */
    .css-1d391kg {
        background-color: #DAA520 !important;
    }
    
    .sidebar .sidebar-content {
        background-color: #DAA520;
    }
    
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #00C805, #00D4AA);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        padding: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 0.5rem 0;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    .prediction-positive {
        color: #00ff88;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .prediction-negative {
        color: #ff4444;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .prediction-neutral {
        color: #ffaa00;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .stSelectbox > div > div {
        background-color: white;
    }
    .stButton > button {
        background: linear-gradient(90deg, #88c2eb, #ff7f0e);
        color: white;
        border: none;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #1565c0, #e65100);
    }
    .model-metrics {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    
    /* Fix text input readability */
    .stTextInput > div > div > input {
        background-color: white !important;
        color: black !important;
        border: 2px solid #e0e0e0 !important;
    }
    .stTextInput > div > div > input:focus {
        border-color: #88c2eb !important;
        box-shadow: 0 0 0 0.2rem rgba(31, 119, 180, 0.25) !important;
    }
    
    /* Fix selectbox readability */
    .stSelectbox > div > div > div {
        background-color: white !important;
        color: black !important;
        border: 2px solid #e0e0e0 !important;
    }
    .stSelectbox > div > div > div:hover {
        border-color: #88c2eb !important;
    }
    
    /* Fix slider readability */
    .stSlider > div > div > div > div {
        background-color: #f0f0f0 !important;
    }
    .stSlider > div > div > div > div > div {
        background-color: #88c2eb !important;
    }
    
    /* Fix checkbox readability */
    .stCheckbox > div > div {
        background-color: white !important;
        color: black !important;
    }
    
    /* Fix date input readability */
    .stDateInput > div > div > input {
        background-color: white !important;
        color: black !important;
        border: 2px solid #e0e0e0 !important;
    }
    
    /* Best options sidebar styling */
    .best-options-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #00ff88;
    }
    .best-options-card.sell {
        border-left-color: #ff4444;
    }
    .best-options-card.hold {
        border-left-color: #ffaa00;
    }
    .option-type {
        font-weight: bold;
        font-size: 1.1rem;
        margin-bottom: 0.5rem;
    }
    .option-details {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    
    /* General text color fixes - only for basic text elements */
    .stMarkdown, .stText, .stWrite {
        color: #333333 !important;
    }
    
    /* AI Analysis Section - Dark Theme */
    .ai-analysis-container {
        background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
        color: white;
        padding: 2rem;
        border-radius: 1rem;
        margin: 1rem 0;
        border: 1px solid #444;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    .ai-insight-card {
        background: linear-gradient(135deg, #2a2a2a 0%, #3a3a3a 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 0.8rem;
        margin: 1rem 0;
        border-left: 4px solid #00ff88;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
    }
    
    .ai-insight-card.warning {
        border-left-color: #ffaa00;
    }
    
    .ai-insight-card.danger {
        border-left-color: #ff4444;
    }
    
    .ai-insight-card.info {
        border-left-color: #88c2eb;
    }
    
    .ai-strategy-card {
        background: linear-gradient(135deg, #2a2a2a 0%, #3a3a3a 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 0.8rem;
        margin: 1rem 0;
        border: 1px solid #444;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
    }
    
    .ai-metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        margin: 0.5rem 0;
        color: #00ff88;
    }
    
    .ai-metric-label {
        font-size: 0.9rem;
        opacity: 0.8;
        color: #ccc;
    }
    
    .ai-section-title {
        color: #00ff88;
        font-size: 1.3rem;
        font-weight: bold;
        margin-bottom: 1rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .ai-explanation {
        background: linear-gradient(135deg, #2a2a2a 0%, #3a3a3a 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 0.8rem;
        margin: 1rem 0;
        border-left: 4px solid #88c2eb;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
    }
    
    .ai-explanation.success {
        border-left-color: #00ff88;
    }
    
    .ai-explanation.error {
        border-left-color: #ff4444;
    }
    
    .ai-explanation.warning {
        border-left-color: #ffaa00;
    }
    
    .ai-explanation.info {
        border-left-color: #88c2eb;
    }
    
    /* Model Comparison Section Styling */
    .model-comparison-container {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        color: #333;
        padding: 2rem;
        border-radius: 1rem;
        margin: 1rem 0;
        border: 1px solid #dee2e6;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
    }
    
    .model-metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        color: #333;
        padding: 1.5rem;
        border-radius: 0.8rem;
        margin: 1rem 0;
        border: 1px solid #dee2e6;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }
    
    .model-metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        margin: 0.5rem 0;
        color: #88c2eb;
    }
    
    .model-metric-label {
        font-size: 0.9rem;
        opacity: 0.8;
        color: #666;
        font-weight: 500;
    }
    
    .model-section-title {
        color: #88c2eb;
        font-size: 1.3rem;
        font-weight: bold;
        margin-bottom: 1rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .model-ranking-table {
        background: white;
        border-radius: 0.5rem;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }
    
    .model-insight-card {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        color: #1565c0;
        padding: 1.5rem;
        border-radius: 0.8rem;
        margin: 1rem 0;
        border-left: 4px solid #2196f3;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }
    
    .model-strategy-card {
        background: linear-gradient(135deg, #f3e5f5 0%, #e1bee7 100%);
        color: #7b1fa2;
        padding: 1.5rem;
        border-radius: 0.8rem;
        margin: 1rem 0;
        border-left: 4px solid #9c27b0;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }
    
    .model-insight-card {
        background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        text-align: center;
        color: white;
        box-shadow: 0 8px 32px rgba(0,0,0,0.2);
        border: 2px solid #3498db;
    }
    
    .model-insight-title {
        font-size: 1rem;
        font-weight: bold;
        margin-bottom: 12px;
        opacity: 1;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
    }
    
    .model-insight-value {
        font-size: 1.6rem;
        font-weight: bold;
        margin-bottom: 8px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
    }
    
    .model-insight-desc {
        font-size: 0.9rem;
        opacity: 1;
        line-height: 1.4;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
    }
    
    .strategy-recommendation {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 25px;
        margin: 20px 0;
        color: white;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    .strategy-title {
        font-size: 1.3rem;
        font-weight: bold;
        margin-bottom: 10px;
    }
    
    .strategy-description {
        font-size: 1rem;
        opacity: 0.9;
        line-height: 1.5;
    }
    
    .decision-matrix-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        text-align: center;
        color: white;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    .decision-title {
        font-size: 0.9rem;
        font-weight: bold;
        margin-bottom: 10px;
        opacity: 0.9;
    }
    
    .decision-value {
        font-size: 1.3rem;
        font-weight: bold;
        margin-bottom: 5px;
    }
    
    .decision-desc {
        font-size: 0.8rem;
        opacity: 0.8;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'stock_data' not in st.session_state:
    st.session_state.stock_data = {}
if 'predictions' not in st.session_state:
    st.session_state.predictions = {}
if 'model_performance' not in st.session_state:
    st.session_state.model_performance = {}
if 'best_options' not in st.session_state:
    st.session_state.best_options = {}

def get_available_models():
    """Get available regression models."""
    return {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=0.1),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
        'SVR': SVR(kernel='rbf')
    }

def train_model_and_predict(df, model_name, target_column='return_5d', test_size=0.2):
    """Train model and make predictions."""
    try:
        # Prepare features
        feature_columns = [col for col in df.columns if col not in 
                         ['Symbol', 'return_1d', 'return_5d', 'return_10d', 'return_20d',
                          'target_1d_up', 'target_5d_up', 'target_10d_up', 'target_20d_up',
                          'target_1d_up_1pct', 'target_5d_up_1pct', 'target_10d_up_1pct', 'target_20d_up_1pct',
                          'target_1d_up_3pct', 'target_5d_up_3pct', 'target_10d_up_3pct', 'target_20d_up_3pct']]
        
        # Remove rows with NaN in target
        df_clean = df.dropna(subset=[target_column])
        
        if len(df_clean) < 50:
            return None, None, None
        
        X = df_clean[feature_columns].fillna(0)
        y = df_clean[target_column]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, shuffle=False)
        
        # Get model
        models = get_available_models()
        model = models[model_name]
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        
        # Calculate success rate (predictions in correct direction)
        success_rate = np.mean((y_pred > 0) == (y_test > 0))
        
        return y_pred, {
            'R²': r2,
            'MSE': mse,
            'Success Rate': success_rate,
            'Test Size': len(y_test)
        }, model
        
    except Exception as e:
        st.error(f"Error training model: {e}")
        return None, None, None

def calculate_position_recommendation(df, prediction, confidence_threshold=0.6):
    """Calculate position recommendation based on prediction and technical indicators."""
    try:
        if df.empty or prediction is None:
            return "HOLD", 0.5, "Insufficient data"
        
        current_price = df['Close'].iloc[-1]
        
        # Technical indicators
        rsi = df['rsi'].iloc[-1] if 'rsi' in df.columns else 50
        macd = df['macd'].iloc[-1] if 'macd' in df.columns else 0
        bb_position = df['bb_position'].iloc[-1] if 'bb_position' in df.columns else 0.5
        
        # Sentiment (if available)
        sentiment_score = df['sentiment_mean'].iloc[-1] if 'sentiment_mean' in df.columns else 0
        
        # Calculate recommendation score
        score = 0
        reasons = []
        
        # Prediction factor (40% weight)
        if prediction > 0.02:  # 2% predicted gain
            score += 0.4
            reasons.append("Strong positive prediction")
        elif prediction > 0:
            score += 0.2
            reasons.append("Moderate positive prediction")
        elif prediction < -0.02:
            score -= 0.4
            reasons.append("Strong negative prediction")
        elif prediction < 0:
            score -= 0.2
            reasons.append("Moderate negative prediction")
        
        # RSI factor (20% weight)
        if rsi < 30:
            score += 0.2
            reasons.append("Oversold (RSI < 30)")
        elif rsi > 70:
            score -= 0.2
            reasons.append("Overbought (RSI > 70)")
        
        # MACD factor (20% weight)
        if macd > 0:
            score += 0.2
            reasons.append("Positive MACD")
        else:
            score -= 0.2
            reasons.append("Negative MACD")
        
        # Bollinger Bands factor (10% weight)
        if bb_position < 0.2:
            score += 0.1
            reasons.append("Near lower Bollinger Band")
        elif bb_position > 0.8:
            score -= 0.1
            reasons.append("Near upper Bollinger Band")
        
        # Sentiment factor (10% weight)
        if sentiment_score > 0.1:
            score += 0.1
            reasons.append("Positive sentiment")
        elif sentiment_score < -0.1:
            score -= 0.1
            reasons.append("Negative sentiment")
        
        # Determine recommendation
        if score > 0.3:
            recommendation = "BUY"
            confidence = min(0.95, 0.5 + abs(score))
        elif score < -0.3:
            recommendation = "SELL"
            confidence = min(0.95, 0.5 + abs(score))
        else:
            recommendation = "HOLD"
            confidence = 0.5
        
        return recommendation, confidence, "; ".join(reasons)
        
    except Exception as e:
        return "HOLD", 0.5, f"Error: {e}"

def generate_options_data(current_price, prediction, volatility, days_to_expiry=30):
    """Generate sample options data based on current price and prediction."""
    try:
        # Calculate predicted price range
        predicted_price = current_price * (1 + prediction)
        price_range = current_price * volatility * np.sqrt(days_to_expiry / 365)
        
        # Generate strike prices around current price
        strikes = np.arange(
            max(current_price * 0.8, current_price - price_range * 2),
            min(current_price * 1.2, current_price + price_range * 2),
            current_price * 0.05
        )
        
        options_data = []
        
        for strike in strikes:
            # Calculate option prices using simplified Black-Scholes approximation
            time_to_expiry = days_to_expiry / 365
            moneyness = current_price / strike
            
            # Call options
            if moneyness > 0.9:  # In-the-money or near-the-money calls
                call_price = max(0.01, current_price - strike + 0.5 * volatility * np.sqrt(time_to_expiry))
                call_delta = min(0.95, max(0.05, 0.5 + (moneyness - 1) * 2))
                call_probability = max(0.1, min(0.9, call_delta))
                
                # Adjust probability based on prediction
                if prediction > 0:
                    call_probability = min(0.95, call_probability + abs(prediction) * 2)
                else:
                    call_probability = max(0.05, call_probability - abs(prediction) * 2)
                
                options_data.append({
                    'type': 'CALL',
                    'strike': strike,
                    'price': call_price,
                    'delta': call_delta,
                    'probability': call_probability,
                    'expiry_days': days_to_expiry,
                    'moneyness': moneyness
                })
            
            # Put options
            if moneyness < 1.1:  # In-the-money or near-the-money puts
                put_price = max(0.01, strike - current_price + 0.5 * volatility * np.sqrt(time_to_expiry))
                put_delta = min(0.95, max(0.05, 0.5 + (1 - moneyness) * 2))
                put_probability = max(0.1, min(0.9, put_delta))
                
                # Adjust probability based on prediction
                if prediction < 0:
                    put_probability = min(0.95, put_probability + abs(prediction) * 2)
                else:
                    put_probability = max(0.05, put_probability - abs(prediction) * 2)
                
                options_data.append({
                    'type': 'PUT',
                    'strike': strike,
                    'price': put_price,
                    'delta': put_delta,
                    'probability': put_probability,
                    'expiry_days': days_to_expiry,
                    'moneyness': moneyness
                })
        
        return pd.DataFrame(options_data)
        
    except Exception as e:
        st.error(f"Error generating options data: {e}")
        return pd.DataFrame()

def calculate_options_recommendations(options_df, current_price, prediction):
    """Calculate options recommendations based on predictions."""
    try:
        if options_df.empty:
            return pd.DataFrame()
        
        recommendations = []
        
        predicted_price = current_price * (1 + prediction)
        
        for _, option in options_df.iterrows():
            # Calculate potential profit/loss
            if option['type'] == 'CALL':
                potential_profit = max(0, predicted_price - option['strike']) - option['price']
                max_loss = option['price']
                breakeven = option['strike'] + option['price']
            else:  # PUT
                potential_profit = max(0, option['strike'] - predicted_price) - option['price']
                max_loss = option['price']
                breakeven = option['strike'] - option['price']
            
            # Calculate risk/reward ratio
            risk_reward = potential_profit / max_loss if max_loss > 0 else 0
            
            # Calculate expected value
            expected_value = (potential_profit * option['probability']) - (max_loss * (1 - option['probability']))
            
            # Determine recommendation
            if expected_value > 0.1 and risk_reward > 2:
                recommendation = "STRONG BUY"
                confidence = min(0.95, option['probability'] + 0.1)
            elif expected_value > 0:
                recommendation = "BUY"
                confidence = option['probability']
            elif expected_value > -0.05:
                recommendation = "HOLD"
                confidence = 0.5
            else:
                recommendation = "SELL"
                confidence = 1 - option['probability']
            
            recommendations.append({
                'type': option['type'],
                'strike': option['strike'],
                'price': option['price'],
                'probability': option['probability'],
                'potential_profit': potential_profit,
                'max_loss': max_loss,
                'risk_reward': risk_reward,
                'expected_value': expected_value,
                'recommendation': recommendation,
                'confidence': confidence,
                'breakeven': breakeven,
                'expiry_days': option['expiry_days']
            })
        
        return pd.DataFrame(recommendations)
        
    except Exception as e:
        st.error(f"Error calculating options recommendations: {e}")
        return pd.DataFrame()

def show_options_analysis(df, prediction, symbol):
    """Display options analysis section."""
    try:
        current_price = df['Close'].iloc[-1]
        volatility = df['price_change'].std() * np.sqrt(252)
        predicted_price = current_price * (1 + prediction)
        
        st.subheader("📊 Options Analysis")
        
        # Options configuration
        col1, col2, col3 = st.columns(3)
        
        with col1:
            expiry_days = st.selectbox(
                "Expiry (Days)",
                options=[7, 14, 30, 45, 60, 90],
                index=2,
                help="Days until option expiration"
            )
        
        with col2:
            min_probability = st.slider(
                "Min Probability",
                min_value=0.1,
                max_value=0.9,
                value=0.3,
                step=0.1,
                help="Minimum probability for recommendations"
            )
        
        with col3:
            min_risk_reward = st.slider(
                "Min Risk/Reward",
                min_value=1.0,
                max_value=5.0,
                value=2.0,
                step=0.5,
                help="Minimum risk/reward ratio"
            )
        
        # Generate options data
        options_df = generate_options_data(current_price, prediction, volatility, expiry_days)
        
        if not options_df.empty:
            # Calculate recommendations
            recommendations_df = calculate_options_recommendations(options_df, current_price, prediction)
            
            if not recommendations_df.empty:
                # Filter recommendations
                filtered_df = recommendations_df[
                    (recommendations_df['probability'] >= min_probability) &
                    (recommendations_df['risk_reward'] >= min_risk_reward)
                ].sort_values(by='expected_value', ascending=False)
                
                # Store best options in session state
                if len(filtered_df) > 0:
                    st.session_state.best_options[symbol] = filtered_df.head(3).to_dict('records')
                
                # Display summary metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "Current Price",
                        f"${current_price:.2f}",
                        f"{prediction*100:+.2f}%"
                    )
                
                with col2:
                    st.metric(
                        "Predicted Price",
                        f"${predicted_price:.2f}",
                        f"in {expiry_days} days"
                    )
                
                with col3:
                    st.metric(
                        "Volatility",
                        f"{volatility*100:.1f}%",
                        "annualized"
                    )
                
                with col4:
                    best_option = filtered_df.iloc[0] if len(filtered_df) > 0 else None
                    if best_option is not None:
                        st.metric(
                            "Best Option",
                            f"{best_option['type']} ${best_option['strike']:.0f}",
                            f"{best_option['recommendation']}"
                        )
                
                # Display top recommendations
                st.subheader("🎯 Top Options Recommendations")
                
                if len(filtered_df) > 0:
                    # Create tabs for calls and puts
                    tab1, tab2 = st.tabs(["📈 Call Options", "📉 Put Options"])
                    
                    with tab1:
                        calls_df = filtered_df[filtered_df['type'] == 'CALL'].head(10)
                        if len(calls_df) > 0:
                            # Format display
                            display_df = calls_df.copy()
                            display_df['Strike'] = pd.Series(display_df['strike']).apply(lambda x: f"${x:.0f}")
                            display_df['Price'] = pd.Series(display_df['price']).apply(lambda x: f"${x:.2f}")
                            display_df['Probability'] = pd.Series(display_df['probability']).apply(lambda x: f"{x:.1%}")
                            display_df['Risk/Reward'] = pd.Series(display_df['risk_reward']).apply(lambda x: f"{x:.1f}")
                            display_df['Expected Value'] = pd.Series(display_df['expected_value']).apply(lambda x: f"${x:.2f}")
                            display_df['Confidence'] = pd.Series(display_df['confidence']).apply(lambda x: f"{x:.1%}")
                            
                            st.dataframe(
                                display_df[['Strike', 'Price', 'Probability', 'Risk/Reward', 'Expected Value', 'recommendation', 'Confidence']],
                                use_container_width=True
                            )
                        else:
                            st.info("No call options meet the criteria.")
                    
                    with tab2:
                        puts_df = filtered_df[filtered_df['type'] == 'PUT'].head(10)
                        if len(puts_df) > 0:
                            # Format display
                            display_df = puts_df.copy()
                            display_df['Strike'] = pd.Series(display_df['strike']).apply(lambda x: f"${x:.0f}")
                            display_df['Price'] = pd.Series(display_df['price']).apply(lambda x: f"${x:.2f}")
                            display_df['Probability'] = pd.Series(display_df['probability']).apply(lambda x: f"{x:.1%}")
                            display_df['Risk/Reward'] = pd.Series(display_df['risk_reward']).apply(lambda x: f"{x:.1f}")
                            display_df['Expected Value'] = pd.Series(display_df['expected_value']).apply(lambda x: f"${x:.2f}")
                            display_df['Confidence'] = pd.Series(display_df['confidence']).apply(lambda x: f"{x:.1%}")
                            
                            st.dataframe(
                                display_df[['Strike', 'Price', 'Probability', 'Risk/Reward', 'Expected Value', 'recommendation', 'Confidence']],
                                use_container_width=True
                            )
                        else:
                            st.info("No put options meet the criteria.")
                    
                    # Enhanced Options Analysis
                    st.subheader("📊 Enhanced Options Analysis & Decision Framework")
                    
                    if len(filtered_df) > 0:
                        # Create enhanced options visualization
                        enhanced_options_fig, recommended_strategy = create_enhanced_options_visualization(
                            filtered_df, current_price, predicted_price, symbol
                        )
                        st.plotly_chart(enhanced_options_fig, use_container_width=True)
                        
                        # Strategy recommendation
                        st.markdown(f"""
                        <div class="strategy-recommendation">
                            <div class="strategy-title">🎯 Recommended Strategy: {recommended_strategy}</div>
                            <div class="strategy-description">
                                Based on current volatility ({volatility*100:.1f}%) and predicted price movement ({prediction*100:+.1f}%), 
                                the optimal options strategy is <strong>{recommended_strategy}</strong>.
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Quick decision matrix
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.markdown(f"""
                            <div class="decision-matrix-card">
                                <div class="decision-title">📈 Market Direction</div>
                                <div class="decision-value">{'Bullish' if prediction > 0 else 'Bearish' if prediction < 0 else 'Neutral'}</div>
                                <div class="decision-desc">Predicted {prediction*100:+.1f}% move</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown(f"""
                            <div class="decision-matrix-card">
                                <div class="decision-title">🌪️ Volatility Level</div>
                                <div class="decision-value">{'High' if volatility > 0.4 else 'Medium' if volatility > 0.2 else 'Low'}</div>
                                <div class="decision-desc">{volatility*100:.1f}% annualized</div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col3:
                            st.markdown(f"""
                            <div class="decision-matrix-card">
                                <div class="decision-title">⚡ Risk Level</div>
                                <div class="decision-value">{'High' if volatility > 0.4 else 'Medium' if volatility > 0.2 else 'Low'}</div>
                                <div class="decision-desc">Based on volatility</div>
                            </div>
                            """, unsafe_allow_html=True)
                
                else:
                    st.warning("No options meet the current criteria. Try adjusting the filters.")
            
            else:
                st.error("Could not calculate options recommendations.")
        
        else:
            st.error("Could not generate options data.")
            
    except Exception as e:
        st.error(f"Error in options analysis: {e}")
        st.info("Options analysis requires valid stock data and predictions.")

def generate_ai_insights(all_models_results, df, symbol, prediction_horizon, recommendation, confidence, reasons):
    """Generate AI-powered insights and explanations."""
    try:
        # Analyze model performance
        best_model = max(all_models_results.keys(), 
                        key=lambda x: all_models_results[x]['metrics']['Success Rate'])
        best_metrics = all_models_results[best_model]['metrics']
        
        # Get current market data
        current_price = df['Close'].iloc[-1]
        price_change_1d = df['Close'].pct_change().iloc[-1] * 100
        price_change_5d = ((df['Close'].iloc[-1] / df['Close'].iloc[-5]) - 1) * 100
        volatility = df['price_change'].std() * np.sqrt(252) * 100
        
        # Technical indicators
        rsi = df['rsi'].iloc[-1] if 'rsi' in df.columns else 50
        macd = df['macd'].iloc[-1] if 'macd' in df.columns else 0
        bb_position = df['bb_position'].iloc[-1] if 'bb_position' in df.columns else 0.5
        
        # Sentiment
        sentiment_score = df['sentiment_mean'].iloc[-1] if 'sentiment_mean' in df.columns else 0
        
        # Generate insights
        insights = []
        
        # Model confidence analysis
        if best_metrics['Success Rate'] > 0.7:
            insights.append("🎯 <strong>High Model Confidence</strong>: The AI models are showing strong predictive accuracy, making this a reliable analysis.")
        elif best_metrics['Success Rate'] > 0.6:
            insights.append("📊 <strong>Moderate Model Confidence</strong>: The models show decent predictive power, but exercise caution.")
        else:
            insights.append("⚠️ <strong>Low Model Confidence</strong>: The models are struggling to predict accurately. Consider waiting for clearer signals.")
        
        # Price trend analysis
        if price_change_1d > 2:
            insights.append("🚀 <strong>Strong Daily Momentum</strong>: The stock is showing strong upward momentum today.")
        elif price_change_1d > 0:
            insights.append("📈 <strong>Positive Daily Movement</strong>: The stock is trending upward today.")
        elif price_change_1d < -2:
            insights.append("📉 <strong>Strong Daily Decline</strong>: The stock is showing significant downward pressure.")
        else:
            insights.append("➡️ <strong>Sideways Movement</strong>: The stock is moving sideways with minimal change.")
        
        # Technical analysis
        if rsi < 30:
            insights.append("🟢 <strong>Oversold Condition</strong>: RSI indicates the stock is oversold, suggesting a potential bounce.")
        elif rsi > 70:
            insights.append("🔴 <strong>Overbought Condition</strong>: RSI indicates the stock is overbought, suggesting potential pullback.")
        else:
            insights.append("⚖️ <strong>Neutral RSI</strong>: The stock is in a balanced technical state.")
        
        if macd > 0:
            insights.append("📈 <strong>Positive MACD</strong>: Momentum indicators suggest upward price movement.")
        else:
            insights.append("📉 <strong>Negative MACD</strong>: Momentum indicators suggest downward price movement.")
        
        # Volatility analysis
        if volatility > 40:
            insights.append("🌪️ <strong>High Volatility</strong>: This stock is highly volatile - expect significant price swings.")
        elif volatility > 25:
            insights.append("📊 <strong>Moderate Volatility</strong>: Standard volatility levels for this stock.")
        else:
            insights.append("🛡️ <strong>Low Volatility</strong>: This stock shows stable, predictable price movements.")
        
        # Sentiment analysis
        if sentiment_score > 0.2:
            insights.append("😊 <strong>Positive Sentiment</strong>: Social media sentiment is bullish on this stock.")
        elif sentiment_score < -0.2:
            insights.append("😞 <strong>Negative Sentiment</strong>: Social media sentiment is bearish on this stock.")
        else:
            insights.append("😐 <strong>Neutral Sentiment</strong>: Social media sentiment is mixed.")
        
        # Recommendation explanation
        if recommendation == "BUY":
            if confidence > 0.8:
                insights.append("💎 <strong>Strong Buy Signal</strong>: Multiple indicators align for a strong buying opportunity.")
            else:
                insights.append("✅ <strong>Buy Signal</strong>: Favorable conditions suggest a buying opportunity.")
        elif recommendation == "SELL":
            if confidence > 0.8:
                insights.append("🚨 <strong>Strong Sell Signal</strong>: Multiple indicators suggest selling or avoiding this stock.")
            else:
                insights.append("⚠️ <strong>Sell Signal</strong>: Unfavorable conditions suggest avoiding or selling.")
        else:
            insights.append("⏸️ <strong>Hold Signal</strong>: Mixed signals suggest waiting for clearer direction.")
        
        # Risk assessment
        if volatility > 40 and abs(price_change_1d) > 5:
            insights.append("🎢 <strong>High Risk</strong>: This stock is experiencing extreme volatility - only for experienced traders.")
        elif volatility > 25:
            insights.append("⚠️ <strong>Moderate Risk</strong>: Standard risk levels - suitable for most investors.")
        else:
            insights.append("🛡️ <strong>Low Risk</strong>: Stable stock suitable for conservative investors.")
        
        # Market timing
        if recommendation == "BUY" and rsi < 40:
            insights.append("⏰ <strong>Good Entry Point</strong>: Technical indicators suggest this is a favorable time to buy.")
        elif recommendation == "SELL" and rsi > 60:
            insights.append("⏰ <strong>Good Exit Point</strong>: Technical indicators suggest this is a favorable time to sell.")
        
        return insights
        
    except Exception as e:
        return [f"Error generating insights: {e}"]

def generate_trading_strategy(recommendation, confidence, all_models_results, df, symbol):
    """Generate specific trading strategy recommendations."""
    try:
        current_price = df['Close'].iloc[-1]
        volatility = df['price_change'].std() * np.sqrt(252) * 100
        
        strategy = {
            'action': recommendation,
            'confidence': confidence,
            'entry_price': current_price,
            'stop_loss': None,
            'take_profit': None,
            'position_size': 'Standard',
            'timeframe': 'Short-term',
            'risk_level': 'Moderate'
        }
        
        # Calculate stop loss and take profit
        if recommendation == "BUY":
            # Conservative stop loss (2% below current price)
            strategy['stop_loss'] = current_price * 0.98
            # Take profit based on volatility
            if volatility > 30:
                strategy['take_profit'] = current_price * 1.05  # 5% for high volatility
            else:
                strategy['take_profit'] = current_price * 1.03  # 3% for low volatility
            
            # Position sizing based on confidence
            if confidence > 0.8:
                strategy['position_size'] = 'Large'
                strategy['risk_level'] = 'Aggressive'
            elif confidence > 0.6:
                strategy['position_size'] = 'Medium'
                strategy['risk_level'] = 'Moderate'
            else:
                strategy['position_size'] = 'Small'
                strategy['risk_level'] = 'Conservative'
                
        elif recommendation == "SELL":
            # For selling, we're looking at downside targets
            strategy['stop_loss'] = current_price * 1.02  # 2% above current price
            if volatility > 30:
                strategy['take_profit'] = current_price * 0.95  # 5% downside for high volatility
            else:
                strategy['take_profit'] = current_price * 0.97  # 3% downside for low volatility
        
        return strategy
        
    except Exception as e:
        return {'error': f"Error generating strategy: {e}"}

def create_enhanced_model_comparison(all_models_results, df, prediction_horizon):
    """Create simplified model comparison with stock price predictions."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    
    # Create simple subplot layout to avoid errors
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Model Performance Radar Chart', 'Stock Price Predictions',
            'Success Rate vs Risk Analysis', 'Combined Predictions Distribution'
        ),
        specs=[
            [{"type": "scatterpolar"}, {"type": "scatter"}],
            [{"type": "scatter"}, {"type": "histogram"}]
        ],
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )
    
    # Update subplot titles for better readability
    fig.update_annotations(
        font=dict(size=14, color='black'),
        bgcolor='rgba(255,255,255,0.9)',
        bordercolor='black',
        borderwidth=1
    )
    
    model_names = list(all_models_results.keys())
    colors = ['#88c2eb', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    # 1. Enhanced Radar Chart with more metrics
    metrics = ['R² Score', 'Success Rate', 'Precision', 'Recall', 'F1-Score', 'Sharpe Ratio']
    
    for i, model_name in enumerate(model_names):
        results = all_models_results[model_name]
        predictions = results['predictions']
        actual_returns = df[f'return_{prediction_horizon}d'].iloc[-len(predictions):] * 100
        
        # Calculate additional metrics
        precision = np.sum((predictions > 0) & (actual_returns > 0)) / max(np.sum(predictions > 0), 1)
        recall = np.sum((predictions > 0) & (actual_returns > 0)) / max(np.sum(actual_returns > 0), 1)
        f1_score = 2 * (precision * recall) / max((precision + recall), 1e-8)
        
        # Calculate Sharpe-like ratio (return vs volatility)
        pred_returns = predictions * 100
        sharpe_ratio = np.mean(pred_returns) / max(np.std(pred_returns), 1e-8)
        sharpe_ratio = min(max(sharpe_ratio, 0), 1)  # Normalize to 0-1
        
        metrics_values = [
            results['metrics']['R²'],
            results['metrics']['Success Rate'],
            precision,
            recall,
            f1_score,
            sharpe_ratio
        ]
        
        fig.add_trace(
            go.Scatterpolar(
                r=metrics_values,
                theta=metrics,
                fill='toself',
                name=model_name,
                line_color=colors[i % len(colors)],
                opacity=0.7,
                line_width=2
            ),
            row=1, col=1
        )
    
    # Update polar chart for better readability
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickfont=dict(size=12, color='black'),
                tickcolor='black',
                linecolor='black',
                gridcolor='rgba(0,0,0,0.3)'
            ),
            angularaxis=dict(
                tickfont=dict(size=11, color='black'),
                tickcolor='black',
                linecolor='black',
                gridcolor='rgba(0,0,0,0.3)'
            ),
            bgcolor='rgba(255,255,255,0.9)'
        ),
        showlegend=False,  # We'll handle legend separately
        title_text="Comprehensive Model Performance Analysis"
    )
    
    # 2. Stock Price Predictions vs Actual
    test_size = len(list(all_models_results.values())[0]['predictions'])
    test_dates = df.index[-test_size:]
    actual_returns = df[f'return_{prediction_horizon}d'].iloc[-test_size:] * 100
    
    # Get current stock prices for the test period
    current_prices = df['Close'].iloc[-test_size:]
    
    # Calculate predicted stock prices for each model
    for i, (model_name, results) in enumerate(all_models_results.items()):
        predictions = results['predictions'] * 100  # Convert to percentage
        
        # Calculate predicted stock prices
        predicted_prices = current_prices * (1 + predictions / 100)
        
        color = colors[i % len(colors)]
        
        # Add predicted stock prices
        fig.add_trace(
            go.Scatter(
                x=test_dates,
                y=predicted_prices,
                mode='lines+markers',
                name=f'{model_name} (Price Prediction)',
                line=dict(color=color, width=2),
                marker=dict(size=4, opacity=0.7),
                hovertemplate='<b>%{x}</b><br>' +
                            f'{model_name} Predicted: $%{{y:.2f}}<br>' +
                            '<extra></extra>'
            ),
            row=1, col=2
        )
    
    # Add actual stock prices
    fig.add_trace(
        go.Scatter(
            x=test_dates,
            y=current_prices,
            mode='lines+markers',
            name='Actual Stock Price',
            line=dict(color='white', width=3),
            marker=dict(size=6, symbol='diamond', color='white'),
            opacity=0.9,
            hovertemplate='<b>%{x}</b><br>' +
                        'Actual Price: $%{y:.2f}<br>' +
                        '<extra></extra>'
        ),
        row=1, col=2
    )
    
    # Add predictions with confidence intervals and enhanced styling
    for i, (model_name, results) in enumerate(all_models_results.items()):
        predictions = results['predictions'] * 100
        
        # Calculate rolling confidence intervals
        window_size = min(10, len(predictions) // 4)
        # Convert numpy array to pandas Series for rolling operations
        predictions_series = pd.Series(predictions)
        rolling_std = predictions_series.rolling(window=window_size, min_periods=1).std()
        upper_bound = predictions + (1.96 * rolling_std)
        lower_bound = predictions - (1.96 * rolling_std)
        
        # Skip confidence intervals for now to avoid color formatting issues
        pass
        
        # Remove duplicate prediction traces - we already have price predictions above
        # This was creating duplicate legend entries
    
    # 3. Enhanced Success Rate vs Risk Analysis
    success_rates = [all_models_results[model]['metrics']['Success Rate'] for model in model_names]
    mse_values = [all_models_results[model]['metrics']['MSE'] for model in model_names]
    
    # Ensure we have valid data for the chart
    if len(success_rates) > 0 and len(mse_values) > 0:
        # Calculate risk-adjusted success rate
        risk_adjusted_success = [sr / max(mse * 10, 0.1) for sr, mse in zip(success_rates, mse_values)]
    else:
        st.warning("No valid data for Success Rate vs Risk Analysis chart")
        risk_adjusted_success = []
    
    if len(risk_adjusted_success) > 0:
        for i, model_name in enumerate(model_names):
            # Size based on risk-adjusted success rate
            marker_size = 15 + (risk_adjusted_success[i] * 25)  # Larger markers
            
            fig.add_trace(
                go.Scatter(
                    x=[mse_values[i]],
                    y=[success_rates[i]],
                    mode='markers+text',
                    name=model_name,
                    text=[model_name],
                    textposition="top center",
                    textfont=dict(size=12, color='black'),  # Black text for better visibility
                    marker=dict(
                        size=marker_size,
                        color=colors[i % len(colors)],
                        symbol='circle',
                        opacity=0.9,  # More opaque
                        line=dict(width=3, color='white')  # Thicker border
                    ),
                    showlegend=False,
                    hovertemplate='<b>%{text}</b><br>' +
                                'Success Rate: %{y:.2%}<br>' +
                                'Risk (MSE): %{x:.4f}<br>' +
                                'Risk-Adjusted: %{marker.size:.1f}<br>' +
                                '<extra></extra>'
                ),
                row=2, col=1
            )
    
    # Add efficiency frontier line
    if len(mse_values) > 1:  # Need at least 2 points for a line
        sorted_data = sorted(zip(mse_values, success_rates))
        frontier_mse = [x[0] for x in sorted_data]
        frontier_success = [x[1] for x in sorted_data]
        
        fig.add_trace(
            go.Scatter(
                x=frontier_mse,
                y=frontier_success,
                mode='lines',
                name='Efficiency Frontier',
                line=dict(color='black', width=3, dash='dash'),  # Black line for better visibility
                showlegend=False,
                hovertemplate='<b>Efficiency Frontier</b><br>' +
                            'Risk (MSE): %{x:.4f}<br>' +
                            'Success Rate: %{y:.2%}<br>' +
                            '<extra></extra>'
            ),
            row=2, col=1
        )
    
    # 4. Model Confidence Intervals Comparison
    confidence_levels = [0.68, 0.95, 0.99]  # 1σ, 2σ, 3σ
    
    for i, (model_name, results) in enumerate(all_models_results.items()):
        predictions = results['predictions'] * 100
        mean_pred = np.mean(predictions)
        std_pred = np.std(predictions)
        
        for j, conf_level in enumerate(confidence_levels):
            z_score = {0.68: 1, 0.95: 1.96, 0.99: 2.58}[conf_level]
            margin = z_score * std_pred
            
            fig.add_trace(
                go.Scatter(
                    x=[model_name],
                    y=[mean_pred + margin],
                    mode='markers',
                    name=f'{conf_level*100}% CI',
                    marker=dict(
                        size=8,
                        color=colors[i % len(colors)],
                        symbol='triangle-up' if j == 0 else 'diamond' if j == 1 else 'star',
                        opacity=0.7
                    ),
                    showlegend=False
                ),
                row=2, col=2
            )
            
            fig.add_trace(
                go.Scatter(
                    x=[model_name],
                    y=[mean_pred - margin],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color=colors[i % len(colors)],
                        symbol='triangle-down' if j == 0 else 'diamond' if j == 1 else 'star',
                        opacity=0.7
                    ),
                    showlegend=False
                ),
                row=2, col=2
            )
    
    # Add mean prediction line
    for i, model_name in enumerate(model_names):
        predictions = all_models_results[model_name]['predictions'] * 100
        mean_pred = np.mean(predictions)
        
        fig.add_trace(
            go.Scatter(
                x=[model_name],
                y=[mean_pred],
                mode='markers',
                name=f'{model_name} Mean',
                marker=dict(
                    size=12,
                    color=colors[i % len(colors)],
                    symbol='circle',
                    line=dict(width=2, color='white')
                ),
                showlegend=False
            ),
            row=2, col=2
        )
    
    # 5. Prediction Distribution Analysis (row 2, col 2) - Combined histogram
    all_predictions = []
    model_labels = []
    
    for model_name, results in all_models_results.items():
        predictions = results['predictions'] * 100
        all_predictions.extend(predictions)
        model_labels.extend([model_name] * len(predictions))
    
    # Create a single histogram with color coding by model
    fig.add_trace(
        go.Histogram(
            x=all_predictions,
            nbinsx=25,
            name='All Predictions',
            marker_color='lightblue',
            opacity=0.7,
            showlegend=False,
            hovertemplate='<b>Return Range</b><br>' +
                        'Count: %{y}<br>' +
                        'Return: %{x:.2f}%<br>' +
                        '<extra></extra>'
        ),
        row=2, col=2
    )
    
    # Add mean line
    mean_prediction = np.mean(all_predictions)
    fig.add_trace(
        go.Scatter(
            x=[mean_prediction, mean_prediction],
            y=[0, max(np.histogram(all_predictions, bins=25)[0]) * 1.1],
            mode='lines',
            line=dict(dash='dash', color='red', width=2),
            name=f'Mean: {mean_prediction:.2f}%',
            showlegend=False
        ),
        row=2, col=2
    )
    
    # Add zero line for reference
    fig.add_trace(
        go.Scatter(
            x=[0, 0],
            y=[0, max(np.histogram(all_predictions, bins=25)[0]) * 1.1],
            mode='lines',
            line=dict(dash='dot', color='black', width=1),
            name='Zero Return',
            showlegend=False
        ),
        row=2, col=2
    )
    
    # 6. Expected Stock Values Over Time (simplified)
    # Get the latest stock price
    latest_price = df['Close'].iloc[-1]
    
    # Calculate future stock values for each model
    future_dates = pd.date_range(start=df.index[-1], periods=prediction_horizon + 1, freq='D')[1:]
    
    for i, (model_name, results) in enumerate(all_models_results.items()):
        # Get the latest prediction for this model
        latest_prediction = results['predictions'][-1] * 100  # Convert to percentage
        
        # Calculate expected stock value progression
        expected_values = []
        for day in range(1, prediction_horizon + 1):
            # Linear interpolation of the prediction over the horizon
            daily_return = latest_prediction / prediction_horizon
            expected_value = latest_price * (1 + (daily_return * day) / 100)
            expected_values.append(expected_value)
        
        color = colors[i % len(colors)]
        
        # Add as overlay to the price predictions chart (row 1, col 2)
        fig.add_trace(
            go.Scatter(
                x=future_dates,
                y=expected_values,
                mode='lines+markers',
                name=f'{model_name} Future Projection',
                line=dict(color=color, width=2, dash='dot'),
                marker=dict(size=4),
                opacity=0.7,
                hovertemplate='<b>%{x}</b><br>' +
                            f'{model_name} Projected: $%{{y:.2f}}<br>' +
                            '<extra></extra>'
            ),
            row=1, col=2
        )
    
    # Add current price line using scatter trace instead of hline
    latest_price = current_prices.iloc[-1]  # Get the latest price
    fig.add_trace(
        go.Scatter(
            x=[test_dates[0], test_dates[-1]],
            y=[latest_price, latest_price],
            mode='lines',
            line=dict(dash='dash', color='black', width=2),
            name=f'Current Price: ${latest_price:.2f}',
            showlegend=False
        ),
        row=1, col=2
    )
    
    # Skip heatmap for now to avoid layout issues
    pass
    
    # Update layout with legend positioned at top, graphs below
    fig.update_layout(
        height=1600,  # Reduced height for 2x2 layout
        title_text="Advanced Model Comparison & Stock Price Predictions Dashboard",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.12,  # Position below the chart area (negative value)
            xanchor="center",
            x=0.5,   # Center horizontally
            font=dict(size=14, color='black'),  # Larger font for better readability
            bgcolor='rgba(255,255,255,0.9)',  # Light background
            bordercolor='rgba(0,0,0,0.3)',
            borderwidth=2,
            itemsizing='constant',  # Consistent legend item sizes
            groupclick="toggleitem"  # Allow individual item toggling
        ),
        font=dict(size=14, color='black'),  # Larger base font and black color
        template="plotly_white",  # Light template for better contrast
        margin=dict(t=140, b=200, l=80, r=80)  # Increased margins for better spacing
    )
    
    # Update axes labels with better styling
    fig.update_xaxes(title_text="Date", row=1, col=2, title_font=dict(size=16, color='black'))
    fig.update_yaxes(title_text="Stock Price ($)", row=1, col=2, title_font=dict(size=16, color='black'))
    fig.update_xaxes(title_text="Mean Squared Error (Risk)", row=2, col=1, title_font=dict(size=16, color='black'))
    fig.update_yaxes(title_text="Success Rate", row=2, col=1, title_font=dict(size=16, color='black'))
    fig.update_xaxes(title_text="Predicted Return (%)", row=2, col=2, title_font=dict(size=16, color='black'))
    fig.update_yaxes(title_text="Frequency", row=2, col=2, title_font=dict(size=16, color='black'))
    
    return fig

def create_enhanced_options_visualization(options_df, current_price, predicted_price, symbol):
    """Create enhanced options visualization with decision framework."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    # Create subplots for comprehensive options analysis
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Options Risk-Reward Matrix', 'Probability Distribution', 
                       'Profit/Loss Scenarios', 'Options Decision Framework'),
        specs=[[{"type": "scatter"}, {"type": "histogram"}],
               [{"type": "scatter"}, {"type": "scatter"}]],
        vertical_spacing=0.1,
        horizontal_spacing=0.1
    )
    
    # 1. Risk-Reward Matrix
    calls_df = options_df[options_df['type'] == 'CALL']
    puts_df = options_df[options_df['type'] == 'PUT']
    
    # Plot calls
    fig.add_trace(
        go.Scatter(
            x=calls_df['risk_reward'],
            y=calls_df['probability'],
            mode='markers+text',
            name='Call Options',
            text=calls_df['strike'].astype(int).astype(str),
            textposition="top center",
            marker=dict(
                size=calls_df['expected_value'] * 10,  # Size based on expected value
                color='green',
                symbol='circle',
                opacity=0.7
            )
        ),
        row=1, col=1
    )
    
    # Plot puts
    fig.add_trace(
        go.Scatter(
            x=puts_df['risk_reward'],
            y=puts_df['probability'],
            mode='markers+text',
            name='Put Options',
            text=puts_df['strike'].astype(int).astype(str),
            textposition="top center",
            marker=dict(
                size=puts_df['expected_value'] * 10,
                color='red',
                symbol='square',
                opacity=0.7
            )
        ),
        row=1, col=1
    )
    
    # Add decision zones
    fig.add_shape(
        type="rect",
        x0=2, y0=0.6, x1=5, y1=1,
        fillcolor="green", opacity=0.2,
        line=dict(color="green", width=2),
        row=1, col=1
    )
    fig.add_annotation(
        x=3.5, y=0.8,
        text="High Conviction Zone",
        showarrow=False,
        font=dict(color="green", size=12),
        row=1, col=1
    )
    
    # 2. Probability Distribution
    all_probabilities = options_df['probability'].values
    fig.add_trace(
        go.Histogram(
            x=all_probabilities,
            nbinsx=20,
            name='Probability Distribution',
            marker_color='lightblue',
            opacity=0.7
        ),
        row=1, col=2
    )
    
    # Add mean line using scatter trace instead of vline
    mean_prob = np.mean(all_probabilities)
    fig.add_trace(
        go.Scatter(
            x=[mean_prob, mean_prob],
            y=[0, max(all_probabilities) * 1.1],
            mode='lines',
            line=dict(dash='dash', color='red', width=2),
            name=f'Mean: {mean_prob:.2f}',
            showlegend=False
        ),
        row=1, col=2
    )
    
    # 3. Profit/Loss Scenarios
    # Select best call and put for scenario analysis
    best_call = calls_df.loc[calls_df['expected_value'].idxmax()] if len(calls_df) > 0 else None
    best_put = puts_df.loc[puts_df['expected_value'].idxmax()] if len(puts_df) > 0 else None
    
    price_range = np.arange(current_price * 0.8, current_price * 1.2, current_price * 0.01)
    
    if best_call is not None:
        call_payoffs = []
        for price in price_range:
            payoff = max(0, price - best_call['strike']) - best_call['price']
            call_payoffs.append(payoff)
        
        fig.add_trace(
            go.Scatter(
                x=price_range,
                y=call_payoffs,
                mode='lines',
                name=f"Call ${best_call['strike']:.0f}",
                line=dict(color='green', width=3)
            ),
            row=2, col=1
        )
    
    if best_put is not None:
        put_payoffs = []
        for price in price_range:
            payoff = max(0, best_put['strike'] - price) - best_put['price']
            put_payoffs.append(payoff)
        
        fig.add_trace(
            go.Scatter(
                x=price_range,
                y=put_payoffs,
                mode='lines',
                name=f"Put ${best_put['strike']:.0f}",
                line=dict(color='red', width=3)
            ),
            row=2, col=1
        )
    
    # Add current and predicted price lines using scatter traces
    # Current price line
    fig.add_trace(
        go.Scatter(
            x=[current_price, current_price],
            y=[min(call_payoffs) if best_call is not None else -10, max(call_payoffs) if best_call is not None else 10],
            mode='lines',
            line=dict(dash='dash', color='blue', width=2),
            name='Current Price',
            showlegend=False
        ),
        row=2, col=1
    )
    
    # Predicted price line
    fig.add_trace(
        go.Scatter(
            x=[predicted_price, predicted_price],
            y=[min(call_payoffs) if best_call is not None else -10, max(call_payoffs) if best_call is not None else 10],
            mode='lines',
            line=dict(dash='dash', color='orange', width=2),
            name='Predicted Price',
            showlegend=False
        ),
        row=2, col=1
    )
    
    # Zero line
    fig.add_trace(
        go.Scatter(
            x=[current_price * 0.8, current_price * 1.2],
            y=[0, 0],
            mode='lines',
            line=dict(dash='dash', color='black', width=2),
            name='Break-even',
            showlegend=False
        ),
        row=2, col=1
    )
    
    # 4. Options Decision Framework
    # Create decision matrix based on market conditions
    volatility_levels = ['Low', 'Medium', 'High']
    trend_directions = ['Bearish', 'Neutral', 'Bullish']
    
    # Simplified decision matrix
    decisions = {
        'Low': {'Bearish': 'Puts', 'Neutral': 'Iron Condor', 'Bullish': 'Calls'},
        'Medium': {'Bearish': 'Bear Spread', 'Neutral': 'Straddle', 'Bullish': 'Bull Spread'},
        'High': {'Bearish': 'Naked Puts', 'Neutral': 'Butterfly', 'Bullish': 'Naked Calls'}
    }
    
    # Determine current conditions
    # Calculate volatility from the options data if available, otherwise use default
    if len(options_df) > 0 and 'delta' in options_df.columns:
        # Estimate volatility from delta values
        avg_delta = options_df['delta'].mean()
        volatility = max(0.1, min(0.8, abs(avg_delta - 0.5) * 2))
    else:
        volatility = 0.2
    price_change = (predicted_price - current_price) / current_price
    
    if volatility < 0.2:
        vol_level = 'Low'
    elif volatility < 0.4:
        vol_level = 'Medium'
    else:
        vol_level = 'High'
    
    if price_change < -0.05:
        trend = 'Bearish'
    elif price_change > 0.05:
        trend = 'Bullish'
    else:
        trend = 'Neutral'
    
    recommended_strategy = decisions[vol_level][trend]
    
    # Create decision framework visualization
    fig.add_trace(
        go.Scatter(
            x=[volatility],
            y=[price_change],
            mode='markers+text',
            name='Current Position',
            text=[recommended_strategy],
            textposition="top center",
            marker=dict(
                size=20,
                color='purple',
                symbol='star'
            )
        ),
        row=2, col=2
    )
    
    # Add decision zones
    fig.add_shape(
        type="rect",
        x0=0, y0=-0.1, x1=0.2, y1=0.1,
        fillcolor="lightblue", opacity=0.3,
        line=dict(color="blue", width=1),
        row=2, col=2
    )
    fig.add_annotation(
        x=0.1, y=0,
        text="Low Volatility",
        showarrow=False,
        font=dict(size=10),
        row=2, col=2
    )
    
    fig.add_shape(
        type="rect",
        x0=0.2, y0=-0.1, x1=0.4, y1=0.1,
        fillcolor="lightgreen", opacity=0.3,
        line=dict(color="green", width=1),
        row=2, col=2
    )
    fig.add_annotation(
        x=0.3, y=0,
        text="Medium Volatility",
        showarrow=False,
        font=dict(size=10),
        row=2, col=2
    )
    
    fig.add_shape(
        type="rect",
        x0=0.4, y0=-0.1, x1=0.6, y1=0.1,
        fillcolor="lightcoral", opacity=0.3,
        line=dict(color="red", width=1),
        row=2, col=2
    )
    fig.add_annotation(
        x=0.5, y=0,
        text="High Volatility",
        showarrow=False,
        font=dict(size=10),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        height=800,
        title_text=f"Enhanced Options Analysis for {symbol}",
        showlegend=True
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Risk/Reward Ratio", row=1, col=1)
    fig.update_yaxes(title_text="Probability", row=1, col=1)
    fig.update_xaxes(title_text="Probability", row=1, col=2)
    fig.update_yaxes(title_text="Count", row=1, col=2)
    fig.update_xaxes(title_text="Stock Price ($)", row=2, col=1)
    fig.update_yaxes(title_text="Profit/Loss ($)", row=2, col=1)
    fig.update_xaxes(title_text="Volatility", row=2, col=2)
    fig.update_yaxes(title_text="Price Change", row=2, col=2)
    
    return fig, recommended_strategy

def main():
    """Main dashboard function with enhanced user experience."""
    
    # Custom CSS for modern, clean styling
    st.markdown("""
    <style>
    /* Modern gradient header */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2.5rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Clean metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        color: white;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    /* Prediction cards */
    .prediction-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        color: white;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* AI insight cards */
    .ai-insight-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        color: white;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Model insight cards */
    .model-insight-card {
        background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        color: white;
        text-align: center;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .model-insight-title {
        font-size: 1rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    
    .model-insight-value {
        font-size: 1.8rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    
    .model-insight-desc {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    
    /* Modern buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2.5rem;
        font-weight: bold;
        font-size: 1rem;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.15);
    }
    
    /* Clean text styling */
    .stMarkdown, .stText, .stWrite {
        color: #333333 !important;
    }
    
    /* Chart containers */
    .stPlotlyChart {
        border-radius: 15px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
        padding: 1rem;
        background: white;
    }
    
    /* Info boxes */
    .stAlert {
        border-radius: 12px;
        border: none;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header with enhanced title bar
    st.markdown("""
    <div class="custom-header">
        <h1>🧠 MarketHacker AI</h1>
        <p>Intelligent Stock Prediction & Portfolio Optimization Platform</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("🎯 Configuration")
        
        # Stock input with dropdown
        st.subheader("Stock Selection")
        
        # Popular stock symbols for dropdown
        popular_stocks = [
            "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "NFLX",
            "JPM", "JNJ", "V", "PG", "UNH", "HD", "MA", "DIS", "PYPL", "ADBE",
            "CRM", "INTC", "VZ", "CMCSA", "PFE", "ABT", "TMO", "COST", "WMT",
            "MRK", "PEP", "KO", "TXN", "AVGO", "QCOM", "HON", "UPS", "LOW",
            "SPY", "QQQ", "IWM", "VTI", "VOO", "ARKK", "TQQQ", "SQQQ"
        ]
        
        # Create a selectbox for popular stocks
        selected_popular = st.selectbox(
            "Quick Select Popular Stocks",
            ["Type manually..."] + popular_stocks,
            help="Select from popular stocks or type manually below"
        )
        
        # Text input for custom stock symbols
        if selected_popular == "Type manually...":
            stock_input = st.text_input(
                "Enter Stock Symbol",
                value="AAPL",
                placeholder="e.g., AAPL, MSFT, GOOGL",
                help="Enter any valid stock symbol"
            ).upper()
        else:
            stock_input = selected_popular
        
        # Check if stock symbol changed and clear cache
        if 'current_stock' not in st.session_state:
            st.session_state.current_stock = stock_input
        elif st.session_state.current_stock != stock_input:
            # Stock changed, clear cached data
            st.session_state.current_stock = stock_input
            if stock_input in st.session_state.stock_data:
                del st.session_state.stock_data[stock_input]
            if stock_input in st.session_state.predictions:
                del st.session_state.predictions[stock_input]
            if stock_input in st.session_state.model_performance:
                del st.session_state.model_performance[stock_input]
            if stock_input in st.session_state.best_options:
                del st.session_state.best_options[stock_input]
            st.rerun()
        
        # Date range
        st.subheader("Date Range")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        
        date_range = st.date_input(
            "Select Date Range",
            value=(start_date, end_date),
            max_value=end_date
        )
        
        if len(date_range) == 2:
            start_date, end_date = date_range
        else:
            start_date = end_date - timedelta(days=365)
        
        # Get available models
        models = get_available_models()
        
        # Prediction horizon
        prediction_horizon = st.selectbox(
            "Prediction Horizon",
            options=[1, 5, 10, 20],
            index=1,
            help="Number of days to predict ahead"
        )
        
        # Confidence threshold
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.5,
            max_value=0.95,
            value=0.7,
            step=0.05,
            help="Minimum confidence for predictions"
        )
        
        # Analysis options
        st.subheader("Analysis Options")
        include_sentiment = st.checkbox("Include Sentiment Analysis", value=True)
        include_technical = st.checkbox("Include Technical Analysis", value=True)
        show_model_metrics = st.checkbox("Show Model Performance", value=True)
        
        # Best Options Recommendations
        if 'best_options' in st.session_state and st.session_state.best_options:
            st.subheader("🏆 Best Options Picks")
            
            for symbol, options in st.session_state.best_options.items():
                if len(options) > 0:
                    best_option = options[0]  # Get the best option
                    
                    # Determine card class based on recommendation
                    card_class = "best-options-card"
                    if "SELL" in best_option['recommendation']:
                        card_class += " sell"
                    elif "HOLD" in best_option['recommendation']:
                        card_class += " hold"
                    
                    st.markdown(f"""
                    <div class="{card_class}">
                        <div class="option-type">{symbol}</div>
                        <div class="option-details">
                            {best_option['type']} ${best_option['strike']:.0f} @ ${best_option['price']:.2f}<br>
                            {best_option['recommendation']} • {best_option['probability']:.1%} probability<br>
                            Risk/Reward: {best_option['risk_reward']:.1f} • EV: ${best_option['expected_value']:.2f}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Clear cache button
        if st.button("🗑️ Clear Cache", help="Clear all cached data and start fresh"):
            st.session_state.stock_data.clear()
            st.session_state.predictions.clear()
            st.session_state.model_performance.clear()
            st.session_state.best_options.clear()
            if 'current_stock' in st.session_state:
                del st.session_state.current_stock
            st.rerun()
        
        # Load data button
        if st.button("🚀 Load & Analyze", type="primary"):
            with st.spinner("Loading data and training models..."):
                try:
                    # Download and process data
                    df = download_stock_data(
                        symbol=stock_input,
                        start_date=start_date.strftime('%Y-%m-%d'),
                        end_date=end_date.strftime('%Y-%m-%d')
                    )
                    
                    if not df.empty:
                        # Clean and create features
                        df = clean_stock_data(df)
                        df = create_features(df)
                        
                        # Add sentiment if requested
                        if include_sentiment:
                            try:
                                sentiment_df = fetch_social_sentiment([stock_input], days_back=30)
                                if not sentiment_df.empty:
                                    # Merge sentiment data with stock data
                                    sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
                                    sentiment_df = sentiment_df.set_index('date')
                                    sentiment_agg = sentiment_df.groupby(sentiment_df.index.date).agg({
                                        'compound': 'mean',
                                        'positive': 'mean',
                                        'negative': 'mean',
                                        'neutral': 'mean'
                                    })
                                    sentiment_agg.columns = ['sentiment_mean', 'sentiment_positive', 'sentiment_negative', 'sentiment_neutral']
                                    df = df.join(sentiment_agg, how='left')
                                    df = df.fillna(0)  # Fill missing sentiment with neutral
                            except Exception as e:
                                st.warning(f"Could not load sentiment data: {e}")
                        
                        # Store in session state
                        st.session_state.stock_data[stock_input] = df
                        
                        # Train all models and make predictions
                        target_col = f'return_{prediction_horizon}d'
                        all_models_results = {}
                        
                        for model_name in models.keys():
                            predictions, metrics, model = train_model_and_predict(df, model_name, target_col)
                            if predictions is not None:
                                all_models_results[model_name] = {
                                    'predictions': predictions,
                                    'metrics': metrics,
                                    'model': model
                                }
                        
                        if all_models_results:
                            st.session_state.predictions[stock_input] = all_models_results
                            st.session_state.model_performance[stock_input] = all_models_results
                            
                            st.success(f"✅ Successfully analyzed {stock_input} with all models!")
                        else:
                            st.error("❌ Failed to train any models. Insufficient data.")
                    else:
                        st.error(f"❌ No data available for {stock_input}")
                        
                except Exception as e:
                    st.error(f"❌ Error: {e}")
    
    # Main content
    if not st.session_state.stock_data:
        st.info("👈 Use the sidebar to select a stock and load data for analysis.")
        return
    
    # Display results for each stock
    for symbol, df in st.session_state.stock_data.items():
        st.markdown(f"## 📊 Analysis for {symbol}")
        
        # Stock info
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            current_price = df['Close'].iloc[-1]
            price_change = df['Close'].pct_change().iloc[-1]
            price_change_pct = price_change * 100
            
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Current Price</div>
                <div class="metric-value">${current_price:.2f}</div>
                <div class="{'prediction-positive' if price_change >= 0 else 'prediction-negative'}">
                    {price_change_pct:+.2f}%
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            ytd_return = ((df['Close'].iloc[-1] / df['Close'].iloc[0]) - 1) * 100
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">YTD Return</div>
                <div class="metric-value">{ytd_return:+.1f}%</div>
                <div class="metric-label">Since {df.index[0].strftime('%Y-%m-%d')}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            volatility = df['price_change'].std() * np.sqrt(252) * 100
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Annual Volatility</div>
                <div class="metric-value">{volatility:.1f}%</div>
                <div class="metric-label">Risk Level</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            volume_avg = df['Volume'].mean()
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Avg Volume</div>
                <div class="metric-value">{volume_avg:,.0f}</div>
                <div class="metric-label">Shares/Day</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Model Performance Metrics
        if show_model_metrics and symbol in st.session_state.model_performance:
            st.subheader("🤖 AI Model Performance Comparison")
            
            all_models_results = st.session_state.model_performance[symbol]
            
            # Create comparison dataframe
            comparison_data = []
            for model_name, results in all_models_results.items():
                metrics = results['metrics']
                comparison_data.append({
                    'Model': model_name,
                    'R² Score': metrics['R²'],
                    'MSE': metrics['MSE'],
                    'Success Rate': metrics['Success Rate'],
                    'Test Size': metrics['Test Size']
                })
            
            comparison_df = pd.DataFrame(comparison_data)
            
            # Display comparison table
            st.dataframe(comparison_df, use_container_width=True)
            
            # Find best model for each metric
            best_r2_model = comparison_df.loc[comparison_df['R² Score'].idxmax(), 'Model']
            best_mse_model = comparison_df.loc[comparison_df['MSE'].idxmin(), 'Model']
            best_success_model = comparison_df.loc[comparison_df['Success Rate'].idxmax(), 'Model']
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Best R² Model", best_r2_model, f"{comparison_df['R² Score'].max():.3f}")
            
            with col2:
                st.metric("Best MSE Model", best_mse_model, f"{comparison_df['MSE'].min():.6f}")
            
            with col3:
                st.metric("Best Success Rate", best_success_model, f"{comparison_df['Success Rate'].max():.1%}")
        
        # Position Recommendation (using best model)
        if symbol in st.session_state.predictions:
            st.subheader("🎯 Position Recommendation")
            
            # Use the model with best success rate for recommendation
            all_models_results = st.session_state.predictions[symbol]
            best_model_name = max(all_models_results.keys(), 
                                key=lambda x: all_models_results[x]['metrics']['Success Rate'])
            
            predictions = all_models_results[best_model_name]['predictions']
            latest_prediction = predictions[-1] if len(predictions) > 0 else 0
            
            recommendation, confidence, reasons = calculate_position_recommendation(
                df, latest_prediction, confidence_threshold
            )
            
            # Generate AI insights
            insights = generate_ai_insights(all_models_results, df, symbol, prediction_horizon, recommendation, confidence, reasons)
            
            # Generate trading strategy
            strategy = generate_trading_strategy(recommendation, confidence, all_models_results, df, symbol)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if recommendation == "BUY":
                    st.markdown(f"""
                    <div class="metric-card" style="background: linear-gradient(135deg, #00ff88 0%, #00cc6a 100%);">
                        <div class="metric-label">RECOMMENDATION</div>
                        <div class="metric-value">🟢 {recommendation}</div>
                        <div class="metric-label">Confidence: {confidence:.1%}</div>
                    </div>
                    """, unsafe_allow_html=True)
                elif recommendation == "SELL":
                    st.markdown(f"""
                    <div class="metric-card" style="background: linear-gradient(135deg, #ff4444 0%, #cc0000 100%);">
                        <div class="metric-label">RECOMMENDATION</div>
                        <div class="metric-value">🔴 {recommendation}</div>
                        <div class="metric-label">Confidence: {confidence:.1%}</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="metric-card" style="background: linear-gradient(135deg, #ffaa00 0%, #ff8800 100%);">
                        <div class="metric-label">RECOMMENDATION</div>
                        <div class="metric-value">🟡 {recommendation}</div>
                        <div class="metric-label">Confidence: {confidence:.1%}</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                pred_pct = latest_prediction * 100
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Predicted Return</div>
                    <div class="metric-value">{pred_pct:+.2f}%</div>
                    <div class="metric-label">Next {prediction_horizon} days</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Key Factors</div>
                    <div style="font-size: 0.9rem; margin-top: 0.5rem;">{reasons}</div>
                </div>
                """, unsafe_allow_html=True)
        
        # AI Insights Section with Dark Theme
        st.markdown("""
        <div class="ai-analysis-container">
            <div class="ai-section-title">🤖 AI-Powered Analysis & Insights</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Display insights in an expandable section
        with st.expander("📊 **Click to see detailed AI analysis**", expanded=True):
            st.markdown("""
            <div class="ai-analysis-container">
                <div class="ai-section-title">🧠 What the AI Models Are Telling Us</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Display insights in styled cards
            for i, insight in enumerate(insights, 1):
                # Determine card type based on insight content
                card_class = "ai-insight-card"
                if any(word in insight.lower() for word in ['warning', 'risk', 'caution', 'decline', 'sell']):
                    card_class += " warning"
                elif any(word in insight.lower() for word in ['danger', 'crash', 'avoid', 'negative']):
                    card_class += " danger"
                elif any(word in insight.lower() for word in ['positive', 'growth', 'opportunity', 'buy']):
                    card_class += " info"
                
                st.markdown(f"""
                <div class="{card_class}">
                    <div style="font-size: 1.1rem; margin-bottom: 0.5rem;"><strong>{i}.</strong></div>
                    <div style="font-size: 1rem; line-height: 1.5;">{insight}</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="ai-analysis-container">
                <div class="ai-section-title">🎯 Your Trading Strategy</div>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div class="ai-strategy-card">
                    <div class="ai-metric-label">Action</div>
                    <div class="ai-metric-value">{strategy['action']}</div>
                    <div class="ai-metric-label">Confidence: {strategy['confidence']:.1%}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="ai-strategy-card">
                    <div class="ai-metric-label">Entry Price</div>
                    <div class="ai-metric-value">${strategy['entry_price']:.2f}</div>
                    <div class="ai-metric-label">Position Size: {strategy['position_size']}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="ai-strategy-card">
                    <div class="ai-metric-label">Risk Level</div>
                    <div class="ai-metric-value">{strategy['risk_level']}</div>
                    <div class="ai-metric-label">Timeframe: {strategy['timeframe']}</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Stop Loss and Take Profit
            col1, col2 = st.columns(2)
            
            with col1:
                if strategy['stop_loss']:
                    stop_loss_pct = ((strategy['stop_loss'] - strategy['entry_price']) / strategy['entry_price'] * 100)
                    st.markdown(f"""
                    <div class="ai-strategy-card">
                        <div class="ai-metric-label">Stop Loss</div>
                        <div class="ai-metric-value">${strategy['stop_loss']:.2f}</div>
                        <div class="ai-metric-label">{stop_loss_pct:+.1f}% from entry</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                if strategy['take_profit']:
                    take_profit_pct = ((strategy['take_profit'] - strategy['entry_price']) / strategy['entry_price'] * 100)
                    st.markdown(f"""
                    <div class="ai-strategy-card">
                        <div class="ai-metric-label">Take Profit</div>
                        <div class="ai-metric-value">${strategy['take_profit']:.2f}</div>
                        <div class="ai-metric-label">{take_profit_pct:+.1f}% from entry</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Simple explanation with styled cards
            st.markdown("""
            <div class="ai-analysis-container">
                <div class="ai-section-title">💡 In Simple Terms</div>
            </div>
            """, unsafe_allow_html=True)
            
            if strategy['action'] == "BUY":
                if strategy['confidence'] > 0.8:
                    st.markdown("""
                    <div class="ai-explanation success">
                        <div style="font-size: 1.2rem; font-weight: bold; margin-bottom: 0.5rem;">🎯 Strong Buy Recommendation</div>
                        <div style="font-size: 1rem; line-height: 1.5;">The AI models are very confident this stock will go up. Consider buying with a larger position size.</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="ai-explanation info">
                        <div style="font-size: 1.2rem; font-weight: bold; margin-bottom: 0.5rem;">✅ Buy Recommendation</div>
                        <div style="font-size: 1rem; line-height: 1.5;">The AI models suggest this stock has upside potential. Consider a moderate position size.</div>
                    </div>
                    """, unsafe_allow_html=True)
            elif strategy['action'] == "SELL":
                if strategy['confidence'] > 0.8:
                    st.markdown("""
                    <div class="ai-explanation error">
                        <div style="font-size: 1.2rem; font-weight: bold; margin-bottom: 0.5rem;">🚨 Strong Sell Recommendation</div>
                        <div style="font-size: 1rem; line-height: 1.5;">The AI models are very confident this stock will go down. Consider selling or avoiding.</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="ai-explanation warning">
                        <div style="font-size: 1.2rem; font-weight: bold; margin-bottom: 0.5rem;">⚠️ Sell Recommendation</div>
                        <div style="font-size: 1rem; line-height: 1.5;">The AI models suggest this stock may decline. Consider reducing position or avoiding.</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="ai-explanation info">
                    <div style="font-size: 1.2rem; font-weight: bold; margin-bottom: 0.5rem;">⏸️ Hold Recommendation</div>
                    <div style="font-size: 1rem; line-height: 1.5;">The AI models are uncertain about direction. Consider waiting for clearer signals.</div>
                </div>
                """, unsafe_allow_html=True)
        
        # Options Analysis
        show_options_analysis(df, latest_prediction, symbol)
        
        # Charts
        st.subheader("📈 Price Charts & Analysis")
        
        # Create tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["Price Chart", "Technical Indicators", "Prediction Analysis", "Model Comparison"])
        
        with tab1:
            # Price chart with predictions
            fig = go.Figure()
            
            # Historical prices
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['Close'],
                mode='lines',
                name='Price',
                line=dict(color='#88c2eb', width=2)
            ))
            
            # Add moving averages
            if 'sma_20' in df.columns:
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df['sma_20'],
                    mode='lines',
                    name='SMA 20',
                    line=dict(color='orange', width=1)
                ))
            
            if 'sma_50' in df.columns:
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df['sma_50'],
                    mode='lines',
                    name='SMA 50',
                    line=dict(color='red', width=1)
                ))
            
            # Add Bollinger Bands
            if all(col in df.columns for col in ['bb_upper', 'bb_middle', 'bb_lower']):
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df['bb_upper'],
                    mode='lines',
                    name='BB Upper',
                    line=dict(color='gray', dash='dash', width=1)
                ))
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df['bb_lower'],
                    mode='lines',
                    name='BB Lower',
                    line=dict(color='gray', dash='dash', width=1),
                    fill='tonexty'
                ))
            
            fig.update_layout(
                title=f"{symbol} - Price Chart with Technical Indicators",
                xaxis_title="Date",
                yaxis_title="Price ($)",
                height=500,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            if include_technical:
                # Technical indicators
                col1, col2 = st.columns(2)
                
                with col1:
                    # RSI
                    if 'rsi' in df.columns:
                        fig_rsi = go.Figure()
                        fig_rsi.add_trace(go.Scatter(
                            x=df.index,
                            y=df['rsi'],
                            mode='lines',
                            name='RSI',
                            line=dict(color='purple', width=2)
                        ))
                        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
                        fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
                        fig_rsi.update_layout(title="RSI", yaxis_range=[0, 100], height=300)
                        st.plotly_chart(fig_rsi, use_container_width=True)
                
                with col2:
                    # MACD
                    if all(col in df.columns for col in ['macd', 'macd_signal']):
                        fig_macd = go.Figure()
                        fig_macd.add_trace(go.Scatter(
                            x=df.index,
                            y=df['macd'],
                            mode='lines',
                            name='MACD',
                            line=dict(color='blue', width=2)
                        ))
                        fig_macd.add_trace(go.Scatter(
                            x=df.index,
                            y=df['macd_signal'],
                            mode='lines',
                            name='Signal',
                            line=dict(color='red', width=2)
                        ))
                        fig_macd.update_layout(title="MACD", height=300)
                        st.plotly_chart(fig_macd, use_container_width=True)
        
        with tab3:
            if symbol in st.session_state.predictions:
                # Prediction analysis (using best model)
                all_models_results = st.session_state.predictions[symbol]
                best_model_name = max(all_models_results.keys(), 
                                    key=lambda x: all_models_results[x]['metrics']['Success Rate'])
                predictions = all_models_results[best_model_name]['predictions']
                
                # Create prediction chart
                fig_pred = go.Figure()
                
                # Get test data dates (last 20% of data)
                test_size = len(predictions)
                test_dates = df.index[-test_size:]
                
                # Actual vs Predicted
                actual_returns = df[f'return_{prediction_horizon}d'].iloc[-test_size:]
                
                fig_pred.add_trace(go.Scatter(
                    x=test_dates,
                    y=actual_returns * 100,
                    mode='lines+markers',
                    name='Actual Returns',
                    line=dict(color='blue', width=2)
                ))
                
                fig_pred.add_trace(go.Scatter(
                    x=test_dates,
                    y=predictions * 100,
                    mode='lines+markers',
                    name=f'{best_model_name} Predictions',
                    line=dict(color='red', width=2)
                ))
                
                fig_pred.update_layout(
                    title=f"Best Model Predictions vs Actual Returns ({prediction_horizon}-day horizon)",
                    xaxis_title="Date",
                    yaxis_title="Return (%)",
                    height=400
                )
                
                st.plotly_chart(fig_pred, use_container_width=True)
                
                # Prediction statistics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    avg_pred = np.mean(predictions) * 100
                    st.metric("Avg Predicted Return", f"{avg_pred:+.2f}%")
                
                with col2:
                    pred_std = np.std(predictions) * 100
                    st.metric("Prediction Std Dev", f"{pred_std:.2f}%")
                
                with col3:
                    positive_preds = np.sum(predictions > 0)
                    total_preds = len(predictions)
                    positive_rate = positive_preds / total_preds
                    st.metric("Positive Predictions", f"{positive_rate:.1%}")
        
        with tab4:
            if symbol in st.session_state.predictions:
                st.markdown("""
                <div class="model-comparison-container">
                    <div class="model-section-title">🤖 Enhanced Model Comparison & Analysis</div>
                </div>
                """, unsafe_allow_html=True)
                
                all_models_results = st.session_state.predictions[symbol]
                
                # Create enhanced model comparison
                enhanced_fig = create_enhanced_model_comparison(all_models_results, df, prediction_horizon)
                st.plotly_chart(enhanced_fig, use_container_width=True)
                
                # Enhanced Model Insights & Analysis
                st.markdown("""
                <div class="model-comparison-container">
                    <div class="model-section-title">💡 Advanced Model Analytics & Insights</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Calculate comprehensive model rankings
                model_names = list(all_models_results.keys())
                model_analytics = {}
                
                for model_name in model_names:
                    results = all_models_results[model_name]
                    predictions = results['predictions']
                    actual_returns = df[f'return_{prediction_horizon}d'].iloc[-len(predictions):] * 100
                    
                    # Calculate advanced metrics
                    precision = np.sum((predictions > 0) & (actual_returns > 0)) / max(np.sum(predictions > 0), 1)
                    recall = np.sum((predictions > 0) & (actual_returns > 0)) / max(np.sum(actual_returns > 0), 1)
                    f1_score = 2 * (precision * recall) / max((precision + recall), 1e-8)
                    
                    pred_returns = predictions * 100
                    sharpe_ratio = np.mean(pred_returns) / max(np.std(pred_returns), 1e-8)
                    sharpe_ratio = min(max(sharpe_ratio, 0), 1)
                    
                    # Calculate risk-adjusted return
                    risk_adjusted_return = results['metrics']['Success Rate'] / max(results['metrics']['MSE'] * 10, 0.1)
                    
                    # Calculate prediction stability (lower std = more stable)
                    prediction_stability = 1 / (1 + np.std(predictions))
                    
                    model_analytics[model_name] = {
                        'R²': results['metrics']['R²'],
                        'Success_Rate': results['metrics']['Success Rate'],
                        'MSE': results['metrics']['MSE'],
                        'Precision': precision,
                        'Recall': recall,
                        'F1_Score': f1_score,
                        'Sharpe_Ratio': sharpe_ratio,
                        'Risk_Adjusted_Return': risk_adjusted_return,
                        'Prediction_Stability': prediction_stability,
                        'Overall_Score': (
                            results['metrics']['R²'] * 0.25 +
                            results['metrics']['Success Rate'] * 0.25 +
                            (1 - results['metrics']['MSE'] * 100) * 0.15 +
                            precision * 0.15 +
                            sharpe_ratio * 0.1 +
                            prediction_stability * 0.1
                        )
                    }
                
                # Find best models for different criteria
                best_overall = max(model_names, key=lambda x: model_analytics[x]['Overall_Score'])
                best_r2_model = max(model_names, key=lambda x: model_analytics[x]['R²'])
                best_success_model = max(model_names, key=lambda x: model_analytics[x]['Success_Rate'])
                best_mse_model = min(model_names, key=lambda x: model_analytics[x]['MSE'])
                best_risk_adjusted = max(model_names, key=lambda x: model_analytics[x]['Risk_Adjusted_Return'])
                most_stable = max(model_names, key=lambda x: model_analytics[x]['Prediction_Stability'])
                
                # Display enhanced model insights
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    overall_score = model_analytics[best_overall]['Overall_Score']
                    st.markdown(f"""
                    <div class="model-insight-card">
                        <div class="model-insight-title">🏆 Best Overall Model</div>
                        <div class="model-insight-value">{best_overall}</div>
                        <div class="model-insight-desc">Score: {overall_score:.3f}</div>
                        <div class="model-insight-desc">Balanced performance across all metrics</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    risk_score = model_analytics[best_risk_adjusted]['Risk_Adjusted_Return']
                    st.markdown(f"""
                    <div class="model-insight-card">
                        <div class="model-insight-title">⚖️ Best Risk-Adjusted</div>
                        <div class="model-insight-value">{best_risk_adjusted}</div>
                        <div class="model-insight-desc">Score: {risk_score:.3f}</div>
                        <div class="model-insight-desc">Optimal risk-reward ratio</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    success_rate = model_analytics[best_success_model]['Success_Rate']
                    st.markdown(f"""
                    <div class="model-insight-card">
                        <div class="model-insight-title">🎯 Highest Success Rate</div>
                        <div class="model-insight-value">{best_success_model}</div>
                        <div class="model-insight-desc">Success: {success_rate:.1%}</div>
                        <div class="model-insight-desc">Most accurate directional predictions</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    stability_score = model_analytics[most_stable]['Prediction_Stability']
                    st.markdown(f"""
                    <div class="model-insight-card">
                        <div class="model-insight-title">📊 Most Stable Predictions</div>
                        <div class="model-insight-value">{most_stable}</div>
                        <div class="model-insight-desc">Stability: {stability_score:.3f}</div>
                        <div class="model-insight-desc">Consistent prediction patterns</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    r2_score = model_analytics[best_r2_model]['R²']
                    st.markdown(f"""
                    <div class="model-insight-card">
                        <div class="model-insight-title">📈 Best Trend Predictor</div>
                        <div class="model-insight-value">{best_r2_model}</div>
                        <div class="model-insight-desc">R²: {r2_score:.3f}</div>
                        <div class="model-insight-desc">Best trend correlation</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    mse_score = model_analytics[best_mse_model]['MSE']
                    st.markdown(f"""
                    <div class="model-insight-card">
                        <div class="model-insight-title">⚡ Lowest Error</div>
                        <div class="model-insight-value">{best_mse_model}</div>
                        <div class="model-insight-desc">MSE: {mse_score:.4f}</div>
                        <div class="model-insight-desc">Minimal prediction variance</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Enhanced Model Rankings & Performance Analysis
                st.markdown("""
                <div class="model-comparison-container">
                    <div class="model-section-title">🏆 Comprehensive Model Rankings & Performance Analysis</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Create comprehensive ranking table
                ranking_data = []
                for model_name in model_names:
                    analytics = model_analytics[model_name]
                    ranking_data.append({
                        'Model': model_name,
                        'Overall Score': analytics['Overall_Score'],
                        'R² Score': analytics['R²'],
                        'Success Rate': analytics['Success_Rate'],
                        'MSE': analytics['MSE'],
                        'Precision': analytics['Precision'],
                        'Recall': analytics['Recall'],
                        'F1 Score': analytics['F1_Score'],
                        'Sharpe Ratio': analytics['Sharpe_Ratio'],
                        'Risk-Adjusted Return': analytics['Risk_Adjusted_Return'],
                        'Prediction Stability': analytics['Prediction_Stability']
                    })
                
                ranking_df = pd.DataFrame(ranking_data)
                ranking_df = ranking_df.sort_values('Overall Score', ascending=False)
                
                # Display ranking table with styling
                st.markdown("""
                <div class="model-ranking-table">
                """, unsafe_allow_html=True)
                st.dataframe(ranking_df, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Model performance summary
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"""
                    <div class="model-metric-card">
                        <div class="model-metric-label">🏆 Best Overall Model</div>
                        <div class="model-metric-value">{best_overall}</div>
                        <div class="model-metric-label">Score: {model_analytics[best_overall]['Overall_Score']:.3f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"""
                    <div class="model-metric-card">
                        <div class="model-metric-label">⚖️ Best Risk-Adjusted</div>
                        <div class="model-metric-value">{best_risk_adjusted}</div>
                        <div class="model-metric-label">Score: {model_analytics[best_risk_adjusted]['Risk_Adjusted_Return']:.3f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="model-metric-card">
                        <div class="model-metric-label">📊 Most Stable</div>
                        <div class="model-metric-value">{most_stable}</div>
                        <div class="model-metric-label">Stability: {model_analytics[most_stable]['Prediction_Stability']:.3f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"""
                    <div class="model-metric-label">Model Performance Summary</div>
                    <div style="font-size: 0.9rem; line-height: 1.4; margin-top: 0.5rem;">
                        • <strong>Best Overall:</strong> {best_overall} - Balanced performance across all metrics<br>
                        • <strong>Best Risk-Adjusted:</strong> {best_risk_adjusted} - Optimal risk-reward ratio<br>
                        • <strong>Most Stable:</strong> {most_stable} - Consistent prediction patterns<br>
                        • <strong>Highest Success:</strong> {best_success_model} - Best directional accuracy
                    </div>
                    """, unsafe_allow_html=True)
                
                # Model comparison insights
                st.markdown("""
                <div class="model-comparison-container">
                    <div class="model-section-title">🔍 Model Comparison Insights</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Calculate model diversity and consensus
                predictions_array = np.array([all_models_results[model]['predictions'] for model in model_names])
                prediction_correlation = np.corrcoef(predictions_array)
                
                # Find most and least correlated models
                correlation_matrix = pd.DataFrame(prediction_correlation, index=model_names, columns=model_names)
                np.fill_diagonal(correlation_matrix.values, 0)  # Remove self-correlation
                
                max_corr = correlation_matrix.max().max()
                min_corr = correlation_matrix.min().min()
                
                # Find model pairs (simplified approach)
                max_corr_models = []
                min_corr_models = []
                
                for i, model1 in enumerate(model_names):
                    for j, model2 in enumerate(model_names):
                        if i != j:
                            corr_val = correlation_matrix.loc[model1, model2]
                            if corr_val == max_corr:
                                max_corr_models = [model1, model2]
                            if corr_val == min_corr:
                                min_corr_models = [model1, model2]
                
                # Ensure we have valid model pairs with proper error handling
                if len(max_corr_models) < 2:
                    if len(model_names) >= 2:
                        max_corr_models = model_names[:2]
                    else:
                        max_corr_models = model_names
                if len(min_corr_models) < 2:
                    if len(model_names) >= 2:
                        min_corr_models = model_names[:2]
                    else:
                        min_corr_models = model_names
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    max_corr_desc = f"Highest correlation: {max_corr_models[0]} & {max_corr_models[1]}" if len(max_corr_models) >= 2 else "Model consensus analysis"
                    st.markdown(f"""
                    <div class="model-insight-card">
                        <div class="model-insight-title">🤝 Model Consensus</div>
                        <div class="model-insight-value">{max_corr:.3f}</div>
                        <div class="model-insight-desc">{max_corr_desc}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    min_corr_desc = f"Lowest correlation: {min_corr_models[0]} & {min_corr_models[1]}" if len(min_corr_models) >= 2 else "Model diversity analysis"
                    st.markdown(f"""
                    <div class="model-insight-card">
                        <div class="model-insight-title">🔄 Model Diversity</div>
                        <div class="model-insight-value">{min_corr:.3f}</div>
                        <div class="model-insight-desc">{min_corr_desc}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    avg_correlation = np.mean(correlation_matrix.values)
                    st.markdown(f"""
                    <div class="model-insight-card">
                        <div class="model-insight-title">📊 Average Correlation</div>
                        <div class="model-insight-value">{avg_correlation:.3f}</div>
                        <div class="model-insight-desc">Overall model agreement level</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Generate AI-powered insights
                insights = generate_ai_insights(all_models_results, df, symbol, prediction_horizon, recommendation, confidence, reasons)
                
                st.markdown("""
                <div class="model-comparison-container">
                    <div class="model-section-title">🤖 AI Insights</div>
                </div>
                """, unsafe_allow_html=True)
                
                for insight in insights:
                    st.markdown(f"""
                    <div class="model-insight-card">
                        <div style="font-size: 1rem; line-height: 1.5;">{insight}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Generate trading strategy
                strategy = generate_trading_strategy(recommendation, confidence, all_models_results, df, symbol)
                
                st.markdown("""
                <div class="model-comparison-container">
                    <div class="model-section-title">🎯 Trading Strategy</div>
                </div>
                """, unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"""
                    <div class="model-strategy-card">
                        <div class="model-metric-label">Action</div>
                        <div class="model-metric-value">{strategy['action']}</div>
                        <div class="model-metric-label">Confidence: {strategy['confidence']:.1%}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"""
                    <div class="model-strategy-card">
                        <div class="model-metric-label">Entry Price</div>
                        <div class="model-metric-value">${strategy['entry_price']:.2f}</div>
                        <div class="model-metric-label">Position Size: {strategy['position_size']}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    if strategy['stop_loss']:
                        st.markdown(f"""
                        <div class="model-strategy-card">
                            <div class="model-metric-label">Stop Loss</div>
                            <div class="model-metric-value">${strategy['stop_loss']:.2f}</div>
                            <div class="model-metric-label">Risk Management</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    if strategy['take_profit']:
                        st.markdown(f"""
                        <div class="model-strategy-card">
                            <div class="model-metric-label">Take Profit</div>
                            <div class="model-metric-value">${strategy['take_profit']:.2f}</div>
                            <div class="model-metric-label">Profit Target</div>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Additional strategy details
                st.markdown(f"""
                <div class="model-strategy-card">
                    <div class="model-metric-label">Strategy Details</div>
                    <div style="font-size: 1rem; line-height: 1.5; margin-top: 0.5rem;">
                        <strong>Timeframe:</strong> {strategy['timeframe']}<br>
                        <strong>Risk Level:</strong> {strategy['risk_level']}<br>
                        <strong>Best Model:</strong> {best_overall}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
            else:
                st.info("No model predictions available. Please run the analysis first.")

if __name__ == "__main__":
    main() 