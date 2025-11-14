# MarketHacker Usage Guide

## Overview

MarketHacker is a comprehensive AI-powered stock prediction platform that uses multiple machine learning models to provide trading recommendations, options analysis, and risk assessments.

## Key Features

### 1. **Multi-Model Prediction System**
- **6 Machine Learning Models**: Random Forest, XGBoost, LightGBM, Gradient Boosting, Ridge, Lasso
- **Ensemble Predictions**: Combines all models for robust predictions
- **Model Comparison**: Side-by-side performance metrics
- **Confidence Intervals**: 95% confidence bounds on predictions

### 2. **Detailed Trading Recommendations**
- **Buy/Sell/Hold Signals**: Clear actionable recommendations
- **Confidence Scores**: Know how reliable each prediction is
- **Price Targets**: Specific target prices and stop-loss levels
- **Detailed Explanations**: Understand WHY the system recommends each action

### 3. **Options Analysis**
- **Best Options Identification**: Find the most profitable call and put options
- **Strategy Recommendations**: Get personalized options strategies
- **Probability Calculations**: Black-Scholes based probability of profit
- **Risk-Reward Analysis**: Understand the potential upside and downside

### 4. **Risk Assessment**
- **Volatility Metrics**: Understand price movement patterns
- **Maximum Drawdown**: Historical worst-case scenarios
- **Sharpe Ratio**: Risk-adjusted returns
- **Warning System**: Get alerted to high-risk conditions

## How to Use

### Running the Dashboard

1. **Start the dashboard:**
   ```bash
   streamlit run dashboard/app.py
   ```

2. **Enter a stock symbol** (e.g., AAPL, TSLA, NVDA)

3. **Adjust settings** if needed:
   - Historical data period (90-730 days)

4. **Click "Run Analysis"** to start the prediction

### Understanding the Results

#### Ensemble Prediction
- **Predicted Return**: Expected price change over 5-10 days
- **Confidence Interval**: Range where actual return is likely to fall (95% probability)
- Green numbers indicate bullish predictions, red indicates bearish

#### Model Performance Comparison
- **Radar Chart**: Compare models across R² Score, Direction Accuracy, and Overall Score
- **Bar Chart**: See individual model predictions side-by-side
- **Performance Table**: Detailed metrics for each model

#### Trading Recommendation

The system provides:
1. **Action Signal**: BUY / SELL / HOLD (with strength: STRONG BUY, etc.)
2. **Confidence Level**: HIGH / MEDIUM / LOW
3. **Target Price**: Where the stock is expected to reach
4. **Stop Loss**: Recommended exit point if trade goes against you

**Explanation Sections:**
- **Summary**: Quick overview of the recommendation
- **Prediction Analysis**: What the models are saying
- **Market Context**: Current market conditions
- **Risk Assessment**: Risk levels and concerns
- **Key Reasons**: Bullet points explaining the logic

#### Individual Model Predictions

See how each model performed and what it predicts:
- **Prediction %**: Expected return percentage
- **Direction**: Bullish or Bearish
- **Strength**: Strong, Moderate, or Weak signal
- **Model Accuracy**: Historical performance score

#### Options Analysis

**Best Call Option:**
- Recommended if prediction is bullish
- Shows strike price, premium, probability of profit

**Best Put Option:**
- Recommended if prediction is bearish
- Shows strike price, premium, probability of profit

**Strategy Recommendations:**
- Long Call/Put: Simple directional plays
- Bull/Bear Spreads: Reduced cost strategies
- Iron Condor: For range-bound predictions
- Straddle: For high volatility expectations

## Interpreting Confidence Levels

### HIGH Confidence (75%+)
- Models show strong agreement
- Historical accuracy is good
- Safer to act on the recommendation

### MEDIUM Confidence (55-75%)
- Moderate model agreement
- Use as one signal among many
- Consider your own research

### LOW Confidence (<55%)
- Models disagree or historical performance is weak
- Exercise extra caution
- May want to wait for better setup

## Risk Management Tips

1. **Never invest more than you can afford to lose**
2. **Always use stop-losses** (the system provides recommended levels)
3. **Diversify** - Don't put all capital in one trade
4. **Check the warnings** - Pay attention to high volatility and drawdown warnings
5. **Do your own research** - Use this as one tool, not the only tool

## Example Workflow

### Bullish Prediction Example

1. **Stock**: AAPL
2. **Ensemble Prediction**: +4.2%
3. **Confidence**: HIGH (78%)
4. **Recommendation**: STRONG BUY
5. **Target**: $185.50
6. **Stop Loss**: $175.20
7. **Best Option**: Call $180 strike, 72% probability of profit

**Action**: Consider buying shares or the recommended call option, with stop loss at $175.20.

### Bearish Prediction Example

1. **Stock**: TSLA
2. **Ensemble Prediction**: -3.8%
3. **Confidence**: MEDIUM (65%)
4. **Recommendation**: SELL
5. **Target**: $245.00
6. **Stop Loss**: $260.00
7. **Best Option**: Put $250 strike, 68% probability of profit

**Action**: Consider selling if you own shares, or buying the recommended put option, with stop loss at $260.

## Common Questions

**Q: How accurate are the predictions?**
A: The ensemble model typically achieves 60-70% directional accuracy. Past performance doesn't guarantee future results.

**Q: What timeframe do predictions cover?**
A: Predictions are for 5-10 trading days ahead.

**Q: Should I follow every recommendation?**
A: No. Use this as one tool in your analysis. Consider your own research, risk tolerance, and market conditions.

**Q: What if models disagree?**
A: Low confidence scores indicate disagreement. In these cases, it's often better to wait or reduce position size.

**Q: Can I use this for day trading?**
A: This system is designed for swing trading (5-10 day holds). It's not optimized for day trading.

## Disclaimer

⚠️ **IMPORTANT**: This tool is for educational purposes only.
- Past performance does not guarantee future results
- Always do your own research before making investment decisions
- Consider consulting with a financial advisor
- Never invest more than you can afford to lose
- The authors are not responsible for any financial losses

## Support

For issues or questions:
- Check the documentation in the repository
- Review the code comments
- Open an issue on GitHub

---

**Happy Trading! 📈**
