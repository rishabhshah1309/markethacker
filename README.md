# 🧠 MarketHacker

MarketHacker is an intelligent stock prediction and options strategy platform that leverages historical data, market indexes, real-time news, and public sentiment to help identify high-potential stocks and options plays. It uses machine learning models to forecast price movements and visualize market trends — all wrapped in a user-friendly dashboard hosted on AWS.

---

## 📊 Features

- 📈 Fetch & analyze historical stock data from Yahoo Finance
- 📰 Ingest and analyze real-time news and social sentiment (Twitter, Reddit)
- 🧠 Train regression and time-series models to predict price movements
- 🧮 Simulate options strategies (covered calls, vertical spreads, etc.)
- 🔎 Index and rank stocks based on volatility, momentum, and sentiment
- 📊 Visual dashboard with predictions, charts, and trade ideas
- ☁️ Deployable on AWS using S3, Lambda, and CloudFront

---

## 📁 Project Structure

```
markethacker/
├── data/                    # Raw and processed datasets
├── notebooks/              # Jupyter notebooks for EDA and modeling
├── src/                    # Modular Python source code
│   ├── data_ingestion/     # Scripts to fetch financial data and sentiment
│   ├── preprocessing/      # Cleaning and feature engineering
│   ├── modeling/           # ML models and options simulations
│   ├── evaluation/         # Backtesting and metrics
│   └── visualization/      # Plots and dashboard utilities
├── dashboard/              # Frontend dashboard (Streamlit or React)
├── tests/                  # Unit and integration tests
├── requirements.txt        # Python dependencies
├── Dockerfile              # Docker container setup
├── README.md               # Project documentation
└── setup.py                # Installable package configuration
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/rishabhshah1309/markethacker.git
cd markethacker
```

### 2. Set up environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Fetch some stock data

```python
from src.data_ingestion.fetch_market_data import download_stock_data
df = download_stock_data("AAPL", "2022-01-01", "2023-01-01")
```

### 4. Run the dashboard

```bash
streamlit run dashboard/app.py
```

---

## 📦 Tech Stack

| Layer           | Tech                                |
|----------------|--------------------------------------|
| Backend         | Python, Flask, FastAPI               |
| Frontend        | Streamlit or React + Plotly.js       |
| Data Sources    | Yahoo Finance, NewsAPI, Reddit, Twitter |
| ML/Modeling     | scikit-learn, XGBoost, Prophet, LSTM |
| Sentiment       | VaderSentiment, FinBERT              |
| Deployment      | AWS (S3, Lambda, CloudFront, RDS)    |

---

## 📈 Example Use Cases

- Predict whether AAPL will move +3% this week and simulate call option profit.
- Identify top 5 high-momentum, low-risk stocks based on sentiment and indicators.
- Visualize sentiment trends alongside price volatility.

---

## ✅ Roadmap

- [x] Set up stock data ingestion
- [x] Add technical indicator features
- [x] Build and evaluate regression models
- [ ] Integrate sentiment from Twitter and Reddit
- [ ] Develop options strategy simulator
- [ ] Build a full dashboard with visual insights
- [ ] Deploy backend APIs and frontend via AWS

---

## 🧪 Contributing

Pull requests and issues are welcome! To contribute:
1. Fork this repo
2. Create a feature branch
3. Commit your changes
4. Submit a PR 🚀

---

## 🛡️ License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## 🙋‍♂️ Maintainer

**Rishabh Shah**  
📫 [LinkedIn](https://linkedin.com/in/rishabhshah1309)  
📬 rishabhshah1309@gmail.com
