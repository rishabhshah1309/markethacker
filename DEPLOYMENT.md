# 🚀 MarketHacker Deployment Guide

This guide will help you deploy MarketHacker to various platforms for sharing and production use.

## 📋 Prerequisites

- Python 3.9 or higher
- Git
- A GitHub account (for version control)

## 🎯 Deployment Options

### 1. **Streamlit Cloud (Recommended for Sharing)**

Streamlit Cloud is the easiest way to deploy and share your dashboard.

#### Setup Steps:

1. **Push to GitHub**
```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/yourusername/markethacker.git
git push -u origin main
```

2. **Deploy to Streamlit Cloud**
- Go to [share.streamlit.io](https://share.streamlit.io)
- Sign in with your GitHub account
- Click "New app"
- Select your repository: `yourusername/markethacker`
- Set the file path: `dashboard/app.py`
- Click "Deploy"

3. **Configure Environment Variables** (Optional)
In Streamlit Cloud settings, add:
```
REDDIT_CLIENT_ID=your_reddit_client_id
REDDIT_CLIENT_SECRET=your_reddit_client_secret
TWITTER_BEARER_TOKEN=your_twitter_bearer_token
```

**✅ Benefits:**
- Free hosting
- Automatic updates from GitHub
- Easy sharing with public URL
- Built-in analytics

### 2. **Heroku Deployment**

For more control and custom domains.

#### Setup Steps:

1. **Create Heroku App**
```bash
# Install Heroku CLI
heroku create markethacker-app
```

2. **Create Procfile**
```bash
echo "web: streamlit run dashboard/app.py --server.port=\$PORT --server.address=0.0.0.0" > Procfile
```

3. **Create runtime.txt**
```bash
echo "python-3.9.18" > runtime.txt
```

4. **Deploy**
```bash
git add .
git commit -m "Add Heroku deployment files"
git push heroku main
```

5. **Open App**
```bash
heroku open
```

### 3. **Docker Deployment**

For containerized deployment.

#### Setup Steps:

1. **Create Dockerfile**
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

2. **Build and Run**
```bash
docker build -t markethacker .
docker run -p 8501:8501 markethacker
```

### 4. **AWS Deployment**

For enterprise-grade hosting.

#### Setup Steps:

1. **Create EC2 Instance**
```bash
# Launch Ubuntu 20.04 instance
# Configure security group to allow port 8501
```

2. **Install Dependencies**
```bash
sudo apt update
sudo apt install python3-pip python3-venv
```

3. **Deploy Application**
```bash
git clone https://github.com/yourusername/markethacker.git
cd markethacker
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

4. **Run with PM2** (for production)
```bash
npm install -g pm2
pm2 start "streamlit run dashboard/app.py --server.port=8501" --name markethacker
pm2 startup
pm2 save
```

## 🔧 Configuration for Production

### Environment Variables
Create a `.env` file for production:

```env
# API Configuration
REDDIT_CLIENT_ID=your_production_reddit_id
REDDIT_CLIENT_SECRET=your_production_reddit_secret
TWITTER_BEARER_TOKEN=your_production_twitter_token

# Model Configuration
PREDICTION_HORIZON=5
CONFIDENCE_THRESHOLD=0.6
MAX_MODELS=6

# Performance Settings
CACHE_TTL=3600
MAX_WORKERS=4
```

### Performance Optimization

1. **Enable Caching**
```python
@st.cache_data(ttl=3600)
def fetch_stock_data(symbol, start_date, end_date):
    # Your data fetching code
    pass
```

2. **Optimize Model Loading**
```python
@st.cache_resource
def load_models():
    # Load and cache models
    return models
```

3. **Configure Streamlit Settings**
Create `.streamlit/config.toml`:
```toml
[server]
maxUploadSize = 200
enableXsrfProtection = false
enableCORS = false

[browser]
gatherUsageStats = false
```

## 📊 Monitoring and Analytics

### 1. **Streamlit Analytics** (Built-in)
- User sessions
- Page views
- Performance metrics

### 2. **Custom Analytics**
```python
import streamlit as st
import time

# Track usage
def track_usage(action):
    st.session_state['usage'] = st.session_state.get('usage', [])
    st.session_state['usage'].append({
        'action': action,
        'timestamp': time.time(),
        'user': st.session_state.get('user_id', 'anonymous')
    })
```

### 3. **Error Monitoring**
```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    # Your code
    pass
except Exception as e:
    logger.error(f"Error: {e}")
    st.error("An error occurred. Please try again.")
```

## 🔒 Security Considerations

### 1. **API Key Management**
- Never commit API keys to version control
- Use environment variables
- Rotate keys regularly

### 2. **Rate Limiting**
```python
import time
from functools import wraps

def rate_limit(calls=60, period=60):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Implement rate limiting logic
            return func(*args, **kwargs)
        return wrapper
    return decorator
```

### 3. **Input Validation**
```python
import re

def validate_stock_symbol(symbol):
    if not re.match(r'^[A-Z]{1,5}$', symbol):
        raise ValueError("Invalid stock symbol")
    return symbol.upper()
```

## 🚀 Scaling Considerations

### 1. **Database Integration**
For high-traffic applications, consider adding a database:

```python
import sqlite3
import pandas as pd

def cache_predictions(symbol, predictions):
    conn = sqlite3.connect('predictions.db')
    predictions.to_sql('predictions', conn, if_exists='replace')
    conn.close()
```

### 2. **Load Balancing**
For multiple instances:
- Use a load balancer (AWS ALB, Nginx)
- Implement session management
- Use Redis for caching

### 3. **CDN Integration**
- Serve static assets via CDN
- Cache API responses
- Optimize image delivery

## 📈 Performance Monitoring

### 1. **Application Metrics**
```python
import psutil
import time

def log_performance():
    cpu_percent = psutil.cpu_percent()
    memory_percent = psutil.virtual_memory().percent
    logger.info(f"CPU: {cpu_percent}%, Memory: {memory_percent}%")
```

### 2. **Response Time Tracking**
```python
import time

def track_response_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"{func.__name__} took {end_time - start_time:.2f} seconds")
        return result
    return wrapper
```

## 🆘 Troubleshooting

### Common Issues:

1. **Port Already in Use**
```bash
# Find and kill process
lsof -ti:8501 | xargs kill -9
```

2. **Memory Issues**
```bash
# Increase memory limit
export STREAMLIT_SERVER_MAX_UPLOAD_SIZE=200
```

3. **API Rate Limits**
```python
# Implement exponential backoff
import time
import random

def api_call_with_retry(func, max_retries=3):
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            time.sleep(2 ** attempt + random.uniform(0, 1))
```

## 📞 Support

For deployment issues:
- Check the [Streamlit documentation](https://docs.streamlit.io/)
- Review [Heroku deployment guide](https://devcenter.heroku.com/)
- Open an issue on GitHub

---

**Happy Deploying! 🚀** 