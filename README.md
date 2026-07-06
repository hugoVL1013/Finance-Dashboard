# Finance Dashboard

An interactive Streamlit dashboard for equity and cross-asset analysis, combining
market data, fundamental ratios, factor models, and machine-learning-based
correlation and price analysis in a single app.

Live app: https://financedashboardhugovl.streamlit.app/

## Features

### Stock Dashboard
- Price history with 30-day rolling volatility and MA20 / MA50 / MA200 moving averages
- 90-day rolling beta and rolling Sharpe ratio against a benchmark
- Key fundamental ratios (P/E, EPS, ROE, debt-to-equity, current ratio, margins)
- Beta, Sharpe ratio, and CAPM expected return versus the market
- Fama-French factor analysis using factor data from the Ken French data library

### Stock Correlation
- Correlation analysis across user-selected assets, indices, and macro variables
- Augmented Dickey-Fuller (ADF) stationarity testing on the series
- Rolling correlation with a custom window, plus automatic window optimisation (20 to 200 days)
- Machine-learning price prediction (Linear Regression, Random Forest, and an optional LSTM) with model comparison, feature importance, and out-of-sample performance metrics
- Historical correlation regime and cycle analysis, seasonal patterns, and Granger causality testing

## Tech stack

Python, Streamlit, yfinance, pandas, NumPy, scikit-learn, statsmodels, SciPy, and Plotly.
An optional TensorFlow/Keras LSTM path is included and degrades gracefully when TensorFlow is not installed.

## Running locally

```bash
git clone https://github.com/hugoVL1013/Finance-Dashboard.git
cd Finance-Dashboard
pip install -r requirements.txt
streamlit run App.py
```

## Configuration

The weather-correlation features use the VisualCrossing API. Provide a key through
Streamlit secrets rather than hardcoding it:

1. Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml`
2. Add your key: `VISUALCROSSING_API_KEY = "your_key_here"`

Alternatively, set a `VISUALCROSSING_API_KEY` environment variable. On Streamlit
Community Cloud, set the key under App settings, then Secrets. The app runs without
a key; the weather-based variables simply fall back to synthetic series.

## Repository layout

- `App.py` is the main application.
- `ML_Stock_Predictor.py` contains the standalone machine-learning prediction module.
- `requirements.txt` lists the dependencies.
