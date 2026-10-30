# Finance Dashboard

An interactive Streamlit app for single-stock and cross-asset analysis: risk and return metrics, fundamental ratios, a Fama-French three-factor model, correlation and causality diagnostics, and a machine-learning return forecaster built around leakage-free validation.

Live app: https://financedashboardhugovl.streamlit.app/

## What it does

### Page 1: Stock Dashboard

**Benchmark matching.** Each ticker is compared against its home index, chosen from the exchange suffix (`.L` to FTSE 100, `.PA` to CAC 40, `.DE` to DAX, `.T` to Nikkei 225). US technology stocks are compared against the Nasdaq Composite and all other US stocks against the S&P 500.

**Risk and return.**
- 30-day rolling volatility, annualised with the square root of 252.
- 90-day rolling beta, estimated by OLS of daily stock returns on benchmark returns over each window.
- 90-day rolling Sharpe ratio on excess returns.
- Full-period beta, Sharpe ratio and CAPM expected return. The equity risk premium is the benchmark's realised annualised return over the selected window minus the risk-free rate.
- The risk-free rate is the live 30-year US Treasury yield (Yahoo `^TYX`), with a 2.5% fallback if the quote is unavailable.

**Fundamentals.** P/E, EPS, ROE, quick ratio, debt-to-equity and current ratio from Yahoo Finance, plus EBIT margin, operating margin and interest coverage computed directly from the latest quarterly income statement.

**Fama-French three-factor model.** Daily factors are downloaded from the Kenneth French data library and compounded to monthly. The stock's monthly excess return (over the one-month T-bill rate in the factor file) is regressed on Mkt-RF, SMB and HML by OLS. The app reports annualised alpha, the three loadings with t-statistics and p-values, R², and a factor-implied expected return.

### Page 2: Stock Correlation

- Daily log-return correlations between a chosen stock and 13 built-in market series (equity, gold, oil, natural gas, broad commodities, 10-year and 20+ year Treasuries, VIX, Brent and WTI crude, the dollar index, Bitcoin and coffee), plus any custom Yahoo ticker.
- Augmented Dickey-Fuller stationarity tests on every series.
- Rolling correlations with a user-set window, and a sweep showing how average correlation changes across windows from 20 to 200 days.
- Granger causality tests between any two series.
- Optional weather series (EU, US and Colombian temperature and rainfall) from the VisualCrossing API.

### Return forecaster

Predicts the stock's return over a chosen horizon (1 to 30 trading days) from technical indicators (moving-average ratios, RSI, MACD, Bollinger band position, momentum, volatility, volume ratios) and rolling correlation features against the selected market series. Models: Random Forest, a two-layer LSTM, or an equal-weight ensemble of both.

## How the models are validated

This is the part of the project I care most about.

The first version of the Random Forest looked predictive. I traced the result to feature selection: the model that ranked features was fitted on the full dataset before the train/test split, so the chosen features already "knew" the test period. The current pipeline fixes this.

- **Chronological holdout.** The last 15% of the sample is a test set the model never sees during development.
- **Embargo.** Each target is the return over the next *h* days, so rows near a split boundary share part of their target window with the other side. An *h*-day gap is dropped between training and test data, and between training and validation data inside each fold.
- **Feature selection inside each fold.** Features are re-ranked on each fold's training rows only, using walk-forward cross-validation (`TimeSeriesSplit`, 5 folds). The final feature set is chosen on development data only.
- **Out-of-sample metrics.** R², RMSE and directional accuracy are reported on the untouched test set alongside the cross-validation average.

Once the leak was removed, the apparent skill disappeared: out-of-sample R² turned negative and directional accuracy fell to roughly a coin flip. That is the expected result for short-horizon return prediction from public price data. The forecaster is kept as a demonstration of a correct validation pipeline, not as a trading signal.

## Known limitations

I would rather list these than have someone find them:

- The LSTM's scalers are fitted on the full sample, including the test period, and its train/test split has no embargo. Both leak information and are next on the list to fix.
- The "predicted price" uses the most recent row that has a known target, which is *h* days old, rather than today's features.
- Fama-French standard errors are plain OLS; heteroskedasticity- and autocorrelation-robust (Newey-West) errors would be more appropriate. US factors are applied to every ticker, including non-US listings, where regional factors should be used.
- The CAPM and factor expected returns use today's 30-year yield as the risk-free rate. A short-term bill rate matched to each period would be more standard for Sharpe ratios.
- The correlation-window sweep and the Granger tests (which check many lags) involve multiple comparisons without correction, so individual "significant" results should be treated as exploratory.
- The correlation pattern section (cycles, seasonality, regime labels) is descriptive and heuristic, not a statistical forecast.
- Weather variables fall back to synthetic seasonal series when the API is unavailable; the app labels this when it happens.

## Tech stack

Python, Streamlit, pandas, NumPy, scikit-learn, statsmodels, SciPy, Plotly and yfinance. The LSTM path uses TensorFlow/Keras and is skipped gracefully if TensorFlow is not installed.

## Running locally

```bash
git clone https://github.com/hugoVL1013/Finance-Dashboard.git
cd Finance-Dashboard
pip install -r requirements.txt
streamlit run App.py
```

The weather features need a VisualCrossing API key, provided through Streamlit secrets (`VISUALCROSSING_API_KEY` in `.streamlit/secrets.toml`) or an environment variable of the same name. Everything else runs without a key.

## How this was built

I built this with AI coding assistance. The analytical design, the choice of methods, the validation approach and the debugging (including finding the feature-selection leak) are my own, and I hold myself to being able to explain and check every part of it.

*For research and education only. Not investment advice.*
