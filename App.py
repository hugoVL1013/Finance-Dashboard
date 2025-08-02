# Finance Dashboard

import yfinance as yf
import streamlit as st
import plotly.express as px
import plotly.graph_objs as go
from datetime import date
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
import requests
import pickle
import hashlib
import os
import zipfile
import io
import shutil
import time
from scipy.stats import skew, kurtosis
import warnings
warnings.filterwarnings('ignore')
from datetime import timedelta
from sklearn.cluster import KMeans
from scipy.signal import find_peaks

# Deep Learning imports
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    DEEP_LEARNING_AVAILABLE = True
except ImportError:
    DEEP_LEARNING_AVAILABLE = False
    print("⚠️ TensorFlow not available. Install with: pip install tensorflow")


# Page title and config
st.set_page_config(page_title="📊 Finance Dashboard", layout="wide")
st.title(":bar_chart: Finance Dashboard")

# Sidebar navigation
st.sidebar.title(":pushpin: Navigation")
page = st.sidebar.radio("Go to", [":chart_with_upwards_trend: Stock Dashboard", ":bar_chart: Stock Correlation"])

# Shared function to load data
@st.cache_data(ttl=600)
def load_data(ticker_symbol, start, end):
    ticker = yf.Ticker(ticker_symbol)
    data = ticker.history(start=start, end=end)
    if data.empty:
        return None
    data = data.reset_index()
    data['Close'] = pd.to_numeric(data['Close'], errors='coerce')
    return data

# Enhanced weather data loading with caching and batch processing
@st.cache_data(ttl=3600)  # Cache for 1 hour
def test_weather_api():
    """Quick test to see if the VisualCrossing API is responding"""
    test_url = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/London,UK/2024-01-01/2024-01-02?unitGroup=metric&key=PH3EAFWDB2MZU6NLGAHHHKRTX&include=days"
    try:
        resp = requests.get(test_url, timeout=2)
        resp.raise_for_status()
        data = resp.json()
        return True, f"API working - got {len(data.get('days', []))} days"
    except requests.exceptions.Timeout:
        return False, "API timeout (service may be slow)"
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 429:
            return False, "Rate limited - API quota exceeded"
        elif e.response.status_code == 401:
            return False, "Authentication failed - check API key"
        else:
            return False, f"HTTP {e.response.status_code} error"
    except Exception as e:
        return False, f"Error: {str(e)[:50]}"

# Enhanced weather data loading with persistent file caching
import os
import pickle
import hashlib

def get_cache_key(region, data_type, start_date, end_date):
    """Generate a unique cache key for the parameters"""
    key_string = f"{region}_{data_type}_{start_date}_{end_date}"
    return hashlib.md5(key_string.encode()).hexdigest()

def load_from_cache(cache_key):
    """Load data from cache file if it exists"""
    cache_dir = "weather_cache"
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    
    cache_file = os.path.join(cache_dir, f"{cache_key}.pkl")
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except:
            return None
    return None

def save_to_cache(cache_key, data):
    """Save data to cache file"""
    cache_dir = "weather_cache"
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    
    cache_file = os.path.join(cache_dir, f"{cache_key}.pkl")
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)
    except Exception as e:
        st.warning(f"Could not save to cache: {e}")

def clear_weather_cache():
    """Clear all cached weather data"""
    cache_dir = "weather_cache"
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
        st.success("🗑️ Weather cache cleared! Next API calls will fetch fresh data.")
    else:
        st.info("No cache to clear.")

def get_cache_info():
    """Get information about cached files"""
    cache_dir = "weather_cache"
    if not os.path.exists(cache_dir):
        return "No cache directory found."
    
    cache_files = [f for f in os.listdir(cache_dir) if f.endswith('.pkl')]
    if not cache_files:
        return "Cache directory is empty."
    
    total_size = sum(os.path.getsize(os.path.join(cache_dir, f)) for f in cache_files)
    return f"📊 Cache: {len(cache_files)} files, {total_size/1024:.1f} KB total"

def load_weather_data(region, data_type, start_date, end_date):
    """
    Load weather data for a region with persistent file caching
    Data is cached forever until manually cleared
    
    Args:
        region: 'EU', 'US', or 'Colombia'
        data_type: 'temperature' or 'rainfall'
        start_date: Start date
        end_date: End date
    
    Returns:
        pandas.Series with daily averages
    """
    # Check persistent cache first
    cache_key = get_cache_key(region, data_type, start_date, end_date)
    cached_data = load_from_cache(cache_key)
    
    if cached_data is not None:
        st.info(f"📁 Using cached {region} {data_type} data (saved API call!)")
        return cached_data
    
    # If not in cache, make API call
    st.info(f"🌐 Fetching fresh {region} {data_type} data from API...")
    
    city_maps = {
        'EU': ["Paris,FR", "Berlin,DE", "Rome,IT"],  # Reduced cities for faster loading
        'US': ["New York,NY", "Chicago,IL", "Los Angeles,CA"],  # Reduced cities for faster loading
        'Colombia': ["Bogota,CO", "Medellin,CO"]  # Reduced cities for faster loading
    }
    
    if region not in city_maps:
        return pd.Series()
    
    cities = city_maps[region]
    all_data = []
    dates = None
    successful_calls = 0
    
    for i, city in enumerate(cities):
        url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{city}/{start_date}/{end_date}?unitGroup=metric&key=PH3EAFWDB2MZU6NLGAHHHKRTX&include=days"
        
        try:
            # Add small delay to avoid rate limiting
            if i > 0:
                time.sleep(1.0)  # Increased delay to 1 second
            
            # Give the API more time - 15 seconds for slower responses
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            weather_data = resp.json()
            
            # Check if API returned valid data
            if 'days' not in weather_data or len(weather_data.get('days', [])) == 0:
                continue
            
            if data_type == 'temperature':
                city_values = [day.get('temp', np.nan) for day in weather_data.get('days', [])]
            else:  # rainfall
                city_values = [day.get('precip', 0) or 0 for day in weather_data.get('days', [])]  # Handle None as 0
            
            all_data.append(city_values)
            successful_calls += 1
            
            if dates is None:
                dates = [day.get('datetime') for day in weather_data.get('days', [])]
            
            # If we get at least one successful call, we can work with that
            if successful_calls >= 1:
                break  # Don't wait for all cities if one works
                
        except requests.exceptions.Timeout:
            continue
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                st.warning(f"🚫 Rate limited for {region} weather data - API quota exceeded")
            elif e.response.status_code == 401:
                st.error(f"🔑 Authentication failed for {region} weather data - check API key")
            else:
                st.warning(f"🌐 HTTP {e.response.status_code} error for {region} weather data")
            continue
        except requests.exceptions.RequestException as e:
            st.warning(f"🌐 Network error for {region} weather data: {str(e)[:50]}...")
            continue
        except Exception as e:
            st.warning(f"❌ Error loading {region} weather data: {str(e)[:50]}...")
            continue
    
    if not all_data or not dates:
        return pd.Series()
    
    # Calculate regional average (even with just one city if that's all we got)
    data_array = np.array(all_data)
    regional_avg = np.nanmean(data_array, axis=0)
    
    # Create pandas Series with proper date index
    date_index = pd.to_datetime(dates)
    result_series = pd.Series(regional_avg, index=date_index)
    
    # Save to persistent cache
    save_to_cache(cache_key, result_series)
    
    # Show success message with the region and data type
    st.success(f"✅ Successfully loaded {region} {data_type} data from {successful_calls} city/cities (cached for future use)")
    
    return result_series

# Alternative data sources for when VisualCrossing fails
def load_alternative_weather_data(region, data_type, start_date, end_date):
    """
    Fallback to synthetic or alternative data sources with persistent caching
    """
    # Check cache first for synthetic data too
    cache_key = get_cache_key(f"{region}_synthetic", data_type, start_date, end_date)
    cached_data = load_from_cache(cache_key)
    
    if cached_data is not None:
        return cached_data
    
    # Generate synthetic weather data as fallback
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    if data_type == 'temperature':
        # Seasonal temperature pattern
        day_of_year = date_range.dayofyear
        seasonal_temp = 15 + 10 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
        noise = np.random.normal(0, 3, len(date_range))
        synthetic_data = seasonal_temp + noise
    else:  # rainfall
        # Random rainfall with seasonal variation
        seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * date_range.dayofyear / 365)
        synthetic_data = np.random.exponential(2, len(date_range)) * seasonal_factor
    
    # Ensure the date_range is timezone-naive
    date_range = date_range.tz_localize(None) if date_range.tz is not None else date_range
    
    result_series = pd.Series(synthetic_data, index=date_range)
    
    # Cache the synthetic data too
    save_to_cache(cache_key, result_series)
    
    return result_series

# --- PAGE 1: STOCK DASHBOARD ---
if page == ":chart_with_upwards_trend: Stock Dashboard":

    # Risk Disclaimer at the top
    st.warning("**Risk Disclaimer:** This is for educational purposes only. Not financial advice. Always do your own research before making investment decisions.")

    # Sidebar inputs
    st.sidebar.subheader("Stock Input")
    ticker = st.sidebar.text_input("Enter a stock ticker:", "AAPL").upper()
    st.header(f":chart_with_upwards_trend: Stock Price Analysis for {ticker}")
    start_date = st.sidebar.date_input("Start Date", value=date(2020, 1, 1))
    end_date = st.sidebar.date_input("End Date", value=date.today())

    st.sidebar.subheader("Chart Options")
    normalize_with_index = st.sidebar.checkbox("Compare with market index", value=False)

    if start_date >= end_date:
        st.sidebar.error("Start date must be before end date!")
        st.stop()

    with st.spinner(f"Loading data for {ticker}..."):
        data = load_data(ticker, start_date, end_date)

    if data is None or data.empty:
        st.error(":x: No data found. Please check your ticker symbol and date range.")
        st.stop()

    with st.container():
        st.subheader(f":mag: Last 5 Days of {ticker}")
        st.dataframe(data.tail(5), use_container_width=True)

    with st.container():
        st.subheader(f":chart_with_downwards_trend: Price Chart for {ticker}")

        # Determine local market benchmark for all beta calculations and charting
        benchmark_for_chart = "^GSPC"
        benchmark_name_chart = "S&P 500"
        if ticker.endswith(".L"):
            benchmark_for_chart = "^FTSE"
            benchmark_name_chart = "FTSE 100"
        elif ticker.endswith(".PA"):
            benchmark_for_chart = "^FCHI"
            benchmark_name_chart = "CAC 40"
        elif ticker.endswith(".DE"):
            benchmark_for_chart = "^GDAXI"
            benchmark_name_chart = "DAX"
        elif ticker.endswith(".T"):
            benchmark_for_chart = "^N225"
            benchmark_name_chart = "Nikkei 225"
        else:
            info_chart = yf.Ticker(ticker).info
            sector_chart = info_chart.get("sector", "").lower()
            if "technology" in sector_chart:
                benchmark_for_chart = "^IXIC"
                benchmark_name_chart = "NASDAQ"

        benchmark_data = load_data(benchmark_for_chart, start_date, end_date)

        if normalize_with_index:
            if benchmark_data is not None and not benchmark_data.empty:
                stock_normalized = (data['Close'] / data['Close'].iloc[0]) * 100
                benchmark_normalized = (benchmark_data['Close'] / benchmark_data['Close'].iloc[0]) * 100
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=data['Date'], y=stock_normalized, name=ticker, line=dict(color='blue', width=2)))
                fig.add_trace(go.Scatter(x=benchmark_data['Date'], y=benchmark_normalized, name=benchmark_name_chart, line=dict(color='red', width=2)))
                fig.update_layout(title=f"{ticker} vs {benchmark_name_chart} (Normalized to 100)", xaxis_title="Date", yaxis_title="Normalized Price (Base = 100)", template="plotly_white", height=500, hovermode='x unified')
            else:
                fig = px.line(data, x='Date', y='Close', title=f"{ticker} Closing Price Over Time (Benchmark data unavailable)", labels={'Close': 'Price (USD)', 'Date': 'Date'})
                fig.update_traces(line_color='blue', line_width=2)
                fig.update_layout(template="plotly_white", height=500, hovermode='x unified')
        else:
            fig = px.line(data, x='Date', y='Close', title=f"{ticker} Closing Price Over Time", labels={'Close': 'Price (USD)', 'Date': 'Date'})
            fig.update_traces(line_color='blue', line_width=2)
            fig.update_layout(template="plotly_white", height=500, hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)

    # Add Rolling Volatility
    with st.container():
        st.subheader(":repeat: Rolling Volatility (30-day)")
        data['Returns'] = data['Close'].pct_change()
        data['Rolling Volatility'] = data['Returns'].rolling(window=30).std() * np.sqrt(252)
        fig_vol = px.line(data, x='Date', y='Rolling Volatility', title="30-day Rolling Volatility", labels={'Rolling Volatility': 'Volatility'})
        fig_vol.update_layout(template="plotly_white", height=400)
        st.plotly_chart(fig_vol, use_container_width=True)

    # Add Moving Averages
    with st.container():
        st.subheader(":globe_with_meridians: Moving Averages (MA20, MA50, MA200)")
        data['MA20'] = data['Close'].rolling(window=20).mean()
        data['MA50'] = data['Close'].rolling(window=50).mean()
        data['MA200'] = data['Close'].rolling(window=200).mean()
        fig_ma = go.Figure()
        fig_ma.add_trace(go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name='Close'))
        fig_ma.add_trace(go.Scatter(x=data['Date'], y=data['MA20'], mode='lines', name='MA20'))
        fig_ma.add_trace(go.Scatter(x=data['Date'], y=data['MA50'], mode='lines', name='MA50'))
        fig_ma.add_trace(go.Scatter(x=data['Date'], y=data['MA200'], mode='lines', name='MA200'))
        fig_ma.update_layout(title="Moving Averages Comparison", template="plotly_white", height=500)
        st.plotly_chart(fig_ma, use_container_width=True)


    # --- Rolling Beta Calculation and Plot ---
    with st.container():
        st.subheader(":arrows_counterclockwise: 90-Day Rolling Beta")

        # Always use the local market benchmark as determined above
        rolling_benchmark = benchmark_for_chart
        rolling_benchmark_name = benchmark_name_chart

        # Prepare returns for stock and benchmark
        stock_returns = data[['Date', 'Close']].copy()
        stock_returns['Return'] = stock_returns['Close'].pct_change()
        stock_returns = stock_returns.dropna(subset=['Return'])

        # Use load_data for benchmark to ensure date alignment
        rolling_bm_data = load_data(rolling_benchmark, start_date, end_date)
        if rolling_bm_data is not None and not rolling_bm_data.empty:
            benchmark_returns = rolling_bm_data[['Date', 'Close']].copy()
            benchmark_returns['Return'] = benchmark_returns['Close'].pct_change()
            benchmark_returns = benchmark_returns.dropna(subset=['Return'])

            # Merge on Date
            merged_returns = pd.merge(
                stock_returns[['Date', 'Return']],
                benchmark_returns[['Date', 'Return']],
                on='Date',
                suffixes=('_stock', '_benchmark')
            )

            window = 90
            betas = []
            beta_dates = []
            for i in range(window, len(merged_returns) + 1):
                window_data = merged_returns.iloc[i - window:i]
                X = window_data['Return_benchmark'].to_numpy().reshape(-1, 1)
                y = window_data['Return_stock'].to_numpy()
                if len(X) == 0 or len(y) == 0 or np.std(X) == 0:
                    betas.append(np.nan)
                else:
                    reg = LinearRegression().fit(X, y)
                    betas.append(reg.coef_[0])
                beta_dates.append(window_data['Date'].iloc[-1])

            rolling_beta_df = pd.DataFrame({
                'Date': beta_dates,
                'Rolling Beta': betas
            })

            fig_beta = px.line(
                rolling_beta_df,
                x='Date',
                y='Rolling Beta',
                title=f"90-Day Rolling Beta for {ticker.upper()} vs {rolling_benchmark_name}",
                labels={'Rolling Beta': 'Beta', 'Date': 'Date'}
            )
            fig_beta.update_layout(template="plotly_white", height=400, hovermode='x unified')
            st.plotly_chart(fig_beta, use_container_width=True)
        else:
            st.warning(f"Benchmark data for {rolling_benchmark_name} unavailable. Rolling beta cannot be displayed.")

    # Add Rolling Sharpe Ratio
    with st.container():
        st.subheader(":chart_with_upwards_trend: Sharpe Ratio Over Time (90-day Rolling)")
        risk_free_rate_daily = 0.025 / 252  # Daily risk-free rate (2.5% annual)
        data['Rolling Return'] = data['Returns'].rolling(window=90).mean()
        data['Rolling Std'] = data['Returns'].rolling(window=90).std()
        # Correct Sharpe ratio: annualized excess return divided by annualized volatility
        data['Sharpe Ratio'] = ((data['Rolling Return'] - risk_free_rate_daily) * 252) / (data['Rolling Std'] * np.sqrt(252))
        fig_sharpe = px.line(data, x='Date', y='Sharpe Ratio', title="Rolling Sharpe Ratio (90-day)", labels={'Sharpe Ratio': 'Sharpe Ratio'})
        fig_sharpe.update_layout(template="plotly_white", height=400)
        st.plotly_chart(fig_sharpe, use_container_width=True)


    with st.container():
        # Calculate daily log returns using 'Close' prices
        data['Log Return'] = np.log(data['Close'] / data['Close'].shift(1))
        data = data.dropna(subset=['Log Return'])  # Drop NaN for plotting

        # Annualized log return (approx. 250 trading days)
        log_return_yearly = data['Log Return'].mean() * 250

        # Use the same local market benchmark as above
        benchmark = benchmark_for_chart
        benchmark_name = benchmark_name_chart + (" (US top 100 tech firms)" if benchmark == "^IXIC" else " (US top 500 firms)" if benchmark == "^GSPC" else "")

        # Load benchmark data and calculate its yearly log return
        bm = load_data(benchmark, start_date, end_date)
        if bm is not None:
            bm['Log Return'] = np.log(bm['Close'] / bm['Close'].shift(1))
            bm = bm.dropna(subset=['Log Return'])
            bm_log_return_yearly = bm['Log Return'].mean() * 250
        else:
            bm_log_return_yearly = None

        st.subheader(f"📉 Daily Log Returns for {ticker}")
        fig_log = px.line(
            data,
            x='Date',
            y='Log Return',
            title=f"Daily Log Returns of {ticker}",
            labels={'Log Return': 'Log Return', 'Date': 'Date'}
        )
        fig_log.update_layout(template="plotly_white", height=400, hovermode='x unified')
        st.plotly_chart(fig_log, use_container_width=True)

        benchmark_text = f" vs. {benchmark_name} ({bm_log_return_yearly:.4f} or {bm_log_return_yearly*100:.2f}%)" if bm_log_return_yearly else ""

        st.markdown(f"""
        **Annualized Log Return:** {log_return_yearly:.4f} (~{log_return_yearly*100:.2f}% per year){benchmark_text}

        Logarithmic returns are preferred in finance because they are time-additive and handle compounding effects more naturally than simple returns. They also tend to be more normally distributed, which helps in modeling and risk analysis.
        """)

    with st.expander("📊 Stock Statistics", expanded=True):
        col1, col2, col3, col4 = st.columns(4)
        current_price = float(data['Close'].iloc[-1])
        previous_price = float(data['Close'].iloc[-2]) if len(data) > 1 else current_price
        change = current_price - previous_price
        percent_change = (change / previous_price * 100) if previous_price != 0 else 0

        col1.metric("Current Price", f"${current_price:.2f}")
        col2.metric("Daily Change", f"${change:.2f}", delta=f"{percent_change:.2f}%")
        col3.metric("Period High", f"${float(data['Close'].max()):.2f}")
        col4.metric("Period Low", f"${float(data['Close'].min()):.2f}")

    # — Financial Ratios & Fundamental Metrics — 
    st.subheader("🧮 Key Financial Ratios & Fundamentals")
    info = yf.Ticker(ticker).info

    pe = info.get("trailingPE", None)
    eps = info.get("trailingEps", None)
    roe = info.get("returnOnEquity", None)
    quick = info.get("quickRatio", None)
    debt_to_equity = info.get("debtToEquity", None)
    current_ratio = info.get("currentRatio", None)

    fins = yf.Ticker(ticker).quarterly_financials

    if fins is not None and "EBIT" in fins.index and "Total Revenue" in fins.index:
        ebit_margin = fins.loc["EBIT"].iloc[0] / fins.loc["Total Revenue"].iloc[0]
    else:
        ebit_margin = None
    if fins is not None and "Operating Income" in fins.index and "Total Revenue" in fins.index:
        opex_margin = fins.loc["Operating Income"].iloc[0] / fins.loc["Total Revenue"].iloc[0]
    else:
        opex_margin = None
    
    # Interest Coverage Ratio calculation
    if (
        fins is not None
        and "EBIT" in fins.index
        and "Interest Expense" in fins.index
        and fins.loc["EBIT"].iloc[0] is not None
        and fins.loc["Interest Expense"].iloc[0] is not None
        and np.isfinite(fins.loc["EBIT"].iloc[0])
        and np.isfinite(fins.loc["Interest Expense"].iloc[0])
    ):
        interest_expense = fins.loc["Interest Expense"].iloc[0]
        try:
            # Robustly extract scalar value from interest_expense
            if isinstance(interest_expense, pd.Series):
                interest_expense_val = float(interest_expense.iloc[0])
            elif isinstance(interest_expense, np.ndarray):
                interest_expense_val = float(interest_expense.item())
            else:
                interest_expense_val = float(interest_expense)
        except Exception:
            interest_expense_val = None
        if interest_expense_val is not None and interest_expense_val != 0:
            interest_coverage = fins.loc["EBIT"].iloc[0] / abs(interest_expense_val)
        else:
            interest_coverage = None
    else:
        interest_coverage = None

    # Display these ratios nicely in columns (centered)
    _, cols1_1, cols1_2, cols1_3, _ = st.columns([1, 2, 2, 2, 1])
    cols1_1.metric("P/E Ratio", f"{pe:.2f}" if pe is not None and np.isfinite(pe) else "N/A")
    cols1_2.metric("EPS", f"{eps:.2f}" if eps is not None and np.isfinite(eps) else "N/A")
    cols1_3.metric("ROE", f"{roe*100:.2f}%" if roe is not None and np.isfinite(roe) else "N/A")

    _, cols2_1, cols2_2, cols2_3, _ = st.columns([1, 2, 2, 2, 1])
    cols2_1.metric("Quick Ratio", f"{quick:.2f}" if quick is not None and np.isfinite(quick) else "N/A")
    cols2_2.metric("Debt/Equity", f"{debt_to_equity:.2f}" if debt_to_equity is not None and np.isfinite(debt_to_equity) else "N/A")
    cols2_3.metric("EBIT Margin", f"{ebit_margin*100:.2f}%" if ebit_margin is not None and np.isfinite(ebit_margin) else "N/A")

    _, cols3_1, cols3_2, cols3_3, _ = st.columns([1, 2, 2, 2, 1])
    cols3_1.metric("OPEX Margin", f"{opex_margin*100:.2f}%" if opex_margin is not None and np.isfinite(opex_margin) else "N/A")
    cols3_2.metric("Current Ratio", f"{current_ratio:.2f}" if current_ratio is not None and np.isfinite(current_ratio) else "N/A")
    cols3_3.metric("Interest Coverage", f"{interest_coverage:.2f}" if interest_coverage is not None and np.isfinite(interest_coverage) else "N/A")

    # — Beta, Sharpe Ratio & CAPM — 
    st.subheader("📊 Beta, Sharpe Ratio & CAPM Expected Return vs Market")

    # Consistent risk-free rate (2.5% annual)
    risk_free_rate_annual = 0.025  # 2.5% annual
    risk_free_rate_daily = risk_free_rate_annual / 252  # Daily equivalent
    
    # Calculate equity risk premium dynamically from actual market return
    if bm_log_return_yearly is not None:
        equity_risk_premium = bm_log_return_yearly - risk_free_rate_annual
    else:
        equity_risk_premium = 0.05  # fallback to 5% if market data unavailable

    if bm is not None:
        data["LogR"] = np.log(data["Close"] / data["Close"].shift(1))
        bm["LogR"] = np.log(bm["Close"] / bm["Close"].shift(1))
        df = pd.merge(data[["Date", "LogR"]], bm[["Date", "LogR"]], on="Date", suffixes=("", "_bm")).dropna()
        if len(df) < 2 or df["LogR_bm"].isnull().all() or df["LogR"].isnull().all():
            beta = None
            sharpe = None
            capm_return = None
        else:
            X = df["LogR_bm"].to_numpy().reshape(-1, 1)
            y = df["LogR"].to_numpy()
            if len(X) == 0 or len(y) == 0 or np.std(X) == 0:
                beta = None
            else:
                beta = LinearRegression().fit(X, y).coef_[0]
            sharpe = (df["LogR"].mean() / df["LogR"].std()) * np.sqrt(252) if df["LogR"].std() != 0 else None
            capm_return = risk_free_rate_annual + beta * equity_risk_premium if beta is not None else None
    else:
        beta = None
        sharpe = None
        capm_return = None

    # Display metrics (centered)
    _, col1, col2, col3, _ = st.columns([1, 2, 2, 2, 1])
    with col1:
        st.metric("Beta", f"{beta:.2f}" if beta is not None and np.isfinite(beta) else "N/A")
    with col2:
        st.metric("Annualized Sharpe", f"{sharpe:.2f}" if sharpe is not None and np.isfinite(sharpe) else "N/A")
    with col3:
        st.metric("CAPM Expected Return", f"{capm_return*100:.2f}%" if capm_return is not None and np.isfinite(capm_return) else "N/A")
    
    # Add visual separator before Fama French analysis
    st.markdown("---")
    st.markdown("")  # Add some spacing
    
    # --- Fama French Factor Analysis ---
    st.subheader("📊 Fama French Factor Analysis")
    
    from statsmodels.api import OLS, add_constant
    import io, zipfile, requests
    
    # Try to fetch Fama-French 3-factor data
    try:
        # Download Fama-French 3-factor daily CSV from Ken French website
        ff_url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip"
        r = requests.get(ff_url, timeout=30)
        r.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            file_name = z.namelist()[0]
            with z.open(file_name) as f:
                ff_raw = f.read().decode('latin1')
                lines = ff_raw.split('\n')
                
                # Find the line with the column headers
                start_idx = next((i for i, line in enumerate(lines) if "Mkt-RF" in line), None)
                if start_idx is None:
                    raise ValueError("Could not find the header row with Mkt-RF")
                
                # Extract the data section
                data_lines = lines[start_idx:]
                ff_csv = '\n'.join(data_lines)
                
                # Read the CSV data
                ff_factors = pd.read_csv(io.StringIO(ff_csv), engine='python', skip_blank_lines=True)
                date_col = ff_factors.columns[0]
                ff_factors = ff_factors.rename(columns={date_col: 'Date'})
                ff_factors = ff_factors.dropna()
                
                # Convert dates and ensure timezone-naive datetime
                ff_factors['Date'] = pd.to_datetime(ff_factors['Date'].astype(str).str.strip(), format='%Y%m%d')
                ff_factors = ff_factors.set_index('Date')
                # Ensure FF factors index is timezone-naive
                if isinstance(ff_factors.index, pd.DatetimeIndex) and ff_factors.index.tz is not None:
                    ff_factors.index = ff_factors.index.tz_localize(None)
                ff_factors = ff_factors / 100  # Convert percent to decimal
                
                # Filter Fama-French data to match stock data date range
                ff_factors = ff_factors.loc[(ff_factors.index >= pd.to_datetime(start_date)) & 
                                          (ff_factors.index <= pd.to_datetime(end_date))]
                
                # Convert daily returns to monthly returns
                ff_factors_monthly = (1 + ff_factors).resample('ME').prod() - 1
                
                # For stock returns, use the data from the stock dashboard
                stock_data_for_ff = data.copy()
                stock_data_for_ff['Date'] = pd.to_datetime(stock_data_for_ff['Date'])
                # Ensure stock data index is timezone-naive before setting as index
                if hasattr(stock_data_for_ff['Date'], 'dt') and hasattr(stock_data_for_ff['Date'].dt, 'tz') and stock_data_for_ff['Date'].dt.tz is not None:
                    stock_data_for_ff['Date'] = stock_data_for_ff['Date'].dt.tz_localize(None)
                stock_data_for_ff = stock_data_for_ff.set_index('Date')
                # Double-check that the index is timezone-naive
                if isinstance(stock_data_for_ff.index, pd.DatetimeIndex) and stock_data_for_ff.index.tz is not None:
                    stock_data_for_ff.index = stock_data_for_ff.index.tz_localize(None)
                
                # Calculate stock monthly returns
                stock_monthly = stock_data_for_ff['Close'].resample('ME').last()
                stock_returns_monthly = stock_monthly.pct_change().dropna()
                
                # Create monthly dataframe for regression
                monthly_df = pd.DataFrame({
                    ticker: stock_returns_monthly
                })
                
                # Ensure both dataframes have timezone-naive indexes before merging
                if isinstance(monthly_df.index, pd.DatetimeIndex) and monthly_df.index.tz is not None:
                    monthly_df.index = monthly_df.index.tz_localize(None)
                if isinstance(ff_factors_monthly.index, pd.DatetimeIndex) and ff_factors_monthly.index.tz is not None:
                    ff_factors_monthly.index = ff_factors_monthly.index.tz_localize(None)
                
                # Merge with monthly FF factors
                ff_regression_df = monthly_df.merge(ff_factors_monthly[['Mkt-RF', 'SMB', 'HML', 'RF']], 
                                         left_index=True, right_index=True, how='inner')
                
                # Calculate excess returns for the stock
                ff_regression_df[f'{ticker}_excess'] = ff_regression_df[ticker] - ff_regression_df['RF']
                
                ff_predictors = ['Mkt-RF', 'SMB', 'HML']
                ff_regression_df = ff_regression_df.dropna()
                
                if len(ff_regression_df) > 10:  # Ensure we have enough data
                    # Run Fama-French 3-factor regression
                    X = ff_regression_df[ff_predictors]
                    y = ff_regression_df[f'{ticker}_excess']
                    X_const = add_constant(X)
                    ff_model = OLS(y, X_const).fit()
                    
                    # Display regression results
                    st.markdown(f"**{ticker} Excess Returns vs Fama-French Factors**")
                    
                    # Extract coefficients
                    alpha = ff_model.params.get('const', 0)
                    beta_market = ff_model.params.get('Mkt-RF', 0)
                    beta_size = ff_model.params.get('SMB', 0)
                    beta_value = ff_model.params.get('HML', 0)
                    
                    # Display factor loadings
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Alpha (α)", f"{alpha*12*100:.2f}%/year")
                        st.caption("Risk-adjusted excess return")
                    
                    with col2:
                        st.metric("Market Beta (β)", f"{beta_market:.3f}")
                        market_interp = "More volatile" if beta_market > 1 else "Less volatile" if beta_market < 1 else "Same volatility"
                        st.caption(f"{market_interp} than market")
                    
                    with col3:
                        st.metric("Size Factor (SMB)", f"{beta_size:.3f}")
                        size_interp = "Small-cap bias" if beta_size > 0 else "Large-cap bias" if beta_size < 0 else "Size neutral"
                        st.caption(size_interp)
                    
                    with col4:
                        st.metric("Value Factor (HML)", f"{beta_value:.3f}")
                        value_interp = "Value bias" if beta_value > 0 else "Growth bias" if beta_value < 0 else "Value neutral"
                        st.caption(value_interp)
                    
                    # Model fit
                    st.markdown(f"**Model R²:** {ff_model.rsquared:.3f} ({ff_model.rsquared*100:.1f}% of excess returns explained)")
                    
                    # Create time series plot of monthly factor returns
                    fig_ff = go.Figure()
                    for factor in ff_predictors:
                        fig_ff.add_trace(go.Scatter(
                            x=ff_factors_monthly.index,
                            y=ff_factors_monthly[factor],
                            name=factor,
                            mode='lines'
                        ))
                    fig_ff.update_layout(
                        title="Fama-French Monthly Factor Returns Over Time",
                        xaxis_title="Date",
                        yaxis_title="Factor Returns",
                        template="plotly_white",
                        height=400,
                        hovermode='x unified'
                    )
                    st.plotly_chart(fig_ff, use_container_width=True)
                    
                    # Create bar chart of factor coefficients
                    coef_vals = [beta_market, beta_size, beta_value]
                    t_vals = [ff_model.tvalues.get('Mkt-RF', 0), ff_model.tvalues.get('SMB', 0), ff_model.tvalues.get('HML', 0)]
                    p_vals = [ff_model.pvalues.get('Mkt-RF', 0), ff_model.pvalues.get('SMB', 0), ff_model.pvalues.get('HML', 0)]
                    
                    colors = ['red' if pval < 0.05 else 'gray' for pval in p_vals]
                    fig_coef = go.Figure()
                    fig_coef.add_trace(go.Bar(
                        x=['Market (Mkt-RF)', 'Size (SMB)', 'Value (HML)'],
                        y=coef_vals,
                        marker_color=colors,
                        text=[f"{coef:.4f}<br>(t={t:.2f}, p={p:.3f})" 
                              for coef, t, p in zip(coef_vals, t_vals, p_vals)],
                        textposition='auto',
                    ))
                    fig_coef.update_layout(
                        title="Fama-French Factor Coefficients (Monthly Returns)",
                        yaxis_title="Coefficient Value",
                        template="plotly_white",
                        height=400,
                        showlegend=False,
                        shapes=[dict(
                            type='line',
                            yref='y',
                            y0=0,
                            y1=0,
                            xref='paper',
                            x0=0,
                            x1=1,
                            line=dict(
                                color='black',
                                width=1,
                                dash='dash'
                            )
                        )]
                    )
                    st.plotly_chart(fig_coef, use_container_width=True)
                    
                    # Factor interpretations
                    with st.expander("📚 Factor Interpretation Guide", expanded=False):
                        st.markdown(f"""
                        **Alpha (α): {alpha*12*100:.2f}%/year**
                        - {'Positive' if alpha > 0 else 'Negative'}: Stock {'generates' if alpha > 0 else 'loses'} excess returns beyond factor predictions
                        
                        **Market Beta: {beta_market:.3f}**
                        - {market_interp} than the market
                        - {'High systematic risk' if beta_market > 1.2 else 'Low systematic risk' if beta_market < 0.8 else 'Moderate systematic risk'}
                        
                        **Size Factor (SMB): {beta_size:.3f}**
                        - {size_interp}
                        - {'Exposed to small-cap risk premiums' if beta_size > 0.2 else 'Exposed to large-cap stability' if beta_size < -0.2 else 'Size-neutral characteristics'}
                        
                        **Value Factor (HML): {beta_value:.3f}**
                        - {value_interp}
                        - {'Exposed to value risk premiums' if beta_value > 0.2 else 'Exposed to growth momentum' if beta_value < -0.2 else 'Value-neutral characteristics'}
                        """)
                    
                    # Calculate Fama-French expected return
                    mkt_rf_ann = ff_factors_monthly['Mkt-RF'].mean() * 12
                    smb_ann = ff_factors_monthly['SMB'].mean() * 12
                    hml_ann = ff_factors_monthly['HML'].mean() * 12
                    rf_annual = 0.025  # 2.5% risk-free rate
                    
                    ff_exp_return = (
                        rf_annual
                        + beta_market * mkt_rf_ann
                        + beta_size * smb_ann
                        + beta_value * hml_ann
                    )
                    
                    st.markdown(f"""
                    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; padding: 20px; margin: 20px 0; text-align: center; color: white; box-shadow: 0 4px 15px rgba(0,0,0,0.2);'>
                        <h3 style='margin: 0; font-size: 1.2em;'>Fama-French Expected Annual Return</h3>
                        <h1 style='margin: 10px 0; font-size: 2.5em; font-weight: bold;'>{ff_exp_return:.2%}</h1>
                        <p style='margin: 0; opacity: 0.9;'>Based on 3-Factor Model</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                else:
                    st.warning("Insufficient data for Fama-French factor analysis (need at least 10 monthly observations).")
                    
    except Exception as e:
        st.error(f"Could not load Fama-French factors: {str(e)[:100]}...")
        st.info("Fama-French factor analysis requires internet connection to download factor data from Ken French's website.")
    

# --- PAGE 2: STOCK CORRELATION ---
elif page == ":bar_chart: Stock Correlation":
    st.header(":bar_chart: Stock Correlation Analysis")

    # User input for a single stock ticker (moved to sidebar)
    st.sidebar.subheader("Stock Input")
    corr_ticker = st.sidebar.text_input("Enter a stock ticker for correlation analysis:", "AAPL").upper()
    corr_start_date = st.sidebar.date_input("Start Date", value=date(2020, 1, 1), key="corr_start")
    corr_end_date = st.sidebar.date_input("End Date", value=date.today(), key="corr_end")

    if corr_start_date >= corr_end_date:
        st.error("Start date must be before end date!")
        st.stop()

    with st.spinner(f"Loading data for {corr_ticker}..."):
        corr_data = load_data(corr_ticker, corr_start_date, corr_end_date)

    if corr_data is None or corr_data.empty:
        st.error(":x: No data found. Please check your ticker symbol and date range.")
        st.stop()

    # Cache management section
    with st.expander("🗂️ Weather Data Cache Management", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.info(get_cache_info())
        
        with col2:
            if st.button("🗑️ Clear Cache"):
                clear_weather_cache()
        
        with col3:
            st.caption("💡 Tip: Cache saves API costs by storing data permanently")

    # List of variables for correlation analysis
    st.subheader("Select Variables for Correlation Analysis")
    available_vars = [
        "SPY (S&P 500 ETF)",
        "GLD (Gold ETF)",
        "USO (Oil ETF)",
        "UNG (Natural Gas ETF)",
        "DBC (Commodities ETF)",
        "TLT (20+ Year Treasury Bond ETF)",
        "IEF (10 Year Treasury Bond ETF)",
        "VIX (Volatility Index)",
        "Global Oil Price (Brent)",
        "Crude Oil Futures (CL=F)",
        "USD Index (DXY)",
        "Bitcoin Price (BTC-USD)",
        "Coffee Futures (KC=F)",
        "EU Daily Rainfall Avg (Historical)",
        "US Daily Rainfall Avg (Historical)",
        "EU Daily Temperature Avg (Historical)",
        "US Daily Temperature Avg (Historical)",
        "Colombia Average Daily Temperature (Historical)",
        "Colombia Average Daily Rainfall (Historical)",    
    ]
    
    # Custom variable input section
    with st.expander("➕ Add Custom Variable (Any Ticker Symbol)", expanded=False):
        st.markdown("Enter any stock ticker, ETF, commodity, or financial instrument symbol:")
        col1, col2 = st.columns([3, 1])
        
        with col1:
            custom_ticker = st.text_input(
                "Custom Ticker Symbol:", 
                placeholder="e.g., TSLA, MSFT, ^TNX, EURUSD=X",
                help="Examples: TSLA (Tesla), MSFT (Microsoft), ^TNX (10-Year Treasury), EURUSD=X (EUR/USD)"
            ).upper().strip()
        
        with col2:
            if st.button("🔍 Test & Add", help="Check if ticker data is available and add to list"):
                if custom_ticker:
                    with st.spinner(f"Testing {custom_ticker}..."):
                        # Test if we can download data for this ticker
                        test_data = load_data(custom_ticker, corr_start_date, corr_end_date)
                        
                        if test_data is not None and not test_data.empty:
                            custom_var_name = f"{custom_ticker} (Custom)"
                            if custom_var_name not in available_vars:
                                available_vars.append(custom_var_name)
                                st.success(f"✅ {custom_ticker} added successfully! Found {len(test_data)} data points.")
                                # Store in session state to persist across reruns
                                if 'custom_vars' not in st.session_state:
                                    st.session_state.custom_vars = []
                                if custom_var_name not in st.session_state.custom_vars:
                                    st.session_state.custom_vars.append(custom_var_name)
                            else:
                                st.info(f"ℹ️ {custom_ticker} is already in the list!")
                        else:
                            st.error(f"❌ Could not find data for {custom_ticker}. Please check the symbol and try again.")
                else:
                    st.warning("Please enter a ticker symbol first.")
    
    # Add any previously added custom variables from session state
    if 'custom_vars' in st.session_state:
        for custom_var in st.session_state.custom_vars:
            if custom_var not in available_vars:
                available_vars.append(custom_var)
    
    # Clear custom variables button
    if 'custom_vars' in st.session_state and st.session_state.custom_vars:
        if st.button("🗑️ Clear All Custom Variables"):
            st.session_state.custom_vars = []
            st.success("Custom variables cleared! Page will refresh.")
            st.rerun()
    
    selected_vars = st.multiselect("Choose variables to check for correlation:", available_vars, default=["SPY (S&P 500 ETF)"])

    if not selected_vars:
        st.warning("Please select at least one variable for correlation analysis from the list above.")
        st.stop()

    # Download selected ETF/commodity data and use log returns
    etf_map = {
        "SPY (S&P 500 ETF)": "SPY",
        "GLD (Gold ETF)": "GLD",
        "USO (Oil ETF)": "USO",
        "UNG (Natural Gas ETF)": "UNG",
        "DBC (Commodities ETF)": "DBC",
        "TLT (20+ Year Treasury Bond ETF)": "TLT",
        "IEF (10 Year Treasury Bond ETF)": "IEF",
        "VIX (Volatility Index)": "^VIX",
        "Global Oil Price (Brent)": "BZ=F",
        "Crude Oil Futures (CL=F)": "CL=F",
        "USD Index (DXY)": "DX-Y.NYB",
        "Bitcoin Price (BTC-USD)": "BTC-USD",
        "Coffee Futures (KC=F)": "KC=F"
    }
    # Calculate log returns for the main ticker
    corr_data = corr_data.set_index('Date')
    
    # Ensure the correlation data index is timezone-naive
    if isinstance(corr_data.index, pd.DatetimeIndex) and corr_data.index.tz is not None:
        corr_data.index = corr_data.index.tz_localize(None)
    
    corr_df = pd.DataFrame()
    corr_df[corr_ticker] = np.log(corr_data['Close'] / corr_data['Close'].shift(1))

    # Progress tracking for API calls
    weather_vars = [var for var in selected_vars if 'Temperature' in var or 'Rainfall' in var or 'Air Quality' in var]
    if weather_vars:
        # Test API connectivity first
        api_working, api_status = test_weather_api()
        if api_working:
            st.info(f"🌤️ Loading weather data for {len(weather_vars)} variables. API test successful: {api_status}")
        else:
            st.warning(f"⚠️ API connectivity issue: {api_status}. Will use synthetic data for weather variables.")
        
        progress_bar = st.progress(0)
        progress_text = st.empty()

    for i, var in enumerate(selected_vars):
        # Update progress for weather variables
        if weather_vars and var in weather_vars:
            progress = (weather_vars.index(var)) / len(weather_vars)
            progress_bar.progress(progress)
            progress_text.text(f"Loading {var}... ({weather_vars.index(var) + 1}/{len(weather_vars)})")
            
        if var == "Colombia Average Daily Temperature (Historical)":
            with st.spinner(f"Fetching Colombia temperature data..."):
                weather_series = load_weather_data('Colombia', 'temperature', corr_start_date, corr_end_date)
            if not weather_series.empty:
                # Align with correlation data index
                aligned_series = weather_series.reindex(corr_data.index, method='nearest')
                if np.any(np.isfinite(aligned_series)):
                    corr_df[var] = aligned_series
                    st.info(f"✅ **{var}**: Using real weather data from API")
                else:
                    st.warning(f"⚠️ **{var}**: No valid API data. Using synthetic seasonal temperature data.")
                    fallback_series = load_alternative_weather_data('Colombia', 'temperature', corr_start_date, corr_end_date)
                    corr_df[var] = fallback_series.reindex(corr_data.index, method='nearest')
            else:
                st.warning(f"⚠️ **{var}**: API unavailable. Using synthetic seasonal temperature data.")
                fallback_series = load_alternative_weather_data('Colombia', 'temperature', corr_start_date, corr_end_date)
                corr_df[var] = fallback_series.reindex(corr_data.index, method='nearest')
                
        elif var == "Colombia Average Daily Rainfall (Historical)":
            with st.spinner(f"Fetching Colombia rainfall data..."):
                weather_series = load_weather_data('Colombia', 'rainfall', corr_start_date, corr_end_date)
            if not weather_series.empty:
                # Align with correlation data index
                aligned_series = weather_series.reindex(corr_data.index, method='nearest')
                if np.any(np.isfinite(aligned_series)):
                    corr_df[var] = aligned_series
                    st.info(f"✅ **{var}**: Using real weather data from API")
                else:
                    st.warning(f"⚠️ **{var}**: No valid API data. Using synthetic seasonal rainfall data.")
                    fallback_series = load_alternative_weather_data('Colombia', 'rainfall', corr_start_date, corr_end_date)
                    corr_df[var] = fallback_series.reindex(corr_data.index, method='nearest')
            else:
                st.warning(f"⚠️ **{var}**: API unavailable. Using synthetic seasonal rainfall data.")
                fallback_series = load_alternative_weather_data('Colombia', 'rainfall', corr_start_date, corr_end_date)
                corr_df[var] = fallback_series.reindex(corr_data.index, method='nearest')
                
        elif var in etf_map:
            etf_ticker = etf_map[var]
            etf_data = load_data(etf_ticker, corr_start_date, corr_end_date)
            if etf_data is not None and not etf_data.empty:
                etf_data = etf_data.set_index('Date')
                
                # Ensure ETF data index is timezone-naive
                if isinstance(etf_data.index, pd.DatetimeIndex) and etf_data.index.tz is not None:
                    etf_data.index = etf_data.index.tz_localize(None)
                
                etf_series = np.log(etf_data['Close'] / etf_data['Close'].shift(1))
                if np.any(np.isfinite(etf_series)):
                    corr_df[var] = etf_series
                    st.info(f"✅ **{var}**: Using real financial data from Yahoo Finance")
                else:
                    st.warning(f"❌ **{var}**: No valid financial data available")
            else:
                st.warning(f"❌ **{var}**: Financial data unavailable from Yahoo Finance")
                
        elif var == "Electricity Demand (Synthetic)":
            np.random.seed(42)
            synthetic = np.cumsum(np.random.normal(0, 1, len(corr_df))) + 100
            synthetic_series = np.log(synthetic / pd.Series(synthetic).shift(1))
            if np.any(np.isfinite(synthetic_series)):
                corr_df[var] = synthetic_series
                st.info(f"🔬 **{var}**: Using synthetic data (random walk simulation)")
                
        elif var == "Gas Demand (Synthetic)":
            np.random.seed(43)
            synthetic = np.cumsum(np.random.normal(0, 1, len(corr_df))) + 50
            synthetic_series = np.log(synthetic / pd.Series(synthetic).shift(1))
            if np.any(np.isfinite(synthetic_series)):
                corr_df[var] = synthetic_series
                st.info(f"🔬 **{var}**: Using synthetic data (random walk simulation)")
                
        elif var == "EU Daily Rainfall Avg (Historical)":
            with st.spinner(f"Fetching EU rainfall data..."):
                weather_series = load_weather_data('EU', 'rainfall', corr_start_date, corr_end_date)
            if not weather_series.empty:
                aligned_series = weather_series.reindex(corr_data.index, method='nearest')
                if np.any(np.isfinite(aligned_series)):
                    corr_df[var] = aligned_series
                    st.info(f"✅ **{var}**: Using real weather data from API")
                else:
                    st.warning(f"⚠️ **{var}**: No valid API data. Using synthetic seasonal rainfall data.")
                    fallback_series = load_alternative_weather_data('EU', 'rainfall', corr_start_date, corr_end_date)
                    corr_df[var] = fallback_series.reindex(corr_data.index, method='nearest')
            else:
                st.warning(f"⚠️ **{var}**: API unavailable. Using synthetic seasonal rainfall data.")
                fallback_series = load_alternative_weather_data('EU', 'rainfall', corr_start_date, corr_end_date)
                corr_df[var] = fallback_series.reindex(corr_data.index, method='nearest')
                
        elif var == "US Daily Rainfall Avg (Historical)":
            with st.spinner(f"Fetching US rainfall data..."):
                weather_series = load_weather_data('US', 'rainfall', corr_start_date, corr_end_date)
            if not weather_series.empty:
                aligned_series = weather_series.reindex(corr_data.index, method='nearest')
                if np.any(np.isfinite(aligned_series)):
                    corr_df[var] = aligned_series
                    st.info(f"✅ **{var}**: Using real weather data from API")
                else:
                    st.warning(f"⚠️ **{var}**: No valid API data. Using synthetic seasonal rainfall data.")
                    fallback_series = load_alternative_weather_data('US', 'rainfall', corr_start_date, corr_end_date)
                    corr_df[var] = fallback_series.reindex(corr_data.index, method='nearest')
            else:
                st.warning(f"⚠️ **{var}**: API unavailable. Using synthetic seasonal rainfall data.")
                fallback_series = load_alternative_weather_data('US', 'rainfall', corr_start_date, corr_end_date)
                corr_df[var] = fallback_series.reindex(corr_data.index, method='nearest')
                
        elif var == "EU Daily Temperature Avg (Historical)":
            with st.spinner(f"Fetching EU temperature data..."):
                weather_series = load_weather_data('EU', 'temperature', corr_start_date, corr_end_date)
            if not weather_series.empty:
                aligned_series = weather_series.reindex(corr_data.index, method='nearest')
                if np.any(np.isfinite(aligned_series)):
                    corr_df[var] = aligned_series
                    st.info(f"✅ **{var}**: Using real weather data from API")
                else:
                    st.warning(f"⚠️ **{var}**: No valid API data. Using synthetic seasonal temperature data.")
                    fallback_series = load_alternative_weather_data('EU', 'temperature', corr_start_date, corr_end_date)
                    corr_df[var] = fallback_series.reindex(corr_data.index, method='nearest')
            else:
                st.warning(f"⚠️ **{var}**: API unavailable. Using synthetic seasonal temperature data.")
                fallback_series = load_alternative_weather_data('EU', 'temperature', corr_start_date, corr_end_date)
                corr_df[var] = fallback_series.reindex(corr_data.index, method='nearest')
                
        elif var == "US Daily Temperature Avg (Historical)":
            with st.spinner(f"Fetching US temperature data..."):
                weather_series = load_weather_data('US', 'temperature', corr_start_date, corr_end_date)
            if not weather_series.empty:
                aligned_series = weather_series.reindex(corr_data.index, method='nearest')
                if np.any(np.isfinite(aligned_series)):
                    corr_df[var] = aligned_series
                    st.info(f"✅ **{var}**: Using real weather data from API")
                else:
                    st.warning(f"⚠️ **{var}**: No valid API data. Using synthetic seasonal temperature data.")
                    fallback_series = load_alternative_weather_data('US', 'temperature', corr_start_date, corr_end_date)
                    corr_df[var] = fallback_series.reindex(corr_data.index, method='nearest')
            else:
                st.warning(f"⚠️ **{var}**: API unavailable. Using synthetic seasonal temperature data.")
                fallback_series = load_alternative_weather_data('US', 'temperature', corr_start_date, corr_end_date)
                corr_df[var] = fallback_series.reindex(corr_data.index, method='nearest')
                
        elif var == "EU Air Quality Daily Avg (Historical)":
            st.warning(f"⚠️ **{var}**: VisualCrossing API doesn't provide air quality data. Using synthetic proxy based on temperature/rainfall correlation.")
            # Create synthetic air quality data based on temperature and rainfall
            try:
                temp_data = load_weather_data('EU', 'temperature', corr_start_date, corr_end_date)
                rain_data = load_weather_data('EU', 'rainfall', corr_start_date, corr_end_date)
                if not temp_data.empty and not rain_data.empty:
                    # Simple air quality proxy: higher temp = worse air, more rain = better air
                    temp_aligned = temp_data.reindex(corr_data.index, method='nearest')
                    rain_aligned = rain_data.reindex(corr_data.index, method='nearest')
                    synthetic_aqi = 50 + (temp_aligned - temp_aligned.mean()) * 2 - rain_aligned * 0.5
                    corr_df[var] = synthetic_aqi
                    st.info(f"🔬 **{var}**: Generated synthetic air quality index based on weather patterns")
                else:
                    corr_df[var] = np.nan
                    st.error(f"❌ **{var}**: Cannot generate synthetic data - weather APIs unavailable")
            except:
                corr_df[var] = np.nan
                st.error(f"❌ **{var}**: Error generating synthetic data")
                
        elif var == "US Air Quality Daily Avg (Historical)":
            st.warning(f"⚠️ **{var}**: VisualCrossing API doesn't provide air quality data. Using synthetic proxy based on temperature/rainfall correlation.")
            try:
                temp_data = load_weather_data('US', 'temperature', corr_start_date, corr_end_date)
                rain_data = load_weather_data('US', 'rainfall', corr_start_date, corr_end_date)
                if not temp_data.empty and not rain_data.empty:
                    temp_aligned = temp_data.reindex(corr_data.index, method='nearest')
                    rain_aligned = rain_data.reindex(corr_data.index, method='nearest')
                    synthetic_aqi = 50 + (temp_aligned - temp_aligned.mean()) * 2 - rain_aligned * 0.5
                    corr_df[var] = synthetic_aqi
                    st.info(f"🔬 **{var}**: Generated synthetic air quality index based on weather patterns")
                else:
                    corr_df[var] = np.nan
                    st.error(f"❌ **{var}**: Cannot generate synthetic data - weather APIs unavailable")
            except:
                corr_df[var] = np.nan
                st.error(f"❌ **{var}**: Error generating synthetic data")
        
        # Handle custom variables (any ticker symbols added by user)
        elif "(Custom)" in var:
            # Extract ticker symbol from the variable name
            custom_ticker = var.replace(" (Custom)", "").strip()
            custom_data = load_data(custom_ticker, corr_start_date, corr_end_date)
            
            if custom_data is not None and not custom_data.empty:
                custom_data = custom_data.set_index('Date')
                
                # Ensure custom data index is timezone-naive
                if isinstance(custom_data.index, pd.DatetimeIndex) and custom_data.index.tz is not None:
                    custom_data.index = custom_data.index.tz_localize(None)
                
                custom_series = np.log(custom_data['Close'] / custom_data['Close'].shift(1))
                if np.any(np.isfinite(custom_series)):
                    corr_df[var] = custom_series
                    st.info(f"✅ **{var}**: Using real financial data for {custom_ticker}")
                else:
                    st.warning(f"❌ **{var}**: No valid data for {custom_ticker}")
                    corr_df[var] = np.nan
            else:
                st.warning(f"❌ **{var}**: Could not load data for {custom_ticker}")
                corr_df[var] = np.nan
        
        # Update progress for weather variables
        if weather_vars and var in weather_vars:
            progress = (weather_vars.index(var) + 1) / len(weather_vars)
            progress_bar.progress(progress)
            progress_text.text(f"Loaded {var} ({weather_vars.index(var) + 1}/{len(weather_vars)})")

    # Clean up progress indicators
    if weather_vars:
        progress_bar.empty()
        progress_text.empty()
        st.success(f"Successfully loaded weather data for {len(weather_vars)} variables!")

    # Ensure all selected variables are present as columns, fill missing with NaN
    for var in selected_vars:
        if var not in corr_df.columns:
            corr_df[var] = np.nan
    # Only drop rows where the main ticker's log return is missing
    corr_df = corr_df.dropna(subset=[corr_ticker])

    # --- Time Series Stationarity Check ---
    st.subheader("Time Series Stationarity (ADF Test)")
    from statsmodels.tsa.stattools import adfuller
    stationarity_results = {}
    for col in corr_df.columns:
        col_data = corr_df[col].dropna()
        finite_vals = col_data[np.isfinite(col_data)]
        unique_vals = np.unique(finite_vals)
        if len(col_data) == 0:
            stationarity_results[col] = {
                'ADF Statistic': None,
                'p-value': None,
                'Stationary': 'No data'
            }
            continue
        if len(unique_vals) <= 1:
            stationarity_results[col] = {
                'ADF Statistic': None,
                'p-value': None,
                'Stationary': 'Constant series'
            }
            continue
        try:
            adf = adfuller(col_data)
            stationarity_results[col] = {
                'ADF Statistic': adf[0],
                'p-value': adf[1],
                'Stationary': 'Yes' if adf[1] < 0.05 else 'No'
            }
        except Exception as e:
            stationarity_results[col] = {
                'ADF Statistic': None,
                'p-value': None,
                'Stationary': f'Error: {e}'
            }
    st.dataframe(pd.DataFrame(stationarity_results).T)


    # --- Rolling Correlation Visualization ---
    st.subheader("Rolling Correlation with Custom Window")
    
    # Auto-optimization section
    st.markdown("#### 🎯 Automatic Window Optimization (20-200 days)")
    
    if st.button("🔍 Find Optimal Rolling Windows", help="Analyzes correlation strength for windows from 20 to 200 days"):
        with st.spinner("Analyzing optimal rolling windows for all variables..."):
            optimal_results = {}
            window_range = range(20, 201)
            
            # Progress tracking
            progress_bar = st.progress(0)
            progress_text = st.empty()
            
            for i, var in enumerate(selected_vars):
                if var in corr_df.columns and var != corr_ticker:
                    progress_text.text(f"Analyzing {var}... ({i+1}/{len(selected_vars)})")
                    
                    avg_correlations = []
                    max_abs_correlations = []
                    
                    for window_size in window_range:
                        rolling_corr = corr_df[corr_ticker].rolling(window_size).corr(corr_df[var])
                        rolling_corr_clean = rolling_corr.dropna()
                        
                        if len(rolling_corr_clean) > 0:
                            avg_corr = rolling_corr_clean.mean()
                            max_abs_corr = rolling_corr_clean.abs().max()
                            avg_correlations.append(avg_corr)
                            max_abs_correlations.append(max_abs_corr)
                        else:
                            avg_correlations.append(0)
                            max_abs_correlations.append(0)
                    
                    # Find optimal windows
                    best_avg_idx = np.argmax(np.abs(avg_correlations))
                    best_max_idx = np.argmax(max_abs_correlations)
                    
                    optimal_results[var] = {
                        'avg_correlations': avg_correlations,
                        'max_correlations': max_abs_correlations,
                        'best_avg_window': best_avg_idx + 20,  # Add 20 because window_range starts at 20
                        'best_avg_corr': avg_correlations[best_avg_idx],
                        'best_max_window': best_max_idx + 20,  # Add 20 because window_range starts at 20
                        'best_max_corr': max_abs_correlations[best_max_idx]
                    }
                    
                    progress_bar.progress((i + 1) / len(selected_vars))
            
            progress_bar.empty()
            progress_text.empty()
        
        # Display optimization results
        st.success("✅ Optimization complete!")
        
        # Create summary table
        summary_data = []
        for var, results in optimal_results.items():
            summary_data.append({
                'Variable': var,
                'Best Avg Window': results['best_avg_window'],
                'Avg Correlation': f"{results['best_avg_corr']:.4f}",
                'Best Max Window': results['best_max_window'], 
                'Max |Correlation|': f"{results['best_max_corr']:.4f}"
            })
        
        summary_df = pd.DataFrame(summary_data)
        st.markdown("#### 📊 Optimal Window Summary")
        st.dataframe(summary_df, use_container_width=True)
        
        # Create visualization of correlation vs window size
        fig_opt = go.Figure()
        
        for var, results in optimal_results.items():
            fig_opt.add_trace(go.Scatter(
                x=list(window_range),
                y=results['avg_correlations'],
                mode='lines',
                name=f"{var} (Avg)",
                line=dict(width=2)
            ))
            
            # Add marker for optimal point
            fig_opt.add_trace(go.Scatter(
                x=[results['best_avg_window']],
                y=[results['best_avg_corr']],
                mode='markers',
                name=f"{var} (Optimal: {results['best_avg_window']}d)",
                marker=dict(size=10, symbol='star')
            ))
        
        fig_opt.update_layout(
            title="Average Correlation vs Rolling Window Size (20-200 days)",
            xaxis_title="Window Size (days)",
            yaxis_title="Average Correlation",
            template="plotly_white",
            height=500,
            hovermode='x unified'
        )
        st.plotly_chart(fig_opt, use_container_width=True)
        
        # Create heatmap for maximum correlations
        fig_heat = go.Figure()
        
        for var, results in optimal_results.items():
            fig_heat.add_trace(go.Scatter(
                x=list(window_range),
                y=results['max_correlations'],
                mode='lines',
                name=f"{var} (Max |Corr|)",
                line=dict(width=2, dash='dash')
            ))
        
        fig_heat.update_layout(
            title="Maximum |Correlation| vs Rolling Window Size",
            xaxis_title="Window Size (days)", 
            yaxis_title="Maximum |Correlation|",
            template="plotly_white",
            height=400,
            hovermode='x unified'
        )
        st.plotly_chart(fig_heat, use_container_width=True)
    
    st.markdown("#### 📈 Custom Rolling Correlation")
    window = st.number_input("Select rolling window size (days):", min_value=5, max_value=250, value=60, step=1, help="Number of days for rolling correlation window.")
    
    import plotly.graph_objs as go
    fig = go.Figure()
    correlation_stats = []
    
    def robust_rolling_corr(series1, series2, window, min_periods=None):
        """
        Calculate rolling correlation that handles NaN values more gracefully.
        Only requires a minimum number of valid pairs within the window.
        """
        if min_periods is None:
            min_periods = max(5, window // 2)  # Require at least half the window or 5 points
        
        result = pd.Series(index=series1.index, dtype=float)
        
        for i in range(len(series1)):
            # Define window bounds
            start_idx = max(0, i - window + 1)
            end_idx = i + 1
            
            # Get window data
            window_s1 = series1.iloc[start_idx:end_idx]
            window_s2 = series2.iloc[start_idx:end_idx]
            
            # Find valid pairs (both series have data)
            valid_mask = (~window_s1.isna()) & (~window_s2.isna())
            valid_s1 = window_s1[valid_mask]
            valid_s2 = window_s2[valid_mask]
            
            # Calculate correlation if enough valid pairs
            if len(valid_s1) >= min_periods:
                try:
                    corr_val = valid_s1.corr(valid_s2)
                    if pd.notna(corr_val):
                        result.iloc[i] = corr_val
                except:
                    pass  # Keep as NaN
        
        return result
    
    # Enhanced ML Functions from ML_Stock_Predictor
    
    def calculate_technical_indicators(data, periods=[5, 10, 20, 50]):
        """Calculate comprehensive technical indicators"""
        df = data.copy()
        
        # Ensure we have the required columns
        if 'Close' not in df.columns:
            raise ValueError("Data must contain 'Close' column")
        
        # Extract close as a proper Series
        close = df['Close'].squeeze()  # Ensure it's a Series, not DataFrame
        
        # Returns
        df['returns'] = close.pct_change()
        df['log_returns'] = np.log(close / close.shift(1))
        
        # Moving averages
        for period in periods:
            # Calculate moving average
            ma_series = close.rolling(window=period).mean()
            df[f'ma_{period}'] = ma_series
            
            # Calculate ratio with explicit Series handling
            ma_values = ma_series.squeeze()  # Ensure Series
            
            # Safe division with proper error handling
            with np.errstate(divide='ignore', invalid='ignore'):
                ratio_values = np.where(
                    (ma_values != 0) & (~np.isnan(ma_values)), 
                    close / ma_values, 
                    1.0
                )
            
            # Ensure the result is a Series with the correct index
            ratio_series = pd.Series(ratio_values, index=close.index, name=f'ma_{period}_ratio')
            df[f'ma_{period}_ratio'] = ratio_series
        
        # Volatility measures
        for period in [5, 10, 20]:
            df[f'volatility_{period}'] = df['returns'].rolling(window=period).std()
        
        # RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        
        # Calculate RSI with safe division
        with np.errstate(divide='ignore', invalid='ignore'):
            rs_values = np.where(
                (loss != 0) & (~np.isnan(loss)), 
                gain / loss, 
                0.0
            )
        rs = pd.Series(rs_values, index=close.index, name='rs')
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        bb_period = 20
        df['bb_middle'] = close.rolling(window=bb_period).mean()
        bb_std = close.rolling(window=bb_period).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        
        # Calculate Bollinger Band position with safe division
        band_width = df['bb_upper'] - df['bb_lower']
        with np.errstate(divide='ignore', invalid='ignore'):
            bb_position_values = np.where(
                (band_width != 0) & (~np.isnan(band_width)), 
                (close - df['bb_lower']) / band_width, 
                0.5
            )
        df['bb_position'] = pd.Series(bb_position_values, index=close.index, name='bb_position')
        
        # MACD
        exp1 = close.ewm(span=12).mean()
        exp2 = close.ewm(span=26).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Price momentum
        for period in [5, 10, 20]:
            df[f'momentum_{period}'] = close / close.shift(period) - 1
        
        # Volume indicators (if available)
        if 'Volume' in df.columns:
            volume_col = df['Volume'].squeeze()  # Ensure Series
            for period in [5, 10, 20]:
                vol_ma = volume_col.rolling(window=period).mean()
                df[f'volume_ma_{period}'] = vol_ma
                
                # Calculate volume ratio with safe division
                with np.errstate(divide='ignore', invalid='ignore'):
                    volume_ratio_values = np.where(
                        (vol_ma != 0) & (~np.isnan(vol_ma)), 
                        volume_col / vol_ma, 
                        1.0
                    )
                df[f'volume_ratio_{period}'] = pd.Series(volume_ratio_values, index=close.index, name=f'volume_ratio_{period}')
        
        # Support and resistance levels
        if 'High' in df.columns and 'Low' in df.columns:
            df['high_20'] = df['High'].rolling(window=20).max()
            df['low_20'] = df['Low'].rolling(window=20).min()
            
            # Calculate price position with safe division
            price_range = df['high_20'] - df['low_20']
            with np.errstate(divide='ignore', invalid='ignore'):
                price_position_values = np.where(
                    (price_range != 0) & (~np.isnan(price_range)), 
                    (close - df['low_20']) / price_range, 
                    0.5
                )
            df['price_position'] = pd.Series(price_position_values, index=close.index, name='price_position')
        
        return df

    # Rolling correlation plotting section
    for var in selected_vars:
        if var in corr_df.columns and var != corr_ticker:
            # Use robust rolling correlation
            rolling_corr = robust_rolling_corr(
                corr_df[corr_ticker], 
                corr_df[var], 
                int(window),
                min_periods=max(5, int(window) // 2)
            )
            rolling_corr_clean = rolling_corr.dropna()
            
            # Track statistics for each variable
            correlation_stats.append({
                'Variable': var,
                'Total Points': len(rolling_corr),
                'Valid Points': len(rolling_corr_clean),
                'NaN Points': len(rolling_corr) - len(rolling_corr_clean),
                'Date Range': f"{str(rolling_corr_clean.index[0])[:10] if len(rolling_corr_clean) > 0 else 'No data'} to {str(rolling_corr_clean.index[-1])[:10] if len(rolling_corr_clean) > 0 else 'No data'}"
            })
            
            if len(rolling_corr_clean) > 0:
                fig.add_trace(go.Scatter(
                    x=rolling_corr.index,
                    y=rolling_corr,
                    mode='lines',
                    name=f"{corr_ticker} vs {var}",
                    connectgaps=False  # Don't connect across NaN gaps
                ))
            else:
                st.warning(f"No valid correlation data for {var} with {int(window)}-day window")
    
    fig.update_layout(
        title=f"Rolling Correlation (window={int(window)})",
        xaxis_title="Date",
        yaxis_title="Correlation",
        template="plotly_white",
        height=400,
        legend_title="Variable"
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Normalized Price Comparison ---
    st.markdown("#### 📊 Price Comparison Chart")
    st.markdown("Compare your stock (normalized to 100) with selected variables (actual prices)")
    
    # Create dual-axis price chart
    fig_prices = go.Figure()
    
    # Add the main ticker (normalized from original close prices) - LEFT Y-AXIS
    if not corr_data.empty:
        ticker_normalized = (corr_data['Close'] / corr_data['Close'].iloc[0]) * 100
        fig_prices.add_trace(go.Scatter(
            x=corr_data.index,
            y=ticker_normalized,
            mode='lines',
            name=f"{corr_ticker} (Normalized)",
            line=dict(width=3, color='#1f77b4'),
            yaxis='y'
        ))
    
    # Add selected variables (actual prices) - RIGHT Y-AXIS
    for var in selected_vars:
        if var in etf_map:
            etf_ticker = etf_map[var]
            etf_data = load_data(etf_ticker, corr_start_date, corr_end_date)
            if etf_data is not None and not etf_data.empty:
                etf_data = etf_data.set_index('Date')
                
                # Ensure timezone consistency
                if isinstance(etf_data.index, pd.DatetimeIndex) and etf_data.index.tz is not None:
                    etf_data.index = etf_data.index.tz_localize(None)
                
                # Use actual prices (not normalized)
                if len(etf_data) > 0:
                    fig_prices.add_trace(go.Scatter(
                        x=etf_data.index,
                        y=etf_data['Close'],
                        mode='lines',
                        name=var,
                        line=dict(width=2),
                        yaxis='y2'
                    ))
    
    fig_prices.update_layout(
        title=f"Price Comparison: {corr_ticker} (Normalized) vs Selected Variables (Actual Prices)",
        xaxis_title="Date",
        template="plotly_white",
        height=500,
        hovermode='x unified',
        legend=dict(
            title="Assets",
            orientation="h",
            yanchor="top",
            y=-0.2,
            xanchor="center",
            x=0.5
        ),
        # Left Y-axis for normalized ticker
        yaxis=dict(
            title=f"{corr_ticker} Normalized Price (Base = 100)",
            side="left",
            color='#1f77b4'
        ),
        # Right Y-axis for actual variable prices
        yaxis2=dict(
            title="Selected Variables Actual Prices ($)",
            side="right",
            overlaying="y",
            color='#ff7f0e'
        )
    )
    st.plotly_chart(fig_prices, use_container_width=True)

    # --- Weather Data Visualization ---
    # Check if any weather variables are selected
    weather_variables = [var for var in selected_vars if 'Temperature' in var or 'Rainfall' in var]
    
    if weather_variables:
        st.markdown("#### � Normalized Weather Patterns")
        st.markdown("Weather data normalized to 0-1 scale for pattern comparison across regions")
        
        # Create a normalized weather comparison (all variables scaled 0-1)
        fig_weather_norm = go.Figure()
        
        for var in weather_variables:
            if var in corr_df.columns:
                region = 'Colombia' if 'Colombia' in var else 'EU' if 'EU' in var else 'US'
                data_type = 'temperature' if 'Temperature' in var else 'rainfall'
                weather_series = load_weather_data(region, data_type, corr_start_date, corr_end_date)
                
                if not weather_series.empty:
                    aligned_series = weather_series.reindex(corr_data.index, method='nearest')
                    
                    # Normalize to 0-1 scale for comparison
                    min_val = aligned_series.min()
                    max_val = aligned_series.max()
                    if max_val > min_val:  # Avoid division by zero
                        normalized = (aligned_series - min_val) / (max_val - min_val)
                        
                        fig_weather_norm.add_trace(go.Scatter(
                            x=aligned_series.index,
                            y=normalized,
                            mode='lines',
                            name=var.replace(' (Historical)', ''),
                            line=dict(width=2)
                        ))
        
        if fig_weather_norm.data:
            fig_weather_norm.update_layout(
                title="Normalized Weather Patterns Comparison (0-1 Scale)",
                xaxis_title="Date",
                yaxis_title="Normalized Value (0 = Min, 1 = Max)",
                template="plotly_white",
                height=400,
                hovermode='x unified',
                legend=dict(
                    orientation="h",
                    yanchor="top",
                    y=-0.2,
                    xanchor="center",
                    x=0.5
                )
            )
            st.plotly_chart(fig_weather_norm, use_container_width=True)
            
            # Add explanation
            with st.expander("💡 Weather Data Explanation", expanded=False):
                st.markdown("""
                **Normalized Comparison**: All weather variables scaled to 0-1 range to compare patterns across different units and scales.
                
                **Data Sources**: 
                - Real data from VisualCrossing Weather API when available
                - Synthetic seasonal data as fallback when API is unavailable
                
                **Regional Coverage**:
                - **EU**: Average of Paris, Berlin, Rome
                - **US**: Average of New York, Chicago, Los Angeles  
                - **Colombia**: Average of Bogotá, Medellín
                """)

    # Enhanced ML Functions from ML_Stock_Predictor
    
    def load_stock_data(ticker, start_date, end_date):
        """Load stock data using yfinance"""
        try:
            data = yf.download(ticker, start=start_date, end=end_date)
            if data is None or data.empty:
                return None
            data = data.reset_index()
            return data
        except Exception as e:
            st.warning(f"Could not load data for {ticker}: {str(e)}")
            return None
    
    def create_correlation_features(main_data, correlation_data, window=60):
        """Create correlation-based features"""
        correlation_features = pd.DataFrame(index=main_data.index)
        
        # Extract main close prices as Series
        main_close = main_data['Close'].squeeze()
        main_log_returns = main_data['log_returns'].squeeze()
        
        for var_name, var_data in correlation_data.items():
            if var_data is None or var_data.empty:
                continue
                
            # Align data
            aligned_data = var_data.reindex(main_data.index, method='nearest')
            if 'log_returns' not in aligned_data.columns:
                aligned_data['log_returns'] = np.log(aligned_data['Close'] / aligned_data['Close'].shift(1))
            
            # Extract aligned close and log returns as Series
            aligned_close = aligned_data['Close'].squeeze()
            aligned_log_returns = aligned_data['log_returns'].squeeze()
            
            # Rolling correlation with safe handling
            rolling_corr = main_log_returns.rolling(window=window).corr(aligned_log_returns)
            correlation_features[f'{var_name}_corr'] = rolling_corr
            
            # Correlation momentum
            correlation_features[f'{var_name}_corr_momentum'] = rolling_corr.diff(5)
            
            # Price ratio with safe division
            with np.errstate(divide='ignore', invalid='ignore'):
                price_ratio_values = np.where(
                    (aligned_close != 0) & (~np.isnan(aligned_close)), 
                    main_close / aligned_close, 
                    1.0
                )
            
            # Create Series with proper index
            price_ratio_series = pd.Series(price_ratio_values, index=main_data.index, name=f'{var_name}_price_ratio')
            correlation_features[f'{var_name}_price_ratio'] = price_ratio_series
            correlation_features[f'{var_name}_price_ratio_ma'] = price_ratio_series.rolling(window=20).mean()
        
        return correlation_features
    
    def prepare_ml_features(main_data, correlation_features=None, prediction_days=5):
        """Prepare features for ML model"""
        # Calculate additional technical indicators
        feature_data = calculate_technical_indicators(main_data)
        
        # Flatten MultiIndex columns if they exist
        if hasattr(feature_data.columns, 'nlevels') and feature_data.columns.nlevels > 1:
            feature_data.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in feature_data.columns]
        
        # Ensure original OHLCV columns are preserved
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in main_data.columns and col not in feature_data.columns:
                feature_data[col] = main_data[col]
        
        # Add correlation features if available
        if correlation_features is not None:
            for col in correlation_features.columns:
                if col not in feature_data.columns:
                    feature_data[col] = correlation_features[col]
        
        # Verify Close column exists before creating target
        if 'Close' not in feature_data.columns:
            raise ValueError("Close column is missing from feature data")
        
        # Create target variable
        feature_data['target'] = feature_data['Close'].pct_change(prediction_days).shift(-prediction_days)
        
        # Select feature columns - ensure exact column names match
        available_columns = list(feature_data.columns)
        
        # Technical indicators with exact matching
        technical_features = []
        for col in available_columns:
            if any(indicator in str(col) for indicator in ['ma_', 'rsi', 'bb_', 'macd', 'volatility_', 'momentum_', 'volume_', 'price_position']):
                technical_features.append(col)
        
        # Correlation features
        correlation_feature_cols = []
        for col in available_columns:
            if any(suffix in str(col) for suffix in ['_corr', '_price_ratio']):
                correlation_feature_cols.append(col)
        
        # Base features
        base_features = []
        for col in ['returns', 'log_returns']:
            if col in available_columns:
                base_features.append(col)
        
        feature_columns = base_features + technical_features + correlation_feature_cols
        
        # Final validation - only include columns that actually exist
        feature_columns = [col for col in feature_columns if col in feature_data.columns]
        
        return feature_data, feature_columns

    def create_lstm_sequences(data, feature_columns, target_column, sequence_length=60, prediction_days=5):
        """Create sequences for LSTM training"""
        if not DEEP_LEARNING_AVAILABLE:
            return None, None, None, None
        
        # Prepare data
        feature_data = data[feature_columns].values
        target_data = data[target_column].values
        
        # Scale features
        feature_scaler = MinMaxScaler()
        target_scaler = MinMaxScaler()
        
        scaled_features = feature_scaler.fit_transform(feature_data)
        scaled_targets = target_scaler.fit_transform(target_data.reshape(-1, 1)).flatten()
        
        # Create sequences
        X_sequences = []
        y_sequences = []
        
        for i in range(sequence_length, len(scaled_features) - prediction_days + 1):
            # Features: sequence_length timesteps of features
            X_sequences.append(scaled_features[i-sequence_length:i])
            # Target: future return after prediction_days
            y_sequences.append(scaled_targets[i + prediction_days - 1])
        
        X_sequences = np.array(X_sequences)
        y_sequences = np.array(y_sequences)
        
        return X_sequences, y_sequences, feature_scaler, target_scaler

    def build_lstm_model(input_shape, lstm_units=[50, 30], dropout_rate=0.2, learning_rate=0.001):
        """Build LSTM model architecture"""
        if not DEEP_LEARNING_AVAILABLE:
            return None
        
        model = Sequential()
        
        # First LSTM layer
        model.add(LSTM(lstm_units[0], return_sequences=True, input_shape=input_shape))
        model.add(Dropout(dropout_rate))
        model.add(BatchNormalization())
        
        # Second LSTM layer
        model.add(LSTM(lstm_units[1], return_sequences=False))
        model.add(Dropout(dropout_rate))
        model.add(BatchNormalization())
        
        # Dense layers
        model.add(Dense(25, activation='relu'))
        model.add(Dropout(dropout_rate))
        model.add(Dense(1, activation='linear'))
        
        # Compile model
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        return model

    def train_lstm_model(X_train, y_train, X_val, y_val, epochs=100, batch_size=32, patience=15):
        """Train LSTM model with early stopping"""
        if not DEEP_LEARNING_AVAILABLE:
            return None, None
        
        try:
            # Build model
            input_shape = (X_train.shape[1], X_train.shape[2])
            model = build_lstm_model(input_shape)
            
            if model is None:
                return None, None
            
            # Callbacks
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=0
            )
            
            reduce_lr = ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=patience//2,
                min_lr=1e-7,
                verbose=0
            )
            
            # Train model
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[early_stopping, reduce_lr],
                verbose=0
            )
            
            return model, history
        except Exception as e:
            st.error(f"LSTM training failed: {str(e)}")
            return None, None

    def analyze_correlation_patterns(main_data, correlation_data, window=60):
        """Analyze historical correlation patterns and detect regimes"""
        correlation_patterns = {}
        
        for var_name, var_data in correlation_data.items():
            if var_data is None or var_data.empty:
                continue
                
            # Align data
            aligned_data = var_data.reindex(main_data.index, method='nearest')
            if 'log_returns' not in aligned_data.columns:
                aligned_data['log_returns'] = np.log(aligned_data['Close'] / aligned_data['Close'].shift(1))
            
            # Calculate rolling correlation
            rolling_corr = main_data['log_returns'].rolling(window=window).corr(aligned_data['log_returns'])
            rolling_corr = rolling_corr.dropna()
            
            if len(rolling_corr) < 50:  # Need sufficient data
                continue
            
            # Pattern analysis
            patterns = {
                'rolling_correlation': rolling_corr,
                'mean_correlation': rolling_corr.mean(),
                'std_correlation': rolling_corr.std(),
                'correlation_regimes': detect_correlation_regimes(rolling_corr),
                'seasonal_patterns': detect_seasonal_correlation_patterns(rolling_corr),
                'cycle_patterns': detect_correlation_cycles(rolling_corr),
                'regime_transitions': detect_regime_transitions(rolling_corr),
                'upcoming_predictions': predict_upcoming_patterns(rolling_corr)
            }
            
            correlation_patterns[var_name] = patterns
        
        return correlation_patterns

    def detect_correlation_regimes(correlation_series, n_regimes=3):
        """Detect correlation regimes using clustering"""
        try:
            # Prepare data for clustering
            corr_values = correlation_series.values.reshape(-1, 1)
            corr_values = corr_values[~np.isnan(corr_values).flatten()]
            
            if len(corr_values) < 20:
                return None
            
            # K-means clustering to identify regimes
            kmeans = KMeans(n_clusters=n_regimes, random_state=42, n_init=10)
            regime_labels = kmeans.fit_predict(corr_values)
            
            # Assign regimes back to time series
            regime_series = pd.Series(index=correlation_series.index, dtype=int)
            valid_indices = correlation_series.dropna().index
            regime_series.loc[valid_indices] = regime_labels
            
            # Analyze regime characteristics
            regimes = {}
            for i in range(n_regimes):
                regime_mask = regime_series == i
                regime_corr = correlation_series[regime_mask]
                
                regimes[f'Regime_{i}'] = {
                    'mean_correlation': regime_corr.mean(),
                    'std_correlation': regime_corr.std(),
                    'duration_avg': calculate_regime_duration(regime_series, i),
                    'frequency': regime_mask.sum() / len(regime_series),
                    'dates': regime_series[regime_mask].index.tolist()
                }
            
            return {
                'regime_series': regime_series,
                'regimes': regimes,
                'cluster_centers': kmeans.cluster_centers_.flatten()
            }
        except Exception as e:
            return None

    def calculate_regime_duration(regime_series, regime_id):
        """Calculate average duration of a specific regime"""
        try:
            regime_mask = regime_series == regime_id
            transitions = regime_mask.astype(int).diff().fillna(0)
            
            starts = regime_series[transitions == 1].index
            ends = regime_series[transitions == -1].index
            
            # Handle edge cases
            if len(starts) == 0:
                return 0
            
            if len(ends) < len(starts):
                ends = ends.tolist() + [regime_series.index[-1]]
            
            durations = [(end - start).days for start, end in zip(starts, ends[:len(starts)])]
            return np.mean(durations) if durations else 0
        except:
            return 0

    def detect_seasonal_correlation_patterns(correlation_series):
        """Detect seasonal patterns in correlations"""
        try:
            df = correlation_series.to_frame('correlation')
            df['month'] = df.index.month
            df['quarter'] = df.index.quarter
            df['day_of_year'] = df.index.dayofyear
            
            # Monthly patterns
            monthly_stats = df.groupby('month')['correlation'].agg(['mean', 'std', 'count'])
            
            # Quarterly patterns
            quarterly_stats = df.groupby('quarter')['correlation'].agg(['mean', 'std', 'count'])
            
            # Find strongest seasonal effects
            monthly_variation = monthly_stats['mean'].std()
            quarterly_variation = quarterly_stats['mean'].std()
            
            return {
                'monthly_patterns': monthly_stats.to_dict(),
                'quarterly_patterns': quarterly_stats.to_dict(),
                'seasonal_strength': {
                    'monthly_variation': monthly_variation,
                    'quarterly_variation': quarterly_variation
                },
                'strongest_month': monthly_stats['mean'].idxmax(),
                'weakest_month': monthly_stats['mean'].idxmin(),
                'strongest_quarter': quarterly_stats['mean'].idxmax(),
                'weakest_quarter': quarterly_stats['mean'].idxmin()
            }
        except Exception as e:
            return None

    def detect_correlation_cycles(correlation_series, min_cycle_days=30, max_cycle_days=365):
        """Detect cyclical patterns in correlations"""
        try:
            # Remove trend
            correlation_detrended = correlation_series - correlation_series.rolling(window=60, center=True).mean()
            correlation_detrended = correlation_detrended.dropna()
            
            if len(correlation_detrended) < 100:
                return None
            
            # Find peaks and troughs
            peaks, _ = find_peaks(correlation_detrended.values, distance=min_cycle_days)
            troughs, _ = find_peaks(-correlation_detrended.values, distance=min_cycle_days)
            
            # Calculate cycle lengths
            peak_dates = correlation_detrended.index[peaks]
            trough_dates = correlation_detrended.index[troughs]
            
            cycles = []
            
            # Peak-to-peak cycles
            if len(peak_dates) > 1:
                peak_cycles = [(peak_dates[i+1] - peak_dates[i]).days 
                            for i in range(len(peak_dates)-1)]
                cycles.extend(peak_cycles)
            
            # Trough-to-trough cycles
            if len(trough_dates) > 1:
                trough_cycles = [(trough_dates[i+1] - trough_dates[i]).days 
                                for i in range(len(trough_dates)-1)]
                cycles.extend(trough_cycles)
            
            # Filter cycles by reasonable length
            valid_cycles = [c for c in cycles if min_cycle_days <= c <= max_cycle_days]
            
            if not valid_cycles:
                return None
            
            return {
                'cycle_lengths': valid_cycles,
                'average_cycle': np.mean(valid_cycles),
                'cycle_std': np.std(valid_cycles),
                'peak_dates': peak_dates.tolist(),
                'trough_dates': trough_dates.tolist(),
                'last_peak': peak_dates[-1] if len(peak_dates) > 0 else None,
                'last_trough': trough_dates[-1] if len(trough_dates) > 0 else None
            }
        except Exception as e:
            return None

    def detect_regime_transitions(correlation_series, threshold=0.15):
        """Detect significant regime transitions"""
        try:
            # Calculate rolling statistics
            rolling_mean = correlation_series.rolling(window=30).mean()
            rolling_std = correlation_series.rolling(window=30).std()
            
            # Standardize
            standardized = (correlation_series - rolling_mean) / rolling_std
            
            # Find significant transitions
            transitions = []
            for i in range(1, len(standardized)):
                if abs(standardized.iloc[i] - standardized.iloc[i-1]) > threshold:
                    transitions.append({
                        'date': standardized.index[i],
                        'from_value': standardized.iloc[i-1],
                        'to_value': standardized.iloc[i],
                        'magnitude': abs(standardized.iloc[i] - standardized.iloc[i-1])
                    })
            
            return sorted(transitions, key=lambda x: x['magnitude'], reverse=True)[:10]  # Top 10
        except Exception as e:
            return []

    def predict_upcoming_patterns(correlation_series):
        """Predict upcoming correlation patterns based on historical cycles"""
        try:
            current_date = correlation_series.index[-1]
            predictions = []
            
            # Analyze recent trend
            recent_trend = correlation_series.tail(30).values
            if len(recent_trend) > 10:
                trend_slope = np.polyfit(range(len(recent_trend)), recent_trend, 1)[0]
            else:
                trend_slope = 0
            
            # Seasonal prediction
            current_month = current_date.month
            current_quarter = (current_month - 1) // 3 + 1
            
            # Historical same-month average
            same_month_data = correlation_series[correlation_series.index.month == current_month]
            if len(same_month_data) > 0:
                seasonal_expectation = same_month_data.mean()
            else:
                seasonal_expectation = correlation_series.mean()
            
            # Cycle-based prediction
            cycles = detect_correlation_cycles(correlation_series)
            cycle_prediction = None
            
            if cycles and cycles['average_cycle']:
                avg_cycle = cycles['average_cycle']
                last_peak = cycles['last_peak']
                last_trough = cycles['last_trough']
                
                if last_peak and last_trough:
                    # Determine cycle position
                    days_since_peak = (current_date - last_peak).days
                    days_since_trough = (current_date - last_trough).days
                    
                    cycle_position = min(days_since_peak, days_since_trough) / avg_cycle
                    
                    # Predict next turning point
                    if last_peak > last_trough:  # Currently in downtrend from peak
                        next_trough_days = avg_cycle/2 - days_since_peak
                        if next_trough_days > 0:
                            cycle_prediction = {
                                'type': 'trough',
                                'expected_date': current_date + timedelta(days=next_trough_days),
                                'confidence': max(0.3, 1 - cycle_position)
                            }
                    else:  # Currently in uptrend from trough
                        next_peak_days = avg_cycle/2 - days_since_trough
                        if next_peak_days > 0:
                            cycle_prediction = {
                                'type': 'peak',
                                'expected_date': current_date + timedelta(days=next_peak_days),
                                'confidence': max(0.3, 1 - cycle_position)
                            }
            
            return {
                'trend_direction': 'up' if trend_slope > 0.001 else 'down' if trend_slope < -0.001 else 'sideways',
                'trend_strength': abs(trend_slope),
                'seasonal_expectation': seasonal_expectation,
                'current_vs_seasonal': correlation_series.iloc[-1] - seasonal_expectation,
                'cycle_prediction': cycle_prediction,
                'next_month_outlook': predict_next_month_correlation(correlation_series),
                'regime_probability': calculate_regime_probabilities(correlation_series)
            }
        except Exception as e:
            return {}

    def predict_next_month_correlation(correlation_series):
        """Predict correlation for next month based on historical patterns"""
        try:
            current_date = correlation_series.index[-1]
            next_month = (current_date.month % 12) + 1
            next_year = current_date.year + (1 if next_month == 1 else 0)
            
            # Historical data for next month
            next_month_historical = correlation_series[correlation_series.index.month == next_month]
            
            if len(next_month_historical) > 0:
                return {
                    'expected_correlation': next_month_historical.mean(),
                    'range_low': next_month_historical.quantile(0.25),
                    'range_high': next_month_historical.quantile(0.75),
                    'historical_count': len(next_month_historical)
                }
            return None
        except:
            return None

    def calculate_regime_probabilities(correlation_series):
        """Calculate probability of being in different correlation regimes"""
        try:
            current_corr = correlation_series.iloc[-1]
            recent_corr = correlation_series.tail(10).mean()
            
            # Define regime thresholds
            high_corr_threshold = 0.3
            low_corr_threshold = -0.1
            
            # Simple regime classification
            if current_corr > high_corr_threshold:
                return {'high_correlation': 0.8, 'medium_correlation': 0.2, 'low_correlation': 0.0}
            elif current_corr < low_corr_threshold:
                return {'high_correlation': 0.0, 'medium_correlation': 0.2, 'low_correlation': 0.8}
            else:
                return {'high_correlation': 0.3, 'medium_correlation': 0.4, 'low_correlation': 0.3}
        except:
            return {'high_correlation': 0.33, 'medium_correlation': 0.34, 'low_correlation': 0.33}

    # Model Configuration Section  
    st.header("🎯 AI/ML Price Prediction Model")
    
    # Information section
    with st.expander("ℹ️ About this Advanced ML Predictor", expanded=False):
        st.markdown("""
        ### Advanced Stock Predictor with Multiple AI Models
        
        **Available Models:**
        - **🌲 Random Forest:** Tree-based ensemble model excellent for feature importance analysis
        - **🧠 LSTM Neural Network:** Deep learning model specialized for time series sequences  
        - **🔄 Ensemble (RF + LSTM):** Combines both models for improved robustness
        
        **Enhanced Features:**
        - **Advanced Lag Features:** 1-365 day historical patterns and momentum acceleration
        - **Technical Indicators:** RSI, MACD, Bollinger Bands, moving averages
        - **Volatility Regimes:** Short vs long-term volatility analysis
        - **Trend Analysis:** Linear trend strength and consistency metrics
        - **Market Regime Detection:** Bull/bear market indicators
        - **Seasonal Patterns:** Day-of-week, monthly, quarterly, and yearly cycles
        - **Correlation Features:** Dynamic correlations with market indices
        
        **LSTM Advantages:**
        - **Sequential Learning:** Captures temporal dependencies in time series
        - **Memory Cells:** Remembers long-term patterns better than traditional ML
        - **Non-linear Modeling:** Complex pattern recognition capabilities
        - **Automatic Feature Engineering:** Learns relevant patterns from sequences
        
        **Ensemble Benefits:**
        - **Reduced Overfitting:** Combines different model strengths
        - **Improved Stability:** More robust predictions across market conditions
        - **Complementary Insights:** RF shows feature importance, LSTM captures sequences
        
        **Technical Implementation:**
        - **Time Series Validation:** Proper walk-forward cross-validation
        - **Feature Selection:** Intelligent selection of most predictive features
        - **Early Stopping:** Prevents LSTM overfitting with validation monitoring
        - **Scaling & Normalization:** Proper data preprocessing for neural networks
        
        **User Input Controls:**
        - **Prediction Horizon:** 1-30 days ahead forecast target
        - **Feature Lookback:** 30-200 days of historical data for feature engineering
        - **Model Parameters:** Fine-tune Random Forest trees, depth, and LSTM architecture
        - **Correlation Variables:** Select market factors to include as predictive features
                        """)
    
    st.info(f"📊 **Using inputs from page header:** Ticker: {corr_ticker} | Date Range: {corr_start_date} to {corr_end_date} | Variables: {len(selected_vars)} selected")

    # Configuration options that aren't already set at page level
    col1, col2 = st.columns(2)
    with col1:
        prediction_days = st.slider("Prediction Horizon (days)", 1, 30, 5)
    with col2:
        lookback_days = st.slider("Feature Lookback (days)", 30, 200, 60)

    st.markdown("---")

    # Model selection section
    st.subheader("Model Selection")
    model_options = ["Random Forest", "LSTM Neural Network", "Ensemble (RF + LSTM)"]
    if not DEEP_LEARNING_AVAILABLE:
        model_options = ["Random Forest"]
        st.warning("LSTM unavailable - install tensorflow")

    selected_model = st.selectbox("Choose Model:", model_options)

    # Initialize all parameters with defaults
    n_estimators = 100
    max_depth = 10
    min_samples_split = 5
    sequence_length = 60
    lstm_units_1 = 50
    lstm_units_2 = 30
    dropout_rate = 0.2
    epochs = 100
    batch_size = 32

    # Model-specific parameters in expandable sections
    if "Random Forest" in selected_model:
        with st.expander("🌲 Random Forest Parameters", expanded=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                n_estimators = st.slider("Number of Trees", 50, 500, 100, 50)
            with col2:
                max_depth = st.slider("Max Depth", 3, 20, 10)
            with col3:
                min_samples_split = st.slider("Min Samples Split", 2, 20, 5)

    if "LSTM" in selected_model and DEEP_LEARNING_AVAILABLE:
        with st.expander("🧠 LSTM Parameters", expanded=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                sequence_length = st.slider("Sequence Length (days)", 30, 120, 60)
                lstm_units_1 = st.slider("LSTM Units (Layer 1)", 20, 100, 50, 10)
            with col2:
                lstm_units_2 = st.slider("LSTM Units (Layer 2)", 10, 60, 30, 5)
                dropout_rate = st.slider("Dropout Rate", 0.1, 0.5, 0.2, 0.05)
            with col3:
                epochs = st.slider("Training Epochs", 50, 200, 100, 25)
                batch_size = st.selectbox("Batch Size", [16, 32, 64], index=1)

    # ML Model Training Section
    st.subheader(f"{selected_model} Training & Prediction")

    if st.button("Train Model & Predict", type="primary"):
        with st.spinner(f"Training {selected_model} model..."):
            try:
                # Load main stock data - using existing page variables
                with st.spinner(f"Loading {corr_ticker} data..."):
                    main_data = load_stock_data(corr_ticker, corr_start_date, corr_end_date)
                
                if main_data is None or main_data.empty:
                    st.error(f"Could not load data for {corr_ticker}")
                    st.stop()
                
                # Set date as index
                main_data['Date'] = pd.to_datetime(main_data['Date'])
                main_data.set_index('Date', inplace=True)
                
                # Add basic returns for correlation calculations
                main_data['log_returns'] = np.log(main_data['Close'] / main_data['Close'].shift(1))
                
                # Load correlation data - using existing page variables and selected_vars
                correlation_data = {}
                if selected_vars:
                    # Map selected_vars to ticker symbols
                    etf_map = {
                        "SPY (S&P 500 ETF)": "SPY",
                        "GLD (Gold ETF)": "GLD",
                        "USO (Oil ETF)": "USO",
                        "UNG (Natural Gas ETF)": "UNG",
                        "DBC (Commodities ETF)": "DBC",
                        "TLT (20+ Year Treasury Bond ETF)": "TLT",
                        "IEF (10 Year Treasury Bond ETF)": "IEF",
                        "VIX (Volatility Index)": "^VIX",
                        "Global Oil Price (Brent)": "BZ=F",
                        "Crude Oil Futures (CL=F)": "CL=F",
                        "USD Index (DXY)": "DX-Y.NYB",
                        "Bitcoin Price (BTC-USD)": "BTC-USD",
                        "Coffee Futures (KC=F)": "KC=F"
                    }
                    
                    # Filter to only financial instruments (not weather data)
                    financial_vars = [var for var in selected_vars if var in etf_map]
                    
                    if financial_vars:
                        progress_bar = st.progress(0)
                        for i, var in enumerate(financial_vars):
                            ticker_symbol = etf_map[var]
                            progress_bar.progress((i + 1) / len(financial_vars))
                            corr_data = load_stock_data(ticker_symbol, corr_start_date, corr_end_date)
                            if corr_data is not None:
                                corr_data['Date'] = pd.to_datetime(corr_data['Date'])
                                corr_data.set_index('Date', inplace=True)
                                corr_data['log_returns'] = np.log(corr_data['Close'] / corr_data['Close'].shift(1))
                                correlation_data[ticker_symbol] = corr_data
                        progress_bar.empty()
                    else:
                        st.info("💡 **Tip**: Select financial instruments (ETFs, indices) from the variables list for enhanced ML features. Weather data is not used in ML models.")
                
                # Create correlation features
                correlation_features = None
                correlation_patterns = None
                if correlation_data:
                    correlation_features = create_correlation_features(
                        main_data, correlation_data, window=60
                    )
                    
                    # Analyze correlation patterns - using financial instruments only
                    st.info("🔍 Analyzing historical correlation patterns...")
                    correlation_patterns = analyze_correlation_patterns(
                        main_data, correlation_data, window=60
                    )
                
                # Prepare ML features
                main_data, feature_columns = prepare_ml_features(
                    main_data, correlation_features, prediction_days
                )
                
                # Debug: Check column structure and flatten if needed
                if hasattr(main_data.columns, 'nlevels') and main_data.columns.nlevels > 1:
                    # Flatten MultiIndex columns
                    main_data.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in main_data.columns]
                    # Update feature columns to match flattened names
                    feature_columns = [col for col in feature_columns if col in main_data.columns]
                
                # Prepare data for ML
                # Target: future price change
                main_data['target'] = main_data['Close'].shift(-prediction_days) / main_data['Close'] - 1
                
                # Remove rows with missing target or features
                ml_data = main_data[feature_columns + ['target']].dropna()
                
                if len(ml_data) < 200:  # Increased minimum for LSTM
                    st.error("Not enough data for training. Please extend the date range or reduce prediction horizon.")
                    st.stop()
                
                # Check for infinite or very large values
                if not np.all(np.isfinite(ml_data[feature_columns].values)):
                    st.warning("Some features contain infinite values. Cleaning data...")
                    ml_data = ml_data.replace([np.inf, -np.inf], np.nan).dropna()
                    
                if len(ml_data) < 100:
                    st.error("After cleaning, not enough data remains for training.")
                    st.stop()
                
                st.success(f"✅ Prepared {len(ml_data)} samples with {len(feature_columns)} features")
                
                # Model training based on selection
                rf_model, rf_pred, rf_r2, rf_mse, rf_directional = None, None, None, None, None
                X_selected, selected_features = None, []
                avg_cv_score, cv_std, avg_cv_directional = 0, 0, 0.5
                
                if selected_model == "Random Forest" or "Ensemble" in selected_model:
                    # Split features and target for Random Forest
                    X = np.array(ml_data[feature_columns].values, dtype=np.float64)
                    y = np.array(ml_data['target'].values, dtype=np.float64)
                    
                    # Time series split for validation
                    tscv = TimeSeriesSplit(n_splits=5, test_size=None)
                    cv_scores = []
                    cv_directional_accuracies = []
                    
                    # Feature selection based on importance
                    st.info("Training Random Forest with feature selection...")
                    
                    # Quick Random Forest to get feature importance for selection
                    rf_selector = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
                    rf_selector.fit(X, y)
                    
                    # Select top features based on importance
                    feature_importance_scores = rf_selector.feature_importances_
                    feature_importance_df = pd.DataFrame({
                        'feature': feature_columns,
                        'importance': feature_importance_scores
                    }).sort_values('importance', ascending=False)
                    
                    # Select top features
                    n_features_to_select = max(20, min(len(feature_columns) // 2, 50))
                    selected_feature_indices = feature_importance_df.head(n_features_to_select).index.tolist()
                    selected_features = [feature_columns[i] for i in selected_feature_indices]
                    
                    # Use only selected features
                    X_selected = X[:, selected_feature_indices]
                    
                    # Cross-validation
                    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_selected)):
                        if len(train_idx) < 100:
                            continue
                            
                        X_train_fold, X_val_fold = X_selected[train_idx], X_selected[val_idx]
                        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
                        
                        # Train model
                        rf_fold = RandomForestRegressor(
                            n_estimators=min(n_estimators, 200),
                            max_depth=max_depth,
                            min_samples_split=max(min_samples_split, 10),
                            min_samples_leaf=5,
                            max_features='sqrt',
                            random_state=42 + fold,
                            n_jobs=-1
                        )
                        rf_fold.fit(X_train_fold, y_train_fold)
                        
                        # Validate
                        y_pred_fold = rf_fold.predict(X_val_fold)
                        fold_r2 = r2_score(y_val_fold, y_pred_fold)
                        fold_directional = np.mean(np.sign(y_val_fold) == np.sign(y_pred_fold))
                        
                        cv_scores.append(fold_r2)
                        cv_directional_accuracies.append(fold_directional)
                    
                    avg_cv_score = np.mean(cv_scores) if cv_scores else 0
                    cv_std = np.std(cv_scores) if cv_scores else 0
                    avg_cv_directional = np.mean(cv_directional_accuracies) if cv_directional_accuracies else 0.5
                    
                    # Final Random Forest training
                    train_size = int(len(X_selected) * 0.85)
                    X_train, X_test = X_selected[:train_size], X_selected[train_size:]
                    y_train, y_test = y[:train_size], y[train_size:]
                    
                    rf_model = RandomForestRegressor(
                        n_estimators=min(n_estimators * 2, 300),
                        max_depth=max_depth,
                        min_samples_split=max(min_samples_split, 10),
                        min_samples_leaf=5,
                        max_features='sqrt',
                        bootstrap=True,
                        oob_score=True,
                        random_state=42,
                        n_jobs=-1
                    )
                    rf_model.fit(X_train, y_train)
                    
                    # Random Forest predictions
                    rf_pred = rf_model.predict(X_test)
                    rf_r2 = r2_score(y_test, rf_pred)
                    rf_mse = mean_squared_error(y_test, rf_pred)
                    rf_directional = np.mean(np.sign(y_test) == np.sign(rf_pred))
                
                # LSTM Training
                lstm_r2, lstm_mse, lstm_directional, lstm_model, lstm_history = None, None, None, None, None
                lstm_pred, y_lstm_test_original = None, None
                feature_scaler, target_scaler = None, None
                
                if ("LSTM" in selected_model or "Ensemble" in selected_model) and DEEP_LEARNING_AVAILABLE:
                    st.info("Training LSTM Neural Network...")
                    
                    # Create LSTM sequences
                    X_lstm, y_lstm, feature_scaler, target_scaler = create_lstm_sequences(
                        ml_data, feature_columns, 'target', 
                        sequence_length=sequence_length, 
                        prediction_days=prediction_days
                    )
                    
                    if (X_lstm is not None and y_lstm is not None and 
                        feature_scaler is not None and target_scaler is not None and 
                        len(X_lstm) > 100):
                        
                        # Split data for LSTM
                        lstm_train_size = int(len(X_lstm) * 0.8)
                        lstm_val_size = int(len(X_lstm) * 0.1)
                        
                        X_lstm_train = X_lstm[:lstm_train_size]
                        y_lstm_train = y_lstm[:lstm_train_size]
                        X_lstm_val = X_lstm[lstm_train_size:lstm_train_size + lstm_val_size]
                        y_lstm_val = y_lstm[lstm_train_size:lstm_train_size + lstm_val_size]
                        X_lstm_test = X_lstm[lstm_train_size + lstm_val_size:]
                        y_lstm_test = y_lstm[lstm_train_size + lstm_val_size:]
                        
                        # Train LSTM
                        lstm_model, lstm_history = train_lstm_model(
                            X_lstm_train, y_lstm_train, X_lstm_val, y_lstm_val,
                            epochs=epochs, batch_size=batch_size
                        )
                        
                        if lstm_model is not None and len(X_lstm_test) > 0:
                            try:
                                # LSTM predictions
                                lstm_pred_scaled = lstm_model.predict(X_lstm_test, verbose=0)
                                lstm_pred = target_scaler.inverse_transform(lstm_pred_scaled.reshape(-1, 1)).flatten()
                                y_lstm_test_original = target_scaler.inverse_transform(y_lstm_test.reshape(-1, 1)).flatten()
                                
                                lstm_r2 = r2_score(y_lstm_test_original, lstm_pred)
                                lstm_mse = mean_squared_error(y_lstm_test_original, lstm_pred)
                                lstm_directional = np.mean(np.sign(y_lstm_test_original) == np.sign(lstm_pred))
                            except Exception as e:
                                st.warning(f"LSTM prediction failed: {str(e)}")
                                lstm_model = None
                    else:
                        st.warning("Not enough data for LSTM training after sequence creation.")
                
                # Ensemble predictions
                if ("Ensemble" in selected_model and rf_model is not None and lstm_model is not None and 
                    rf_pred is not None and lstm_pred is not None and 
                    len(rf_pred) > 0 and len(lstm_pred) > 0):
                    
                    # Align prediction lengths
                    min_length = min(len(rf_pred), len(lstm_pred))
                    rf_pred_aligned = rf_pred[:min_length]
                    lstm_pred_aligned = lstm_pred[:min_length]
                    y_test_aligned = y_test[:min_length]
                    
                    # Simple ensemble: average predictions
                    ensemble_pred = (rf_pred_aligned + lstm_pred_aligned) / 2
                    ensemble_r2 = r2_score(y_test_aligned, ensemble_pred)
                    ensemble_mse = mean_squared_error(y_test_aligned, ensemble_pred)
                    ensemble_directional = np.mean(np.sign(y_test_aligned) == np.sign(ensemble_pred))
                    
                    # Use ensemble metrics
                    r2, mse, directional_accuracy = ensemble_r2, ensemble_mse, ensemble_directional
                    y_pred = ensemble_pred
                    y_test = y_test_aligned
                elif "LSTM" in selected_model and lstm_model is not None and lstm_pred is not None:
                    # Use LSTM metrics
                    r2, mse, directional_accuracy = lstm_r2, lstm_mse, lstm_directional
                    y_pred = lstm_pred
                    y_test = y_lstm_test_original
                elif rf_model is not None and rf_pred is not None:
                    # Use Random Forest metrics
                    r2, mse, directional_accuracy = rf_r2, rf_mse, rf_directional
                    y_pred = rf_pred
                else:
                    # Default fallback
                    r2, mse, directional_accuracy = 0, 1, 0.5
                    y_pred, y_test = np.array([]), np.array([])
                
                # Display results
                st.header(f"📊 {selected_model} Model Performance")
                
                # Main metrics with safe None handling
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Test R² Score", f"{r2:.3f}" if r2 is not None else "N/A")
                with col2:
                    if "Random Forest" in selected_model:
                        st.metric("CV R² (Avg)", f"{avg_cv_score:.3f}")
                    else:
                        st.metric("RMSE", f"{np.sqrt(mse):.4f}" if mse is not None else "N/A")
                with col3:
                    st.metric("Directional Accuracy", f"{directional_accuracy:.2%}" if directional_accuracy is not None else "N/A")
                with col4:
                    if "Random Forest" in selected_model and rf_model is not None and hasattr(rf_model, 'oob_score_'):
                        st.metric("OOB Score", f"{rf_model.oob_score_:.3f}")
                    else:
                        st.metric("Training Samples", len(ml_data))
                
                # Model comparison for ensemble
                if "Ensemble" in selected_model and rf_model is not None and lstm_model is not None:
                    st.subheader("🔄 Model Comparison")
                    comparison_df = pd.DataFrame({
                        'Model': ['Random Forest', 'LSTM', 'Ensemble'],
                        'R² Score': [rf_r2 or 0, lstm_r2 or 0, r2 or 0],
                        'RMSE': [np.sqrt(rf_mse) if rf_mse else 0, np.sqrt(lstm_mse) if lstm_mse else 0, np.sqrt(mse) if mse else 0],
                        'Directional Acc': [rf_directional or 0, lstm_directional or 0, directional_accuracy or 0]
                    })
                    st.dataframe(comparison_df, use_container_width=True)
                
                # Feature importance (for Random Forest models)
                if ("Random Forest" in selected_model and rf_model is not None and 
                    selected_features and len(selected_features) > 0):
                    st.subheader("🌟 Top Feature Importance")
                    
                    feature_importance = pd.DataFrame({
                        'feature': selected_features,
                        'importance': rf_model.feature_importances_
                    }).sort_values('importance', ascending=False).head(15)
                    
                    fig_importance = go.Figure(go.Bar(
                        x=feature_importance['importance'],
                        y=feature_importance['feature'],
                        orientation='h',
                        marker_color='skyblue'
                    ))
                    fig_importance.update_layout(
                        title="Top 15 Most Important Features",
                        xaxis_title="Importance",
                        yaxis_title="Features",
                        template="plotly_white",
                        height=500
                    )
                    st.plotly_chart(fig_importance, use_container_width=True)
                
                # LSTM training history
                if ("LSTM" in selected_model or "Ensemble" in selected_model) and lstm_history is not None:
                    st.subheader("🧠 LSTM Training History")
                    
                    fig_history = go.Figure()
                    fig_history.add_trace(go.Scatter(
                        y=lstm_history.history['loss'],
                        mode='lines',
                        name='Training Loss',
                        line=dict(color='blue')
                    ))
                    fig_history.add_trace(go.Scatter(
                        y=lstm_history.history['val_loss'],
                        mode='lines',
                        name='Validation Loss',
                        line=dict(color='red')
                    ))
                    fig_history.update_layout(
                        title="LSTM Training Progress",
                        xaxis_title="Epoch",
                        yaxis_title="Loss",
                        template="plotly_white",
                        height=400
                    )
                    st.plotly_chart(fig_history, use_container_width=True)
                
                # Make future prediction
                current_price = main_data['Close'].iloc[-1]
                future_prediction = 0
                predicted_price = current_price
                
                if ("Random Forest" in selected_model and rf_model is not None and 
                    X_selected is not None and len(X_selected) > 0):
                    latest_features = X_selected[-1].reshape(1, -1)
                    future_prediction = rf_model.predict(latest_features)[0]
                elif ("LSTM" in selected_model and lstm_model is not None and 
                    feature_scaler is not None and target_scaler is not None):
                    # Create sequence for LSTM prediction
                    seq_len = sequence_length if 'sequence_length' in locals() else 60
                    if len(ml_data) >= seq_len:
                        try:
                            latest_sequence = ml_data[feature_columns].iloc[-seq_len:].values
                            latest_sequence_scaled = feature_scaler.transform(latest_sequence)
                            latest_sequence_lstm = latest_sequence_scaled.reshape(1, seq_len, len(feature_columns))
                            
                            lstm_pred_scaled = lstm_model.predict(latest_sequence_lstm, verbose=0)[0]
                            future_prediction = target_scaler.inverse_transform(np.array([[lstm_pred_scaled]]))[0][0]
                        except Exception as e:
                            st.warning(f"LSTM prediction failed: {str(e)}")
                            future_prediction = 0
                
                predicted_price = current_price * (1 + future_prediction)
                
                st.header("🔮 Price Prediction")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Current Price", f"${current_price:.2f}")
                with col2:
                    st.metric(f"Predicted Price ({prediction_days}d)", f"${predicted_price:.2f}")
                with col3:
                    st.metric("Predicted Change", f"{future_prediction:.2%}")
                
                # Enhanced confidence assessment with None checking
                r2_safe = r2 if r2 is not None else 0
                mse_safe = mse if mse is not None else 1
                directional_safe = directional_accuracy if directional_accuracy is not None else 0.5
                
                confidence_score = max(0, r2_safe)
                
                if "Random Forest" in selected_model:
                    cv_confidence = max(0.0, float(avg_cv_score))
                    directional_confidence = avg_cv_directional
                    overall_confidence = (confidence_score * 0.4 + cv_confidence * 0.4 + directional_confidence * 0.2)
                else:
                    overall_confidence = (confidence_score * 0.6 + directional_safe * 0.4)
                
                if overall_confidence > 0.4:
                    confidence = "High"
                    confidence_color = "🟢"
                elif overall_confidence > 0.2:
                    confidence = "Medium"
                    confidence_color = "🟡"
                else:
                    confidence = "Low"
                    confidence_color = "🔴"
                
                st.info(f"**Prediction Confidence:** {confidence_color} {confidence} (Score: {overall_confidence:.3f})")
                st.info(f"**Model Quality:** R² = {r2_safe:.3f}, RMSE = {np.sqrt(mse_safe):.4f}, Directional = {directional_safe:.2%}")
                
                # Model insights
                if r2_safe > 0.1:
                    st.success("✅ Model shows positive predictive power!")
                elif r2_safe > 0:
                    st.warning("⚠️ Model shows weak predictive power. Consider more data or different features.")
                else:
                    st.warning("⚠️ Model performance is poor. Try extending the date range or using ensemble methods.")
                
                # Prediction vs actual chart
                if y_test is not None and y_pred is not None and len(y_test) > 0:
                    st.subheader("📈 Model Performance Visualization")
                    
                    fig_perf = go.Figure()
                    
                    # Scatter plot
                    fig_perf.add_trace(go.Scatter(
                        x=y_test,
                        y=y_pred,
                        mode='markers',
                        name='Predictions',
                        marker=dict(color='blue', opacity=0.6)
                    ))
                    
                    # Perfect prediction line
                    y_test_np = np.array(y_test, dtype=np.float64)
                    y_pred_np = np.array(y_pred, dtype=np.float64)
                    min_val = min(float(np.min(y_test_np)), float(np.min(y_pred_np)))
                    max_val = max(float(np.max(y_test_np)), float(np.max(y_pred_np)))
                    fig_perf.add_trace(go.Scatter(
                        x=[min_val, max_val],
                        y=[min_val, max_val],
                        mode='lines',
                        name='Perfect Prediction',
                        line=dict(color='red', dash='dash')
                    ))
                    
                    fig_perf.update_layout(
                        title=f"Actual vs Predicted Returns ({prediction_days}-day horizon)",
                        xaxis_title="Actual Returns",
                        yaxis_title="Predicted Returns",
                        template="plotly_white",
                        height=500
                    )
                    st.plotly_chart(fig_perf, use_container_width=True)
                
                # === CORRELATION PATTERN ANALYSIS ===
                if correlation_patterns:
                    st.header("🔍 Historical Correlation Pattern Analysis")
                    
                    # Create tabs for different aspects
                    pattern_tabs = st.tabs([" Correlation Cycles", "📅 Seasonal Patterns", "🔮 Upcoming Predictions"])
                    
                    with pattern_tabs[0]:  # Correlation Cycles
                        st.subheader("Correlation Cycle Analysis")
                        
                        for var_name, patterns in correlation_patterns.items():
                            if patterns and 'cycle_patterns' in patterns and patterns['cycle_patterns']:
                                cycles = patterns['cycle_patterns']
                                
                                st.write(f"**{var_name} Cycles:**")
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    st.metric("Average Cycle Length", f"{cycles.get('average_cycle', 0):.0f} days")
                                with col2:
                                    st.metric("Cycle Variability", f"±{cycles.get('cycle_std', 0):.0f} days")
                                with col3:
                                    if cycles.get('last_peak'):
                                        days_since_peak = (main_data.index[-1] - cycles['last_peak']).days
                                        st.metric("Days Since Last Peak", f"{days_since_peak}")
                                
                                # Cycle visualization
                                if 'rolling_correlation' in patterns:
                                    fig_cycle = go.Figure()
                                    
                                    corr_data = patterns['rolling_correlation']
                                    fig_cycle.add_trace(go.Scatter(
                                        x=corr_data.index,
                                        y=corr_data.values,
                                        mode='lines',
                                        name='Correlation',
                                        line=dict(color='blue')
                                    ))
                                    
                                    # Mark peaks and troughs
                                    if cycles.get('peak_dates'):
                                        peak_corrs = [corr_data.loc[date] for date in cycles['peak_dates'] if date in corr_data.index]
                                        fig_cycle.add_trace(go.Scatter(
                                            x=cycles['peak_dates'][:len(peak_corrs)],
                                            y=peak_corrs,
                                            mode='markers',
                                            name='Peaks',
                                            marker=dict(color='red', size=8, symbol='triangle-up')
                                        ))
                                    
                                    if cycles.get('trough_dates'):
                                        trough_corrs = [corr_data.loc[date] for date in cycles['trough_dates'] if date in corr_data.index]
                                        fig_cycle.add_trace(go.Scatter(
                                            x=cycles['trough_dates'][:len(trough_corrs)],
                                            y=trough_corrs,
                                            mode='markers',
                                            name='Troughs',
                                            marker=dict(color='green', size=8, symbol='triangle-down')
                                        ))
                                    
                                    fig_cycle.update_layout(
                                        title=f"{var_name} Correlation Cycles",
                                        xaxis_title="Date",
                                        yaxis_title="Correlation",
                                        template="plotly_white",
                                        height=400
                                    )
                                    st.plotly_chart(fig_cycle, use_container_width=True)
                                
                                st.markdown("---")
                    
                    with pattern_tabs[1]:  # Seasonal Patterns
                        st.subheader("Seasonal Correlation Patterns")
                        
                        # Create comprehensive seasonal analysis
                        seasonal_assets = []
                        all_monthly_data = {}
                        all_quarterly_data = {}
                        
                        for var_name, patterns in correlation_patterns.items():
                            if patterns and 'seasonal_patterns' in patterns and patterns['seasonal_patterns']:
                                seasonal = patterns['seasonal_patterns']
                                
                                # Get month names for display
                                strongest_month_num = seasonal.get('strongest_month', 0)
                                weakest_month_num = seasonal.get('weakest_month', 0)
                                
                                month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                            'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                                
                                strongest_month_name = month_names[strongest_month_num - 1] if 1 <= strongest_month_num <= 12 else 'N/A'
                                weakest_month_name = month_names[weakest_month_num - 1] if 1 <= weakest_month_num <= 12 else 'N/A'
                                
                                seasonal_assets.append({
                                    'Asset': var_name,
                                    'Strongest Month': strongest_month_name,
                                    'Weakest Month': weakest_month_name,
                                    'Monthly Variation': f"{seasonal.get('seasonal_strength', {}).get('monthly_variation', 0):.3f}",
                                    'Strongest Quarter': f"Q{seasonal.get('strongest_quarter', 'N/A')}",
                                    'Quarterly Variation': f"{seasonal.get('seasonal_strength', {}).get('quarterly_variation', 0):.3f}"
                                })
                                
                                # Store monthly data for visualization
                                monthly_patterns = seasonal.get('monthly_patterns', {})
                                if monthly_patterns and 'mean' in monthly_patterns:
                                    monthly_corrs = []
                                    for month in range(1, 13):
                                        corr_val = monthly_patterns['mean'].get(month, 0)
                                        monthly_corrs.append(corr_val)
                                    all_monthly_data[var_name] = monthly_corrs
                                
                                # Store quarterly data
                                quarterly_patterns = seasonal.get('quarterly_patterns', {})
                                if quarterly_patterns and 'mean' in quarterly_patterns:
                                    quarterly_corrs = []
                                    for quarter in range(1, 5):
                                        corr_val = quarterly_patterns['mean'].get(quarter, 0)
                                        quarterly_corrs.append(corr_val)
                                    all_quarterly_data[var_name] = quarterly_corrs
                        
                        # Display summary table
                        if seasonal_assets:
                            st.write("**📊 Seasonal Pattern Summary:**")
                            seasonal_df = pd.DataFrame(seasonal_assets)
                            st.dataframe(seasonal_df, use_container_width=True)
                            
                            st.markdown("---")
                            
                            # Enhanced visualizations
                            if all_monthly_data:
                                # Monthly patterns - Multiple assets heatmap
                                st.write("**📅 Monthly Correlation Patterns:**")
                                
                                # Prepare data for heatmap
                                month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                            'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                                
                                heatmap_data = []
                                asset_names = []
                                
                                for asset, monthly_corrs in all_monthly_data.items():
                                    heatmap_data.append(monthly_corrs)
                                    asset_names.append(asset)
                                
                                # Create enhanced heatmap
                                fig_monthly = go.Figure(data=go.Heatmap(
                                    z=heatmap_data,
                                    x=month_names,
                                    y=asset_names,
                                    colorscale='RdBu',
                                    zmid=0,
                                    text=[[f"{val:.3f}" for val in row] for row in heatmap_data],
                                    texttemplate="%{text}",
                                    textfont={"size": 10},
                                    hoverongaps=False,
                                    colorbar=dict(
                                        title="Correlation"
                                    )
                                ))
                                
                                fig_monthly.update_layout(
                                    title="Monthly Correlation Heatmap - All Assets",
                                    xaxis_title="Month",
                                    yaxis_title="Assets",
                                    template="plotly_white",
                                    height=max(200, len(asset_names) * 60),
                                    font=dict(size=12)
                                )
                                st.plotly_chart(fig_monthly, use_container_width=True)
                                
                                # Monthly line chart for better trend visualization
                                st.write("**📈 Monthly Correlation Trends:**")
                                fig_monthly_lines = go.Figure()
                                
                                colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                                        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
                                
                                for i, (asset, monthly_corrs) in enumerate(all_monthly_data.items()):
                                    fig_monthly_lines.add_trace(go.Scatter(
                                        x=month_names,
                                        y=monthly_corrs,
                                        mode='lines+markers',
                                        name=asset,
                                        line=dict(color=colors[i % len(colors)], width=3),
                                        marker=dict(size=8)
                                    ))
                                
                                fig_monthly_lines.update_layout(
                                    title="Monthly Correlation Patterns - Trend Lines",
                                    xaxis_title="Month",
                                    yaxis_title="Correlation",
                                    template="plotly_white",
                                    height=400,
                                    hovermode='x unified',
                                    legend=dict(
                                        orientation="h",
                                        yanchor="bottom",
                                        y=1.02,
                                        xanchor="right",
                                        x=1
                                    )
                                )
                                st.plotly_chart(fig_monthly_lines, use_container_width=True)
                            
                            # Quarterly patterns visualization
                            if all_quarterly_data:
                                st.write("**📊 Quarterly Correlation Patterns:**")
                                
                                # Quarterly bar chart
                                quarter_names = ['Q1', 'Q2', 'Q3', 'Q4']
                                fig_quarterly = go.Figure()
                                
                                for i, (asset, quarterly_corrs) in enumerate(all_quarterly_data.items()):
                                    fig_quarterly.add_trace(go.Bar(
                                        x=quarter_names,
                                        y=quarterly_corrs,
                                        name=asset,
                                        text=[f"{val:.3f}" for val in quarterly_corrs],
                                        textposition='auto',
                                        marker_color=colors[i % len(colors)]
                                    ))
                                
                                fig_quarterly.update_layout(
                                    title="Quarterly Correlation Patterns",
                                    xaxis_title="Quarter",
                                    yaxis_title="Average Correlation",
                                    template="plotly_white",
                                    height=400,
                                    barmode='group',
                                    legend=dict(
                                        orientation="h",
                                        yanchor="bottom",
                                        y=1.02,
                                        xanchor="right",
                                        x=1
                                    )
                                )
                                st.plotly_chart(fig_quarterly, use_container_width=True)
                                
                                # Seasonal insights
                                st.write("**🔍 Seasonal Insights:**")
                                
                                insights_cols = st.columns(2)
                                
                                with insights_cols[0]:
                                    st.write("**Strongest Seasonal Effects:**")
                                    for asset_info in seasonal_assets:
                                        monthly_var = float(asset_info['Monthly Variation'])
                                        if monthly_var > 0.05:  # Significant monthly variation
                                            st.write(f"• **{asset_info['Asset']}**: Peak in {asset_info['Strongest Month']}, "
                                                f"Low in {asset_info['Weakest Month']} "
                                                f"(Variation: {asset_info['Monthly Variation']})")
                                
                                with insights_cols[1]:
                                    st.write("**Quarterly Trends:**")
                                    for asset_info in seasonal_assets:
                                        quarterly_var = float(asset_info['Quarterly Variation'])
                                        if quarterly_var > 0.03:  # Significant quarterly variation
                                            st.write(f"• **{asset_info['Asset']}**: Strongest in {asset_info['Strongest Quarter']} "
                                                f"(Variation: {asset_info['Quarterly Variation']})")
                            
                            else:
                                st.info("📊 No significant seasonal patterns detected in the current data.")
                        else:
                            st.info("📅 No seasonal pattern data available for the selected assets.")
                    
                    with pattern_tabs[2]:  # Upcoming Predictions
                        st.subheader("🔮 Upcoming Correlation Predictions")
                        
                        current_date = main_data.index[-1]
                        
                        for var_name, patterns in correlation_patterns.items():
                            if patterns and 'upcoming_predictions' in patterns:
                                predictions = patterns['upcoming_predictions']
                                
                                st.write(f"**{var_name} Outlook:**")
                                
                                col1, col2, col3 = st.columns(3)
                                
                                with col1:
                                    trend = predictions.get('trend_direction', 'sideways')
                                    trend_emoji = "📈" if trend == 'up' else "📉" if trend == 'down' else "➡️"
                                    st.metric("Current Trend", f"{trend_emoji} {trend.title()}")
                                
                                with col2:
                                    current_corr = patterns['rolling_correlation'].iloc[-1] if len(patterns['rolling_correlation']) > 0 else 0
                                    seasonal_exp = predictions.get('seasonal_expectation', current_corr)
                                    vs_seasonal = predictions.get('current_vs_seasonal', 0)
                                    seasonal_status = "Above" if vs_seasonal > 0.02 else "Below" if vs_seasonal < -0.02 else "Near"
                                    st.metric("vs Seasonal Average", f"{seasonal_status} ({vs_seasonal:+.3f})")
                                
                                with col3:
                                    regime_probs = predictions.get('regime_probability', {})
                                    dominant_regime = max(regime_probs.items(), key=lambda x: x[1])[0] if regime_probs else 'Unknown'
                                    st.metric("Likely Regime", dominant_regime.replace('_', ' ').title())
                                
                                # Cycle prediction
                                cycle_pred = predictions.get('cycle_prediction')
                                if cycle_pred:
                                    pred_date = cycle_pred['expected_date']
                                    days_until = (pred_date - current_date).days
                                    confidence = cycle_pred['confidence']
                                    
                                    st.info(f"🔄 **Next {cycle_pred['type'].title()} Expected:** "
                                        f"{pred_date.strftime('%Y-%m-%d')} ({days_until} days) "
                                        f"- Confidence: {confidence:.1%}")
                                
                                # Next month prediction
                                next_month = predictions.get('next_month_outlook')
                                if next_month:
                                    st.success(f"📅 **Next Month Expectation:** "
                                            f"{next_month['expected_correlation']:.3f} "
                                            f"(Range: {next_month['range_low']:.3f} to {next_month['range_high']:.3f})")
                                
                
            except Exception as e:
                st.error(f"❌ Model training failed: {str(e)}")
                st.exception(e)

    st.markdown("---")

    # --- Granger Causality Analysis ---
    st.subheader("Granger Causality Test")
    granger_vars = list(corr_df.columns)
    default_x = granger_vars[0] if granger_vars else None
    default_y = granger_vars[1] if len(granger_vars) > 1 else None
    col_gc1, col_gc2 = st.columns(2)
    with col_gc1:
        granger_x = st.selectbox("Select variable X (cause)", granger_vars, index=0 if default_x else 0, key="granger_x")
    with col_gc2:
        granger_y = st.selectbox("Select variable Y (effect)", granger_vars, index=1 if default_y else 0, key="granger_y")
    max_lag = st.slider("Max lag for Granger test", min_value=1, max_value=365, value=3, step=1)
    if granger_x and granger_y and granger_x != granger_y:
        from statsmodels.tsa.stattools import grangercausalitytests
        test_df = corr_df[[granger_y, granger_x]].dropna()
        try:
            st.write(f"Testing if {granger_x} Granger-causes {granger_y} (lags 1 to {max_lag})")
            gc_results = grangercausalitytests(test_df, max_lag, verbose=False)
            gc_table = []
            for lag in range(1, max_lag+1):
                pval = gc_results[lag][0]['ssr_ftest'][1]
                significant = "Yes" if pval < 0.05 else "No"
                gc_table.append({"Lag": lag, "p-value": pval, "Significant": significant})
            import pandas as pd
            # Only show first 3 and last 3 lags
            if max_lag > 6:
                display_table = gc_table[:3] + gc_table[-3:]
            else:
                display_table = gc_table
            st.dataframe(pd.DataFrame(display_table), use_container_width=True)
            # Check if any displayed lag is significant
            any_significant = any(row['Significant'] == "Yes" for row in gc_table)
            if any_significant:
                st.success(f"{granger_x} Granger-causes {granger_y} at one or more lags (p < 0.05)")
            else:
                st.info(f"No significant Granger causality detected from {granger_x} to {granger_y} (all p >= 0.05)")
        except Exception as e:
            st.warning(f"Granger causality test failed: {e}")
    else:
        st.info("Select two different variables for Granger causality analysis.")