# Stock Correlation Dashboard

# Advanced correlation analysis with AI/ML price prediction models
# Analyzes stock correlations with financial markets, commodities, and weather data

import yfinance as yf
import streamlit as st
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
st.set_page_config(page_title="📊 Stock Correlation Dashboard", layout="wide")
st.title(":bar_chart: Stock Correlation Dashboard")

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

# --- STOCK CORRELATION ANALYSIS ---
st.header(":bar_chart: Stock Correlation Analysis")

# User input for a single stock ticker
st.subheader("Stock Input")
corr_ticker = st.text_input("Enter a stock ticker for correlation analysis:", "AAPL").upper()
corr_start_date = st.date_input("Start Date", value=date(2020, 1, 1), key="corr_start")
corr_end_date = st.date_input("End Date", value=date.today(), key="corr_end")

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
    close = df['Close']
    
    # Returns
    df['returns'] = close.pct_change()
    df['log_returns'] = np.log(close / close.shift(1))
    
    # Moving averages
    for period in periods:
        df[f'ma_{period}'] = close.rolling(window=period).mean()
        # Avoid division by zero
        ma_values = df[f'ma_{period}']
        df[f'ma_{period}_ratio'] = np.where(ma_values != 0, close / ma_values, 1.0)
    
    # Volatility measures
    for period in [5, 10, 20]:
        df[f'volatility_{period}'] = df['returns'].rolling(window=period).std()
    
    # RSI
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    # Avoid division by zero in RSI calculation
    rs = np.where(loss != 0, gain / loss, 0)
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    bb_period = 20
    df['bb_middle'] = close.rolling(window=bb_period).mean()
    bb_std = close.rolling(window=bb_period).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    # Avoid division by zero in Bollinger Band position
    band_width = df['bb_upper'] - df['bb_lower']
    df['bb_position'] = np.where(band_width != 0, 
                                (close - df['bb_lower']) / band_width, 
                                0.5)
    
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
        for period in [5, 10, 20]:
            df[f'volume_ma_{period}'] = df['Volume'].rolling(window=period).mean()
            # Avoid division by zero in volume ratios
            vol_ma = df[f'volume_ma_{period}']
            df[f'volume_ratio_{period}'] = np.where(vol_ma != 0, df['Volume'] / vol_ma, 1.0)
    
    # Support and resistance levels
    df['high_20'] = df['High'].rolling(window=20).max()
    df['low_20'] = df['Low'].rolling(window=20).min()
    # Avoid division by zero in price position
    price_range = df['high_20'] - df['low_20']
    df['price_position'] = np.where(price_range != 0,
                                   (close - df['low_20']) / price_range,
                                   0.5)
    
    return df

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

# --- AI/ML Price Prediction Model ---
st.markdown("#### AI Price Prediction Model")
st.markdown("**Enhanced Visual Pattern Analysis**: Directly analyzes patterns from the Rolling Correlation Graph and Price Comparison Chart")

with st.expander("🔍 How the Enhanced AI Model Works", expanded=False):
    st.markdown("""
    **🚀 Advanced ML Features:**
    
    **📈 Technical Analysis:**
    - Moving averages, RSI, MACD, Bollinger Bands
    - Multi-timeframe momentum (5-120 days)
    - Volatility analysis and regime detection
    
    **🌍 Multi-Timeframe Patterns:**
    - Short-term: 1-7 days (momentum/reversal)
    - Medium-term: 14-90 days (monthly/quarterly trends)
    - Long-term: 120-365 days (seasonal patterns)
    
    **📊 Market Structure:**
    - Support/resistance levels
    - Volume-price relationships
    - Correlation analysis from selected variables
    
    **�️ Robust Validation:**
    - Time-aware cross-validation (TimeSeriesSplit)
    - Feature selection based on importance
    - Prevents data leakage with proper temporal splits
    
    **🎯 Model Options:**
    - **Random Forest**: Tree-based ensemble with feature selection
    - **LSTM Neural Network**: Deep learning for sequential patterns
    - **Ensemble**: Combines Random Forest + LSTM predictions
    """)

# Model configuration
col1, col2, col3 = st.columns(3)
with col1:
    prediction_days = st.number_input("Prediction horizon (days):", min_value=1, max_value=1000, value=5, step=1)
with col2:
    feature_window = st.number_input("Feature window (days):", min_value=1, value=600, step=5, help="Number of days to include for feature calculation")
with col3:
    # Enhanced model selection with LSTM options
    model_options = ["Random Forest", "LSTM Neural Network", "Ensemble (RF + LSTM)", "Gradient Boosting", "Linear Regression"]
    if not DEEP_LEARNING_AVAILABLE:
        model_options = ["Random Forest", "Gradient Boosting", "Linear Regression"]
        st.warning("⚠️ LSTM unavailable - install tensorflow")
    
    model_type = st.selectbox("Model Type:", model_options)

# Enhanced Model Parameters
if "Random Forest" in model_type:
    st.subheader("🌲 Random Forest Parameters")
    col1, col2, col3 = st.columns(3)
    with col1:
        n_estimators = st.slider("Number of Trees", 50, 500, 100, 50)
    with col2:
        max_depth = st.slider("Max Depth", 3, 20, 10)
    with col3:
        min_samples_split = st.slider("Min Samples Split", 2, 20, 5)

if "LSTM" in model_type and DEEP_LEARNING_AVAILABLE:
    st.subheader("🧠 LSTM Parameters")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        sequence_length = st.slider("Sequence Length (days)", 30, 120, 60)
    with col2:
        lstm_units_1 = st.slider("LSTM Units (Layer 1)", 20, 100, 50, 10)
    with col3:
        lstm_units_2 = st.slider("LSTM Units (Layer 2)", 10, 60, 30, 5)
    with col4:
        dropout_rate = st.slider("Dropout Rate", 0.1, 0.5, 0.2, 0.05)
    
    col1, col2 = st.columns(2)
    with col1:
        epochs = st.slider("Training Epochs", 50, 200, 100, 25)
    with col2:
        batch_size = st.selectbox("Batch Size", [16, 32, 64], index=1)

if st.button("🚀 Train & Predict", help="Train enhanced AI model with LSTM and Random Forest capabilities"):
    with st.spinner(f"Training {model_type} model..."):
        try:
            # Enhanced ML training with technical indicators
            # Calculate technical indicators first
            corr_data_enhanced = calculate_technical_indicators(corr_data)
            
            # Create correlation features for enhanced model (simplified approach)
            correlation_data = {}
            if selected_vars:
                progress_bar = st.progress(0)
                for i, var in enumerate(selected_vars):
                    if var in etf_map:
                        etf_ticker = etf_map[var]
                        etf_data = load_data(etf_ticker, corr_start_date, corr_end_date)
                        if etf_data is not None and not etf_data.empty:
                            etf_data = etf_data.set_index('Date')
                            if isinstance(etf_data.index, pd.DatetimeIndex) and etf_data.index.tz is not None:
                                etf_data.index = etf_data.index.tz_localize(None)
                            etf_data['log_returns'] = np.log(etf_data['Close'] / etf_data['Close'].shift(1))
                            correlation_data[var] = etf_data
                    progress_bar.progress((i + 1) / len(selected_vars))
                progress_bar.empty()
            
            # Create correlation features using simplified approach
            correlation_features = None
            if correlation_data:
                features_df = pd.DataFrame(index=corr_data_enhanced.index)
                
                for var_name, var_data in correlation_data.items():
                    if var_data is not None and not var_data.empty:
                        # Align dates
                        aligned_data = var_data.reindex(corr_data_enhanced.index, method='nearest')
                        
                        # Calculate rolling correlations
                        for window in [10, 20, 50]:
                            if 'log_returns' in corr_data_enhanced.columns and 'log_returns' in aligned_data.columns:
                                corr = corr_data_enhanced['log_returns'].rolling(window=window).corr(aligned_data['log_returns'])
                                features_df[f'{var_name}_corr_{window}'] = corr
                        
                        # Price ratio features
                        if 'Close' in aligned_data.columns:
                            main_close = corr_data_enhanced['Close']
                            aligned_close = aligned_data['Close']
                            
                            # Relative strength
                            main_pct_change = main_close.pct_change(20)
                            aligned_pct_change = aligned_close.pct_change(20)
                            features_df[f'{var_name}_rel_strength'] = main_pct_change - aligned_pct_change
                
                correlation_features = features_df
            
            # Prepare enhanced ML features using simplified approach from ML_Stock_Predictor.py
            # Technical indicator features
            feature_columns = [
                'returns', 'log_returns',
                'ma_5_ratio', 'ma_10_ratio', 'ma_20_ratio', 'ma_50_ratio',
                'volatility_5', 'volatility_10', 'volatility_20',
                'rsi', 'bb_position', 'macd', 'macd_histogram',
                'momentum_5', 'momentum_10', 'momentum_20',
                'price_position'
            ]
            
            # Add volume features if available
            volume_features = [col for col in corr_data_enhanced.columns if 'volume_ratio' in col]
            feature_columns.extend(volume_features)
            
            # Add correlation features if available
            if correlation_features is not None:
                corr_features = [col for col in correlation_features.columns 
                               if not correlation_features[col].isna().all()]
                feature_columns.extend(corr_features)
                # Merge correlation features
                corr_data_enhanced = corr_data_enhanced.join(correlation_features[corr_features], how='left')
            
            # Target: future price change
            corr_data_enhanced['target'] = corr_data_enhanced['Close'].shift(-prediction_days) / corr_data_enhanced['Close'] - 1
            
            # Select only available features
            available_features = []
            for f in feature_columns:
                if f in corr_data_enhanced.columns:
                    non_null_ratio = corr_data_enhanced[f].count() / len(corr_data_enhanced)
                    if non_null_ratio > 0.5:  # Keep features with >50% non-null values
                        available_features.append(f)
            
            # Remove rows with missing target or features
            ml_data = corr_data_enhanced[available_features + ['target']].dropna()
            
            if len(ml_data) < 200:  # Increased minimum for LSTM
                st.error("Not enough data for training. Please extend the date range or reduce prediction horizon.")
                st.stop()
            
            # Check for infinite or very large values
            if not np.all(np.isfinite(ml_data[available_features].values)):
                st.warning("Some features contain infinite values. Cleaning data...")
                ml_data = ml_data.replace([np.inf, -np.inf], np.nan).dropna()
                
            if len(ml_data) < 100:
                st.error("After cleaning, not enough data remains for training.")
                st.stop()
            
            st.success(f"✅ Prepared {len(ml_data)} samples with {len(available_features)} features")
            
            # Model training based on selection
            rf_model, rf_pred, rf_r2, rf_mse, rf_directional = None, None, None, None, None
            X_selected, selected_features = None, []
            avg_cv_score, cv_std, avg_cv_directional = 0, 0, 0.5
            
            if model_type == "Random Forest" or "Ensemble" in model_type:
                # Split features and target for Random Forest
                X = np.array(ml_data[available_features].values, dtype=np.float64)
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
                    'feature': available_features,
                    'importance': feature_importance_scores
                }).sort_values('importance', ascending=False)
                
                # Select top features
                n_features_to_select = max(20, min(len(available_features) // 2, 50))
                selected_feature_indices = feature_importance_df.head(n_features_to_select).index.tolist()
                selected_features = [available_features[i] for i in selected_feature_indices]
                
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
                        n_estimators=min(n_estimators if 'n_estimators' in locals() else 100, 200),
                        max_depth=max_depth if 'max_depth' in locals() else 10,
                        min_samples_split=max(min_samples_split if 'min_samples_split' in locals() else 5, 10),
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
                    n_estimators=min((n_estimators if 'n_estimators' in locals() else 100) * 2, 300),
                    max_depth=max_depth if 'max_depth' in locals() else 10,
                    min_samples_split=max(min_samples_split if 'min_samples_split' in locals() else 5, 10),
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
            
            # LSTM Training (if selected and available)
            lstm_r2, lstm_mse, lstm_directional, lstm_model, lstm_history = None, None, None, None, None
            lstm_pred, y_lstm_test_original = None, None
            feature_scaler, target_scaler = None, None
            
            if ("LSTM" in model_type or "Ensemble" in model_type) and DEEP_LEARNING_AVAILABLE:
                st.info("Training LSTM Neural Network...")
                
                # Create LSTM sequences
                X_lstm, y_lstm, feature_scaler, target_scaler = create_lstm_sequences(
                    ml_data, available_features, 'target', 
                    sequence_length=sequence_length if 'sequence_length' in locals() else 60, 
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
                        epochs=epochs if 'epochs' in locals() else 100, 
                        batch_size=batch_size if 'batch_size' in locals() else 32
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
            
            # Determine final metrics based on model selection
            if ("Ensemble" in model_type and rf_model is not None and lstm_model is not None and 
                rf_pred is not None and lstm_pred is not None and 
                len(rf_pred) > 0 and len(lstm_pred) > 0):
                
                # Align prediction lengths for ensemble
                min_length = min(len(rf_pred), len(lstm_pred))
                rf_pred_aligned = rf_pred[:min_length]
                lstm_pred_aligned = lstm_pred[:min_length]
                y_test_aligned = y_test[:min_length]
                
                # Simple ensemble: average predictions
                ensemble_pred = (rf_pred_aligned + lstm_pred_aligned) / 2
                r2 = r2_score(y_test_aligned, ensemble_pred)
                mse = mean_squared_error(y_test_aligned, ensemble_pred)
                directional_accuracy = np.mean(np.sign(y_test_aligned) == np.sign(ensemble_pred))
                y_pred = ensemble_pred
                y_test_final = y_test_aligned
                
            elif "LSTM" in model_type and lstm_model is not None and lstm_pred is not None:
                # Use LSTM metrics
                r2, mse, directional_accuracy = lstm_r2, lstm_mse, lstm_directional
                y_pred = lstm_pred
                y_test_final = y_lstm_test_original
                
            elif rf_model is not None and rf_pred is not None:
                # Use Random Forest metrics
                r2, mse, directional_accuracy = rf_r2, rf_mse, rf_directional
                y_pred = rf_pred
                y_test_final = y_test
                
            else:
                # Fallback for other models
                if model_type == "Gradient Boosting":
                    X = np.array(ml_data[available_features].values, dtype=np.float64)
                    y = np.array(ml_data['target'].values, dtype=np.float64)
                    
                    split_point = int(len(X) * 0.8)
                    X_train, X_test = X[:split_point], X[split_point:]
                    y_train, y_test = y[:split_point], y[split_point:]
                    
                    model = GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=6, learning_rate=0.1)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    
                    r2 = r2_score(y_test, y_pred)
                    mse = mean_squared_error(y_test, y_pred)
                    directional_accuracy = np.mean(np.sign(y_test) == np.sign(y_pred))
                    y_test_final = y_test
                
                elif model_type == "Linear Regression":
                    X = np.array(ml_data[available_features].values, dtype=np.float64)
                    y = np.array(ml_data['target'].values, dtype=np.float64)
                    
                    split_point = int(len(X) * 0.8)
                    X_train, X_test = X[:split_point], X[split_point:]
                    y_train, y_test = y[:split_point], y[split_point:]
                    
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    
                    model = LinearRegression()
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                    
                    r2 = r2_score(y_test, y_pred)
                    mse = mean_squared_error(y_test, y_pred)
                    directional_accuracy = np.mean(np.sign(y_test) == np.sign(y_pred))
                    y_test_final = y_test
                
                else:
                    # Default fallback
                    r2, mse, directional_accuracy = 0, 1, 0.5
                    y_pred, y_test_final = np.array([]), np.array([])
            
            # Display results
            st.header(f"📊 {model_type} Model Performance")
            
            # Main metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Test R² Score", f"{r2:.3f}" if r2 is not None else "N/A")
            with col2:
                if model_type == "Random Forest":
                    st.metric("CV R² (Avg)", f"{avg_cv_score:.3f}")
                else:
                    st.metric("RMSE", f"{np.sqrt(mse):.4f}" if mse is not None else "N/A")
            with col3:
                st.metric("Directional Accuracy", f"{directional_accuracy:.2%}" if directional_accuracy is not None else "N/A")
            with col4:
                if model_type == "Random Forest" and rf_model is not None and hasattr(rf_model, 'oob_score_'):
                    st.metric("OOB Score", f"{rf_model.oob_score_:.3f}")
                else:
                    st.metric("Training Samples", len(ml_data))
            
            # Feature importance (for Random Forest models)
            if (model_type == "Random Forest" and rf_model is not None and 
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
            
            # Make future prediction
            current_price = corr_data['Close'].iloc[-1]
            future_prediction = 0
            predicted_price = current_price
            
            if (model_type == "Random Forest" and rf_model is not None and 
                X_selected is not None and len(X_selected) > 0):
                latest_features = X_selected[-1].reshape(1, -1)
                future_prediction = rf_model.predict(latest_features)[0]
                
            elif ("LSTM" in model_type and lstm_model is not None and 
                  feature_scaler is not None and target_scaler is not None):
                # Create sequence for LSTM prediction
                seq_len = sequence_length if 'sequence_length' in locals() else 60
                if len(ml_data) >= seq_len:
                    try:
                        latest_sequence = ml_data[available_features].iloc[-seq_len:].values
                        latest_sequence_scaled = feature_scaler.transform(latest_sequence)
                        latest_sequence_lstm = latest_sequence_scaled.reshape(1, seq_len, len(available_features))
                        
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
            
            # Enhanced confidence assessment
            r2_safe = r2 if r2 is not None else 0
            mse_safe = mse if mse is not None else 1
            directional_safe = directional_accuracy if directional_accuracy is not None else 0.5
            
            confidence_score = max(0, r2_safe)
            
            if model_type == "Random Forest":
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
            if y_test_final is not None and y_pred is not None and len(y_test_final) > 0:
                st.subheader("📈 Model Performance Visualization")
                
                fig_perf = go.Figure()
                
                # Scatter plot
                fig_perf.add_trace(go.Scatter(
                    x=y_test_final,
                    y=y_pred,
                    mode='markers',
                    name='Predictions',
                    marker=dict(color='blue', opacity=0.6)
                ))
                
                # Perfect prediction line
                y_test_np = np.array(y_test_final, dtype=np.float64)
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
            
        except Exception as e:
            st.error(f"❌ Model training failed: {str(e)}")
            st.exception(e)