import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import date, timedelta, datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from scipy import stats
from scipy.signal import find_peaks
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
    st.warning("⚠️ TensorFlow not available. Install with: pip install tensorflow")

# Page configuration
st.set_page_config(page_title="ML Stock Predictor", layout="wide")
st.title("AI Stock Price Predictor")
st.markdown("**Advanced Random Forest Model** - Predicts stock prices using technical indicators and market correlations")

# Cached data loading function
@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_stock_data(ticker, start_date, end_date):
    """Load stock data with error handling"""
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(start=start_date, end=end_date)
        if data.empty:
            return None
        data = data.reset_index()
        return data
    except Exception as e:
        st.error(f"Error loading {ticker}: {str(e)}")
        return None

# Technical indicators calculation
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
        # Avoid division by zero with proper Series alignment
        ma_values = df[f'ma_{period}']
        ratio_result = close / ma_values
        df[f'ma_{period}_ratio'] = ratio_result.fillna(1.0)
    
    # Volatility measures
    for period in [5, 10, 20]:
        df[f'volatility_{period}'] = df['returns'].rolling(window=period).std()
    
    # RSI
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    # Avoid division by zero in RSI calculation
    rs_calc = gain / loss
    rs = rs_calc.fillna(0)
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    bb_period = 20
    df['bb_middle'] = close.rolling(window=bb_period).mean()
    bb_std = close.rolling(window=bb_period).std()
    df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
    df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
    # Avoid division by zero in Bollinger Band position
    band_width = df['bb_upper'] - df['bb_lower']
    bb_position_calc = (close - df['bb_lower']) / band_width
    df['bb_position'] = bb_position_calc.fillna(0.5)
    
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
            volume_ratio_calc = df['Volume'] / vol_ma
            df[f'volume_ratio_{period}'] = volume_ratio_calc.fillna(1.0)
    
    # Support and resistance levels
    df['high_20'] = df['High'].rolling(window=20).max()
    df['low_20'] = df['Low'].rolling(window=20).min()
    # Avoid division by zero in price position
    price_range = df['high_20'] - df['low_20']
    price_position_calc = (close - df['low_20']) / price_range
    df['price_position'] = price_position_calc.fillna(0.5)
    
    return df

def create_correlation_features(main_data, correlation_data, correlation_vars):
    """Create correlation-based features"""
    features_df = pd.DataFrame(index=main_data.index)
    
    for var_name, var_data in correlation_data.items():
        if var_data is not None and not var_data.empty:
            # Align dates
            aligned_data = var_data.reindex(main_data.index, method='nearest')
            
            # Calculate rolling correlations
            for window in [10, 20, 50]:
                corr = main_data['log_returns'].rolling(window=window).corr(
                    aligned_data['log_returns'] if 'log_returns' in aligned_data.columns 
                    else aligned_data['Close'].pct_change()
                )
                features_df[f'{var_name}_corr_{window}'] = corr
            
            # Price ratio features
            if 'Close' in aligned_data.columns:
                # Normalize both to same starting point for comparison
                main_close = main_data['Close']
                aligned_close = aligned_data['Close']
                
                # Avoid division by zero in normalization
                main_first = main_close.iloc[0] if len(main_close) > 0 else 1.0
                aligned_first = aligned_close.iloc[0] if len(aligned_close) > 0 else 1.0
                
                if main_first != 0 and aligned_first != 0:
                    main_norm = main_close / main_first
                    var_norm = aligned_close / aligned_first
                    # Avoid division by zero in ratio calculation
                    features_df[f'{var_name}_price_ratio'] = np.where(var_norm != 0, 
                                                                     main_norm / var_norm, 1.0)
                else:
                    features_df[f'{var_name}_price_ratio'] = 1.0
                
                # Relative strength
                main_pct_change = main_close.pct_change(20)
                aligned_pct_change = aligned_close.pct_change(20)
                features_df[f'{var_name}_rel_strength'] = main_pct_change - aligned_pct_change
    
    return features_df

def prepare_ml_features(data, correlation_features, lookback_days=60):
    """Prepare features for ML model with advanced lag-based features"""
    feature_columns = []
    
    # Technical indicator features
    technical_features = [
        'returns', 'log_returns',
        'ma_5_ratio', 'ma_10_ratio', 'ma_20_ratio', 'ma_50_ratio',
        'volatility_5', 'volatility_10', 'volatility_20',
        'rsi', 'bb_position', 'macd', 'macd_histogram',
        'momentum_5', 'momentum_10', 'momentum_20',
        'price_position'
    ]
    
    # Add volume features if available
    volume_features = [col for col in data.columns if 'volume_ratio' in col]
    technical_features.extend(volume_features)
    
    # ENHANCED LAG FEATURES - Multi-timeframe analysis
    lag_periods = [1, 2, 3, 5, 7, 10, 14, 21, 30, 60, 90, 120, 200, 365]  # 1 day to 1 year
    lag_features = []
    
    # Price-based lag features
    close_prices = data['Close']
    for lag in lag_periods:
        if lag < len(data):
            # Price returns over different periods
            lag_col = f'return_lag_{lag}d'
            data[lag_col] = (close_prices / close_prices.shift(lag) - 1).fillna(0)
            lag_features.append(lag_col)
            
            # Price momentum acceleration (change in momentum)
            if lag > 1:
                momentum_col = f'momentum_accel_{lag}d'
                current_momentum = close_prices / close_prices.shift(lag) - 1
                prev_momentum = close_prices.shift(lag) / close_prices.shift(lag*2) - 1
                data[momentum_col] = (current_momentum - prev_momentum).fillna(0)
                lag_features.append(momentum_col)
    
    # Technical indicator lag features
    for feature in ['returns', 'log_returns', 'rsi', 'bb_position', 'macd']:
        if feature in data.columns:
            for lag in [1, 2, 3, 5, 7, 14, 30, 60]:  # Shorter lags for technical indicators
                lag_col = f'{feature}_lag_{lag}'
                data[lag_col] = data[feature].shift(lag).fillna(data[feature].mean())
                lag_features.append(lag_col)
    
    # SEASONAL AND CYCLICAL FEATURES
    seasonal_features = []
    
    # Add date-based features
    if hasattr(data.index, 'dayofweek'):
        data['day_of_week'] = data.index.dayofweek
        data['month'] = data.index.month
        data['quarter'] = data.index.quarter
        seasonal_features.extend(['day_of_week', 'month', 'quarter'])
    
    # Yearly patterns (if we have enough data)
    if len(data) > 365:
        # Year-over-year comparisons
        data['yoy_return'] = (close_prices / close_prices.shift(365) - 1).fillna(0)
        seasonal_features.append('yoy_return')
        
        # Seasonal momentum patterns
        for period in [90, 180, 270]:  # Quarterly, semi-annual, 9-month
            if period < len(data):
                seasonal_col = f'seasonal_momentum_{period}d'
                data[seasonal_col] = (close_prices / close_prices.shift(period) - 1).fillna(0)
                seasonal_features.append(seasonal_col)
    
    # VOLATILITY REGIME FEATURES
    volatility_features = []
    for short_period, long_period in [(5, 20), (10, 50), (20, 100), (30, 200)]:
        if long_period < len(data):
            short_vol = data['returns'].rolling(short_period).std()
            long_vol = data['returns'].rolling(long_period).std()
            vol_ratio_col = f'vol_ratio_{short_period}_{long_period}'
            data[vol_ratio_col] = (short_vol / long_vol).fillna(1.0)
            volatility_features.append(vol_ratio_col)
            
            # Volatility regime indicator
            vol_regime_col = f'vol_regime_{short_period}_{long_period}'
            data[vol_regime_col] = (short_vol > 1.5 * long_vol).astype(int)
            volatility_features.append(vol_regime_col)
    
    # TREND STRENGTH FEATURES
    trend_features = []
    for period in [10, 20, 50, 100, 200]:
        if period < len(data):
            # Linear trend strength
            trend_col = f'trend_strength_{period}d'
            rolling_returns = data['returns'].rolling(period)
            trend_strength = rolling_returns.apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == period else 0)
            data[trend_col] = trend_strength.fillna(0)
            trend_features.append(trend_col)
            
            # Trend consistency (how often price moves in same direction)
            consistency_col = f'trend_consistency_{period}d'
            sign_changes = data['returns'].rolling(period).apply(lambda x: np.sum(np.diff(np.sign(x)) != 0))
            data[consistency_col] = (period - sign_changes) / period  # Higher = more consistent
            data[consistency_col] = data[consistency_col].fillna(0.5)
            trend_features.append(consistency_col)
    
    # MARKET REGIME FEATURES
    regime_features = []
    
    # Bull/Bear market indicators
    for period in [50, 100, 200]:
        if period < len(data):
            ma_long = close_prices.rolling(period).mean()
            bull_col = f'bull_market_{period}d'
            data[bull_col] = (close_prices > ma_long).astype(int)
            regime_features.append(bull_col)
            
            # Distance from long-term MA
            distance_col = f'ma_distance_{period}d'
            data[distance_col] = ((close_prices - ma_long) / ma_long).fillna(0)
            regime_features.append(distance_col)
    
    # PRICE LEVEL FEATURES
    level_features = []
    
    # Distance from recent highs/lows
    for period in [20, 50, 100, 252]:  # 1 month, ~2.5 months, ~5 months, 1 year
        if period < len(data):
            high_period = close_prices.rolling(period).max()
            low_period = close_prices.rolling(period).min()
            
            high_distance_col = f'high_distance_{period}d'
            low_distance_col = f'low_distance_{period}d'
            
            data[high_distance_col] = ((high_period - close_prices) / close_prices).fillna(0)
            data[low_distance_col] = ((close_prices - low_period) / close_prices).fillna(0)
            
            level_features.extend([high_distance_col, low_distance_col])
            
            # Percentile position within period
            percentile_col = f'percentile_position_{period}d'
            data[percentile_col] = close_prices.rolling(period).rank(pct=True).fillna(0.5)
            level_features.append(percentile_col)
    
    # Combine all features
    all_features = (technical_features + lag_features + seasonal_features + 
                   volatility_features + trend_features + regime_features + level_features)
    
    # Add correlation features
    if correlation_features is not None:
        corr_features = [col for col in correlation_features.columns 
                        if not correlation_features[col].isna().all()]
        all_features.extend(corr_features)
        
        # Merge correlation features
        data = data.join(correlation_features[corr_features], how='left')
    
    # Select features that exist in the data and have sufficient non-null values
    available_features = []
    for f in all_features:
        if f in data.columns:
            non_null_ratio = data[f].count() / len(data)
            if non_null_ratio > 0.5:  # Keep features with >50% non-null values
                available_features.append(f)
    
    return data, available_features

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
st.header("🎯 Model Configuration")

# Basic configuration in columns
col1, col2, col3, col4 = st.columns(4)
with col1:
    ticker = st.text_input("Stock Ticker", value="AAPL").upper()
with col2:
    start_date = st.date_input("Start Date", value=date.today() - timedelta(days=365*3))
with col3:
    end_date = st.date_input("End Date", value=date.today())
with col4:
    prediction_days = st.slider("Prediction Horizon (days)", 1, 30, 5)

st.markdown("---")

# Model selection section
st.subheader("Model Selection")
model_options = ["Random Forest", "LSTM Neural Network", "Ensemble (RF + LSTM)"]
if not DEEP_LEARNING_AVAILABLE:
    model_options = ["Random Forest"]
    st.warning("LSTM unavailable - install tensorflow")

col1, col2 = st.columns([1, 2])
with col1:
    selected_model = st.selectbox("Choose Model:", model_options)
with col2:
    # Correlation variables selection
    correlation_options = {
        "SPY": "S&P 500 ETF",
        "QQQ": "NASDAQ ETF", 
        "GLD": "Gold ETF",
        "USO": "Oil ETF",
        "TLT": "Treasury Bond ETF",
        "^VIX": "Volatility Index",
        "DXY": "US Dollar Index",
        "BTC-USD": "Bitcoin"
    }
    
    selected_correlations = st.multiselect(
        "Select correlation variables:",
        options=list(correlation_options.keys()),
        default=["SPY", "^VIX"],
        format_func=lambda x: correlation_options[x]
    )

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

st.markdown("---")

# ML Model Training Section
st.header(f" {selected_model} Training & Prediction")

if st.button("🚀 Train Model & Predict", type="primary"):
    with st.spinner(f"Training {selected_model} model..."):
        try:
            # Load main stock data
            with st.spinner(f"Loading {ticker} data..."):
                main_data = load_stock_data(ticker, start_date, end_date)
            
            if main_data is None or main_data.empty:
                st.error(f"Could not load data for {ticker}")
                st.stop()
            
            # Set date as index
            main_data['Date'] = pd.to_datetime(main_data['Date'])
            main_data.set_index('Date', inplace=True)
            
            # Calculate technical indicators
            main_data = calculate_technical_indicators(main_data)
            
            # Load correlation data
            correlation_data = {}
            if selected_correlations:
                progress_bar = st.progress(0)
                for i, corr_ticker in enumerate(selected_correlations):
                    progress_bar.progress((i + 1) / len(selected_correlations))
                    corr_data = load_stock_data(corr_ticker, start_date, end_date)
                    if corr_data is not None:
                        corr_data['Date'] = pd.to_datetime(corr_data['Date'])
                        corr_data.set_index('Date', inplace=True)
                        corr_data['log_returns'] = np.log(corr_data['Close'] / corr_data['Close'].shift(1))
                        correlation_data[corr_ticker] = corr_data
                progress_bar.empty()
            
            # Create correlation features
            correlation_features = None
            correlation_patterns = None
            if correlation_data:
                correlation_features = create_correlation_features(
                    main_data, correlation_data, selected_correlations
                )
                
                # Analyze correlation patterns
                st.info("🔍 Analyzing historical correlation patterns...")
                correlation_patterns = analyze_correlation_patterns(
                    main_data, correlation_data, window=60
                )
            
            # Prepare ML features
            main_data, feature_columns = prepare_ml_features(
                main_data, correlation_features, lookback_days=60
            )
            
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

