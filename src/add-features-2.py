import pandas as pd
import numpy as np
import os

# Configuration
INPUT_PATH = 'data/raw.csv'
OUTPUT_PATH = 'data/engineered.csv'

def load_data(filepath):
    """
    Loads data and ensures Date is datetime and sorted.
    """
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by=['Ticker', 'Date'])
    return df

# ==========================================
# 1. Trend & Momentum Features
# ==========================================

def add_log_returns(df):
    """
    Calculates Logarithmic Returns.
    Why: Log returns are time-additive and statistically preferable to simple % change.
    """
    # Log(Current Price / Previous Price)
    df['Log_Ret'] = np.log(df['Adj Close'] / df['Adj Close'].shift(1))
    return df

def add_sma(df, window=50):
    """
    Simple Moving Average (SMA).
    Why: Smooths out price data to identify the trend direction.
    """
    df[f'SMA_{window}'] = df['Adj Close'].rolling(window=window).mean()
    # Feature: Distance from SMA (Are we overextended?)
    df[f'Dist_SMA_{window}'] = (df['Adj Close'] - df[f'SMA_{window}']) / df[f'SMA_{window}']
    return df

def add_ema(df, span=20):
    """
    Exponential Moving Average (EMA).
    Why: Similar to SMA but gives more weight to recent prices. Reacts faster.
    """
    df[f'EMA_{span}'] = df['Adj Close'].ewm(span=span, adjust=False).mean()
    return df

def add_rsi(df, window=14):
    """
    Relative Strength Index (RSI).
    Why: Measures the speed and change of price movements. 
    Values > 70 indicate overbought, < 30 indicate oversold.
    """
    delta = df['Adj Close'].diff()
    
    # Separate gains and losses
    gain = (delta.where(delta > 0, 0))
    loss = (-delta.where(delta < 0, 0))

    # Use Exponential Weighted Moving Average for Wilder's Smoothing
    avg_gain = gain.ewm(alpha=1/window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/window, min_periods=window, adjust=False).mean()

    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df

def add_macd(df):
    """
    Moving Average Convergence Divergence (MACD).
    Why: A trend-following momentum indicator. 
    Consists of the MACD line (fast - slow EMA) and the Signal line.
    """
    ema_12 = df['Adj Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Adj Close'].ewm(span=26, adjust=False).mean()
    
    df['MACD'] = ema_12 - ema_26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal'] # The histogram
    return df

# ==========================================
# 2. Volatility Features
# ==========================================

def add_rolling_std(df, window=21):
    """
    Rolling Standard Deviation (Volatility).
    Why: Measures how 'wild' the price swings are. 21 days approx 1 trading month.
    """
    df[f'Volatility_{window}d'] = df['Log_Ret'].rolling(window=window).std()
    return df

def add_bollinger_bands(df, window=20, num_std=2):
    """
    Bollinger Bands.
    Why: Identifies when prices are high (upper band) or low (lower band) relative to recent range.
    """
    sma = df['Adj Close'].rolling(window=window).mean()
    std = df['Adj Close'].rolling(window=window).std()
    
    df['BB_Upper'] = sma + (std * num_std)
    df['BB_Lower'] = sma - (std * num_std)
    
    # Feature: %B (Percent Bandwidth) - Where is price relative to bands?
    # 1.0 = at upper band, 0.0 = at lower band
    df['BB_Percent_B'] = (df['Adj Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
    return df

def add_atr(df, window=14):
    """
    Average True Range (ATR).
    Why: A measure of market volatility that takes gaps (high-low) into account.
    """
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Adj Close'].shift(1))
    low_close = np.abs(df['Low'] - df['Adj Close'].shift(1))
    
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    
    df['ATR'] = true_range.rolling(window=window).mean()
    return df

# ==========================================
# 3. Volume Features
# ==========================================

def add_volume_features(df):
    """
    Volume Analysis.
    Why: Volume confirms trends.
    """
    # Rate of change in volume
    df['Vol_Change'] = df['Volume'].pct_change()
    
    # Relative Volume: Current volume vs 20-day average
    df['Vol_SMA_20'] = df['Volume'].rolling(window=20).mean()
    df['Rel_Vol'] = df['Volume'] / df['Vol_SMA_20']
    return df

# ==========================================
# Main Pipeline
# ==========================================

def process_ticker_data(ticker_df):
    """
    Applies all feature engineering functions to a single ticker's dataframe.
    """
    # Make a copy to avoid SettingWithCopy warnings
    df = ticker_df.copy()
    
    # Apply functions sequentially
    df = add_log_returns(df)
    df = add_sma(df, window=50)       # Mid-term trend
    df = add_sma(df, window=200)      # Long-term trend
    df = add_ema(df, span=20)
    df = add_rsi(df)
    df = add_macd(df)
    df = add_rolling_std(df)
    df = add_bollinger_bands(df)
    df = add_atr(df)
    df = add_volume_features(df)
    
    return df

def main():
    # 1. Setup
    if not os.path.exists('data'):
        os.makedirs('data')
        
    # 2. Load Data
    try:
        df = load_data(INPUT_PATH)
    except FileNotFoundError:
        print(f"Error: {INPUT_PATH} not found.")
        return

    # 3. Apply Features Per Ticker
    print("Engineering features per ticker...")
    df_engineered = df.groupby('Ticker', group_keys=False).apply(process_ticker_data)

    # 4. Clean up Data
    # Drop NaNs created by rolling windows (the "warm-up" period)
    initial_rows = len(df_engineered)
    df_engineered.dropna(inplace=True)
    
    # --- NEW STEP: REMOVE 2010 ---
    # Since we don't have 2009 data, the indicators for 2010 are either 
    # NaN (handled above) or statistically unstable (calculated on too few days).
    # We start the dataset cleanly from 2011.
    print("Removing data from 2010 (insufficient history for lag features)...")
    df_engineered = df_engineered[df_engineered['Date'].dt.year >= 2011]
    
    rows_dropped = initial_rows - len(df_engineered)
    print(f"Dropped {rows_dropped} rows total (NaNs + Year 2010).")

    # 5. Save
    df_engineered.to_csv(OUTPUT_PATH, index=False)
    print(f"Success! Feature engineering complete. Saved to {OUTPUT_PATH}")
    print(f"Data range: {df_engineered['Date'].min().date()} to {df_engineered['Date'].max().date()}")
    print(f"Columns created: {list(df_engineered.columns)}")

if __name__ == "__main__":
    main()