import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import yfinance as yf


# Relative Strenght Index calculation:
def calculate_rsi(data, window=14):
    """
    Uses the Exponential Weighted Moving Average (EWMA) method (Wilder's Smoothing),
    which is considered the standard for RSI calculation.
    """
    delta = data.diff()
    
    # Separate gains (positive changes) and losses (negative changes)
    gain = (delta.where(delta > 0, 0))
    loss = (-delta.where(delta < 0, 0))
    
    # Use EWM for smoothing (alpha = 1/window)
    avg_gain = gain.ewm(alpha=1/window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/window, min_periods=window, adjust=False).mean()

    # Calculate Relative Strength (RS)
    rs = avg_gain / avg_loss
    
    # Calculate RSI
    return 100 - (100 / (1 + rs))


# Return calculation:
def add_return_features(df):

    # Percentage change returns (1, 3, 6 and 12 months)
    df['return_1m'] = df['Close'].pct_change(21)
    df['return_3m'] = df['Close'].pct_change(63)
    df['return_6m'] = df['Close'].pct_change(126)
    df['return_12m'] = df['Close'].pct_change(252)
    
    # Cumulative return (for better aggregation/compounding)
    df['cum_return_3m'] = (1 + df['Close'].pct_change()).rolling(63).apply(lambda x: (1+x).prod() - 1, raw=True)
    
    # Momentum/Acceleration (Are returns speeding up or slowing down?)
    df['return_acceleration'] = df['return_1m'] - df['return_3m']
    
    # Logarithmic returns (for use in statistical calculations like volatility/beta)
    df['Log_Ret'] = np.log(df['Close'] / df['Close'].shift(1))
    
    return df


# Volatility, ATR and Bollinger Bands:
def add_volatility_features(df):
    """Add volatility-based features"""
    daily_returns = df['Close'].pct_change()
    
    # Rolling Standard Deviation (Annualized volatility)
    df['volatility_1m'] = daily_returns.rolling(21).std() * np.sqrt(252)
    df['volatility_3m'] = daily_returns.rolling(63).std() * np.sqrt(252)
    df['volatility_6m'] = daily_returns.rolling(126).std() * np.sqrt(252)
    df['vol_ratio'] = df['volatility_1m'] / df['volatility_6m'] # Squeeze/Expansion indicator
    
    # High-low range volatility
    df['hl_ratio'] = (df['High'] - df['Low']) / df['Close']
    df['hl_volatility'] = df['hl_ratio'].rolling(21).mean()
    
    # Average True Range (better for handling price gaps)
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift(1))
    low_close = np.abs(df['Low'] - df['Close'].shift(1))
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    df['ATR_14'] = true_range.rolling(14).mean()
    df['ATR_normalized'] = df['ATR_14'] / df['Close']
    
    # Bollinger Bands (Calculated on 20-day SMA, 2-StdDev)
    sma = df['Close'].rolling(window=20).mean()
    std = df['Close'].rolling(window=20).std()
    upper = sma + (std * 2)
    lower = sma - (std * 2)
    df['Bollinger_PctB'] = (df['Close'] - lower) / (upper - lower)
    df['Bollinger_Width'] = (upper - lower) / sma
    
    return df

# Volume:
def add_volume_features(df):
    # Relative volume
    df['volume_ratio_1m'] = df['Volume'] / df['Volume'].rolling(21).mean()
    df['volume_ratio_3m'] = df['Volume'] / df['Volume'].rolling(63).mean()
    
    # Price-Volume correlation (Is volume confirming the recent price move?)
    df['pv_corr_1m'] = df['Close'].pct_change().rolling(21).corr(df['Volume'].pct_change())
    df['volume_change_1m'] = df['Volume'].pct_change(21)
    
    return df

# SMA, RSI, MACD:
def add_technical_indicators(df):
    """Add core trend/momentum indicators"""

    # Simple Moving Averages
    df['sma_20'] = df['Close'].rolling(20).mean()
    df['sma_50'] = df['Close'].rolling(50).mean()
    df['sma_200'] = df['Close'].rolling(200).mean()
    
    # Ratios (Normalized trend features)
    df['price_to_sma20'] = df['Close'] / df['sma_20'] - 1
    df['price_to_sma50'] = df['Close'] / df['sma_50'] - 1
    df['price_to_sma200'] = df['Close'] / df['sma_200'] - 1
    
    # Crossover feature (Long-term regime)
    df['ma_cross_50_200'] = (df['sma_50'] / df['sma_200']) - 1
    
    # Momentum indicators
    df['rsi_14'] = calculate_rsi(df['Close'], 14)

    # Moving Average Convergence Divergence:
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    df['MACD_hist'] = macd - signal
    df['MACD_hist_norm'] = df['MACD_hist'] / df['Close']
    
    return df

# Time-based features (seasonality):
def add_time_features(df):
    df['month'] = df['Date'].dt.month
    df['quarter'] = df['Date'].dt.quarter
    df['day_of_week'] = df['Date'].dt.dayofweek
    
    # Sin/Cos transformation for cyclical features
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    return df

# Cummax and drawdown:
def add_drawdown_features(df):
    """Add drawdown features (Risk/Max Loss)"""
    df['cummax'] = df['Close'].cummax()
    df['drawdown'] = (df['Close'] - df['cummax']) / df['cummax']
    # Drop intermediate column
    df = df.drop(columns=['cummax'])
    return df

# Lag features (short-term memory):
def add_lag_features(df):
    # Lagged returns
    df['return_lag_1'] = df['Close'].pct_change().shift(1)
    df['return_lag_2'] = df['Close'].pct_change().shift(2)
    df['return_lag_3'] = df['Close'].pct_change().shift(3)
    
    # Lagged volume changes
    df['vol_change_lag_1'] = df['Volume'].pct_change().shift(1)
    
    return df

# Nasdaq, S&P500 (ETF) and Volatility index features:
def add_market_features(stock_df, market_data):
    """
    Add market context features (Relative Strength, Beta, VIX)
    NOTE: Uses 'Close' for all calculations, as 'Adj Close' is not guaranteed 
    in all datasets/download methods.
    """
    # NASDAQ
    nasdaq = market_data[market_data['Ticker'] == '^IXIC'][['Date', 'Close']].copy()
    nasdaq.columns = ['Date', 'nasdaq_close']
    nasdaq['nasdaq_return_1m'] = nasdaq['nasdaq_close'].pct_change(21)
    
    # SPY (S&P 500 ETF)
    spy = market_data[market_data['Ticker'] == 'SPY'][['Date', 'Close']].copy()
    spy.columns = ['Date', 'spy_close']
    spy['spy_return_1m'] = spy['spy_close'].pct_change(21)
    
    # Calculate SPY-200 day SMA ratio (market regime indicator)
    spy_sma_raw = spy['spy_close'].rolling(200).mean()
    spy['spy_sma200_ratio'] = (spy['spy_close'] / spy_sma_raw) - 1
    
    # VIX (Volatility index)
    vix = market_data[market_data['Ticker'] == '^VIX'][['Date', 'Close']].copy()
    vix.columns = ['Date', 'vix_level']
    vix['vix_change_1m'] = vix['vix_level'].pct_change(21)
    
    # Merge data
    stock_df = stock_df.merge(nasdaq[['Date', 'nasdaq_return_1m', 'nasdaq_close']], on='Date', how='left')
    stock_df = stock_df.merge(spy[['Date', 'spy_return_1m', 'spy_sma200_ratio']], on='Date', how='left')
    stock_df = stock_df.merge(vix[['Date', 'vix_level', 'vix_change_1m']], on='Date', how='left')
    
    # Fill NaNs (Market data might have different holidays/trading days)
    for col in ['nasdaq_return_1m', 'spy_return_1m', 'vix_level', 'vix_change_1m', 'spy_sma200_ratio']:
        stock_df[col] = stock_df[col].fillna(method='ffill')
    
    # Relative strength (Stock return minus Market return)
    stock_df['relative_strength_nasdaq'] = stock_df['return_1m'] - stock_df['nasdaq_return_1m']
    stock_df['relative_strength_spy'] = stock_df['return_1m'] - stock_df['spy_return_1m']
    
    # Beta calculation (Rolling 63-day Beta vs Nasdaq)
    stock_returns = stock_df['Close'].pct_change()
    nasdaq_returns = stock_df['nasdaq_close'].pct_change()
    stock_df['beta_nasdaq'] = stock_returns.rolling(63).cov(nasdaq_returns) / nasdaq_returns.rolling(63).var()
    
    # Drop raw price columns (cleanup)
    stock_df = stock_df.drop(['nasdaq_close'], axis=1)
    
    return stock_df

# Target variable (= future 1-month return):
def add_target_variable(df):
    # Calculates the percent change 21 days *in the future*
    df['target_return_1m'] = df['Close'].pct_change(21).shift(-21)
    return df

# MAIN PIPELINE

def engineer_features_for_ticker(ticker_df, market_data):
    """Engineer all features for a single ticker"""
    df = ticker_df.copy()
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Note: Order matters for some features (e.g., return_1m must exist before market_features)
    df = add_return_features(df)
    df = add_volatility_features(df)
    df = add_volume_features(df)
    df = add_technical_indicators(df)
    df = add_time_features(df)
    df = add_drawdown_features(df)
    df = add_lag_features(df) # Lag features should be near the end
    
    # Market features rely on 'return_1m' and 'Date' for merging
    df = add_market_features(df, market_data)
    
    # Target variable relies on all previous 'Close' data
    df = add_target_variable(df)
    
    return df

def main():
    print("=" * 70)
    print("CONSOLIDATED FEATURE ENGINEERING PIPELINE")
    print("=" * 70)
    
    # Configuration (assuming standard file paths)
    INPUT_PATH = 'data/raw.csv'
    OUTPUT_PATH = 'data/engineered.csv'
    
    if not os.path.exists('data'):
        os.makedirs('data')

    # Load raw data
    print("\n1. Loading raw data from data/raw.csv...")
    try:
        raw_data = pd.read_csv(INPUT_PATH)
        raw_data['Date'] = pd.to_datetime(raw_data['Date'])
    except FileNotFoundError:
        print(f"Error: {INPUT_PATH} not found. Please ensure the file exists.")
        return
        
    # Separate tech stocks from market indices
    market_tickers = ['^IXIC', '^GSPC', 'SPY', '^VIX']
    tech_stocks = raw_data[~raw_data['Ticker'].isin(market_tickers)].copy()
    market_data = raw_data[raw_data['Ticker'].isin(market_tickers)].copy()
    
    tech_tickers = tech_stocks['Ticker'].unique()
    print(f"   Processing {len(tech_tickers)} tech stocks...")
    
    # Process each ticker
    engineered_data = []
    
    # Use the full `Close` price for all calculations as 'Adj Close' is not consistently
    # available across all source files, and a simple merger should maintain consistency.
    # NOTE: The original files used a mix of 'Adj Close' and 'Close'. The merged file uses 'Close' for uniformity.

    print("2. Engineering features per ticker...")
    for ticker in tqdm(tech_tickers, desc="Adding all consolidated features"):
        ticker_df = tech_stocks[tech_stocks['Ticker'] == ticker].copy()
        ticker_engineered = engineer_features_for_ticker(ticker_df, market_data)
        engineered_data.append(ticker_engineered)
    
    # Combine
    final_data = pd.concat(engineered_data, ignore_index=True)
    final_data = final_data.sort_values(['Date', 'Ticker']).reset_index(drop=True)
    
    # Clean and filter
    initial_rows = len(final_data)
    print("\n3. Cleaning up data (Dropping NaNs)...")
    final_data.dropna(inplace=True)
    
    print("4. Filtering data (2011 onwards)...")
    data_2011 = final_data[final_data['Date'] >= '2011-01-01'].copy()
    
    rows_dropped = initial_rows - len(data_2011)
    print(f"   Dropped {rows_dropped:,} rows total (NaNs + Pre-2011).")

    # Save to CSV
    print(f"5. Saving to {OUTPUT_PATH}...")
    data_2011.to_csv(OUTPUT_PATH, index=False)
  
    print("\n" + "=" * 70)
    print("SUCCESS! CONSOLIDATED FEATURE ENGINEERING COMPLETE.")
    print(f"File saved: {OUTPUT_PATH}")
    print(f"Final rows: {len(data_2011):,}")
    print(f"Final columns: {len(data_2011.columns)}")
    print("=" * 70)

if __name__ == "__main__":
    main()