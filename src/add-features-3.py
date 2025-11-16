import pandas as pd
import numpy as np
import yfinance as yf
import os

# Configuration
INPUT_PATH = 'data/raw.csv'
OUTPUT_PATH = 'data/engineered.csv'

def load_data(filepath):
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by=['Ticker', 'Date'])
    return df

# ==========================================
# NEW: Market Data & Relative Features
# ==========================================

def get_market_data(start_date, end_date):
    """
    Downloads SPY (S&P 500), QQQ (Nasdaq), and VIX (Volatility).
    Returns a clean dataframe with these benchmarks.
    """
    print("Downloading Market Data (SPY, QQQ, VIX)...")
    tickers = ['SPY', 'QQQ', '^VIX']
    
    # buffer start date by 6 months for rolling windows
    buffer_start = pd.to_datetime(start_date) - pd.DateOffset(months=6)
    
    # --- FIX IS HERE ---
    # We must set auto_adjust=False to force yfinance to give us the 'Adj Close' column
    # We also add multi_level_index=False (if available in your version) or handle the columns manually
    try:
        mkt_data = yf.download(tickers, start=buffer_start, end=end_date, progress=False, auto_adjust=False)
    except TypeError:
        # Fallback for older versions if auto_adjust param causes issues, though unlikely
        mkt_data = yf.download(tickers, start=buffer_start, end=end_date, progress=False)

    # Extract Adj Close. 
    # The structure is usually MultiIndex: Level 0 = Price Type, Level 1 = Ticker
    if 'Adj Close' in mkt_data.columns.get_level_values(0):
        df_mkt = mkt_data['Adj Close']
    else:
        # Fallback: If Adj Close is missing (newer defaults), use Close
        # This handles the case where yfinance ignores the flag or structure changes
        print("Warning: 'Adj Close' not found, using 'Close' as proxy.")
        df_mkt = mkt_data['Close']
    
    # Clean up columns (Removes MultiIndex if present)
    df_mkt.columns.name = None 
    
    # Rename columns for clarity
    df_mkt = df_mkt.rename(columns={'SPY': 'SPY_Close', 'QQQ': 'QQQ_Close', '^VIX': 'VIX_Close'})
    
    # Calculate Market Returns
    df_mkt['SPY_Ret'] = np.log(df_mkt['SPY_Close'] / df_mkt['SPY_Close'].shift(1))
    df_mkt['QQQ_Ret'] = np.log(df_mkt['QQQ_Close'] / df_mkt['QQQ_Close'].shift(1))
    
    # VIX is already a percentage-like number, but change is useful
    df_mkt['VIX_Change'] = df_mkt['VIX_Close'].diff()
    
    # Market Moving Averages (Trend Regime)
    df_mkt['SPY_SMA_200'] = df_mkt['SPY_Close'].rolling(window=200).mean()
    df_mkt['Is_Market_Bullish'] = (df_mkt['SPY_Close'] > df_mkt['SPY_SMA_200']).astype(int)
    
    return df_mkt

def add_beta_and_relative_metrics(df):
    """
    Calculates Beta and Relative Strength. 
    Requires 'Log_Ret' and 'SPY_Ret' to exist in the dataframe.
    """
    # 1. Excess Return (Alpha proxy)
    df['Excess_Ret_SPY'] = df['Log_Ret'] - df['SPY_Ret']
    df['Excess_Ret_QQQ'] = df['Log_Ret'] - df['QQQ_Ret']
    
    # 2. Relative Strength (Price Ratio)
    df['Rel_Str_SPY'] = df['Adj Close'] / df['SPY_Close']
    
    # 3. Rolling Beta (Sensitivity)
    # We use a 60-day (approx 3-month) rolling window
    window = 60
    
    cov = df['Log_Ret'].rolling(window=window).cov(df['SPY_Ret'])
    var = df['SPY_Ret'].rolling(window=window).var()
    
    df['Rolling_Beta'] = cov / var
    
    return df

# ==========================================
# Previous Standard Features
# ==========================================
def add_technical_indicators(df):
    # Returns
    df['Log_Ret'] = np.log(df['Adj Close'] / df['Adj Close'].shift(1))
    
    # Moving Averages
    df['SMA_50'] = df['Adj Close'].rolling(window=50).mean()
    df['Dist_SMA_50'] = (df['Adj Close'] - df['SMA_50']) / df['SMA_50']
    df['SMA_200'] = df['Adj Close'].rolling(window=200).mean()
    
    # RSI
    delta = df['Adj Close'].diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/14, min_periods=14).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, min_periods=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Volatility
    df['Volatility_21d'] = df['Log_Ret'].rolling(window=21).std()
    
    # Volume
    df['Rel_Vol'] = df['Volume'] / df['Volume'].rolling(window=20).mean()
    
    return df

# ==========================================
# Main Pipeline
# ==========================================

def main():
    # 1. Load Stock Data
    if not os.path.exists(INPUT_PATH):
        print(f"Error: {INPUT_PATH} not found.")
        return
    df = load_data(INPUT_PATH)
    
    # 2. Load Market Data (SPY/QQQ/VIX)
    min_date = df['Date'].min()
    max_date = df['Date'].max()
    
    # The fixed function is called here
    df_market = get_market_data(min_date, max_date)
    
    # 3. Merge Market Data onto Stock Data
    print("Merging Market Data...")
    df = df.merge(df_market, left_on='Date', right_index=True, how='left')

    # 4. Engineering Per Ticker
    print("Calculating Stock-Specific Features...")
    
    def process_ticker(group):
        group = add_technical_indicators(group)
        group = add_beta_and_relative_metrics(group)
        return group

    df_engineered = df.groupby('Ticker', group_keys=False).apply(process_ticker)

    # 5. Clean up (Remove NaNs and 2010)
    initial_rows = len(df_engineered)
    df_engineered.dropna(inplace=True)
    
    print("Removing data prior to 2011...")
    df_engineered = df_engineered[df_engineered['Date'].dt.year >= 2011]
    
    rows_dropped = initial_rows - len(df_engineered)
    print(f"Dropped {rows_dropped} rows (NaNs + Pre-2011).")

    # 6. Save
    df_engineered.to_csv(OUTPUT_PATH, index=False)
    print(f"Success! Saved to {OUTPUT_PATH}")
    print(f"New Market Features: 'Rolling_Beta', 'Rel_Str_SPY', 'VIX_Close', 'Excess_Ret_SPY'")

if __name__ == "__main__":
    main()