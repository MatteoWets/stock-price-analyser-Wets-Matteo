import pandas as pd
import numpy as np
from tqdm import tqdm

def calculate_rsi(data, window=14):
    """Calculate Relative Strength Index"""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def add_return_features(df):
    """Add return-based features"""
    df['return_1m'] = df['Close'].pct_change(21)
    df['return_3m'] = df['Close'].pct_change(63)
    df['return_6m'] = df['Close'].pct_change(126)
    df['return_12m'] = df['Close'].pct_change(252)
    df['cum_return_3m'] = (1 + df['Close'].pct_change()).rolling(63).apply(lambda x: (1+x).prod() - 1, raw=True)
    df['return_acceleration'] = df['return_1m'] - df['return_3m']
    return df

def add_volatility_features(df):
    """Add volatility-based features"""
    daily_returns = df['Close'].pct_change()
    df['volatility_1m'] = daily_returns.rolling(21).std() * np.sqrt(252)
    df['volatility_3m'] = daily_returns.rolling(63).std() * np.sqrt(252)
    df['volatility_6m'] = daily_returns.rolling(126).std() * np.sqrt(252)
    df['vol_ratio'] = df['volatility_1m'] / df['volatility_6m']
    df['hl_ratio'] = (df['High'] - df['Low']) / df['Close']
    df['hl_volatility'] = df['hl_ratio'].rolling(21).mean()
    return df

def add_volume_features(df):
    """Add volume-based features"""
    df['volume_ratio_1m'] = df['Volume'] / df['Volume'].rolling(21).mean()
    df['volume_ratio_3m'] = df['Volume'] / df['Volume'].rolling(63).mean()
    df['pv_corr_1m'] = df['Close'].pct_change().rolling(21).corr(df['Volume'].pct_change())
    df['volume_change_1m'] = df['Volume'].pct_change(21)
    return df

def add_technical_indicators(df):
    """Add technical indicators"""
    df['sma_20'] = df['Close'].rolling(20).mean()
    df['sma_50'] = df['Close'].rolling(50).mean()
    df['sma_200'] = df['Close'].rolling(200).mean()
    df['price_to_sma20'] = df['Close'] / df['sma_20'] - 1
    df['price_to_sma50'] = df['Close'] / df['sma_50'] - 1
    df['price_to_sma200'] = df['Close'] / df['sma_200'] - 1
    df['ma_cross_50_200'] = (df['sma_50'] / df['sma_200']) - 1
    df['rsi_14'] = calculate_rsi(df['Close'], 14)
    return df

def add_time_features(df):
    """Add time-based features"""
    df['month'] = df['Date'].dt.month
    df['quarter'] = df['Date'].dt.quarter
    df['day_of_week'] = df['Date'].dt.dayofweek
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    return df

def add_drawdown_features(df):
    """Add drawdown features"""
    df['cummax'] = df['Close'].cummax()
    df['drawdown'] = (df['Close'] - df['cummax']) / df['cummax']
    return df

def add_market_features(stock_df, market_data):
    """Add market context features"""
    # Get Nasdaq data
    nasdaq = market_data[market_data['Ticker'] == '^IXIC'][['Date', 'Close']].copy()
    nasdaq.columns = ['Date', 'nasdaq_close']
    nasdaq['nasdaq_return_1m'] = nasdaq['nasdaq_close'].pct_change(21)
    
    # Get SPY data
    spy = market_data[market_data['Ticker'] == 'SPY'][['Date', 'Close']].copy()
    spy.columns = ['Date', 'spy_close']
    spy['spy_return_1m'] = spy['spy_close'].pct_change(21)
    
    # Get VIX data
    vix = market_data[market_data['Ticker'] == '^VIX'][['Date', 'Close']].copy()
    vix.columns = ['Date', 'vix_level']
    vix['vix_change_1m'] = vix['vix_level'].pct_change(21)
    
    # Merge market data with stock data
    stock_df = stock_df.merge(nasdaq[['Date', 'nasdaq_return_1m', 'nasdaq_close']], on='Date', how='left')
    stock_df = stock_df.merge(spy[['Date', 'spy_return_1m', 'spy_close']], on='Date', how='left')
    stock_df = stock_df.merge(vix[['Date', 'vix_level', 'vix_change_1m']], on='Date', how='left')
    
    # Forward fill missing values (market not open on some days)
    stock_df['nasdaq_return_1m'] = stock_df['nasdaq_return_1m'].fillna(method='ffill')
    stock_df['spy_return_1m'] = stock_df['spy_return_1m'].fillna(method='ffill')
    stock_df['vix_level'] = stock_df['vix_level'].fillna(method='ffill')
    stock_df['vix_change_1m'] = stock_df['vix_change_1m'].fillna(method='ffill')
    
    # Relative strength (stock vs market)
    stock_df['relative_strength_nasdaq'] = stock_df['return_1m'] - stock_df['nasdaq_return_1m']
    stock_df['relative_strength_spy'] = stock_df['return_1m'] - stock_df['spy_return_1m']
    
    # Calculate beta (rolling 63-day window)
    stock_returns = stock_df['Close'].pct_change()
    nasdaq_returns = stock_df['nasdaq_close'].pct_change()
    
    # Beta calculation using rolling window
    stock_df['beta_nasdaq'] = stock_returns.rolling(63).cov(nasdaq_returns) / nasdaq_returns.rolling(63).var()
    
    # Drop intermediate columns
    stock_df = stock_df.drop(['nasdaq_close', 'spy_close'], axis=1)
    
    return stock_df

def add_target_variable(df):
    """Add target variable (future 1-month return)"""
    df['target_return_1m'] = df['Close'].pct_change(21).shift(-21)
    return df

def engineer_features_for_ticker(ticker_df, market_data):
    """Engineer all features for a single ticker"""
    # Make a copy to avoid modifying original
    df = ticker_df.copy()
    
    # Sort by date
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Add all features
    df = add_return_features(df)
    df = add_volatility_features(df)
    df = add_volume_features(df)
    df = add_technical_indicators(df)
    df = add_time_features(df)
    df = add_drawdown_features(df)
    df = add_market_features(df, market_data)
    df = add_target_variable(df)
    
    return df

def main():
    print("=" * 70)
    print("FEATURE ENGINEERING FROM RAW DATA")
    print("=" * 70)
    
    # Load raw data
    print("\n1. Loading raw data from data/raw.csv...")
    # Ensure you have the raw file, or this will error
    raw_data = pd.read_csv('data/raw.csv')
    raw_data['Date'] = pd.to_datetime(raw_data['Date'])
    
    print(f"   Total rows: {len(raw_data):,}")
    print(f"   Date range: {raw_data['Date'].min()} to {raw_data['Date'].max()}")
    print(f"   Unique tickers: {raw_data['Ticker'].nunique()}")
    
    # Separate tech stocks from market indices
    market_tickers = ['^IXIC', '^GSPC', 'SPY', '^VIX']
    tech_stocks = raw_data[~raw_data['Ticker'].isin(market_tickers)].copy()
    market_data = raw_data[raw_data['Ticker'].isin(market_tickers)].copy()
    
    tech_tickers = tech_stocks['Ticker'].unique()
    print(f"\n2. Processing {len(tech_tickers)} tech stocks...")
    
    # Process each ticker
    engineered_data = []
    
    for ticker in tqdm(tech_tickers, desc="Engineering features"):
        ticker_df = tech_stocks[tech_stocks['Ticker'] == ticker].copy()
        ticker_engineered = engineer_features_for_ticker(ticker_df, market_data)
        engineered_data.append(ticker_engineered)
    
    # Combine all engineered data
    print("\n3. Combining all engineered data...")
    final_data = pd.concat(engineered_data, ignore_index=True)
    
    # Sort by date and ticker
    final_data = final_data.sort_values(['Date', 'Ticker']).reset_index(drop=True)
    
    # ---------------------------------------------------------
    # MODIFICATION START: Filter for 2011 onwards
    # ---------------------------------------------------------
    print("\n4. Filtering data (2011 onwards)...")
    
    # Filter rows where Date is greater than or equal to 2011-01-01
    data_2011 = final_data[final_data['Date'] >= '2011-01-01'].copy()
    
    print("5. Saving engineered data...")
    # Save to the requested filename
    data_2011.to_csv('data/engineered.csv', index=False)
    # ---------------------------------------------------------
    # MODIFICATION END
    # ---------------------------------------------------------
  
    print("\n" + "=" * 70)
    print("FEATURE ENGINEERING COMPLETE!")
    print("=" * 70)
    print(f"\nFull dataset:")
    print(f"  - File: data/engineered.csv")
    print(f"  - Rows: {len(data_2011):,}") # Updated to show count of filtered data
    print(f"  - Columns: {len(data_2011.columns)}")
    print(f"  - Date range: {data_2011['Date'].min()} to {data_2011['Date'].max()}")
  
    # Show feature summary
    print(f"\n" + "=" * 70)
    print("FEATURES CREATED")
    print("=" * 70)
    # Use data_2011 here to match the saved file
    feature_cols = [col for col in data_2011.columns if col not in ['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']]
    print(f"Total engineered features: {len(feature_cols)}")
    print("\nFeature list:")
    for i, feat in enumerate(feature_cols, 1):
        print(f"  {i:2d}. {feat}")
    
    # Data quality check
    print(f"\n" + "=" * 70)
    print("DATA QUALITY CHECK")
    print("=" * 70)
    
    print("\n✅ All files saved successfully!")
    print("\nNext steps:")
    print("  1. Load the engineered data: pd.read_csv('data/engineered.csv')")
    print("  2. Drop rows with NaN: df.dropna()")
    print("  3. Separate features (X) and target (y)")
    print("  4. Train your models!")

if __name__ == "__main__":
    main()