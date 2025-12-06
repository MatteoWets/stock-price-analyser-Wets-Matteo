import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def calculate_rsi(prices, period=14):
    """Calculate RSI"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_base_features(df):
    """Calculate basic features from raw price data"""
    df = df.sort_values(['Ticker', 'Date']).reset_index(drop=True)
    
    for ticker in df['Ticker'].unique():
        mask = df['Ticker'] == ticker
        ticker_data = df[mask].copy()
        
        # Returns at multiple frequencies
        ticker_data['return_1d'] = ticker_data['Adj Close'].pct_change(fill_method=None)
        ticker_data['return_5d'] = ticker_data['Adj Close'].pct_change(5, fill_method=None)
        ticker_data['return_10d'] = ticker_data['Adj Close'].pct_change(10, fill_method=None)
        ticker_data['return_21d'] = ticker_data['Adj Close'].pct_change(21, fill_method=None)
        ticker_data['return_63d'] = ticker_data['Adj Close'].pct_change(63, fill_method=None)
        ticker_data['return_126d'] = ticker_data['Adj Close'].pct_change(126, fill_method=None)
        ticker_data['return_252d'] = ticker_data['Adj Close'].pct_change(252, fill_method=None)
        
        # Moving averages
        for period in [5, 10, 20, 50, 100, 200]:
            ticker_data[f'sma_{period}'] = ticker_data['Adj Close'].rolling(period).mean()
            # Note: price_to_sma is (price / sma) in this script
            ticker_data[f'price_to_sma{period}'] = ticker_data['Adj Close'] / ticker_data[f'sma_{period}']
        
        # Volume features
        ticker_data['volume_sma_20'] = ticker_data['Volume'].rolling(20).mean()
        ticker_data['volume_ratio'] = ticker_data['Volume'] / ticker_data['volume_sma_20']
        ticker_data['dollar_volume'] = ticker_data['Close'] * ticker_data['Volume']
        ticker_data['dollar_vol_20'] = ticker_data['dollar_volume'].rolling(20).mean()
        
        # Volatility at multiple scales (std dev of daily returns)
        for period in [5, 10, 21, 63, 126]:
            ticker_data[f'volatility_{period}d'] = ticker_data['return_1d'].rolling(period).std()
        
        # Volatility ratios
        ticker_data['vol_ratio_short'] = ticker_data['volatility_5d'] / ticker_data['volatility_21d']
        ticker_data['vol_ratio_medium'] = ticker_data['volatility_21d'] / ticker_data['volatility_63d']
        ticker_data['vol_ratio_long'] = ticker_data['volatility_63d'] / ticker_data['volatility_126d']
        
        # Price patterns
        ticker_data['high_low_ratio'] = ticker_data['High'] / ticker_data['Low'] - 1
        # Close position within the daily range
        ticker_data['close_to_high'] = (ticker_data['Close'] - ticker_data['Low']) / (ticker_data['High'] - ticker_data['Low'])
        ticker_data['gap'] = ticker_data['Open'] / ticker_data['Close'].shift(1) - 1
        
        # RSI at multiple periods
        for period in [7, 14, 21]:
            ticker_data[f'rsi_{period}'] = calculate_rsi(ticker_data['Adj Close'], period)
        
        # Bollinger Bands
        bb_period = 20
        bb_std = ticker_data['Adj Close'].rolling(bb_period).std()
        bb_mean = ticker_data['Adj Close'].rolling(bb_period).mean()
        ticker_data['bb_upper'] = bb_mean + (2 * bb_std)
        ticker_data['bb_lower'] = bb_mean - (2 * bb_std)
        ticker_data['bb_position'] = (ticker_data['Adj Close'] - ticker_data['bb_lower']) / (ticker_data['bb_upper'] - ticker_data['bb_lower'])
        ticker_data['bb_width'] = (ticker_data['bb_upper'] - ticker_data['bb_lower']) / bb_mean
        
        # MACD
        ema_12 = ticker_data['Adj Close'].ewm(span=12, adjust=False).mean()
        ema_26 = ticker_data['Adj Close'].ewm(span=26, adjust=False).mean()
        ticker_data['macd'] = ema_12 - ema_26
        ticker_data['macd_signal'] = ticker_data['macd'].ewm(span=9, adjust=False).mean()
        ticker_data['macd_hist'] = ticker_data['macd'] - ticker_data['macd_signal']
        
        # Distance from highs/lows
        # 52w high/low look at the 252 trading days window
        ticker_data['dist_from_52w_high'] = ticker_data['Close'] / ticker_data['Close'].rolling(252).max() - 1
        ticker_data['dist_from_52w_low'] = ticker_data['Close'] / ticker_data['Close'].rolling(252).min() - 1
        # 1m high/low look at the 21 trading days window
        ticker_data['dist_from_1m_high'] = ticker_data['Close'] / ticker_data['Close'].rolling(21).max() - 1
        ticker_data['dist_from_1m_low'] = ticker_data['Close'] / ticker_data['Close'].rolling(21).min() - 1
        
        # Update main dataframe
        for col in ticker_data.columns:
            if col not in ['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume', 'Dividends', 'Stock Splits']:
                df.loc[mask, col] = ticker_data[col].values
    
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

def add_market_regime_features(df):
    """Add market-wide features"""
    print("  Adding market regime features...")
    
    # Calculate market returns (using mean of all stocks as proxy)
    market_returns = df.groupby('Date')['return_1d'].mean()
    market_returns.name = 'market_return_1d'
    df = df.merge(market_returns, left_on='Date', right_index=True, how='left')
    
    # Market volatility (std dev of daily returns across all stocks on that day)
    market_vol = df.groupby('Date')['return_1d'].std()
    market_vol.name = 'market_volatility'
    df = df.merge(market_vol, left_on='Date', right_index=True, how='left')
    
    # Market breadth (percentage of stocks up on that day)
    breadth = df.groupby('Date').apply(lambda x: (x['return_1d'] > 0).mean())
    breadth.name = 'market_breadth'
    df = df.merge(breadth, left_on='Date', right_index=True, how='left')
    
    # Correlation and Beta
    for ticker in df['Ticker'].unique():
        mask = df['Ticker'] == ticker
        # 63d (3-month) rolling correlation
        df.loc[mask, 'correlation_market'] = df.loc[mask, 'return_1d'].rolling(63).corr(df.loc[mask, 'market_return_1d'])
        # Beta: Covariance / Market Variance
        df.loc[mask, 'beta'] = df.loc[mask, 'return_1d'].rolling(63).cov(df.loc[mask, 'market_return_1d']) / df.loc[mask, 'market_return_1d'].rolling(63).var()
    
    # Relative strength (stock vs market median)
    df['relative_strength'] = df['return_21d'] - df.groupby('Date')['return_21d'].transform('median')
    
    return df

def add_cross_sectional_features(df):
    """Add features comparing stocks within each date (Rank and Z-Score)"""
    print("  Adding cross-sectional features...")
    
    rank_features = ['return_1d', 'return_5d', 'return_21d', 'return_63d', 
                    'volume_ratio', 'volatility_21d', 'rsi_14', 'dollar_volume']
    
    for feature in rank_features:
        if feature in df.columns:
            # Rank (Percentile Rank: 0 to 1)
            df[f'{feature}_rank'] = df.groupby('Date')[feature].rank(pct=True, method='dense')
            # Z-Score (How many standard deviations from the mean on that day)
            df[f'{feature}_zscore'] = df.groupby('Date')[feature].transform(lambda x: (x - x.mean()) / x.std())
    
    return df

def add_momentum_and_reversal_signals(df):
    """Add custom interaction and momentum features."""
    
    # Momentum (interaction between returns and trading activity)
    df['momentum_1m'] = df['return_21d'] * df['volume_ratio']
    df['momentum_3m'] = df['return_63d'] * df['volume_ratio']
    
    # Reversal signal (high distance from 1m high combined with overbought RSI)
    # Note: Requires dist_from_1m_high and rsi_14 to be calculated first
    df['reversal_signal'] = -df['dist_from_1m_high'] * (df['rsi_14'] > 70).astype(int)
    
    # Trend consistency (How many moving averages are correctly stacked)
    df['trend_consistency'] = (
        (df['sma_5'] > df['sma_10']).astype(int) +
        (df['sma_10'] > df['sma_20']).astype(int) +
        (df['sma_20'] > df['sma_50']).astype(int) +
        (df['sma_50'] > df['sma_200']).astype(int)
    ) / 4
    
    return df

def create_advanced_targets(df):
    """Create multiple types of prediction targets"""
    print("  Creating advanced targets...")
    
    df = df.sort_values(['Ticker', 'Date']).reset_index(drop=True)
    
    for ticker in df['Ticker'].unique():
        mask = df['Ticker'] == ticker
        ticker_data = df[mask].copy()
        
        # Forward returns at multiple horizons
        # Map periods: 1d, 5d, 21d, 3m(63d), 6m(126d), 12m(252d)
        for horizon, periods in [('1d', 1), ('5d', 5), ('21d', 21), ('3m', 63), ('6m', 126), ('12m', 252)]:
            ticker_data[f'target_return_{horizon}'] = (
                ticker_data['Adj Close'].shift(-periods) / ticker_data['Adj Close'] - 1
            )
            
            # Direction (binary classification) - PRIMARY TARGET
            # Note: We classify targets only for model evaluation, not raw features
            ticker_data[f'target_direction_{horizon}'] = (ticker_data[f'target_return_{horizon}'] > 0).astype(int)
        
        # Update main dataframe
        for col in ticker_data.columns:
            if col.startswith('target_'):
                df.loc[mask, col] = ticker_data[col].values
    
    return df

def engineer_features(df):
    """Main feature engineering pipeline"""
    df = df.sort_values(['Date', 'Ticker']).reset_index(drop=True)
    
    # 1. Calculate base features if not present
    if 'return_1d' not in df.columns:
        print("  Calculating base features...")
        df = calculate_base_features(df)
    
    # 2. Add advanced market/cross-sectional features
    df = add_market_regime_features(df)
    df = add_cross_sectional_features(df)
    
    # 3. Add momentum and trend signals
    df = add_momentum_and_reversal_signals(df)
    
    # 4. Create all targets (should be the last step before cleaning NaNs)
    df = create_advanced_targets(df)
    
    return df

# Helper functions below are for model integration and should not be modified
def get_feature_columns(df=None):
    """Return list of columns to exclude from features"""
    exclude = ['Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Adj Close', 
               'Volume', 'Dividends', 'Stock Splits']
    
    if df is not None:
        exclude.extend([col for col in df.columns if col.startswith('target_')])
    
    # Columns that were explicitly calculated for ratio/position but shouldn't be used raw
    exclude.extend(['bb_upper', 'bb_lower', 'macd', 'macd_signal',
                   'volume_sma_20', 'dollar_volume', 'dollar_vol_20', 'sma_5', 'sma_10', 'sma_20', 'sma_50', 'sma_100', 'sma_200'])
    
    # Remove duplicates from the list
    return list(set(exclude))

def get_scalable_features():
    """Features that need normalization (used for reference, not run in this script)"""
    return [
        'dollar_volume_rank', 'dollar_volume_zscore', 
        'market_volatility', 'beta', 'correlation_market',
        'return_1d', 'return_5d', 'return_21d', 'return_63d',
        'rsi_14', 'bb_position', 'macd_hist', 'dist_from_52w_high'
    ]