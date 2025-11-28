import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
import pickle

def normalize_features(df, train_end_date='2022-01-01', scalable_features=None):
    """Normalize features per ticker using RobustScaler (better for outliers)"""
    scalers = {}
    
    for ticker in df['Ticker'].unique():
        ticker_mask = df['Ticker'] == ticker
        train_mask = ticker_mask & (df['Date'] < train_end_date)
        
        if train_mask.sum() < 10:
            continue
        
        # Use RobustScaler instead of StandardScaler
        scaler = RobustScaler()
        scalers[ticker] = scaler
        
        # Fit on training data
        train_data = df.loc[train_mask, scalable_features].fillna(0)
        scaler.fit(train_data)
        
        # Transform all data for this ticker
        df.loc[ticker_mask, scalable_features] = scaler.transform(
            df.loc[ticker_mask, scalable_features].fillna(0)
        )
    
    return df, scalers

def save_scalers(scalers, filename='scalers.pkl'):
    """Save fitted scalers"""
    with open(filename, 'wb') as f:
        pickle.dump(scalers, f)

def load_scalers(filename='scalers.pkl'):
    """Load saved scalers"""
    with open(filename, 'rb') as f:
        return pickle.load(f)

def apply_scalers(df, scalers, scalable_features):
    """Apply pre-fitted scalers to new data"""
    for ticker, scaler in scalers.items():
        mask = df['Ticker'] == ticker
        if mask.sum() > 0:
            df.loc[mask, scalable_features] = scaler.transform(
                df.loc[mask, scalable_features].fillna(0)
            )
    return df

def get_scalable_features():
    """Updated list of features to normalize"""
    return ['dollar_volume', 'dollar_vol_20', 'amihud', 'Volume',
            'sma_5', 'sma_10', 'sma_20', 'sma_50', 'sma_100', 'sma_200',
            'macd_hist', 'bid_ask_proxy']
