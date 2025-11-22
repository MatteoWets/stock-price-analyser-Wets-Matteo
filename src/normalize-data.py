import pandas as pd
from sklearn.preprocessing import RobustScaler
import numpy as np

INPUT_FILE = 'data/engineered.csv'
OUTPUT_FILE = 'data/normalized.csv'

def normalize_data_robust():
    print("=" * 70)
    print("NORMALIZING ALL FEATURES USING ROBUST SCALER (FIXED EXCLUSIONS)")
    print("=" * 70)
    
    # Load data
    try:
        df = pd.read_csv(INPUT_FILE)
        # Note: We must drop NaNs here again before scaling to ensure clean data
        df = df.dropna().reset_index(drop=True)
    except FileNotFoundError:
        print(f"Error: Input file '{INPUT_FILE}' not found. Did you run add_advanced_features.py?")
        return


    # Identify feature columns (X) and non-feature columns
    # We must exclude ALL raw price levels, whether they are direct prices or derived averages.
    NON_FEATURE_COLS_BASE = [
        'Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Adj Close', 
        'Volume', 'Dividends', 'Stock Splits', 'target_return_1m', # Base exclusions
        
        # RAW PRICE LEVEL DERIVATIVES (MUST BE EXCLUDED)
        'sma_20',       # Raw price level 
        'sma_50',       # Raw price level 
        'sma_200',      # Raw price level
        'cummax',       # Raw price level
        # Note: VIX is correctly included for scaling as it is an indicator level.
    ]
    
    # Filter only columns that exist in the DataFrame
    features_to_exclude = [col for col in NON_FEATURE_COLS_BASE if col in df.columns]
    
    # Feature columns are everything else that is not excluded
    feature_cols = [col for col in df.columns if col not in features_to_exclude]
    NON_FEATURE_COLS = features_to_exclude
    
    X = df[feature_cols].copy()
    
    print(f"Total features found for scaling: {len(feature_cols)}")
    print(f"Total raw price/metadata columns excluded: {len(NON_FEATURE_COLS)}")
    
    # Initialize, fit, and transform scaler
    scaler = RobustScaler()
    
    print("\nFitting and transforming data with RobustScaler...")
    X_scaled = scaler.fit_transform(X)
    
    # Reassemble DataFrame
    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_cols, index=df.index)
    
    # Recombine with original non-feature columns
    df_normalized = pd.concat([df[NON_FEATURE_COLS].reset_index(drop=True), X_scaled_df.reset_index(drop=True)], axis=1)

    # Save normalized file
    df_normalized.to_csv(OUTPUT_FILE, index=False)
    
    print("\nRobust Normalization finished.")
    print(f"✅ Full normalized dataset saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    normalize_data_robust()