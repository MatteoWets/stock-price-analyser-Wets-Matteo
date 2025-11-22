import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor

def check_feature_quality():
    print("="*70)
    print("ANALYZING FEATURE IMPORTANCE & QUALITY")
    print("="*70)

    # Load data
    print("Loading data/normalized.csv...")
    try:
        df = pd.read_csv('data/normalized.csv')
    except FileNotFoundError:
        print("Error: data/normalized.csv not found. Did you run the feature engineering and normalized script?")
        return

    # Preprocessing
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(['Ticker', 'Date'])
    
    # Drop NaNs created by rolling windows
    df = df.dropna()
    
    target_col = 'target_return_1m'
    
    # Exclude raw price columns and metadata
    excluded_cols = [
        'Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Adj Close', 
        'Volume', 'Dividends', 'Stock Splits', target_col
    ]
    
    # Get list of features
    feature_cols = [c for c in df.columns if c not in excluded_cols]
    
    # Ensure SMA_20 is in the list (it should be based on your previous code)
    if 'sma_20' not in feature_cols:
        print(" 'sma_20' not found in columns. Adding it to check list if available...")
        if 'sma_20' in df.columns:
            feature_cols.append('sma_20')
    
    X = df[feature_cols]
    y = df[target_col]

    print(f"\nAnalyzing {len(feature_cols)} features...")


    # TEST 1: Correlation with target
    print("\n--- Test 1: Linear Correlation with Target ---")
    correlations = X.apply(lambda x: x.corr(y))
    abs_corrs = correlations.abs().sort_values(ascending=False)
    print(abs_corrs.head(15)) # Showing top 15 for better context


    # TEST 2: Random Forest importance
    print("\n--- Test 2: Random Forest Importance ---")
    rf = RandomForestRegressor(n_estimators=50, max_depth=10, n_jobs=-1, random_state=42)
    rf.fit(X, y)
    
    importances = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': rf.feature_importances_
    }).sort_values('Importance', ascending=False)

    print(importances.head(15).to_string(index=False)) # Showing top 15


    # Special deep dive: SMA_20 vs Price_to_SMA20
    print("\n" + "="*70)
    print("DEEP DIVE: SMA 20 ANALYSIS")
    print("="*70)
    
    # Check raw SMA 20 stats
    if 'sma_20' in df.columns:
        sma_corr = df['sma_20'].corr(df[target_col])
        # Find sma_20 importance from the calculated importances table
        sma_imp = importances[importances['Feature'] == 'sma_20']['Importance'].values[0] if 'sma_20' in importances['Feature'].values else 0
        print(f"Feature: sma_20")
        print(f"  - Correlation with Target: {sma_corr:.4f}")
        print(f"  - RF Model Importance:     {sma_imp:.4f}")
    
    # Check derived Ratio stats
    if 'price_to_sma20' in df.columns:
        ratio_corr = df['price_to_sma20'].corr(df[target_col])
        # Find price_to_sma20 importance from the calculated importances table
        ratio_imp = importances[importances['Feature'] == 'price_to_sma20']['Importance'].values[0] if 'price_to_sma20' in importances['Feature'].values else 0
        print(f"Feature: price_to_sma20")
        print(f"  - Correlation with Target: {ratio_corr:.4f}")
        print(f"  - RF Model Importance:     {ratio_imp:.4f}")
    

    # Feature report saving:
    print("\n" + "="*70)
    print("FEATURE REPORT SAVING")
    print("="*70)

    # Create the final summary dataframe that combines correlation and importance
    # Use the features from X as the index
    final_report = pd.DataFrame(index=X.columns)
    
    # Add Correlation
    final_report['Abs_Correlation'] = X.corrwith(y).abs()
    
    # Map the RF Importance scores to the index
    importance_map = importances.set_index('Feature')['Importance']
    final_report['RF_Importance'] = final_report.index.map(importance_map).fillna(0)
    
    # Save the file the final_prep script is looking for
    report_file_path = 'data/feature_importance.csv'
    final_report.to_csv(report_file_path, index=True)
    
    print(f"✅ Full Feature Report saved to: {report_file_path}")
    
    # Visualizations:
    print("\nGenerating visualizations...")
    
    # Feature Importance Plot
    plt.figure(figsize=(10, 8))
    sns.barplot(x='Importance', y='Feature', data=importances.head(20))
    plt.title('Top 20 Features Importance')
    plt.tight_layout()
    plt.savefig('data/feature_importance_plot.png')
    
    # ... (SMA Visualization code remains the same) ...

if __name__ == "__main__":
    check_feature_quality()
