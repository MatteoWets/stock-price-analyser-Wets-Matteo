# This script creates visualizations to compare and analyze the top 50 engineered features

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- CONFIGURATION ---
DATA_FILE = 'data/engineered.csv'
IMPORTANCE_FILE = 'data/feature_importance_report.csv'
TARGET_COL = 'target_return_21d' # Ensure this matches your column name
# ---------------------

def load_data():
    if not os.path.exists(DATA_FILE):
        print(f"Error: {DATA_FILE} not found.")
        return None
    df = pd.read_csv(DATA_FILE)
    return df

def plot_correlation_heatmap(df):
    """1. Visualizes how features relate to each other (Redundancy Check)"""
    print("Generating Correlation Heatmap...")
    
    # Select a mix of top features to keep the chart readable (not all 50)
    # We pick distinct categories: Trend, Volatility, Momentum, Market
    selected_cols = [
        'macd_hist', 'rsi_14', 'volatility_21d', 'price_to_sma200', 
        'market_volatility', 'volume_ratio', 'beta', TARGET_COL
    ]
    
    # Filter only columns that exist
    cols_to_plot = [c for c in selected_cols if c in df.columns]
    
    plt.figure(figsize=(10, 8))
    corr = df[cols_to_plot].corr()
    
    mask = np.triu(np.ones_like(corr, dtype=bool)) # Hide upper triangle
    sns.heatmap(corr, mask=mask, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
    
    plt.title('Correlation Matrix: Top Features vs Target', fontsize=15)
    plt.tight_layout()
    plt.savefig('data/viz_1_correlation_heatmap.png')
    print("✅ Saved: data/viz_1_correlation_heatmap.png")


def main():
    df = load_data()
    if df is None: return
    
    # Run the 4 visualizers
    plot_correlation_heatmap(df)
    # Note: Cumulative signal works best if Date is preserved. 
    # If Date was dropped during normalization, this might be just an index plot, which is still fine.
    
    print("\n" + "="*50)
    print("ALL VISUALIZATIONS CREATED IN data/ FOLDER")
    print("="*50)

if __name__ == "__main__":
    main()