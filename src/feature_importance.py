# This script creates a graph

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

INPUT = "data/feature_importance.csv"
OUTPUT = "src/feature_importance_normalized.csv"
TARGET = "direction"  # change if your target column has a different name

def normalize_csv(input_path=INPUT, output_path=OUTPUT, target_col=TARGET):
    df = pd.read_csv(input_path)
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in {input_path}")

    # Keep non-numeric columns as-is (e.g. dates, ids)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Ensure we don't accidentally scale id-like columns; remove common id/date names if present
    for c in ["id", "index", "date", "timestamp"]:
        if c in numeric_cols:
            numeric_cols.remove(c)

    # Handle target
    target_is_binary = df[target_col].nunique() <= 2
    features_to_scale = [c for c in numeric_cols if c != target_col]

    scaler = MinMaxScaler()
    if features_to_scale:
        df[features_to_scale] = scaler.fit_transform(df[features_to_scale])

    if target_is_binary:
        # map binary values to 0/1 (keeps class labels balanced, avoids ~0.7 float)
        uniques = sorted(df[target_col].unique())
        mapping = {uniques[0]: 0, uniques[1]: 1}
        df[target_col] = df[target_col].map(mapping)
    else:
        # scale continuous target to 0-1
        df[[target_col]] = MinMaxScaler().fit_transform(df[[target_col]])

    df.to_csv(output_path, index=False)
    print(f"Saved normalized data to {output_path}")

if __name__ == "__main__":
    normalize_csv()
