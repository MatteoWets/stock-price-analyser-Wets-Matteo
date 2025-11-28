"""
Robust feature comparison and visualization.

- Auto-detects feature list from src/feature_engineering.py (FEATURES / FEATURE_LIST / features / get_features)
- Falls back to numeric columns in the CSV (excluding ids/dates/target)
- Optionally normalizes features (MinMax or Standard)
- Produces:
    - Horizontal bar chart of correlation with target (sorted by absolute correlation) - TOP 20 ONLY
    - Heatmap of pairwise correlations for top-N features by abs correlation
Outputs saved to the specified output directory.
"""

import os
import sys
import argparse
import traceback
import importlib.util
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
# Use non-interactive backend so script works in headless environments
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler

sns.set(style="whitegrid")

DEFAULT_INPUT = "data/engineered.csv"
FEATURE_MODULE = "src/feature_engineering.py"
DEFAULT_OUTDIR = "data"
DEFAULT_TARGET_NAMES = ["target_return_21d", "target", "direction", "target_return", "y"]


def try_load_feature_list(path):
    """Attempt to import file and extract a feature list variable or callable."""
    if not os.path.exists(path):
        return None
    try:
        spec = importlib.util.spec_from_file_location("feature_engineering", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        # common names
        for name in ("FEATURES", "FEATURE_LIST", "FEATURE_COLUMNS", "features"):
            if hasattr(mod, name):
                val = getattr(mod, name)
                if isinstance(val, (list, tuple)):
                    return list(val)
        # functions that return list
        for fname in ("get_features", "feature_list"):
            if hasattr(mod, fname) and callable(getattr(mod, fname)):
                try:
                    v = getattr(mod, fname)()
                    if isinstance(v, (list, tuple)):
                        return list(v)
                except Exception:
                    continue
    except Exception:
        # If import fails, return None (we'll fallback)
        return None
    return None


def detect_target_column(df, provided=None):
    if provided and provided in df.columns:
        return provided
    for t in DEFAULT_TARGET_NAMES:
        if t in df.columns:
            return t
    # fallback: last numeric column
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric:
        return numeric[-1]
    raise ValueError("No numeric target column found; provide --target explicitly.")


def choose_features(df, target_col, features_from_module=None):
    if features_from_module:
        # keep only features that exist in df and exclude target
        feats = [f for f in features_from_module if f in df.columns and f != target_col]
        if feats:
            return feats
    # fallback: all numeric columns except target and common ids/dates
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    blacklist = {target_col, "id", "index", "timestamp"}
    for d in ("date", "Date", "datetime"):
        if d in df.columns:
            blacklist.add(d)
    feats = [c for c in numeric if c not in blacklist]
    return sorted(feats)


def normalize_features(df, features, method="minmax"):
    if not features:
        return df
    if method == "minmax":
        scaler = MinMaxScaler()
    elif method == "standard":
        scaler = StandardScaler()
    else:
        raise ValueError("Unknown normalization method: choose 'minmax' or 'standard'")
    df[features] = scaler.fit_transform(df[features])
    return df


def plot_corr_with_target(df, features, target, out_path, top_n=20):
    corrs = df[features + [target]].corr()[target].drop(labels=[target]).copy()
    # sort by absolute correlation but display signed values
    corrs = corrs.reindex(corrs.abs().sort_values(ascending=False).index)
    # Take only top N features by absolute correlation
    corrs_top = corrs.head(top_n)
    
    plt.figure(figsize=(10, max(4, len(corrs_top) * 0.25)))
    palette = ["#d73027" if v < 0 else "#1a9850" for v in corrs_top.values]
    ax = sns.barplot(x=corrs_top.values, y=corrs_top.index, palette=palette)
    ax.set_xlabel(f"Pearson correlation with '{target}'")
    ax.set_ylabel("")
    ax.set_title(f"Top {len(corrs_top)} features by correlation with target")
    # annotate values
    for i, v in enumerate(corrs_top.values):
        ax.text(v, i, f" {v:.3f}", va="center", color="black", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return corrs


def plot_heatmap_top_features(df, features, out_path, top_n=30):
    n = min(len(features), top_n)
    if n == 0:
        raise ValueError("No features for heatmap")
    top_feats = features[:n]
    corrmat = df[top_feats].corr()
    figsize = (min(18, 0.5 * n + 6), min(14, 0.5 * n + 6))
    plt.figure(figsize=figsize)
    sns.heatmap(corrmat, cmap="RdBu_r", center=0, linewidths=0.4)
    plt.title(f"Feature correlation matrix (top {n})")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return corrmat


def save_summary(corr_series, out_csv):
    df = corr_series.rename("corr_with_target").to_frame()
    df["abs_corr"] = df["corr_with_target"].abs()
    df = df.sort_values("abs_corr", ascending=False)
    df.drop(columns=["abs_corr"]).to_csv(out_csv)


def main(argv):
    ap = argparse.ArgumentParser(description="Compare features and plot correlations.")
    ap.add_argument("--input", "-i", default=DEFAULT_INPUT, help="Engineered CSV input")
    ap.add_argument("--target", "-t", default=None, help="Target column name (auto-detected if omitted)")
    ap.add_argument("--outdir", "-o", default=DEFAULT_OUTDIR, help="Output directory for graphs and CSV")
    ap.add_argument("--top", type=int, default=30, help="Max features to show in heatmap")
    ap.add_argument("--top-bar", type=int, default=20, help="Number of top features to show in bar chart")
    ap.add_argument("--normalize", choices=["none", "minmax", "standard"], default="none",
                    help="Normalize features before analysis")
    ap.add_argument("--feature-module", default=FEATURE_MODULE,
                    help="Path to feature_engineering.py to discover feature list")
    args = ap.parse_args(argv)

    try:
        if not os.path.exists(args.input):
            print(f"Input file not found: {args.input}")
            return 1

        os.makedirs(args.outdir, exist_ok=True)

        # load data
        df = pd.read_csv(args.input, low_memory=False)
        if df.empty:
            print("Input CSV is empty")
            return 1

        # detect target
        target_col = detect_target_column(df, args.target)
        # drop rows missing target
        df = df.dropna(subset=[target_col])

        # try to load feature list from module
        features_from_module = try_load_feature_list(args.feature_module)
        features = choose_features(df, target_col, features_from_module)
        if not features:
            print("No features detected to analyze.")
            return 1

        # optionally normalize features (not the target)
        if args.normalize != "none":
            df = normalize_features(df, features, method=args.normalize)

        # prepare numeric df for correlation
        df_num = df[features + [target_col]].dropna()

        # compute correlations with target and plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_bar = os.path.join(args.outdir, f"feature_target_correlation_{timestamp}.png")
        out_heat = os.path.join(args.outdir, f"feature_correlation_heatmap_{timestamp}.png")
        out_csv = os.path.join(args.outdir, f"feature_correlation_summary_{timestamp}.csv")

        corrs = plot_corr_with_target(df_num, features, target_col, out_bar, top_n=args.top_bar)
        # reorder features by abs corr for heatmap selection
        ordered_feats = corrs.abs().sort_values(ascending=False).index.tolist()
        corrmat = plot_heatmap_top_features(df_num, ordered_feats, out_heat, top_n=args.top)

        save_summary(corrs, out_csv)

        print(f"Saved correlation bar: {out_bar}")
        print(f"Saved heatmap: {out_heat}")
        print(f"Saved summary CSV: {out_csv}")
        return 0

    except Exception as exc:
        print("ERROR:", str(exc))
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
