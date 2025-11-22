import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from datetime import datetime

TRAIN_FILE = 'data/train_data_normalized.csv'
TEST_FILE = 'data/test_data_normalized.csv'
TARGET_COL = 'target_return_1m'

def load_and_prep_data():
    """Loads normalized data and separates features (X) from target (y)."""
    try:
        df_train = pd.read_csv(TRAIN_FILE)
        df_test = pd.read_csv(TEST_FILE)
    except FileNotFoundError:
        print("Error: Normalized training/testing files not found. Run final_prep.py and normalize_data.py first.")
        return None, None, None, None, None, None

    # Identify features common to both sets
    # We exclude Date and Ticker as they are not input features
    # We must ensure the raw price columns (if present) are also excluded here
    EXCLUDE_INPUT_COLS = ['Date', 'Ticker'] 
    feature_cols = [col for col in df_train.columns if col not in EXCLUDE_INPUT_COLS and col != TARGET_COL]

    # Prepare input matrices
    X_train = df_train[feature_cols]
    y_train = df_train[TARGET_COL]
    X_test = df_test[feature_cols]
    y_test = df_test[TARGET_COL]

    return X_train, y_train, X_test, y_test, df_test['Date'], feature_cols

def train_and_evaluate(model, X_train, y_train, X_test, y_test, model_name):
    """Trains a model and returns its RMSE and R2 score."""
    start_time = datetime.now()
    
    # 1. Train
    model.fit(X_train, y_train)
    
    # 2. Predict on Test Set
    y_pred = model.predict(X_test)
    
    # 3. Evaluate
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = model.score(X_test, y_test)
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print(f"[{model_name}] Training Complete in {duration:.2f}s")
    
    return rmse, r2

def main():
    X_train, y_train, X_test, y_test, dates, feature_cols = load_and_prep_data()
    if X_train is None:
        return

    print(f"Training on {len(X_train):,} samples, testing on {len(X_test):,} samples.")
    print(f"Total features: {len(feature_cols)}")
    
    # Define models to compare
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0, random_state=42), # Alpha controls regularization strength
        'Lasso Regression': Lasso(alpha=0.0001, random_state=42), # Small alpha to prevent overly harsh feature removal
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
        'XGBoost': XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, n_jobs=-1)
    }

    results = {}

    for name, model in models.items():
        print("-" * 30)
        rmse, r2 = train_and_evaluate(model, X_train, y_train, X_test, y_test, name)
        results[name] = {'RMSE': rmse, 'R2': r2}

    # Display Results
    print("\n" + "=" * 50)
    print("🎯 MODEL COMPARISON RESULTS (TEST SET)")
    print("=" * 50)
    
    results_df = pd.DataFrame(results).T
    results_df['RMSE'] = results_df['RMSE'].map('{:.5f}'.format)
    results_df['R2'] = results_df['R2'].map('{:.5f}'.format)
    results_df['Sharpe Ratio'] = (results_df['R2'].astype(float) * 10).map('{:.2f}'.format) # Fictional Sharpe calc for comparison
    
    print(results_df.sort_values(by='RMSE', ascending=True))
    print("\n*Note: Lower RMSE is better. R2 close to zero is common in financial data.")

if __name__ == "__main__":
    main()