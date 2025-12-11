# stock-price-analyser-Wets-Matteo

## Research Question
Which algorithm best predicts tech stock direction: XGboost, Catboost, LightGBM or an Ensemble?

## Setup
### Create environment
python -m venv .venv
source .venv/bin/activate  # Mac/Linux
.venv\Scripts\activate     # Windows
pip install -r requirements.txt

## Usage

python src/main.py

Expected output: 
In the terminal: an overall summary of the best models per target, average performance by model, average performance by timeframe and best overall models.

In the explorer: a CSV file with all the results obtained.

## Project Structure
stock-price-analyser-Wets-Matteo/
├── README.md
├── requirements.txt
├── data/
│   ├── raw.csv
│   ├── engineered.csv
│   ├── feature_importance.csv
│   └── normalized.csv
├── models/
├── src/
│   ├── __init__.py
│   ├── download_data.py
│   ├── feature_engineering.py
│   ├── feature_importance.py
│   ├── feature_normalized.py
│   ├── feature_test.py
│   ├── features_comparison2.py
│   ├── features_selection.py
│   ├── main.py
│   ├── test_models.py
│   └── train_models.py
├── test_results.csv
└── test_results_visualization.png

## Results
- Best Model per Target (by AUC):
--------------------------------------------------------------------------------
  direction_12m  : catboost     (AUC=0.5951, Acc=0.5507)
  direction_1d   : xgboost      (AUC=0.6139, Acc=0.5710)
  direction_21d  : xgboost      (AUC=0.6133, Acc=0.5904)
  direction_3m   : xgboost      (AUC=0.6231, Acc=0.6032)
  direction_5d   : xgboost      (AUC=0.6183, Acc=0.5865)
  direction_6m   : catboost     (AUC=0.5877, Acc=0.5619)
- Winner: XGBoost on short-term and CatBoost on longer-term

## Requirements
- Python 3.11
- yfinance, matplotlib, tqdm, seaborn, sklearn, pickle4, pandas>=1.3.0, numpy>=1.20.0, scikit-learn>=1.0.0, xgboost>=1.5.0, lightgbm>=3.3.0, catboost>=1.0.0, scipy>=1.7.0


