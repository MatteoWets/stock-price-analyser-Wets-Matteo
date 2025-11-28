import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
import pickle
import os
import sys

import warnings
warnings.filterwarnings('ignore')

np.set_printoptions(threshold=np.inf)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class BinaryClassificationTrainer:
    def __init__(self):
        self.models = {}
        self.feature_selectors = {}
        self.selected_features = {}
        self.class_distributions = {}
        
    def check_class_balance(self, y, target_name):
        """Check class distribution and return whether to use class weights"""
        unique, counts = np.unique(y, return_counts=True)
        distribution = dict(zip(unique, counts))
        self.class_distributions[target_name] = distribution
        
        if len(unique) != 2:
            print(f"    WARNING: Expected 2 classes, got {len(unique)}")
            return False, None
        
        ratio = min(counts) / max(counts)
        print(f"    Class distribution: {distribution} (ratio: {ratio:.3f})")
        
        # Use class weights if imbalanced (< 0.7 ratio)
        if ratio < 0.7:
            print(f"    Imbalanced detected, using class weights")
            class_weights = compute_class_weight('balanced', classes=unique, y=y)
            return True, dict(zip(unique, class_weights))
        
        return False, None
    
    def train_classification_models(self, X_train, y_train, target_name, fast_mode=False):
        """Train binary classification models - gradient boosting focus"""
        print(f"  Training classification models for {target_name}...")
        
        # Check class balance
        use_weights, class_weights = self.check_class_balance(y_train, target_name)
        
        models = {}
        
        # Compute scale_pos_weight for XGBoost (ratio of negative to positive)
        scale_pos_weight = 1.0
        if use_weights:
            neg_count = (y_train == 0).sum()
            pos_count = (y_train == 1).sum()
            scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
        
        # XGBoost - usually best for tabular data
        print("    Training XGBoost...")
        xgb_model = xgb.XGBClassifier(
            n_estimators=300 if not fast_mode else 150,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            n_jobs=-1,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        xgb_model.fit(X_train, y_train)
        models['xgboost'] = xgb_model
        
        # LightGBM - fast and competitive
        print("    Training LightGBM...")
        lgb_params = {
            'n_estimators': 300 if not fast_mode else 150,
            'max_depth': 4,
            'learning_rate': 0.05,
            'num_leaves': 31,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }
        if use_weights:
            lgb_params['class_weight'] = 'balanced'
        
        lgb_model = lgb.LGBMClassifier(**lgb_params)
        lgb_model.fit(X_train, y_train)
        models['lightgbm'] = lgb_model
        
        # CatBoost - handles categorical well, robust
        print("    Training CatBoost...")
        cb_params = {
            'iterations': 300 if not fast_mode else 150,
            'depth': 4,
            'learning_rate': 0.05,
            'l2_leaf_reg': 3,
            'subsample': 0.8,
            'colsample_bylevel': 0.8,
            'random_seed': 42,
            'verbose': False
        }
        if use_weights:
            cb_params['auto_class_weights'] = 'Balanced'
        
        cb_model = cb.CatBoostClassifier(**cb_params)
        cb_model.fit(X_train, y_train)
        models['catboost'] = cb_model
        
        # Ensemble - soft voting with top 3 boosters
        if not fast_mode:
            print("    Training ensemble...")
            voting = VotingClassifier(
                estimators=[
                    ('xgb', xgb_model),
                    ('lgb', lgb_model),
                    ('cb', cb_model)
                ],
                voting='soft'
            )
            voting.fit(X_train, y_train)
            models['ensemble'] = voting
        
        return models
    
    def train_all_targets(self, df, feature_cols, train_end_date='2022-01-01', fast_mode=False):
        """Train binary classification models for all direction targets"""
        from feature_selection import FeatureSelector
        
        train_df = df[df['Date'] < train_end_date].copy()
        
        # Define binary classification targets for direction prediction
        # Timeframes: 1d, 5d, 21d (1 month), 3m (63d), 6m (126d), 12m (252d)
        targets_config = {
            'direction_1d': 30,    # More features for short-term noise
            'direction_5d': 35,
            'direction_21d': 40,   # 1 month
            'direction_3m': 40,    # 3 months  
            'direction_6m': 35,    # 6 months
            'direction_12m': 30,   # 12 months (long-term trends)
        }
        
        for target_name, n_features in targets_config.items():
            target_col = f'target_{target_name}'
            
            if target_col not in train_df.columns:
                print(f"  Warning: {target_col} not found, skipping")
                continue
            
            # Clean data - drop NaN targets
            train_clean = train_df.dropna(subset=[target_col])
            if len(train_clean) < 1000:
                print(f"  Skipping {target_name}: insufficient data ({len(train_clean)} samples)")
                continue
            
            X_train_full = train_clean[feature_cols].fillna(0)
            y_train = train_clean[target_col].astype(int)
            
            print(f"\nTraining {target_name}: {len(X_train_full)} samples")
            
            # Feature selection using importance-based method
            selector = FeatureSelector(task_type='classification', method='importance', n_features=n_features)
            X_train = selector.fit_transform(X_train_full, y_train)
            
            self.feature_selectors[target_name] = selector
            self.selected_features[target_name] = selector.selected_features
            
            print(f"    Selected {len(selector.selected_features)} features")
            
            # Train models
            self.models[target_name] = self.train_classification_models(X_train, y_train, target_name, fast_mode)
            
            # Quick validation on training set (sanity check)
            print(f"    Training metrics:")
            for model_name, model in self.models[target_name].items():
                y_pred = model.predict(X_train)
                y_prob = model.predict_proba(X_train)
                acc = accuracy_score(y_train, y_pred)
                try:
                    auc = roc_auc_score(y_train, y_prob[:, 1])
                    print(f"      {model_name:12s}: Acc={acc:.3f}, AUC={auc:.3f}")
                except:
                    print(f"      {model_name:12s}: Acc={acc:.3f}")
    
    def save_all(self, directory='models'):
        """Save models, selectors, and configurations"""
        os.makedirs(directory, exist_ok=True)
        
        # Save models
        for target_name, models_dict in self.models.items():
            for model_name, model in models_dict.items():
                filename = f"{directory}/{target_name}_{model_name}.pkl"
                with open(filename, 'wb') as f:
                    pickle.dump(model, f)
        
        # Save feature selectors and configurations
        with open(f"{directory}/feature_selectors.pkl", 'wb') as f:
            pickle.dump(self.feature_selectors, f)
        
        with open(f"{directory}/selected_features.pkl", 'wb') as f:
            pickle.dump(self.selected_features, f)
        
        with open(f"{directory}/class_distributions.pkl", 'wb') as f:
            pickle.dump(self.class_distributions, f)
        
        print(f"\nModels and configurations saved to {directory}/")
        print(f"Total targets trained: {len(self.models)}")
        print(f"Total models: {sum(len(m) for m in self.models.values())}")

def train_all_models(csv_path, train_end_date='2022-01-01', fast_mode=False):
    """Main training pipeline"""
    from feature_engineering import engineer_features, get_feature_columns, get_scalable_features
    from normalize_data import normalize_features, save_scalers
    
    # Load data
    print("Loading data...")
    df = pd.read_csv(csv_path)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Engineer features
    print("Engineering features...")
    df = engineer_features(df)
    
    # Save engineered data
    df.to_csv('data/engineered.csv', index=False)
    print("Saved: data/engineered.csv")
    
    # Get feature columns
    exclude_cols = get_feature_columns(df)
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    print(f"Initial features: {len(feature_cols)}")
    
    # Normalize
    print("Normalizing features...")
    scalable_features = get_scalable_features()
    scalable_features = [f for f in scalable_features if f in df.columns]
    df, scalers = normalize_features(df, scalable_features=scalable_features, train_end_date=train_end_date)
    save_scalers(scalers)
    
    # Save normalized data
    df.to_csv('data/normalized.csv', index=False)
    print("Saved: data/normalized.csv")
    
    # Train models
    trainer = BinaryClassificationTrainer()
    trainer.train_all_targets(df, feature_cols, train_end_date, fast_mode)
    trainer.save_all()
    
    return trainer

if __name__ == "__main__":
    # Use fast_mode=True for testing, False for production
    trainer = train_all_models('data/raw.csv', fast_mode=False)
