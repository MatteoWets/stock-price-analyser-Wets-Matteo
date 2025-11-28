import pandas as pd
import numpy as np
from sklearn.feature_selection import (
    SelectKBest, f_regression, f_classif, mutual_info_regression, 
    mutual_info_classif, RFE, SelectFromModel, VarianceThreshold
)
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

class FeatureSelector:
    def __init__(self, task_type='regression', method='all', n_features=30):
        """
        task_type: 'regression' or 'classification'
        method: 'variance', 'correlation', 'mutual_info', 'importance', 'rfe', 'all'
        n_features: target number of features to select
        """
        self.task_type = task_type
        self.method = method
        self.n_features = n_features
        self.selected_features = None
        
    def remove_low_variance(self, X, threshold=0.01):
        """Remove features with low variance"""
        print(f"    Removing low variance features (threshold={threshold})...")
        selector = VarianceThreshold(threshold=threshold)
        selector.fit(X)
        return X.columns[selector.get_support()].tolist()
    
    def remove_correlated(self, X, threshold=0.95):
        """Remove highly correlated features"""
        print(f"    Removing correlated features (threshold={threshold})...")
        corr_matrix = X.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        return [col for col in X.columns if col not in to_drop]
    
    def select_by_mutual_info(self, X, y, n_features=30):
        """Select features by mutual information"""
        print(f"    Selecting top {n_features} by mutual information...")
        if self.task_type == 'regression':
            mi_scores = mutual_info_regression(X, y, random_state=42)
        else:
            mi_scores = mutual_info_classif(X, y, random_state=42)
        
        mi_scores = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)
        return mi_scores.head(n_features).index.tolist()
    
    def select_by_importance(self, X, y, n_features=30):
        """Select features by tree-based importance"""
        print(f"    Selecting top {n_features} by feature importance...")
        
        if self.task_type == 'regression':
            model = lgb.LGBMRegressor(
                n_estimators=100, 
                max_depth=5,
                random_state=42,
                verbose=-1
            )
        else:
            model = lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=42,
                verbose=-1
            )
        
        model.fit(X, y)
        importance = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
        return importance.head(n_features).index.tolist()
    
    def select_by_rfe(self, X, y, n_features=30):
        """Recursive Feature Elimination"""
        print(f"    Selecting {n_features} features by RFE...")
        
        if self.task_type == 'regression':
            estimator = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42, n_jobs=-1)
        else:
            estimator = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42, n_jobs=-1)
        
        selector = RFE(estimator, n_features_to_select=n_features, step=5)
        selector.fit(X, y)
        return X.columns[selector.support_].tolist()
    
    def select_features(self, X, y):
        """Main feature selection method"""
        print(f"  Feature selection: {len(X.columns)} features -> {self.n_features}")
        
        # Start with all features
        selected = list(X.columns)
        
        if self.method in ['variance', 'all']:
            selected = [f for f in selected if f in self.remove_low_variance(X[selected])]
        
        if self.method in ['correlation', 'all']:
            selected = [f for f in selected if f in self.remove_correlated(X[selected])]
        
        # For the remaining methods, we need to handle NaN
        X_clean = X[selected].fillna(0)
        y_clean = y.fillna(0)
        
        if self.method in ['mutual_info', 'all']:
            mi_features = self.select_by_mutual_info(X_clean, y_clean, self.n_features)
            if self.method == 'mutual_info':
                selected = mi_features
            else:
                selected = [f for f in selected if f in mi_features]
        
        if self.method in ['importance', 'all']:
            imp_features = self.select_by_importance(X_clean, y_clean, self.n_features)
            if self.method == 'importance':
                selected = imp_features
            else:
                # Combine with existing selection
                selected = list(set(selected) & set(imp_features))
        
        if self.method == 'rfe':
            selected = self.select_by_rfe(X_clean, y_clean, self.n_features)
        
        # Ensure we have at least n_features
        if len(selected) > self.n_features:
            # Use importance to pick the final set
            selected = self.select_by_importance(X[selected].fillna(0), y_clean, self.n_features)
        
        self.selected_features = selected
        print(f"    Selected {len(selected)} features")
        
        return selected
    
    def fit_transform(self, X, y):
        """Fit selector and transform X"""
        selected = self.select_features(X, y)
        return X[selected]
    
    def transform(self, X):
        """Transform X using already selected features"""
        if self.selected_features is None:
            raise ValueError("Must call fit_transform first")
        return X[self.selected_features]
