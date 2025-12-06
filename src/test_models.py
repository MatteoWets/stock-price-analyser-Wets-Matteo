import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score, f1_score,
    log_loss, confusion_matrix, classification_report
)
import pickle
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class BinaryClassificationTester:
    def __init__(self, models_dir='models'):
        self.models_dir = models_dir
        self.results = {}
        self.model_types = ['xgboost', 'lightgbm', 'catboost', 'ensemble']
        self.load_configurations()
        
    def load_configurations(self):
        """Load feature selectors and selected features"""
        try:
            with open(f"{self.models_dir}/feature_selectors.pkl", 'rb') as f:
                self.feature_selectors = pickle.load(f)
            with open(f"{self.models_dir}/selected_features.pkl", 'rb') as f:
                self.selected_features = pickle.load(f)
            with open(f"{self.models_dir}/class_distributions.pkl", 'rb') as f:
                self.class_distributions = pickle.load(f)
        except FileNotFoundError as e:
            print(f"Warning: Could not load configuration: {e}")
            self.feature_selectors = {}
            self.selected_features = {}
            self.class_distributions = {}
    
    def load_model(self, target_name, model_type):
        """Load a saved model"""
        filename = f"{self.models_dir}/{target_name}_{model_type}.pkl"
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                return pickle.load(f)
        return None
    
    def test_binary_classification(self, y_true, y_pred, y_prob):
        """Calculate comprehensive binary classification metrics"""
        if len(y_true) == 0:
            return {}
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0)
        }
        
        # AUC and log loss
        if y_prob is not None and len(np.unique(y_true)) == 2:
            try:
                metrics['auc'] = roc_auc_score(y_true, y_prob[:, 1])
                metrics['log_loss'] = log_loss(y_true, y_prob)
            except Exception as e:
                print(f"      Warning: Could not compute AUC/log_loss: {e}")
        
        # Confusion matrix for additional insights
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics['true_neg'] = tn
        metrics['false_pos'] = fp
        metrics['false_neg'] = fn
        metrics['true_pos'] = tp
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        return metrics
    
    def test_all_models(self, df, feature_cols, test_start='2022-01-01'):
        """Test all binary classification models"""
        test_df = df[df['Date'] >= test_start].copy()
        
        print("\n" + "="*80)
        print("BINARY CLASSIFICATION TEST RESULTS")
        print("="*80)
        
        # Target timeframes
        target_names = ['direction_1d', 'direction_5d', 'direction_21d', 
                       'direction_3m', 'direction_6m', 'direction_12m']
        
        results_summary = []
        
        for target_name in target_names:
            if target_name not in self.selected_features:
                print(f"\n{target_name}: No trained model found")
                continue
            
            target_col = f'target_{target_name}'
            if target_col not in test_df.columns:
                print(f"\n{target_name}: Target column not found in test data")
                continue
            
            # Get selected features for this target
            selected_features = self.selected_features[target_name]
            test_clean = test_df.dropna(subset=[target_col])
            
            if len(test_clean) < 100:
                print(f"\n{target_name}: Insufficient test data ({len(test_clean)} samples)")
                continue
            
            X_test = test_clean[selected_features].fillna(0)
            y_test = test_clean[target_col].astype(int)
            
            print(f"\n{'='*80}")
            print(f"Target: {target_name.upper()}")
            print(f"Test samples: {len(y_test)}")
            
            # Get class distribution in test set
            test_dist = dict(zip(*np.unique(y_test, return_counts=True)))
            print(f"Test distribution: {test_dist}")
            if target_name in self.class_distributions:
                print(f"Train distribution: {self.class_distributions[target_name]}")
            print(f"{'='*80}")
            
            # Test each model type
            target_results = {}
            for model_type in self.model_types:
                model = self.load_model(target_name, model_type)
                if model is None:
                    continue
                
                try:
                    y_pred = model.predict(X_test)
                    y_prob = None
                    if hasattr(model, 'predict_proba'):
                        y_prob = model.predict_proba(X_test)
                    
                    metrics = self.test_binary_classification(y_test, y_pred, y_prob)
                    target_results[model_type] = metrics
                    
                    # Print results
                    acc = metrics['accuracy']
                    prec = metrics['precision']
                    rec = metrics['recall']
                    f1 = metrics['f1']
                    
                    if 'auc' in metrics:
                        auc = metrics['auc']
                        ll = metrics['log_loss']
                        print(f"  {model_type:12s}: Acc={acc:.4f} | Prec={prec:.4f} | Rec={rec:.4f} | F1={f1:.4f} | AUC={auc:.4f} | LogLoss={ll:.4f}")
                    else:
                        print(f"  {model_type:12s}: Acc={acc:.4f} | Prec={prec:.4f} | Rec={rec:.4f} | F1={f1:.4f}")
                    
                    # Store for summary
                    results_summary.append({
                        'target': target_name,
                        'model': model_type,
                        'accuracy': acc,
                        'auc': metrics.get('auc', 0),
                        'f1': f1
                    })
                    
                except Exception as e:
                    print(f"  {model_type:12s}: ERROR - {str(e)[:60]}")
            
            # Find best model for this target
            if target_results:
                best_model = max(target_results.items(), 
                               key=lambda x: x[1].get('auc', x[1].get('accuracy', 0)))
                print(f"\n  BEST: {best_model[0]} (AUC={best_model[1].get('auc', 0):.4f})")
                
                # Show confusion matrix for best model
                m = best_model[1]
                print(f"  Confusion Matrix: TP={m['true_pos']}, FP={m['false_pos']}, TN={m['true_neg']}, FN={m['false_neg']}")
                print(f"  Specificity: {m['specificity']:.4f}")
            
            self.results[target_name] = target_results
        
        # Print overall summary
        self.print_summary(results_summary)
        
        # Save results to CSV
        if results_summary:
            results_df = pd.DataFrame(results_summary)
            results_df = results_df.sort_values(['target', 'auc'], ascending=[True, False])
            results_df.to_csv('test_results.csv', index=False)
            print(f"\nDetailed results saved to: test_results.csv")
    
    def print_summary(self, results_summary):
        """Print overall performance summary"""
        if not results_summary:
            return
        
        print("\n" + "="*80)
        print("OVERALL SUMMARY")
        print("="*80)
        
        df = pd.DataFrame(results_summary)
        
        # Best models per target
        print("\nBest Model per Target (by AUC):")
        print("-" * 80)
        best_per_target = df.loc[df.groupby('target')['auc'].idxmax()]
        for _, row in best_per_target.iterrows():
            print(f"  {row['target']:15s}: {row['model']:12s} (AUC={row['auc']:.4f}, Acc={row['accuracy']:.4f})")
        
        # Model type comparison
        print("\nAverage Performance by Model Type:")
        print("-" * 80)
        model_avg = df.groupby('model')[['accuracy', 'auc', 'f1']].mean().sort_values('auc', ascending=False)
        for model, row in model_avg.iterrows():
            print(f"  {model:12s}: AUC={row['auc']:.4f}, Acc={row['accuracy']:.4f}, F1={row['f1']:.4f}")
        
        # Timeframe comparison
        print("\nAverage Performance by Timeframe:")
        print("-" * 80)
        target_avg = df.groupby('target')[['accuracy', 'auc', 'f1']].mean().sort_values('auc', ascending=False)
        for target, row in target_avg.iterrows():
            print(f"  {target:15s}: AUC={row['auc']:.4f}, Acc={row['accuracy']:.4f}, F1={row['f1']:.4f}")
        
        # Top 5 overall
        print("\nTop 5 Overall Models (by AUC):")
        print("-" * 80)
        top5 = df.nlargest(5, 'auc')
        for i, (_, row) in enumerate(top5.iterrows(), 1):
            print(f"  {i}. {row['target']:15s} ({row['model']:12s}): AUC={row['auc']:.4f}, Acc={row['accuracy']:.4f}")

def test_all_models(csv_path=None, use_saved=True):
    """Main testing pipeline"""
    from feature_engineering import get_feature_columns
    
    if use_saved and os.path.exists('data/normalized.csv'):
        print("Loading pre-processed test data...")
        df = pd.read_csv('data/normalized.csv')
        df['Date'] = pd.to_datetime(df['Date'])
    else:
        print("Processing test data from scratch...")
        from feature_engineering import engineer_features
        from feature_normalized import load_scalers, apply_scalers, get_scalable_features
        
        df = pd.read_csv(csv_path)
        df['Date'] = pd.to_datetime(df['Date'])
        df = engineer_features(df)
        
        # Normalize
        scalers = load_scalers()
        scalable_features = get_scalable_features()
        scalable_features = [f for f in scalable_features if f in df.columns]
        df = apply_scalers(df, scalers, scalable_features)
    
    # Get all feature columns
    exclude_cols = get_feature_columns(df)
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    # Test models
    tester = BinaryClassificationTester()
    tester.test_all_models(df, feature_cols)
    
    return tester

if __name__ == "__main__":
    tester = test_all_models()
