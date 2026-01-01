#!/usr/bin/env python3
"""Main pipeline to run everything with full reproducibility"""
import sys
import os
import runpy
import argparse
import traceback

def set_all_seeds(seed=42):
    """Set all random seeds for complete reproducibility"""
    import random
    import numpy as np
    
    # Set all seeds
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Scikit-learn
    try:
        import sklearn
        sklearn.set_config(random_state=seed)
    except:
        pass
    
    # TensorFlow
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
    except:
        pass
    
    # PyTorch
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except:
        pass

# Set seeds at module level (before any imports)
set_all_seeds(42)

SRC_DIR = os.path.dirname(__file__)
EXCLUDE = {
    "main.py",  # avoid re-running this launcher
    "feature_engineeringV1.py",
    "features_comparison.py",
    "final_visualizations.py"
}

def run_all_scripts(dry_run=False):
    """Run all scripts in alphabetical order with seed reset before each"""
    files = sorted(f for f in os.listdir(SRC_DIR) if f.endswith(".py") and f not in EXCLUDE)
    
    print(f"Found {len(files)} scripts to run:")
    for f in files:
        print(f" -> {f}")
    
    if dry_run:
        return
    
    for f in files:
        path = os.path.join(SRC_DIR, f)
        print("\n" + "=" * 60)
        print(f"Running: {f}")
        print("=" * 60)
        
        # CRITICAL: Reset seeds before EACH script
        set_all_seeds(42)
        
        try:
            # Execute the script as __main__ so any guarded main() will run
            runpy.run_path(path, run_name="__main__")
            print(f"✓ {f} completed successfully")
        except Exception as e:
            print(f"Error running {f}: {e}")
            traceback.print_exc()
            # Continue to next script (don't stop the whole process)
            continue

# Add src directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    """Main pipeline execution"""
    # Reset seeds at the start
    set_all_seeds(42)
    
    # Get project root directory (parent of src/)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Set working directory to project root
    os.chdir(project_root)
    print(f"Working directory: {os.getcwd()}")
    
    # Check if data exists
    data_path = 'data/raw.csv'
    if not os.path.exists(data_path):
        print(f"ERROR: {data_path} not found!")
        print("Please ensure your data file is in the data/ directory")
        sys.exit(1)
    
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    try:
        # Import after setting seeds
        from train_models import train_all_models
        from test_models import test_all_models
        
        # TRAINING PHASE
        print("="*60)
        print("TRAINING PHASE")
        print("="*60)
        
        # Reset seeds before training
        set_all_seeds(42)
        
        trainer = train_all_models(
            data_path,
        )
        
        # TESTING PHASE
        print("\n" + "="*60)
        print("TESTING PHASE")
        print("="*60)
        
        # Reset seeds before testing
        set_all_seeds(42)
        
        tester = test_all_models(
            csv_path=data_path,
            use_saved=True  # Use the normalized data from training
        )
        
        print("\n" + "="*60)
        print("PIPELINE COMPLETE")
        print("="*60)
        print("Generated files:")
        print("  - data/raw_with_targets.csv")
        print("  - data/engineered.csv")
        print("  - data/normalized.csv")
        print("  - data/training.csv")
        print("  - data/testing.csv")
        print("  - test_results.csv")
        print("  - models/*.pkl (trained models)")
        print("  - scalers.pkl (normalization parameters)")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    # Parse arguments
    ap = argparse.ArgumentParser(description="Run all src/*.py (except excluded)")
    ap.add_argument("--dry-run", action="store_true", help="List scripts without running")
    ap.add_argument("--run-all", action="store_true", help="Run all scripts in order first")
    args = ap.parse_args()
    
    # Set seeds before anything else
    set_all_seeds(42)
    
    if args.run_all:
        run_all_scripts(dry_run=args.dry_run)
    else:
        main()