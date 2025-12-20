#!/usr/bin/env python3
"""Main pipeline to run everything"""
import sys
import os
import os
import runpy
import argparse
import traceback

SRC_DIR = os.path.dirname(__file__)

EXCLUDE = {
    "main.py",           # avoid re-running this launcher
    "train_models.py",
    "test_models.py",
    "feature_engineeringV1.py",
    "features_comparison.py",
    "final_visualitzation.py"
}

def run_all_scripts(dry_run=False):
    files = sorted(f for f in os.listdir(SRC_DIR) if f.endswith(".py") and f not in EXCLUDE)
    print(f"Found {len(files)} scripts to run:")
    for f in files:
        path = os.path.join(SRC_DIR, f)
        print(f" -> {f}")
    if dry_run:
        return

    for f in files:
        path = os.path.join(SRC_DIR, f)
        print("\n" + "=" * 60)
        print(f"Running: {f}")
        print("=" * 60)
        try:
            # execute the script as __main__ so any guarded main() will run
            runpy.run_path(path, run_name="__main__")
        except Exception as e:
            print(f"Error running {f}: {e}")
            traceback.print_exc()
            # continue to next script (don't stop the whole process)
            continue

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run all src/*.py (except excluded)")
    ap.add_argument("--dry-run", action="store_true", help="List scripts without running")
    args = ap.parse_args()
    run_all_scripts(dry_run=args.dry_run)

# Add src directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train_models import train_all_models
from test_models import test_all_models

def main():
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
        # TRAINING PHASE
        print("="*60)
        print("TRAINING PHASE")
        print("="*60)
        trainer = train_all_models(
            data_path,
        )
        
        # TESTING PHASE
        print("\n" + "="*60)
        print("TESTING PHASE")
        print("="*60)
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
    main()
