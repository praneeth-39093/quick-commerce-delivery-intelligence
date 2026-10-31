"""
Command-Line Interface (CLI) Entrypoint for Quick-Commerce Delivery Intelligence.
Provides unified commands for data inspection, cleaning, feature transformation,
exploratory analysis, evaluation reporting, and application launching.
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import config
from src.data_loader import DataLoader
from src.data_cleaning import DataCleaner
from src.feature_engineering import FeatureEngineer
from src.eda import ExploratoryDataAnalysis


def print_banner():
    print("=" * 80)
    print("  QUICK-COMMERCE DELIVERY INTELLIGENCE")
    print("  AI-Powered ETA Prediction, Late-Order Detection & Delivery Optimization")
    print("=" * 80)


def cmd_inspect():
    """Inspects the raw dataset and verifies schema."""
    print("\n🔍 Running Dataset Inspection & Schema Audit...")
    loader = DataLoader()
    try:
        df_raw = loader.load_raw_data()
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return

    is_valid, missing = loader.validate_raw_schema(df_raw)
    summary = loader.inspect_dataset(df_raw)

    print("\n--- Summary Statistics ---")
    print(f"Total Rows:     {summary['total_rows']:,}")
    print(f"Total Columns:  {summary['total_columns']}")
    print(f"Memory Usage:   {summary['memory_usage_mb']:.2f} MB")
    print(f"Schema Valid:   {is_valid}")
    if missing:
        print(f"Missing Expected Columns: {missing}")

    print("\nMissing Values by Column:")
    for col, count in summary["null_counts"].items():
        if count > 0:
            pct = (count / summary["total_rows"]) * 100
            print(f"  • {col:30s}: {count:5d} ({pct:5.2f}%)")


def cmd_clean():
    """Cleans raw dataset and saves cleaned intermediate artifact."""
    print("\n🧹 Executing Data Cleaning Pipeline...")
    loader = DataLoader()
    df_raw = loader.load_raw_data()

    cleaner = DataCleaner()
    df_cleaned = cleaner.clean(df_raw, is_training=True)

    out_path = config.CLEANED_DATA_PATH
    loader.save_processed_data(df_cleaned, out_path)
    print(f"✅ Cleaned data successfully written to: {out_path} (Shape: {df_cleaned.shape})")


def cmd_features():
    """Generates features and SLA breach targets."""
    print("\n⚙️ Executing Feature Engineering Pipeline...")
    loader = DataLoader()

    cleaned_path = config.CLEANED_DATA_PATH
    if not cleaned_path.exists():
        print("Cleaned data not found. Running cleaning first...")
        cmd_clean()

    df_cleaned = loader.load_raw_data(cleaned_path)
    engineer = FeatureEngineer(sla_threshold_min=config.DEFAULT_SLA_MINUTES)
    df_featured = engineer.transform_features(df_cleaned, is_training=True)

    out_path = config.PROCESSED_DATA_PATH
    loader.save_processed_data(df_featured, out_path)
    print(f"✅ Featured data successfully written to: {out_path} (Shape: {df_featured.shape})")


def cmd_eda():
    """Executes EDA visualization generation."""
    print("\n📈 Generating Exploratory Data Analysis Plots...")
    loader = DataLoader()

    proc_path = config.PROCESSED_DATA_PATH
    if proc_path.exists():
        df = loader.load_raw_data(proc_path)
    elif config.CLEANED_DATA_PATH.exists():
        df = loader.load_raw_data(config.CLEANED_DATA_PATH)
    else:
        print("Processed or cleaned data not found. Preparing features first...")
        cmd_features()
        df = loader.load_raw_data(config.PROCESSED_DATA_PATH)

    eda = ExploratoryDataAnalysis()
    saved = eda.run_full_eda(df)

    print(f"✅ Generated {len(saved)} analytical figures in: {config.FIGURES_DIR}")
    for name, path in saved.items():
        print(f"  • {name:25s} -> {path.name}")


def cmd_app():
    """Launches the Streamlit interactive dashboard."""
    import subprocess
    print("\n🚀 Launching Streamlit Web Application...")
    app_script = config.APP_DIR / "streamlit_app.py"
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_script)])


def main():
    print_banner()

    parser = argparse.ArgumentParser(
        description="Quick-Commerce Delivery Intelligence CLI Entrypoint"
    )
    parser.add_argument("--inspect", action="store_true", help="Inspect raw dataset schema and missing values")
    parser.add_argument("--clean", action="store_true", help="Execute data cleaning pipeline")
    parser.add_argument("--features", action="store_true", help="Generate engineered features and targets")
    parser.add_argument("--eda", action="store_true", help="Generate and save exploratory visualization figures")
    parser.add_argument("--train", action="store_true", help="Model training status check")
    parser.add_argument("--app", action="store_true", help="Launch Streamlit dashboard application")

    args = parser.parse_args()

    if args.inspect:
        cmd_inspect()
    elif args.clean:
        cmd_clean()
    elif args.features:
        cmd_features()
    elif args.eda:
        cmd_eda()
    elif args.train:
        print("\nℹ️ Model training is staged and configured in src/regression.py & src/classification.py.")
        print("As per project instructions, models are not trained yet during architecture setup.")
        print("To initiate model training in Phase 2, use the designated training workflows.")
    elif args.app:
        cmd_app()
    else:
        print("\nAvailable CLI workflows:")
        print("  python main.py --inspect   : Profile and audit raw dataset")
        print("  python main.py --clean     : Clean raw dataset and save standardized CSV")
        print("  python main.py --features  : Compute Haversine distance and analytical features")
        print("  python main.py --eda       : Generate exploratory figures to reports/figures/")
        print("  python main.py --train     : Inspect model training pipeline")
        print("  python main.py --app       : Launch interactive Streamlit application")
        print("\nRunning default inspection...")
        cmd_inspect()


if __name__ == "__main__":
    main()

