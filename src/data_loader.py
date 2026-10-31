"""
Data loading and dataset validation module.
Responsible for ingesting raw data formats (.xlsx, .csv), verifying structural integrity,
and handling train/test data splitting.
"""

from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import logging
import pandas as pd
from sklearn.model_selection import train_test_split

import config

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class DataLoader:
    """Handles raw and processed dataset ingestion and schema validation."""

    def __init__(self, raw_path: Optional[Path] = None):
        self.raw_path = Path(raw_path) if raw_path else config.RAW_DATA_PATH

    def load_raw_data(self, filepath: Optional[Path] = None) -> pd.DataFrame:
        """
        Loads raw delivery dataset from Excel (.xlsx) or CSV.

        Args:
            filepath: Optional override for dataset path.

        Returns:
            pd.DataFrame: Loaded raw dataset.
        """
        target_path = Path(filepath) if filepath else self.raw_path

        if not target_path.exists():
            raise FileNotFoundError(f"Dataset file not found at: {target_path}")

        logger.info(f"Loading raw dataset from: {target_path}")

        if target_path.suffix.lower() in [".xlsx", ".xls"]:
            df = pd.read_excel(target_path)
        elif target_path.suffix.lower() == ".csv":
            df = pd.read_csv(target_path)
        else:
            raise ValueError(f"Unsupported file format: {target_path.suffix}")

        logger.info(f"Successfully loaded dataset with shape: {df.shape}")
        return df

    def validate_raw_schema(self, df: pd.DataFrame) -> Tuple[bool, list]:
        """
        Validates presence of essential columns in raw dataframe.

        Args:
            df: Raw DataFrame to check.

        Returns:
            Tuple[bool, list]: (is_valid, list of missing columns)
        """
        expected_cols = (
            config.RAW_ID_COLUMNS
            + config.RAW_NUMERICAL_COLUMNS
            + config.RAW_CATEGORICAL_COLUMNS
            + config.RAW_DATETIME_COLUMNS
            + [config.RAW_TARGET_COLUMN]
        )

        missing = [col for col in expected_cols if col not in df.columns]
        is_valid = len(missing) == 0

        if not is_valid:
            logger.warning(f"Schema validation warning: Missing columns {missing}")
        else:
            logger.info("Schema validation passed: All expected columns are present.")

        return is_valid, missing

    def inspect_dataset(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generates comprehensive structural report of the dataset.

        Args:
            df: DataFrame to profile.

        Returns:
            Dict[str, Any]: Metadata dictionary.
        """
        null_counts = df.isnull().sum().to_dict()
        dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}

        summary = {
            "total_rows": int(df.shape[0]),
            "total_columns": int(df.shape[1]),
            "columns": list(df.columns),
            "data_types": dtypes,
            "null_counts": null_counts,
            "memory_usage_mb": float(round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2))
        }

        logger.info(
            f"Dataset Summary: {summary['total_rows']} rows, {summary['total_columns']} columns, "
            f"{summary['memory_usage_mb']} MB"
        )
        return summary

    def split_data(
        self,
        df: pd.DataFrame,
        target_col: str,
        test_size: float = config.MODEL_CONFIG.test_size,
        random_state: int = config.MODEL_CONFIG.random_state,
        stratify: bool = False
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Splits data into training and test sets.

        Args:
            df: Cleaned/Featured DataFrame.
            target_col: Target column name.
            test_size: Proportion of test split.
            random_state: Seed for reproducibility.
            stratify: Whether to perform stratified sampling.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (train_df, test_df)
        """
        strat_col = df[target_col] if stratify and target_col in df.columns else None

        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=strat_col
        )

        logger.info(f"Split dataset into Train: {train_df.shape}, Test: {test_df.shape}")
        return train_df, test_df

    def split_chronological(
        self,
        df: pd.DataFrame,
        date_col: str = "order_date",
        train_end_date: str = config.MODEL_CONFIG.train_end_date,
        val_end_date: str = config.MODEL_CONFIG.val_end_date
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Splits data chronologically into train, validation, and test sets.
        Ensures strict temporal ordering: models train on historical data, tune on intermediate validation,
        and evaluate on unseen future holdout test data.

        Args:
            df: Featured DataFrame.
            date_col: Name of order date column.
            train_end_date: Cutoff date string (YYYY-MM-DD) for training data.
            val_end_date: Cutoff date string (YYYY-MM-DD) for validation data.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: (train_df, val_df, test_df)
        """
        df = df.copy()
        date_series = pd.to_datetime(df[date_col], errors="coerce")

        train_cutoff = pd.to_datetime(train_end_date)
        val_cutoff = pd.to_datetime(val_end_date)

        train_mask = date_series < train_cutoff
        val_mask = (date_series >= train_cutoff) & (date_series < val_cutoff)
        test_mask = date_series >= val_cutoff

        train_df = df.loc[train_mask].copy().reset_index(drop=True)
        val_df = df.loc[val_mask].copy().reset_index(drop=True)
        test_df = df.loc[test_mask].copy().reset_index(drop=True)

        logger.info(
            f"Chronological split: Train ({len(train_df)} rows, < {train_end_date}), "
            f"Val ({len(val_df)} rows, {train_end_date} to {val_end_date}), "
            f"Test ({len(test_df)} rows, >= {val_end_date})"
        )
        return train_df, val_df, test_df

    def save_processed_data(self, df: pd.DataFrame, destination_path: Path) -> Path:
        """Saves processed dataframe to CSV."""
        destination_path = Path(destination_path)
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(destination_path, index=False)
        logger.info(f"Saved processed dataset to: {destination_path}")
        return destination_path

