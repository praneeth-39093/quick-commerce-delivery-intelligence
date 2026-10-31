"""
Feature Engineering module for Quick-Commerce Delivery Intelligence.
Extracts geospatial distance, preparation latency, temporal indicators, traffic ordinals,
and constructs targets for both ETA regression and late-delivery classification.
"""

from typing import Tuple, List, Optional
import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

import config

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class FeatureEngineer:
    """Computes domain-specific features and provides scikit-learn compatible transformation pipelines."""

    def __init__(self, sla_threshold_min: float = config.DEFAULT_SLA_MINUTES):
        # NOTE: sla_threshold_min is a configurable demonstration benchmark, not an official platform SLA.
        self.sla_threshold_min = sla_threshold_min
        self.preprocessor: Optional[ColumnTransformer] = None
        self.feature_names_: List[str] = []
        self.city_median_distances_: dict = {}

    def transform_features(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """
        Calculates all engineered features from cleaned data.

        Args:
            df: Cleaned DataFrame.
            is_training: Whether processing training split.

        Returns:
            pd.DataFrame: DataFrame augmented with analytical features.
        """
        logger.info("Computing engineered features...")
        df = df.copy()

        df = self._calculate_haversine_distance(df)
        df = self._calculate_prep_time(df)
        df = self._extract_temporal_features(df)
        df = self._extract_city_code(df)
        df = self._encode_traffic_density(df)
        df = self._create_late_delivery_target(df)

        logger.info(f"Feature engineering completed. Resulting shape: {df.shape}")
        return df

    def _calculate_haversine_distance(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates great-circle distance between restaurant and delivery location in kilometers.
        Only calculates distance when coordinates are valid and in bounds.
        For invalid/missing coordinates, imputes distance using city-level median distance.
        """
        req_cols = [
            "restaurant_latitude", "restaurant_longitude",
            "delivery_location_latitude", "delivery_location_longitude"
        ]
        if all(col in df.columns for col in req_cols):
            # Mask valid coordinates within geographical boundaries
            lat_min, lat_max = config.LATITUDE_BOUNDS
            lon_min, lon_max = config.LONGITUDE_BOUNDS

            valid_coords = (
                df["restaurant_latitude"].between(lat_min, lat_max) &
                df["restaurant_longitude"].between(lon_min, lon_max) &
                df["delivery_location_latitude"].between(lat_min, lat_max) &
                df["delivery_location_longitude"].between(lon_min, lon_max)
            )

            distances = pd.Series(np.nan, index=df.index, dtype=float)

            if valid_coords.any():
                v_df = df.loc[valid_coords]
                lat1 = np.radians(v_df["restaurant_latitude"].values)
                lon1 = np.radians(v_df["restaurant_longitude"].values)
                lat2 = np.radians(v_df["delivery_location_latitude"].values)
                lon2 = np.radians(v_df["delivery_location_longitude"].values)

                dlat = lat2 - lat1
                dlon = lon2 - lon1

                a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
                c = 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
                d = config.EARTH_RADIUS_KM * c
                distances.loc[valid_coords] = np.clip(np.round(d, 2), 0.1, 50.0)

            # Impute invalid/missing distance using city median, fallback to global median
            global_median_dist = float(distances.median(skipna=True)) if distances.notnull().any() else 5.0

            if "city" in df.columns:
                city_medians = distances.groupby(df["city"]).transform("median")
                df["distance_km"] = distances.fillna(city_medians).fillna(global_median_dist)
            else:
                df["distance_km"] = distances.fillna(global_median_dist)

            df["distance_imputed"] = (~valid_coords).astype(int)
            logger.info("Haversine distance calculated (invalid GPS rows imputed via city/global distance median).")
        else:
            df["distance_km"] = 5.0
            df["distance_imputed"] = 1
        return df

    def _calculate_prep_time(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates kitchen/merchant preparation latency: (time_picked_min - time_ordered_min).
        Handles midnight transitions.
        
        CRITICAL DATA LEAKAGE PREVENTION:
        'prep_time_min' and 'time_picked_min' are strictly exploratory / descriptive telemetry metrics.
        They require 'Time_Order_picked' which is only known post-pickup.
        THEY MUST NEVER ENTER MODEL TRAINING FEATURE MATRICES (X) FOR ORDER-PLACEMENT PREDICTIONS.
        """
        if "time_picked_min" in df.columns and "time_ordered_min" in df.columns:
            diff = df["time_picked_min"] - df["time_ordered_min"]
            # Handle midnight crossover (e.g. ordered 23:55 [1435 min], picked 00:10 [10 min])
            diff = np.where(diff < 0, diff + 1440.0, diff)
            # Clip between 1 and 60 minutes
            df["prep_time_min"] = np.clip(np.round(diff, 1), 1.0, 60.0)
            logger.info("Kitchen preparation latency ('prep_time_min') calculated for exploratory analytics.")
        else:
            df["prep_time_min"] = np.nan
        return df

    def _extract_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Derives hour of day, day of week, weekend indicator, and peak rush hour flags."""
        if "time_ordered_min" in df.columns:
            df["order_hour"] = (df["time_ordered_min"] // 60).astype(int) % 24
        elif "order_date" in df.columns and pd.api.types.is_datetime64_any_dtype(df["order_date"]):
            df["order_hour"] = df["order_date"].dt.hour
        else:
            df["order_hour"] = 18

        if "order_date" in df.columns and pd.api.types.is_datetime64_any_dtype(df["order_date"]):
            df["order_day_of_week"] = df["order_date"].dt.dayofweek
            df["is_weekend"] = df["order_day_of_week"].isin([5, 6]).astype(int)
        else:
            df["order_day_of_week"] = 3
            df["is_weekend"] = 0

        # High-volume rush periods in quick-commerce: lunch (12-14) and dinner (19-22)
        df["is_rush_hour"] = df["order_hour"].isin([12, 13, 14, 19, 20, 21, 22]).astype(int)
        return df

    def _extract_city_code(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extracts city abbreviation from delivery_person_id (e.g., 'INDORES13DEL02' -> 'INDO')."""
        if "delivery_person_id" in df.columns:
            df["city_code"] = df["delivery_person_id"].astype(str).str.extract(r"^([A-Z]+)RES")[0]
            df["city_code"] = df["city_code"].fillna("UNKNOWN")
        return df

    def _encode_traffic_density(self, df: pd.DataFrame) -> pd.DataFrame:
        """Maps traffic density to ordinal scale."""
        if "road_traffic_density" in df.columns:
            df["traffic_density_ordinal"] = (
                df["road_traffic_density"]
                .map(config.TRAFFIC_ORDINAL_MAP)
                .fillna(2)
                .astype(int)
            )
        return df

    def _create_late_delivery_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates binary classification label:
        1 if actual delivery time exceeds SLA threshold, 0 otherwise.
        """
        if config.CLEANED_TARGET_REGRESSION in df.columns:
            df[config.CLEANED_TARGET_CLASSIFICATION] = (
                df[config.CLEANED_TARGET_REGRESSION] > self.sla_threshold_min
            ).astype(int)
            late_rate = df[config.CLEANED_TARGET_CLASSIFICATION].mean() * 100
            logger.info(f"Target '{config.CLEANED_TARGET_CLASSIFICATION}' created. Late Delivery Rate: {late_rate:.2f}%")
        return df

    def build_preprocessing_pipeline(
        self,
        numerical_cols: List[str] = config.MODEL_CONFIG.numerical_features,
        categorical_cols: List[str] = config.MODEL_CONFIG.categorical_features
    ) -> ColumnTransformer:
        """
        Instantiates scikit-learn ColumnTransformer for model feeding.

        Args:
            numerical_cols: Numerical feature columns to standardize.
            categorical_cols: Categorical feature columns to one-hot encode.

        Returns:
            ColumnTransformer
        """
        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numerical_cols),
                ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols)
            ],
            remainder="drop"
        )
        return self.preprocessor

    def get_feature_matrices(
        self,
        df: pd.DataFrame,
        task: str = "regression"
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Extracts feature matrix X and target vector y for training or evaluation.
        Strictly guarantees leakage prevention: post-outcome features and target variables
        are forbidden from entering X.

        Args:
            df: Featured DataFrame.
            task: 'regression' or 'classification'.

        Returns:
            Tuple[pd.DataFrame, pd.Series]: (X, y)
        """
        available_num = [c for c in config.MODEL_CONFIG.numerical_features if c in df.columns]
        available_cat = [c for c in config.MODEL_CONFIG.categorical_features if c in df.columns]
        feature_cols = available_num + available_cat

        # Remove any quarantined leakage columns that might accidentally be present
        feature_cols = [c for c in feature_cols if c not in config.LEAKAGE_COLUMNS_QUARANTINE]

        X = df[feature_cols].copy()

        # Target selection and isolation
        if task == "regression":
            target_col = config.CLEANED_TARGET_REGRESSION
        elif task == "classification":
            target_col = config.CLEANED_TARGET_CLASSIFICATION
        else:
            raise ValueError(f"Unknown task: {task}. Must be 'regression' or 'classification'.")

        # Explicit verification: ensure target and leakage features never enter X
        for forbidden in config.LEAKAGE_COLUMNS_QUARANTINE:
            if forbidden in X.columns:
                logger.warning(f"Removing quarantined leakage feature from X: {forbidden}")
                X = X.drop(columns=[forbidden])

        if target_col in X.columns:
            X = X.drop(columns=[target_col])

        if target_col in df.columns:
            y = df[target_col].copy()
        else:
            y = pd.Series(dtype=float)

        return X, y

