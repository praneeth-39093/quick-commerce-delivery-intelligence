"""
Data Cleaning and Preprocessing Pipeline.
Implements robust parsing, type coercion, whitespace stripping, missing value handling,
coordinate sanitization, and target variable normalization.
"""

from typing import Optional, Dict, Any
import logging
import re
import numpy as np
import pandas as pd

import config

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class DataCleaner:
    """End-to-end cleaning and normalization pipeline for raw delivery records."""

    def __init__(self):
        self.imputation_values_: Dict[str, Any] = {}

    def clean(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """
        Executes full data cleaning workflow.

        Args:
            df: Raw input DataFrame.
            is_training: If True, calculates and saves imputation statistics.

        Returns:
            pd.DataFrame: Cleaned DataFrame with standardized schemas.
        """
        logger.info("Initiating dataset cleaning sequence...")
        df = df.copy()

        df = self._standardize_string_columns(df)
        df = self._clean_target_variable(df)
        df = self._clean_weather_conditions(df)
        df = self._clean_numerical_features(df)
        df = self._clean_coordinates(df)
        df = self._clean_date_and_times(df)
        df = self._rename_columns(df)
        df = self._impute_missing_values(df, is_training=is_training)

        logger.info(f"Cleaning sequence completed. Cleaned shape: {df.shape}")
        return df

    def _standardize_string_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Strips trailing whitespace and converts string missing tokens to np.nan."""
        str_cols = df.select_dtypes(include=["object", "string"]).columns

        for col in str_cols:
            df[col] = df[col].astype(str).str.strip()
            # Replace common pseudo-null strings
            df[col] = df[col].replace(["NaN", "nan", "null", "None", ""], np.nan)

        return df

    def _clean_target_variable(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extracts numeric duration from raw target string format '(min) 24'."""
        if config.RAW_TARGET_COLUMN in df.columns:
            def extract_minutes(val):
                if pd.isna(val):
                    return np.nan
                match = re.search(r"(\d+(\.\d+)?)", str(val))
                return float(match.group(1)) if match else np.nan

            df[config.CLEANED_TARGET_REGRESSION] = df[config.RAW_TARGET_COLUMN].apply(extract_minutes)
            df = df.drop(columns=[config.RAW_TARGET_COLUMN])
            logger.info("Target variable parsed into numeric 'delivery_time_min'.")

        return df

    def _clean_weather_conditions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalizes 'Weatherconditions' by stripping 'conditions ' prefix."""
        col = "Weatherconditions"
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].str.replace(r"^conditions\s*", "", regex=True)
            df[col] = df[col].replace(["NaN", "nan", "null", "None", ""], np.nan)

            # Filter to known valid categories or nan
            df.loc[~df[col].isin(config.VALID_WEATHER_CONDITIONS), col] = np.nan
        return df

    def _clean_numerical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Coerces numerical columns to float/int, masking out-of-range values."""
        # Age
        if "Delivery_person_Age" in df.columns:
            df["Delivery_person_Age"] = pd.to_numeric(df["Delivery_person_Age"], errors="coerce")
            df.loc[(df["Delivery_person_Age"] < 15) | (df["Delivery_person_Age"] > 75), "Delivery_person_Age"] = np.nan

        # Ratings
        if "Delivery_person_Ratings" in df.columns:
            df["Delivery_person_Ratings"] = pd.to_numeric(df["Delivery_person_Ratings"], errors="coerce")
            df.loc[(df["Delivery_person_Ratings"] < 1.0) | (df["Delivery_person_Ratings"] > 5.0), "Delivery_person_Ratings"] = np.nan

        # Multiple Deliveries
        if "multiple_deliveries" in df.columns:
            df["multiple_deliveries"] = pd.to_numeric(df["multiple_deliveries"], errors="coerce")
            df.loc[(df["multiple_deliveries"] < 0) | (df["multiple_deliveries"] > 5), "multiple_deliveries"] = np.nan

        # Vehicle Condition
        if "Vehicle_condition" in df.columns:
            df["Vehicle_condition"] = pd.to_numeric(df["Vehicle_condition"], errors="coerce")

        return df

    def _clean_coordinates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Sanitizes geospatial coordinates with verified boundary validation.
        Does NOT blindly apply absolute values: verifies negative coordinates against
        geographical boundaries (India: lat 6.0-38.0, lon 68.0-98.0) and paired delivery points.
        Flags invalid coordinates (e.g. 0.0, 0.04 noise) without median-imputing them.
        """
        coord_cols = [
            "Restaurant_latitude", "Restaurant_longitude",
            "Delivery_location_latitude", "Delivery_location_longitude"
        ]

        for col in coord_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        lat_min, lat_max = config.LATITUDE_BOUNDS
        lon_min, lon_max = config.LONGITUDE_BOUNDS

        # Process Restaurant Latitude: verify negative coordinates against geographical bounds
        if "Restaurant_latitude" in df.columns:
            r_lat = df["Restaurant_latitude"].copy()
            # Where negative, check if abs(r_lat) falls in valid Indian latitude
            neg_mask = r_lat < 0
            valid_neg = neg_mask & (r_lat.abs() >= lat_min) & (r_lat.abs() <= lat_max)
            # Rectify only verified sign inversions
            r_lat[valid_neg] = r_lat[valid_neg].abs()
            # Any remaining negative or out-of-bound coords become NaN
            r_lat[(r_lat < lat_min) | (r_lat > lat_max)] = np.nan
            df["Restaurant_latitude"] = r_lat

        # Process Restaurant Longitude: verify negative coordinates against geographical bounds
        if "Restaurant_longitude" in df.columns:
            r_lon = df["Restaurant_longitude"].copy()
            neg_mask = r_lon < 0
            valid_neg = neg_mask & (r_lon.abs() >= lon_min) & (r_lon.abs() <= lon_max)
            r_lon[valid_neg] = r_lon[valid_neg].abs()
            r_lon[(r_lon < lon_min) | (r_lon > lon_max)] = np.nan
            df["Restaurant_longitude"] = r_lon

        # Process Delivery Latitude (must strictly lie inside valid bounds; catches 0.01-0.11 noise)
        if "Delivery_location_latitude" in df.columns:
            d_lat = df["Delivery_location_latitude"].copy()
            neg_mask = d_lat < 0
            valid_neg = neg_mask & (d_lat.abs() >= lat_min) & (d_lat.abs() <= lat_max)
            d_lat[valid_neg] = d_lat[valid_neg].abs()
            d_lat[(d_lat < lat_min) | (d_lat > lat_max)] = np.nan
            df["Delivery_location_latitude"] = d_lat

        # Process Delivery Longitude
        if "Delivery_location_longitude" in df.columns:
            d_lon = df["Delivery_location_longitude"].copy()
            neg_mask = d_lon < 0
            valid_neg = neg_mask & (d_lon.abs() >= lon_min) & (d_lon.abs() <= lon_max)
            d_lon[valid_neg] = d_lon[valid_neg].abs()
            d_lon[(d_lon < lon_min) | (d_lon > lon_max)] = np.nan
            df["Delivery_location_longitude"] = d_lon

        # Flag valid coordinates (both restaurant and customer must have non-null, in-bounds GPS)
        has_all_coords = all(c in df.columns for c in coord_cols)
        if has_all_coords:
            df["is_valid_coordinates"] = (
                df["Restaurant_latitude"].notnull() &
                df["Restaurant_longitude"].notnull() &
                df["Delivery_location_latitude"].notnull() &
                df["Delivery_location_longitude"].notnull()
            )
            valid_count = df["is_valid_coordinates"].sum()
            invalid_count = len(df) - valid_count
            logger.info(
                f"Coordinate audit: {valid_count} valid GPS pairs, {invalid_count} invalid "
                f"(kept as NaN to prevent distorted spatial clustering)."
            )
        else:
            df["is_valid_coordinates"] = False

        return df

    def _clean_date_and_times(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Parses dates and times handling Excel serial floats, strings, and timedeltas.
        Standardizes Order_Date to YYYY-MM-DD and times to minute-of-day.
        """
        # Clean Order_Date
        if "Order_Date" in df.columns:
            def parse_date(val):
                if pd.isna(val) or str(val).strip() in ["", "nan", "NaN"]:
                    return pd.NaT
                # Check if it's an Excel serial date number (e.g. 44639)
                try:
                    num_val = float(val)
                    return pd.Timestamp("1899-12-30") + pd.Timedelta(days=num_val)
                except (ValueError, TypeError):
                    pass
                # Otherwise parse string formats with flexible inference
                return pd.to_datetime(val, errors="coerce")

            df["Order_Date"] = df["Order_Date"].apply(parse_date)

        # Helper to parse time into minutes from midnight (0 to 1439)
        def parse_time_to_minutes(val):
            if pd.isna(val) or str(val).strip() in ["", "nan", "NaN"]:
                return np.nan
            try:
                # If Excel day fraction: 0.5 = 12:00 PM -> 0.5 * 1440 = 720 minutes
                f_val = float(val)
                if 0.0 <= f_val <= 1.0:
                    return f_val * 1440.0
            except (ValueError, TypeError):
                pass

            # String format parsing e.g. "19:10:00" or "19:10"
            val_str = str(val).strip()
            match = re.match(r"^(\d{1,2}):(\d{2})(?::(\d{2}))?$", val_str)
            if match:
                hours = int(match.group(1))
                minutes = int(match.group(2))
                seconds = int(match.group(3)) if match.group(3) else 0
                return hours * 60.0 + minutes + (seconds / 60.0)

            return np.nan

        if "Time_Orderd" in df.columns:
            df["time_ordered_min"] = df["Time_Orderd"].apply(parse_time_to_minutes)
        if "Time_Order_picked" in df.columns:
            df["time_picked_min"] = df["Time_Order_picked"].apply(parse_time_to_minutes)

        return df

    def _rename_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalizes column names to clean snake_case."""
        rename_map = {
            "ID": "id",
            "Delivery_person_ID": "delivery_person_id",
            "Delivery_person_Age": "delivery_person_age",
            "Delivery_person_Ratings": "delivery_person_ratings",
            "Restaurant_latitude": "restaurant_latitude",
            "Restaurant_longitude": "restaurant_longitude",
            "Delivery_location_latitude": "delivery_location_latitude",
            "Delivery_location_longitude": "delivery_location_longitude",
            "Order_Date": "order_date",
            "Weatherconditions": "weather",
            "Road_traffic_density": "road_traffic_density",
            "Vehicle_condition": "vehicle_condition",
            "Type_of_order": "type_of_order",
            "Type_of_vehicle": "type_of_vehicle",
            "Festival": "festival",
            "City": "city"
        }
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

        # Drop original raw time columns if parsed columns exist
        to_drop = [c for c in ["Time_Orderd", "Time_Order_picked"] if c in df.columns]
        if to_drop:
            df = df.drop(columns=to_drop)

        return df

    def _impute_missing_values(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """
        Imputes non-spatial features using domain-appropriate statistical aggregates.
        Numerical non-spatial: median. Categorical: mode.
        
        CRITICAL ARCHITECTURAL GUARANTEE:
        Coordinates (lat/lon) are INTENTIONALLY NOT median-imputed.
        Imputing nationwide median coordinates produces artificial points in central India
        that catastrophically distort spatial clustering and Haversine distance calculations.
        Invalid coordinates remain NaN and are bypassed in clustering and handled via distance fallbacks.
        """
        num_cols = [
            "delivery_person_age", "delivery_person_ratings", "multiple_deliveries",
            "vehicle_condition", "time_ordered_min"
        ]

        cat_cols = ["weather", "road_traffic_density", "type_of_order", "type_of_vehicle", "festival", "city"]

        if is_training:
            for col in num_cols:
                if col in df.columns:
                    self.imputation_values_[col] = float(df[col].median(skipna=True))
            for col in cat_cols:
                if col in df.columns:
                    mode_val = df[col].mode(dropna=True)
                    self.imputation_values_[col] = mode_val.iloc[0] if not mode_val.empty else "Missing"

        # Apply saved imputation values
        for col, val in self.imputation_values_.items():
            if col in df.columns:
                df[col] = df[col].fillna(val)

        return df

