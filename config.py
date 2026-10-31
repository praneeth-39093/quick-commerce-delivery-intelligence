"""
Configuration module for Quick-Commerce Delivery Intelligence project.
Centralizes paths, column mappings, data schemas, hyperparameter grids, and operational parameters.
All filesystem paths are dynamically derived relative to the repository root.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any

# Root repository directory
BASE_DIR: Path = Path(__file__).resolve().parent

# Directory hierarchy
DATA_DIR: Path = BASE_DIR / "data"
RAW_DATA_PATH: Path = DATA_DIR / "train.xlsx"
CLEANED_DATA_PATH: Path = DATA_DIR / "cleaned_delivery_data.csv"
PROCESSED_DATA_PATH: Path = DATA_DIR / "featured_delivery_data.csv"

NOTEBOOKS_DIR: Path = BASE_DIR / "notebooks"
MODELS_DIR: Path = BASE_DIR / "models"
REPORTS_DIR: Path = BASE_DIR / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"
METRICS_PATH: Path = REPORTS_DIR / "model_comparison.csv"
APP_DIR: Path = BASE_DIR / "app"
TESTS_DIR: Path = BASE_DIR / "tests"

# Ensure runtime directories exist
for directory in [DATA_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


# ==============================================================================
# DATASET SCHEMA DEFINITIONS
# ==============================================================================

RAW_ID_COLUMNS: List[str] = [
    "ID",
    "Delivery_person_ID"
]

RAW_NUMERICAL_COLUMNS: List[str] = [
    "Delivery_person_Age",
    "Delivery_person_Ratings",
    "Restaurant_latitude",
    "Restaurant_longitude",
    "Delivery_location_latitude",
    "Delivery_location_longitude",
    "multiple_deliveries",
    "Vehicle_condition"
]

RAW_CATEGORICAL_COLUMNS: List[str] = [
    "Weatherconditions",
    "Road_traffic_density",
    "Type_of_order",
    "Type_of_vehicle",
    "Festival",
    "City"
]

RAW_DATETIME_COLUMNS: List[str] = [
    "Order_Date",
    "Time_Orderd",
    "Time_Order_picked"
]

RAW_TARGET_COLUMN: str = "Time_taken(min)"

# Standardized column names after cleaning
CLEANED_TARGET_REGRESSION: str = "delivery_time_min"
CLEANED_TARGET_CLASSIFICATION: str = "is_late"

# Standard Quick-Commerce SLA benchmark threshold (in minutes)
# NOTE: This threshold is an empirical demonstration benchmark / operational baseline
# adopted for late-delivery risk detection, NOT an official contractual or platform-wide SLA.
# It is fully configurable per city, service tier, or experimentation cohort.
DEFAULT_SLA_MINUTES: float = 30.0

# Earth radius in kilometers for Haversine distance
EARTH_RADIUS_KM: float = 6371.0

# Geographic coordinate bounding box for validation (India region)
LATITUDE_BOUNDS = (6.0, 38.0)
LONGITUDE_BOUNDS = (68.0, 98.0)

# Ordinal mappings for traffic density
TRAFFIC_ORDINAL_MAP: Dict[str, int] = {
    "Low": 1,
    "Medium": 2,
    "High": 3,
    "Jam": 4
}

# Weather condition categories
VALID_WEATHER_CONDITIONS: List[str] = [
    "Sunny",
    "Stormy",
    "Sandstorms",
    "Cloudy",
    "Fog",
    "Windy"
]

# Post-outcome and order-placement leakage quarantine list
# These columns MUST NEVER enter model training features (X) for order-placement predictions.
# 'multiple_deliveries' is quarantined because stacking/batching decisions occur downstream during dispatch,
# not at order placement, and creates artificial predictive shortcuts.
LEAKAGE_COLUMNS_QUARANTINE: List[str] = [
    "Time_Order_picked",
    "time_picked_min",
    "prep_time_min",
    "multiple_deliveries",
    "Time_taken(min)",
    "delivery_time_min",
    "is_late",
    "speed_kmh"
]


# ==============================================================================
# MODEL CONFIGURATION
# ==============================================================================

@dataclass
class ModelConfig:
    random_state: int = 42
    test_size: float = 0.20
    val_size: float = 0.10
    cv_folds: int = 5
    n_jobs: int = -1

    # Chronological dataset splitting boundaries (dates: Feb 11 to April 06, 2022)
    # Train: 2022-02-11 to 2022-03-23 (~67.4%, 30,739 rows)
    # Val:   2022-03-24 to 2022-03-30 (~16.5%, 7,525 rows)
    # Test:  2022-03-31 to 2022-04-06 (~16.1%, 7,329 rows)
    train_end_date: str = "2022-03-23"
    val_end_date: str = "2022-03-30"
    
    # Feature columns used for training (Strictly leakage-free at order placement)
    # NOTE: 'prep_time_min', 'time_picked_min', and 'multiple_deliveries' are strictly quarantined.
    numerical_features: List[str] = field(default_factory=lambda: [
        "delivery_person_age",
        "delivery_person_ratings",
        "distance_km",
        "order_hour",
        "order_day_of_week",
        "is_weekend",
        "is_rush_hour",
        "vehicle_condition",
        "traffic_density_ordinal"
    ])
    
    categorical_features: List[str] = field(default_factory=lambda: [
        "weather",
        "road_traffic_density",
        "type_of_order",
        "type_of_vehicle",
        "festival",
        "city"
    ])


@dataclass
class ClusteringConfig:
    n_clusters_zone: int = 5
    random_state: int = 42


MODEL_CONFIG = ModelConfig()
CLUSTERING_CONFIG = ClusteringConfig()

