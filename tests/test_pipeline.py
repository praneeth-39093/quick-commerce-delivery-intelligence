"""
Unit and Integration Tests for Quick-Commerce Delivery Intelligence Pipeline.
Validates configuration integrity, data cleaning logic, feature engineering formulas,
heuristics, leakage prevention, GPS coordinate validation, and inference pipelines.
"""

import sys
from pathlib import Path
import pytest
import numpy as np
import pandas as pd

# Add repository root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import config
from src.data_cleaning import DataCleaner
from src.feature_engineering import FeatureEngineer
from src.recommendations import DeliveryOptimizer
from src.model_evaluation import ModelEvaluator
from src.prediction import DeliveryInferencePipeline
from src.clustering import DeliveryOptimizerClustering


class TestPipelineConfiguration:
    """Tests file and path configuration resolution."""

    def test_paths_relative(self):
        assert config.BASE_DIR.exists()
        assert config.DATA_DIR.exists()
        assert config.REPORTS_DIR.exists()
        assert config.RAW_DATA_PATH.name == "train.xlsx"

    def test_traffic_map(self):
        assert config.TRAFFIC_ORDINAL_MAP["Low"] < config.TRAFFIC_ORDINAL_MAP["Jam"]
        assert config.DEFAULT_SLA_MINUTES == 30.0

    def test_leakage_features_quarantined_in_config(self):
        """Verifies prep_time_min, time_picked_min, and multiple_deliveries are strictly absent from model training features."""
        assert "prep_time_min" not in config.MODEL_CONFIG.numerical_features
        assert "time_picked_min" not in config.MODEL_CONFIG.numerical_features
        assert "Time_Order_picked" not in config.MODEL_CONFIG.numerical_features
        assert "multiple_deliveries" not in config.MODEL_CONFIG.numerical_features


class TestDataCleaner:
    """Tests data cleaning and normalization transforms."""

    @pytest.fixture
    def sample_raw_data(self):
        return pd.DataFrame({
            "ID": ["0x101 ", "0x102 "],
            "Delivery_person_ID": ["BLRRES01DEL01 ", "DELRES02DEL02 "],
            "Delivery_person_Age": ["28 ", "35 "],
            "Delivery_person_Ratings": ["4.8 ", "NaN "],
            "Restaurant_latitude": [-12.9716, 28.7041],
            "Restaurant_longitude": [-77.5946, 77.1025],
            "Delivery_location_latitude": [12.9916, 28.7241],
            "Delivery_location_longitude": [77.6146, 77.1225],
            "Order_Date": ["2022-03-15", "2022-03-16"],
            "Time_Orderd": ["19:15:00", "0.50"],
            "Time_Order_picked": ["19:30:00", "0.51"],
            "Weatherconditions": ["conditions Sunny ", "conditions Stormy "],
            "Road_traffic_density": ["High ", "Jam "],
            "Vehicle_condition": ["1", "2"],
            "Type_of_order": ["Meal ", "Snack "],
            "Type_of_vehicle": ["motorcycle ", "scooter "],
            "multiple_deliveries": ["0", "1"],
            "Festival": ["No ", "No "],
            "City": ["Metropolitian ", "Urban "],
            "Time_taken(min)": ["(min) 24", "(min) 38"]
        })

    def test_clean_transformations(self, sample_raw_data):
        cleaner = DataCleaner()
        cleaned_df = cleaner.clean(sample_raw_data, is_training=True)

        # Check target column extraction
        assert "delivery_time_min" in cleaned_df.columns
        assert cleaned_df["delivery_time_min"].iloc[0] == 24.0
        assert cleaned_df["delivery_time_min"].iloc[1] == 38.0

        # Check weather prefix removal
        assert cleaned_df["weather"].iloc[0] == "Sunny"
        assert cleaned_df["weather"].iloc[1] == "Stormy"

        # Check coordinate sign rectification for valid Indian latitudes
        assert cleaned_df["restaurant_latitude"].iloc[0] == 12.9716
        assert cleaned_df["restaurant_longitude"].iloc[0] == 77.5946

        # Check missing rating imputed
        assert not cleaned_df["delivery_person_ratings"].isnull().any()


class TestGPSValidation:
    """Dedicated validation of geographic boundary checks and coordinate integrity."""

    def test_zero_coordinates_flagged_invalid(self):
        """Ensures (0,0) and near-zero telemetry noise are flagged invalid and not median-imputed."""
        df_zero = pd.DataFrame({
            "Restaurant_latitude": [0.0, 12.97],
            "Restaurant_longitude": [0.0, 77.59],
            "Delivery_location_latitude": [0.08, 12.99],
            "Delivery_location_longitude": [0.08, 77.61]
        })
        cleaner = DataCleaner()
        cleaned = cleaner.clean(df_zero, is_training=True)

        assert pd.isna(cleaned["restaurant_latitude"].iloc[0])
        assert pd.isna(cleaned["delivery_location_latitude"].iloc[0])
        assert cleaned["is_valid_coordinates"].iloc[0] == False
        assert cleaned["is_valid_coordinates"].iloc[1] == True

    def test_out_of_boundary_negatives_rejected(self):
        """Ensures negative values outside India bounds (e.g. -50.0) are NOT converted to valid positives."""
        df_out = pd.DataFrame({
            "Restaurant_latitude": [-50.0],
            "Restaurant_longitude": [77.59],
            "Delivery_location_latitude": [12.99],
            "Delivery_location_longitude": [77.61]
        })
        cleaner = DataCleaner()
        cleaned = cleaner.clean(df_out, is_training=True)
        assert pd.isna(cleaned["restaurant_latitude"].iloc[0])

    def test_no_coordinate_median_imputation(self):
        """Verifies that missing coordinates remain NaN and are NEVER median-imputed."""
        df_missing = pd.DataFrame({
            "Restaurant_latitude": [12.97, np.nan],
            "Restaurant_longitude": [77.59, np.nan],
            "Delivery_location_latitude": [12.99, np.nan],
            "Delivery_location_longitude": [77.61, np.nan],
            "delivery_person_ratings": [4.8, np.nan]
        })
        cleaner = DataCleaner()
        cleaned = cleaner.clean(df_missing, is_training=True)

        # Non-spatial features are imputed
        assert not pd.isna(cleaned["delivery_person_ratings"].iloc[1])
        # Coordinates must strictly remain NaN to avoid polluting spatial clustering
        assert pd.isna(cleaned["restaurant_latitude"].iloc[1])
        assert pd.isna(cleaned["delivery_location_latitude"].iloc[1])


class TestFeatureEngineerAndLeakagePrevention:
    """Tests feature generation math, target separation, and strict leakage prevention."""

    @pytest.fixture
    def sample_cleaned_df(self):
        return pd.DataFrame({
            "restaurant_latitude": [12.9716],
            "restaurant_longitude": [77.5946],
            "delivery_location_latitude": [12.9916],
            "delivery_location_longitude": [77.6146],
            "time_ordered_min": [1155.0],  # 19:15
            "time_picked_min": [1170.0],   # 19:30 -> prep = 15 min
            "delivery_time_min": [35.0],   # Late order (> 30 min)
            "road_traffic_density": ["High"],
            "delivery_person_id": ["BLRRES01DEL01"],
            "city": ["Metropolitian"]
        })

    def test_feature_transformations(self, sample_cleaned_df):
        engineer = FeatureEngineer(sla_threshold_min=30.0)
        df_feat = engineer.transform_features(sample_cleaned_df, is_training=True)

        assert "distance_km" in df_feat.columns
        assert df_feat["distance_km"].iloc[0] > 0.0
        assert "prep_time_min" in df_feat.columns
        assert df_feat["prep_time_min"].iloc[0] == 15.0
        assert "traffic_density_ordinal" in df_feat.columns
        assert df_feat["traffic_density_ordinal"].iloc[0] == 3  # High = 3

        # Check late delivery target flag
        assert "is_late" in df_feat.columns
        assert df_feat["is_late"].iloc[0] == 1  # 35.0 > 30.0 SLA

    def test_strict_leakage_prevention_in_feature_matrix(self, sample_cleaned_df):
        """Verifies that NO post-outcome or target-derived features enter X."""
        engineer = FeatureEngineer(sla_threshold_min=30.0)
        df_feat = engineer.transform_features(sample_cleaned_df, is_training=True)

        X_reg, y_reg = engineer.get_feature_matrices(df_feat, task="regression")
        X_clf, y_clf = engineer.get_feature_matrices(df_feat, task="classification")

        for forbidden in config.LEAKAGE_COLUMNS_QUARANTINE:
            assert forbidden not in X_reg.columns, f"Leakage violation: {forbidden} found in X_reg!"
            assert forbidden not in X_clf.columns, f"Leakage violation: {forbidden} found in X_clf!"

        assert "delivery_time_min" not in X_reg.columns
        assert "is_late" not in X_clf.columns


class TestClusteringValidation:
    """Tests clustering coordinate safety and deterministic partner profiling."""

    def test_spatial_clustering_bypasses_invalid_coords(self):
        clustering = DeliveryOptimizerClustering(n_clusters=2)
        df_coords = pd.DataFrame({
            "delivery_location_latitude": [12.97, 13.00, 0.08, np.nan],
            "delivery_location_longitude": [77.59, 77.62, 0.08, np.nan]
        })
        clustered = clustering.fit_spatial_clusters(df_coords, n_clusters=2)

        # Invalid rows should be tagged with cluster label -1
        assert clustered["delivery_zone_cluster"].iloc[2] == -1
        assert clustered["delivery_zone_cluster"].iloc[3] == -1
        # Valid rows receive non-negative cluster ID
        assert clustered["delivery_zone_cluster"].iloc[0] >= 0
        assert clustered["delivery_zone_cluster"].iloc[1] >= 0

    def test_meaningful_partner_tier_ranking(self):
        clustering = DeliveryOptimizerClustering()
        df_riders = pd.DataFrame({
            "delivery_person_id": ["R1", "R1", "R1", "R2", "R2", "R2", "R3", "R3", "R3"],
            "delivery_time_min": [18.0, 19.0, 17.0, 28.0, 29.0, 30.0, 45.0, 46.0, 48.0],
            "delivery_person_ratings": [4.9, 4.8, 5.0, 4.5, 4.4, 4.5, 3.8, 3.9, 3.7]
        })
        profiled, segment_summary = clustering.profile_delivery_partners(df_riders)

        r1_tier = profiled.loc[profiled["delivery_person_id"] == "R1", "efficiency_tier"].iloc[0]
        r3_tier = profiled.loc[profiled["delivery_person_id"] == "R3", "efficiency_tier"].iloc[0]

        # R1 (fastest, highest rated) MUST be Tier 0 (Rank 1 High Efficiency)
        # R3 (slowest, lowest rated) MUST be Tier 2 (At Risk)
        assert r1_tier == 0
        assert r3_tier == 2


class TestDeliveryOptimizer:
    """Tests operational dispatch and SLA rules."""

    def test_dynamic_eta(self):
        optimizer = DeliveryOptimizer()
        result = optimizer.compute_dynamic_eta(
            predicted_eta_min=20.0,
            weather="Stormy",
            traffic="Jam",
            safety_margin_min=3.0
        )

        assert result["promised_sla_eta_min"] > 20.0
        assert result["weather_buffer_min"] == 8.0
        assert result["traffic_buffer_min"] == 10.0

    def test_fleet_recommendation(self):
        optimizer = DeliveryOptimizer()
        rec = optimizer.recommend_fleet_dispatch(distance_km=2.5, traffic="Jam", weather="Sunny")
        assert "scooter" in rec["recommended_vehicle"] or "bicycle" in rec["recommended_vehicle"]


class TestModelEvaluator:
    """Tests evaluation metrics calculation."""

    def test_regression_metrics(self):
        evaluator = ModelEvaluator()
        y_true = np.array([20.0, 30.0, 40.0])
        y_pred = np.array([22.0, 29.0, 41.0])

        metrics = evaluator.evaluate_regression(y_true, y_pred)
        assert metrics["mae"] > 0
        assert metrics["rmse"] > 0
        assert metrics["r2_score"] > 0.90


class TestInferencePipeline:
    """Tests end-to-end inference flow without post-outcome dependencies."""

    def test_single_order_inference_without_picked_time(self):
        pipeline = DeliveryInferencePipeline()
        # Strictly order-placement payload: no Time_Order_picked or prep_time_min provided
        order_dict = {
            "Delivery_person_Age": 29,
            "Delivery_person_Ratings": 4.6,
            "restaurant_latitude": 12.97,
            "restaurant_longitude": 77.59,
            "delivery_location_latitude": 12.99,
            "delivery_location_longitude": 77.61,
            "weather": "Sunny",
            "road_traffic_density": "Medium",
            "type_of_vehicle": "motorcycle",
            "type_of_order": "Meal",
            "multiple_deliveries": 0,
            "festival": "No",
            "city": "Metropolitian",
            "vehicle_condition": 1,
            "time_ordered_min": 720.0
        }

        output = pipeline.predict_single(order_dict)

        assert "predicted_eta_minutes" in output
        assert "late_risk_probability" in output
        assert "dynamic_sla" in output
        assert "fleet_recommendation" in output
        assert output["predicted_eta_minutes"] > 0.0


class TestModelArtifactRoundtrip:
    """Verifies that all Phase 3 serialized models deserialize and operate accurately."""

    def test_best_regressor_artifact_loads(self):
        import joblib
        reg_path = config.MODELS_DIR / "best_eta_regressor.joblib"
        assert reg_path.exists(), "best_eta_regressor.joblib does not exist!"
        bundle = joblib.load(reg_path)
        assert "preprocessor" in bundle
        assert "model" in bundle
        assert bundle["algorithm"] == "xgboost"
        assert bundle["test_mae"] < 4.0

    def test_best_classifier_artifact_loads(self):
        import joblib
        clf_path = config.MODELS_DIR / "best_late_classifier.joblib"
        assert clf_path.exists(), "best_late_classifier.joblib does not exist!"
        bundle = joblib.load(clf_path)
        assert "preprocessor" in bundle
        assert "model" in bundle
        assert "threshold" in bundle
        assert bundle["test_f1"] > 0.85

    def test_dark_store_and_rider_clustering_artifacts_load(self):
        import joblib
        ds_path = config.MODELS_DIR / "dark_store_kmeans.joblib"
        rider_path = config.MODELS_DIR / "rider_segmentation_kmeans.joblib"
        assert ds_path.exists()
        assert rider_path.exists()

        ds_bundle = joblib.load(ds_path)
        rider_bundle = joblib.load(rider_path)

        assert len(ds_bundle["centroids"]) == 5
        assert len(rider_bundle["segment_profiles"]) == 3
