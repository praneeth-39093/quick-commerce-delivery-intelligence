"""
Unit tests for Streamlit application view modules and data service.
"""

import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pytest
import pandas as pd
import numpy as np

import config
from app.data_service import (
    load_featured_dataset,
    load_model_comparison_df,
    load_regressor_bundle,
    load_classifier_bundle,
    load_dark_store_bundle,
    load_rider_segmentation_bundle,
    calculate_haversine_km,
    execute_eta_inference,
    execute_late_risk_inference
)


class TestStreamlitDataService:
    """Verifies all caching and data service methods for Streamlit."""

    def test_load_featured_dataset(self):
        df = load_featured_dataset()
        assert not df.empty
        assert len(df) == 45593
        assert "delivery_time_min" in df.columns
        assert "is_late" in df.columns

    def test_load_model_comparison_df(self):
        df = load_model_comparison_df()
        assert not df.empty
        assert "algorithm" in df.columns
        assert "task_type" in df.columns
        assert "xgboost" in df["algorithm"].values

    def test_load_all_model_bundles(self):
        reg = load_regressor_bundle()
        clf = load_classifier_bundle()
        ds = load_dark_store_bundle()
        rider = load_rider_segmentation_bundle()

        assert reg is not None
        assert clf is not None
        assert ds is not None
        assert rider is not None

        assert reg["algorithm"] == "xgboost"
        assert clf["threshold"] == 0.59
        assert len(ds["centroids"]) == 5
        assert len(rider["segment_profiles"]) == 3

    def test_calculate_haversine_km(self):
        # Bangalore Dark Store to Indiranagar: approx 4-5 km
        dist = calculate_haversine_km(12.9716, 77.5946, 12.9985, 77.6250)
        assert 3.5 <= dist <= 5.5

    def test_execute_eta_inference(self):
        sample = {
            "restaurant_latitude": 12.9716,
            "restaurant_longitude": 77.5946,
            "delivery_location_latitude": 13.0116,
            "delivery_location_longitude": 77.6346,
            "weather": "Sunny",
            "road_traffic_density": "Medium",
            "type_of_order": "Meal",
            "city": "Metropolitian",
            "delivery_person_ratings": 4.8,
            "delivery_person_age": 28,
            "type_of_vehicle": "motorcycle",
            "vehicle_condition": 2,
            "order_hour": 14,
            "order_day_of_week": 3,
            "is_weekend": 0,
            "festival": "No",
            "order_date": "2022-04-01"
        }
        res = execute_eta_inference(sample)
        assert res["predicted_eta_min"] > 0
        assert res["distance_km"] > 0
        assert "dynamic_sla" in res
        assert "fleet_recommendation" in res
        assert not res["is_fallback"]

    def test_execute_late_risk_inference(self):
        sample = {
            "distance_km": 6.5,
            "road_traffic_density": "High",
            "weather": "Fog",
            "city": "Metropolitian",
            "delivery_person_ratings": 4.2,
            "vehicle_condition": 0,
            "delivery_person_age": 30,
            "type_of_vehicle": "motorcycle",
            "order_hour": 20,
            "order_day_of_week": 4,
            "is_weekend": 0,
            "type_of_order": "Meal",
            "festival": "No",
            "order_date": "2022-04-01"
        }
        res = execute_late_risk_inference(sample)
        assert 0.0 <= res["late_probability"] <= 1.0
        assert res["threshold_used"] == 0.59
        assert res["risk_category"] in ["LOW", "MODERATE", "HIGH", "CRITICAL"]
        assert "mitigation_plan" in res
        assert not res["is_fallback"]


class TestStreamlitViewModules:
    """Verifies that all 8 view modules load and define render_page function."""

    def test_all_views_callable(self):
        from app.views import (
            page_executive,
            page_data_explorer,
            page_eta_prediction,
            page_late_risk,
            page_demand_zones,
            page_rider_segmentation,
            page_model_performance,
            page_business_recommendations
        )
        views = [
            page_executive,
            page_data_explorer,
            page_eta_prediction,
            page_late_risk,
            page_demand_zones,
            page_rider_segmentation,
            page_model_performance,
            page_business_recommendations
        ]
        for v in views:
            assert hasattr(v, "render_page")
            assert callable(v.render_page)

    def test_threshold_application_and_calibration(self):
        # Test low friction order
        low_risk_sample = {
            "distance_km": 2.0,
            "road_traffic_density": "Low",
            "weather": "Sunny",
            "city": "Metropolitian",
            "delivery_person_ratings": 4.9,
            "vehicle_condition": 2,
            "delivery_person_age": 25,
            "type_of_vehicle": "scooter",
            "order_hour": 11,
            "order_day_of_week": 2,
            "is_weekend": 0,
            "type_of_order": "Snack",
            "festival": "No",
            "order_date": "2022-04-01"
        }
        res_low = execute_late_risk_inference(low_risk_sample)
        assert res_low["threshold_used"] == 0.59
        assert res_low["late_probability"] < 0.59
        assert res_low["is_late_decision"] == 0

        # Test high friction adverse order
        high_risk_sample = {
            "distance_km": 18.0,
            "road_traffic_density": "Jam",
            "weather": "Fog",
            "city": "Metropolitian",
            "delivery_person_ratings": 4.0,
            "vehicle_condition": 0,
            "delivery_person_age": 35,
            "type_of_vehicle": "motorcycle",
            "order_hour": 19,
            "order_day_of_week": 4,
            "is_weekend": 0,
            "type_of_order": "Meal",
            "festival": "No",
            "order_date": "2022-04-01"
        }
        res_high = execute_late_risk_inference(high_risk_sample)
        assert res_high["threshold_used"] == 0.59
        assert res_high["late_probability"] >= 0.59
        assert res_high["is_late_decision"] == 1
        assert res_high["risk_category"] in ["HIGH", "CRITICAL"]

    def test_cluster_integrity_in_data_service(self):
        ds = load_dark_store_bundle()
        assert len(ds["centroids"]) == 5
        # Verify valid latitude/longitude ranges for all 5 centroids
        lat_min, lat_max = config.LATITUDE_BOUNDS
        lon_min, lon_max = config.LONGITUDE_BOUNDS
        for _, row in ds["centroids"].iterrows():
            assert lat_min <= row["hub_latitude"] <= lat_max
            assert lon_min <= row["hub_longitude"] <= lon_max

        rider = load_rider_segmentation_bundle()
        assert len(rider["segment_profiles"]) == 3
        tiers = rider["segment_profiles"]["efficiency_tier_name"].tolist()
        assert any("Tier 1" in t for t in tiers)
        assert any("Tier 2" in t for t in tiers)
        assert any("Tier 3" in t for t in tiers)

