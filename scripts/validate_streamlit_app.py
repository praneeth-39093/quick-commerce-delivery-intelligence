"""
Validation and Smoke-Test Script for Streamlit Dashboard.
Verifies all 8 pages, model loading, inference functions, map generation,
and styling templates.
"""

import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import tempfile
import os
os.environ["MPLCONFIGDIR"] = tempfile.gettempdir()

def test_imports():
    print("[1/5] Testing View and Service Imports...")
    import app.styles
    import app.data_service
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
    print("  ✅ All 8 page view modules imported cleanly.")

def test_data_service():
    print("[2/5] Testing Data and Model Loading...")
    from app.data_service import (
        load_featured_dataset,
        load_model_comparison_df,
        load_regressor_bundle,
        load_classifier_bundle,
        load_dark_store_bundle,
        load_rider_segmentation_bundle
    )
    df = load_featured_dataset()
    assert len(df) == 45593, f"Expected 45,593 rows, got {len(df)}"
    print(f"  ✅ Featured dataset loaded: {len(df):,} rows.")

    comp = load_model_comparison_df()
    assert len(comp) >= 7, f"Expected at least 7 model comparison rows, got {len(comp)}"
    print(f"  ✅ Model comparison ledger loaded: {len(comp)} models.")

    reg = load_regressor_bundle()
    assert reg is not None and "model" in reg and reg["algorithm"] == "xgboost"
    print("  ✅ XGBoost Regressor bundle loaded.")

    clf = load_classifier_bundle()
    assert clf is not None and "model" in clf and clf["threshold"] == 0.59
    print("  ✅ XGBoost Classifier bundle loaded (Threshold = 0.59).")

    ds = load_dark_store_bundle()
    assert ds is not None and len(ds["centroids"]) == 5
    print("  ✅ Dark store clustering bundle loaded (5 centroids).")

    rider = load_rider_segmentation_bundle()
    assert rider is not None and len(rider["segment_profiles"]) == 3
    print("  ✅ Rider segmentation bundle loaded (3 tiers).")

def test_inference():
    print("[3/5] Testing Interactive Inference Functions...")
    from app.data_service import execute_eta_inference, execute_late_risk_inference
    
    sample_order = {
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

    eta_res = execute_eta_inference(sample_order)
    assert eta_res["predicted_eta_min"] > 0
    assert not eta_res["is_fallback"]
    print(f"  ✅ ETA Inference: {eta_res['predicted_eta_min']} mins (Model: {eta_res['model_name']})")

    risk_res = execute_late_risk_inference(sample_order)
    assert 0.0 <= risk_res["late_probability"] <= 1.0
    assert risk_res["threshold_used"] == 0.59
    assert not risk_res["is_fallback"]
    print(f"  ✅ Late-Order Risk Inference: P(Late) = {risk_res['late_probability_pct']}% (Risk: {risk_res['risk_category']})")

def test_geospatial_rendering():
    print("[4/5] Testing Geospatial Folium Map Construction...")
    import folium
    from app.data_service import load_dark_store_bundle
    bundle = load_dark_store_bundle()
    centroids_df = bundle["centroids"]
    
    m = folium.Map(location=[21.0, 78.5], zoom_start=5)
    for _, row in centroids_df.iterrows():
        folium.Marker(location=[row["hub_latitude"], row["hub_longitude"]]).add_to(m)
    html = m.get_root().render()
    assert len(html) > 1000
    print("  ✅ Folium Map successfully generated HTML representation.")

def test_figures_exist():
    print("[5/5] Testing Report Figure Assets...")
    import config
    required_figures = [
        "regression_actual_vs_predicted.png",
        "classification_roc_pr_curves.png",
        "shap_summary_regression.png",
        "shap_summary_classification.png",
        "permutation_importance_regression.png",
        "dark_store_elbow_silhouette.png",
        "dark_store_clusters.png",
        "rider_segments.png"
    ]
    for fig in required_figures:
        p = config.FIGURES_DIR / fig
        assert p.exists(), f"Figure missing: {p}"
    print(f"  ✅ All {len(required_figures)} required figure assets verified.")

if __name__ == "__main__":
    print("=" * 70)
    print("RUNNING STREAMLIT APPLICATION VALIDATION SUITE")
    print("=" * 70)
    test_imports()
    test_data_service()
    test_inference()
    test_geospatial_rendering()
    test_figures_exist()
    print("=" * 70)
    print("ALL STREAMLIT VALIDATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)

