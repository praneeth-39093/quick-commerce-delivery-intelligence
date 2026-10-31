# Technical Audit Report: Pre-Training Readiness & Pipeline Integrity
**Project:** Quick-Commerce Delivery Intelligence – AI-Powered ETA Prediction, Late-Order Detection, and Delivery Optimization  
**Role:** Senior Machine Learning Engineer  
**Date:** October 31, 2026  
**Status:** Certified Ready for Training (All Critical Checks Passed)

---

## 1. Executive Summary

A comprehensive architectural and algorithmic technical audit of the **Quick-Commerce Delivery Intelligence** project was conducted prior to model training. The audit evaluated data ingestion, sanitization, feature engineering, spatial clustering, cross-validation boundaries, and inference services across 12 project files.

All identified vulnerabilities—including **data leakage from post-pickup timestamps**, **flawed coordinate median imputation**, **unverified coordinate sign rectifications**, and **arbitrary cluster tiering**—have been remediated and verified through an expanded unit test suite (15/15 tests passing, 0 errors, 0 warnings).

The project is certified **READY FOR MODEL TRAINING**.

---

## 2. Issue Log & Severity Matrix

| ID | Domain | Issue Description | Original Severity | Status |
| :--- | :--- | :--- | :--- | :--- |
| **AUD-01** | **Data Leakage** | `prep_time_min` (derived from post-event `Time_Order_picked`) was present in `MODEL_CONFIG.numerical_features`. | **CRITICAL** | **RESOLVED** |
| **AUD-02** | **GPS Coordinates** | `Restaurant_latitude` and `Restaurant_longitude` were subject to blind `.abs()` calls without boundary or pairing verification. | **CRITICAL** | **RESOLVED** |
| **AUD-03** | **GPS Coordinates** | `_impute_missing_values` median-imputed invalid coordinates, injecting artificial central-India centroids into spatial clustering. | **CRITICAL** | **RESOLVED** |
| **AUD-04** | **GPS Coordinates** | 3,640 records with `(0.0, 0.0)` restaurant GPS and near-zero `(0.01 - 0.11)` delivery noise were not isolated. | **HIGH** | **RESOLVED** |
| **AUD-05** | **Clustering** | Delivery partner clustering assigned fixed tier labels (Tier 1 vs Tier 3) to arbitrary KMeans cluster indices. | **MEDIUM** | **RESOLVED** |
| **AUD-06** | **SLA Definition** | Default SLA was described as standard without explicit demonstration benchmark disclaimers. | **MEDIUM** | **RESOLVED** |
| **AUD-07** | **Inference Pipeline** | Single-order prediction heuristic relied on `prep_time_min` which is unknown at order placement. | **HIGH** | **RESOLVED** |

---

## 3. Deep Dive: Technical Findings & Corrections Made

### 3.1. GPS Coordinates & Boundary Sanitization
* **Audit Finding:**
  Inspection of `data/train.xlsx` revealed:
  - 431 instances of negative `Restaurant_latitude` (e.g., `-27.163303`) and 162 instances of negative `Restaurant_longitude` (e.g., `-73.804855`).
  - 3,640 records where `Restaurant_latitude == 0.0` and `Delivery_location_latitude` contained near-zero noise (`0.01` to `0.11`), representing failed GPS sensor locks.
  - Blindly calling `.abs()` would convert nonsensical negative coordinates into valid Indian coordinates.
  - Median-imputing coordinates resulted in placing 3,640 deliveries at the geographic median of India (~22.5°N, 77.8°E in Madhya Pradesh), severely distorting spatial clustering and creating fake dark store candidates.
* **Corrections Implemented in [`src/data_cleaning.py`](file:///d:/Quick-Commerce-Delivery-Intelligence/src/data_cleaning.py):**
  1. Negative coordinates are only rectified if their absolute values fall strictly within India boundaries (`LATITUDE_BOUNDS = (6.0, 38.0)`, `LONGITUDE_BOUNDS = (68.0, 98.0)`). Out-of-boundary values are set to `np.nan`.
  2. Records with `(0.0, 0.0)` or near-zero delivery dropoff noise (`< 6.0` latitude) are quarantined and assigned `np.nan`.
  3. Added an explicit boolean flag `is_valid_coordinates`.
  4. Coordinates are **strictly excluded** from median imputation in `_impute_missing_values`.
  5. In [`src/feature_engineering.py`](file:///d:/Quick-Commerce-Delivery-Intelligence/src/feature_engineering.py), Haversine distance is calculated only on valid coordinate pairs. Rows with missing GPS have their `distance_km` imputed via city-level median trip distance with a boolean audit column `distance_imputed = 1`.

### 3.2. Data Leakage Elimination
* **Audit Finding:**
  In a real quick-commerce production environment, customer ETA prediction and initial late-delivery risk scoring occur at **order placement**. At this timestamp, the order has not yet been prepared or picked up (`Time_Order_picked` does not exist).
  - `prep_time_min` was previously included in `MODEL_CONFIG.numerical_features`.
  - Using `prep_time_min` in training features allowed target information and post-placement latency to leak into predictive models.
* **Corrections Implemented in [`config.py`](file:///d:/Quick-Commerce-Delivery-Intelligence/config.py) and [`src/feature_engineering.py`](file:///d:/Quick-Commerce-Delivery-Intelligence/src/feature_engineering.py):**
  1. Removed `prep_time_min` from `MODEL_CONFIG.numerical_features`.
  2. Defined `LEAKAGE_COLUMNS_QUARANTINE = ["Time_Order_picked", "time_picked_min", "prep_time_min", "Time_taken(min)", "delivery_time_min", "is_late", "speed_kmh"]`.
  3. `FeatureEngineer.get_feature_matrices` explicitly purges quarantined features from `X`.
  4. Target separation enforced: `delivery_time_min` is solely used as regression target `y`; `is_late` is solely used as classification target `y`.

### 3.3. Configurable SLA Threshold & Contractual Disclaimer
* **Audit Finding:**
  The project previously used a hardcoded 30-minute threshold without documenting that this was an empirical demonstration benchmark.
* **Corrections Implemented in [`config.py`](file:///d:/Quick-Commerce-Delivery-Intelligence/config.py):**
  1. Documented `DEFAULT_SLA_MINUTES = 30.0` as an empirical demonstration benchmark / operational baseline adopted for late-delivery risk detection, rather than an official platform SLA.
  2. `FeatureEngineer` and `LateOrderClassifier` accept `sla_threshold_min` as a parameter to support city-level and tier-level customization.

### 3.4. Training Validation & Preprocessing Isolation
* **Audit Finding:**
  Ensuring preprocessing does not leak test set information into training folds.
* **Corrections Implemented:**
  1. `ETARegressor` and `LateOrderClassifier` assemble preprocessing (`StandardScaler` and `OneHotEncoder`) into an encapsulated `sklearn.pipeline.Pipeline`.
  2. During cross-validation, `cross_val_score` fits the preprocessor exclusively on training splits and transforms validation splits.
  3. Class imbalance adjustment in `LateOrderClassifier` dynamically computes `scale_pos_weight` based on training split class distributions.
  4. Train/test splitting in `DataLoader.split_data` supports deterministic random state control for fair model comparisons.

### 3.5. Spatial Clustering & Partner Profiling Integrity
* **Audit Finding:**
  - `DeliveryOptimizerClustering` previously clustered raw coordinates without excluding invalid or near-zero points.
  - `profile_delivery_partners` previously mapped arbitrary KMeans labels (`{0: "Tier 1", 1: "Tier 2", 2: "Tier 3"}`) without verifying whether cluster 0 was actually the highest performing.
* **Corrections Implemented in [`src/clustering.py`](file:///d:/Quick-Commerce-Delivery-Intelligence/src/clustering.py):**
  1. `fit_spatial_clusters` filters strictly for points within `LATITUDE_BOUNDS` and `LONGITUDE_BOUNDS`. Invalid rows are assigned cluster label `-1`.
  2. Candidate dark store centroids are computed solely from verified geographic points.
  3. In `profile_delivery_partners`, cluster statistics are computed and sorted by `avg_delivery_time` ascending and `avg_rating` descending. Tiers are assigned dynamically based on composite ranking, ensuring Tier 1 is always the highest-performing partner tier.

### 3.6. Order-Placement Inference Pipeline
* **Audit Finding:**
  In `src/prediction.py`, the heuristic fallback required `prep_time_min` as an input parameter.
* **Corrections Implemented in [`src/prediction.py`](file:///d:/Quick-Commerce-Delivery-Intelligence/src/prediction.py):**
  1. Heuristic ETA prediction calculates kitchen prep time based on order category (`type_of_order`), which is known at order placement.
  2. `predict_single` accepts order-placement payloads without requiring `Time_Order_picked` or `time_picked_min`.
  3. Model input matrices `X` are generated via `FeatureEngineer.get_feature_matrices`, guaranteeing identical preprocessing and feature sets between training and inference.

---

## 4. Automated Test Results

The unit test suite in [`tests/test_pipeline.py`](file:///d:/Quick-Commerce-Delivery-Intelligence/tests/test_pipeline.py) was executed via Pytest with the following results:

```text
============================= test session starts =============================
platform win32 -- Python 3.15.1, pytest-9.2.0, pluggy-1.7.0 -- .\venv\Scripts\python.exe
cachedir: .pytest_cache
rootdir: D:\Quick-Commerce-Delivery-Intelligence
collected 15 items

tests/test_pipeline.py::TestPipelineConfiguration::test_paths_relative PASSED             [  6%]
tests/test_pipeline.py::TestPipelineConfiguration::test_traffic_map PASSED                [ 13%]
tests/test_pipeline.py::TestPipelineConfiguration::test_leakage_features_quarantined_in_config PASSED [ 20%]
tests/test_pipeline.py::TestDataCleaner::test_clean_transformations PASSED                [ 26%]
tests/test_pipeline.py::TestGPSValidation::test_zero_coordinates_flagged_invalid PASSED   [ 33%]
tests/test_pipeline.py::TestGPSValidation::test_out_of_boundary_negatives_rejected PASSED [ 40%]
tests/test_pipeline.py::TestGPSValidation::test_no_coordinate_median_imputation PASSED   [ 46%]
tests/test_pipeline.py::TestFeatureEngineerAndLeakagePrevention::test_feature_transformations PASSED [ 53%]
tests/test_pipeline.py::TestFeatureEngineerAndLeakagePrevention::test_strict_leakage_prevention_in_feature_matrix PASSED [ 60%]
tests/test_pipeline.py::TestClusteringValidation::test_spatial_clustering_bypasses_invalid_coords PASSED [ 66%]
tests/test_pipeline.py::TestClusteringValidation::test_meaningful_partner_tier_ranking PASSED [ 73%]
tests/test_pipeline.py::TestDeliveryOptimizer::test_dynamic_eta PASSED                   [ 80%]
tests/test_pipeline.py::TestDeliveryOptimizer::test_fleet_recommendation PASSED          [ 86%]
tests/test_pipeline.py::TestModelEvaluator::test_regression_metrics PASSED               [ 93%]
tests/test_pipeline.py::TestInferencePipeline::test_single_order_inference_without_picked_time PASSED [100%]

============================= 15 passed in 3.50s ==============================
```

---

## 5. Remaining Limitations & Edge Cases

1. **GPS Sensor Noise in Historical Data:** Approximately 7.98% (3,640 records) of the historical dataset lacks valid restaurant/delivery coordinates. In the current implementation, these records have their distance imputed using city-level medians. For spatial clustering and dark store siting, these records are excluded.
2. **Order-Date Cardinality:** Orders in the dataset span a multi-week period during Feb–April 2022. Cross-validation should use K-Fold or Stratified K-Fold. If temporal drift is suspected across seasons, rolling-window validation may be introduced.
3. **Kitchen Prep Latency Variance:** In actual operations, kitchen prep latency varies by merchant load. The pipeline currently estimates prep times by order category (Snack, Meal, Buffet, Drinks). Integrating merchant-specific historical prep latency tables is recommended for future iterations.

---

## 6. Final Readiness Status

| Audit Check | Status | Verification Detail |
| :--- | :--- | :--- |
| **GPS Coordinates** | **PASSED** | Boundary verification enforced; zero-noise flagged; no median-imputing of coordinates. |
| **Data Leakage** | **PASSED** | `Time_Order_picked` and `prep_time_min` quarantined from predictive feature matrix. |
| **SLA Integrity** | **PASSED** | SLA documented as configurable demonstration benchmark. |
| **Training Validation**| **PASSED** | Scikit-learn Pipeline architecture guarantees strict isolation of preprocessing. |
| **Clustering Safety** | **PASSED** | Out-of-bounds coordinates excluded; partner efficiency tiers dynamically ranked. |
| **Prediction Parity** | **PASSED** | Single/batch inference uses identical feature schemas without post-outcome inputs. |
| **Unit Test Coverage** | **PASSED** | 15/15 unit and integration tests passing cleanly. |

### Certification:
The Quick-Commerce Delivery Intelligence codebase satisfies all architectural, statistical, and operational requirements. The project is officially certified **READY FOR MODEL TRAINING**.

