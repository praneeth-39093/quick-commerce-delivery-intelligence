# Streamlit Dashboard Integration & Technical Validation Report

**Project Title:** QUICK-COMMERCE DELIVERY INTELLIGENCE  
**Subtitle:** AI-Powered ETA Prediction, Late-Order Detection and Delivery Optimization  
**Phase:** Phase 4 — Full Stack Web Application & Interactive Dashboard  
**Date:** 2026-10-31  
**Status:** **PASSED & PRODUCTION-READY**  

---

## 1. Executive Summary & Design System

The Quick-Commerce Delivery Intelligence web application has been constructed and validated as a modular, 8-page interactive enterprise dashboard.

### Design System Highlights:
- **Palette:** Cyber-executive dark navy (`#0B0F19`, `#111827`, `#1E293B`) paired with cyan (`#00F2FE`), electric blue (`#3B82F6`), and purple (`#8B5CF6`) accent glowing headers.
- **KPI Cards:** Custom glassmorphic metric cards with color-coded top borders indicating status (cyan for volume, blue for duration, amber/red for late risk, green for top performance).
- **Interactive Visualizations:** Built on dark-themed Plotly charts with unified layout styling, custom tooltips, and interactive legends.
- **Geospatial Mapping:** Folium interactive dark-matter maps integrated via `streamlit-folium` displaying nationwide macro demand cluster centroids and volume-scaled radius bubbles.
- **Strict Anti-Fabrication Guarantee:** Zero mock data; every metric, chart, and model output is sourced directly from verified logs (`featured_delivery_data.csv`), serialized artifacts (`models/*.joblib`), and the audit ledger (`reports/model_comparison.csv`).

---

## 2. Page-by-Page Technical Implementation & Verification

### Page 1: Executive Dashboard ([`app/views/page_executive.py`](file:///D:/Quick-Commerce-Delivery-Intelligence/app/views/page_executive.py))
- **Verified Metrics:**
  - Total Orders: $45,593$
  - Average Delivery Time: $26.2\text{ min}$
  - Late-Order Percentage: $29.8\%$ (using configurable $30.0\text{ min}$ SLA)
  - Average Distance: $9.8\text{ km}$
  - Best Regressor: XGBoost Regressor ($\text{MAE} = 3.32\text{ min}$, $R^2 = 0.801$)
  - Best Classifier: XGBoost Classifier ($\text{F1} = 0.871$, $\text{Recall} = 88.3\%$)
- **Visualizations:**
  - Delivery duration histogram with dashed $30.0\text{ min}$ SLA threshold line.
  - Road traffic volume donut chart (Low, Medium, High, Jam).
  - Late-order rate by traffic density bar chart (highlighting Jam traffic at $50.6\%$ late rate).
  - Late-order rate by weather condition bar chart (highlighting Fog and Stormy conditions).

---

### Page 2: Data Explorer ([`app/views/page_data_explorer.py`](file:///D:/Quick-Commerce-Delivery-Intelligence/app/views/page_data_explorer.py))
- **Multi-Dimensional Filter Controls:**
  - Order date range picker (Feb 11 – Apr 06, 2022).
  - Weather condition multi-select (Sunny, Cloudy, Windy, Fog, Stormy, Sandstorms).
  - Road traffic density multi-select (Low, Medium, High, Jam).
  - City classification multi-select (Metropolitian, Urban, Semi-Urban).
- **Tabs:**
  - *Dataset Preview & Schema:* High-performance paginated table with summary statistics (`describe().T`).
  - *Data Quality & Missing Values:* Documentation of raw text stripping, zero-coordinate quarantine ($3,640$ unmapped records), and post-cleaning completeness check.
  - *Delivery Latency Analytics:* Interactive distance vs duration scatter plot colored by traffic congestion; hourly order volume and late rate dual-axis trend line.

---

### Page 3: ETA Prediction ([`app/views/page_eta_prediction.py`](file:///D:/Quick-Commerce-Delivery-Intelligence/app/views/page_eta_prediction.py))
- **Inference Pipeline:** Loads [`models/best_eta_regressor.joblib`](file:///D:/Quick-Commerce-Delivery-Intelligence/models/best_eta_regressor.joblib) strictly via `@st.cache_resource`.
- **Verified Decision Point:** Courier Assignment & Dispatch Time (Stage 2).
- **Features Handled:** 15 verified features (restaurant/destination coordinates, weather, traffic density, vehicle type, vehicle condition, courier rating, courier age, order hour, weekend flag, order type, festival).
- **Geodesic Calculation:** Dynamically calculates great-circle Haversine distance in km from store and destination GPS coordinates.
- **Output Display:**
  - Predicted delivery duration (e.g. $15.1\text{ min}$).
  - Live clock arrival time (e.g. "01:07 PM IST").
  - Dynamic SLA buffer breakdown: Base ETA + Weather Buffer + Traffic Buffer + Safety Margin.
  - Fleet dispatch recommendation: Recommended vehicle type and batching policy.

---

### Page 4: Late-Order Risk ([`app/views/page_late_risk.py`](file:///D:/Quick-Commerce-Delivery-Intelligence/app/views/page_late_risk.py))
- **Inference Pipeline:** Loads [`models/best_late_classifier.joblib`](file:///D:/Quick-Commerce-Delivery-Intelligence/models/best_late_classifier.joblib).
- **Calibrated Threshold:** Implements verified cutoff **$T^* = 0.59$** (calibrated on validation data to optimize positive class weighting).
- **Diagnostic Output:**
  - Late breach probability ($P(\text{Late})$) rendered via an interactive Plotly bullet gauge.
  - Risk categorization: Low ($<30\%$), Moderate ($30-59\%$), High ($59-80\%$), Critical ($>80\%$).
  - Decision banner: ON-TIME EXPECTED vs SLA BREACH ALERT.
  - Educational callout explaining why threshold $0.59$ eliminates false alarm fatigue while capturing $88.3\%$ of breaches.
  - Automated operational mitigation checklist based on risk severity.

---

### Page 5: Demand Zone Analysis ([`app/views/page_demand_zones.py`](file:///D:/Quick-Commerce-Delivery-Intelligence/app/views/page_demand_zones.py))
- **Inference Pipeline:** Loads [`models/dark_store_kmeans.joblib`](file:///D:/Quick-Commerce-Delivery-Intelligence/models/dark_store_kmeans.joblib).
- **Geographic Mapping:** Renders a Folium CartoDB Dark Matter map with custom centroid icons and semi-transparent volume-scaled radius bubbles for the 5 nationwide demand zones:
  - Eastern Hub (Kolkata/Ranchi): $3,832$ orders ($9.1\%$).
  - Western Hub (Mumbai/Pune/Gujarat): $14,782$ orders ($35.2\%$).
  - Central-South Hub (Hyderabad): $3,790$ orders ($9.0\%$).
  - Northern Hub (Delhi-NCR/Jaipur): $6,326$ orders ($15.1\%$).
  - Southern Hub (Bangalore/Chennai): $13,223$ orders ($31.5\%$).
- **Mandatory Disclaimer Displayed:** Clearly emphasizes that clusters represent **macro-level nationwide demand regions**, not micro-fulfillment dark stores within a single neighborhood.

---

### Page 6: Rider Segmentation ([`app/views/page_rider_segmentation.py`](file:///D:/Quick-Commerce-Delivery-Intelligence/app/views/page_rider_segmentation.py))
- **Inference Pipeline:** Loads [`models/rider_segmentation_kmeans.joblib`](file:///D:/Quick-Commerce-Delivery-Intelligence/models/rider_segmentation_kmeans.joblib).
- **Courier Profiles:** Displays all 3 behavioral tiers across $1,320$ active couriers:
  - *Tier 1 (High Efficiency / Express):* $396$ couriers ($30.0\%$), $25.0\text{ min}$ ETA, $4.68$ rating, $22.7\%$ late rate.
  - *Tier 2 (Standard Operations):* $550$ couriers ($41.7\%$), $25.9\text{ min}$ ETA, $4.64$ rating, $28.1\%$ late rate, $41.98$ avg total deliveries.
  - *Tier 3 (At-Risk / Needs Coaching):* $374$ couriers ($28.3\%$), $28.4\text{ min}$ ETA, $4.60$ rating, $40.5\%$ late rate.
- **Interactive Visualizations:** Late rate vs transit latency scatter plot by tier; fleet share donut chart; targeted dispatch policy and coaching recommendation cards.

---

### Page 7: Model Performance ([`app/views/page_model_performance.py`](file:///D:/Quick-Commerce-Delivery-Intelligence/app/views/page_model_performance.py))
- **Regression Tab:** Full test set comparison table (Mean Baseline, Ridge, Random Forest, XGBoost) and actual vs predicted residual plot.
- **Classification Tab:** Full test set comparison table (Logistic Regression, Random Forest, XGBoost) and XGBoost confusion matrix breakdown ($\text{TN}=5,685, \text{FP}=353, \text{FN}=284, \text{TP}=2,148$). Static and interactive ROC and PR curve images.
- **Explainability Tab:** Gini feature importance, Permutation feature importance, and SHAP beeswarm plots for regression and classification.
- **Clustering Tab:** Inertia elbow and silhouette score curves ($K=2$ to $K=8$).

---

### Page 8: Business Recommendations ([`app/views/page_business_recommendations.py`](file:///D:/Quick-Commerce-Delivery-Intelligence/app/views/page_business_recommendations.py))
- Displays exactly three prioritized, data-backed strategic recommendations:
  1. **Dynamic Weather & Traffic SLA Buffer System:** +8m buffer for Jam, +6m for Fog/Stormy, projected to cut perceived SLA breaches by $38.2\%$.
  2. **Courier Tier-Based Algorithmic Dispatch Routing:** Route high-risk ($P > 0.59$) orders exclusively to Tier 1 Express riders; restrict Tier 3 riders to short-radius off-peak orders with route coaching.
  3. **Strict Route Batching Quarantine During Adverse Conditions:** Completely ban multi-order batching during High/Jam traffic and Fog/Stormy weather to prevent the $100\%$ breach failure mode.

---

## 3. Technical Verification & Automated Testing

### Test Suite Execution
An expanded automated test suite covering configuration, cleaning, feature engineering, model roundtrip, data service caching, inference pipelines, and view modules was executed:

```
tests/test_pipeline.py::TestPipelineConfiguration::test_paths_relative PASSED
tests/test_pipeline.py::TestPipelineConfiguration::test_traffic_map PASSED
tests/test_pipeline.py::TestPipelineConfiguration::test_leakage_features_quarantined_in_config PASSED
tests/test_pipeline.py::TestDataCleaner::test_clean_transformations PASSED
tests/test_pipeline.py::TestGPSValidation::test_zero_coordinates_flagged_invalid PASSED
tests/test_pipeline.py::TestGPSValidation::test_out_of_boundary_negatives_rejected PASSED
tests/test_pipeline.py::TestGPSValidation::test_no_coordinate_median_imputation PASSED
tests/test_pipeline.py::TestFeatureEngineerAndLeakagePrevention::test_feature_transformations PASSED
tests/test_pipeline.py::TestFeatureEngineerAndLeakagePrevention::test_strict_leakage_prevention_in_feature_matrix PASSED
tests/test_pipeline.py::TestClusteringValidation::test_spatial_clustering_bypasses_invalid_coords PASSED
tests/test_pipeline.py::TestClusteringValidation::test_meaningful_partner_tier_ranking PASSED
tests/test_pipeline.py::TestDeliveryOptimizer::test_dynamic_eta PASSED
tests/test_pipeline.py::TestDeliveryOptimizer::test_fleet_recommendation PASSED
tests/test_pipeline.py::TestModelEvaluator::test_regression_metrics PASSED
tests/test_pipeline.py::TestInferencePipeline::test_single_order_inference_without_picked_time PASSED
tests/test_pipeline.py::TestModelArtifactRoundtrip::test_best_regressor_artifact_loads PASSED
tests/test_pipeline.py::TestModelArtifactRoundtrip::test_best_classifier_artifact_loads PASSED
tests/test_pipeline.py::TestModelArtifactRoundtrip::test_dark_store_and_rider_clustering_artifacts_load PASSED
tests/test_streamlit_views.py::TestStreamlitDataService::test_load_featured_dataset PASSED
tests/test_streamlit_views.py::TestStreamlitDataService::test_load_model_comparison_df PASSED
tests/test_streamlit_views.py::TestStreamlitDataService::test_load_all_model_bundles PASSED
tests/test_streamlit_views.py::TestStreamlitDataService::test_calculate_haversine_km PASSED
tests/test_streamlit_views.py::TestStreamlitDataService::test_execute_eta_inference PASSED
tests/test_streamlit_views.py::TestStreamlitDataService::test_execute_late_risk_inference PASSED
tests/test_streamlit_views.py::TestStreamlitViewModules::test_all_views_callable PASSED
============================= 25 passed in 4.71s ==============================
```

### Live Server Probing:
The Streamlit application was spun up and probed on port 8501 via `Invoke-WebRequest`:
```
HTTP Status: 200 (OK)
```
The server initialized tornado, loaded preprocessors and models into cache, rendered the custom styles, and served traffic cleanly.

---

## 4. How to Launch the Application

To launch and interact with the Streamlit dashboard:

```powershell
# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Run the Streamlit web application
streamlit run app/streamlit_app.py
```

The application will be accessible in your web browser at:
`http://localhost:8501`

