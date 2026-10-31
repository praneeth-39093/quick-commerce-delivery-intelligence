# Quick-Commerce Delivery Intelligence ⚡
### AI-Powered ETA Prediction, Late-Order Detection and Delivery Optimization

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Architecture-Modular%20Pipeline-orange.svg)]()
[![ML](https://img.shields.io/badge/ML-Scikit--Learn%20%7C%20XGBoost%20%7C%20SHAP-success.svg)](https://xgboost.readthedocs.io/)
[![Frontend](https://img.shields.io/badge/Frontend-Streamlit%201.66-red.svg)](https://streamlit.io/)
[![Tests](https://img.shields.io/badge/Tests-27%20Passed-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📌 Executive Summary & Problem Formulation

In quick-commerce fulfillment (10-to-30 minute delivery promises), delivery duration precision directly impacts customer retention, driver dispatch efficiency, and unit economics. Unpredictable traffic bottlenecks, sudden adverse weather, and non-optimal courier matching can cascade into widespread service-level agreement (SLA) breaches and high refund overhead.

This project delivers an enterprise-grade, end-to-end Machine Learning intelligence engine that:
1. **Accurately Predicts Continuous ETA (Regression):** Continuous estimation of delivery time from order placement to customer doorstep using XGBoost Regressor ($\text{MAE} = 3.32\text{ min}$, $R^2 = 0.8013$), reducing baseline error by $55.8\%$.
2. **Proactively Flags Late-Order Breaches (Classification):** Early probabilistic detection of orders poised to exceed a demonstration SLA threshold of $30.0\text{ minutes}$ using an XGBoost Classifier ($\text{F1} = 0.8709$, $\text{Recall} = 88.32\%$, $\text{ROC-AUC} = 0.9814$, $\text{PR-AUC} = 0.9614$) at a validation-calibrated decision threshold ($T^* = 0.59$).
3. **Identifies Regional Demand Zones (Spatial Clustering):** $K\text{-Means}$ spatial clustering ($K=5$, Silhouette Score $0.7180$) identifying macro demand centers across $41,953$ valid GPS deliveries across India.
4. **Segments Courier Fleet Performance (Behavioral Clustering):** $K\text{-Means}$ courier segmentation ($K=3$) profiling $1,320$ active couriers into Tier 1 (Express, $22.7\%$ late rate), Tier 2 (Standard Operations, $41.98$ avg deliveries), and Tier 3 (At-Risk / Needs Coaching, $40.5\%$ late rate).
5. **Interactive Cyber-Navy Enterprise Dashboard:** A multi-page Streamlit application providing real-time dispatch simulation, Folium map rendering, SHAP explainability, and strategic business interventions.

---

## 🏗️ Project Architecture & Repository Structure

```
Quick-Commerce-Delivery-Intelligence/
│
├── data/
│   ├── train.xlsx                      <- Raw source dataset (45,593 records, 20 columns)
│   ├── cleaned_delivery_data.csv       <- Standardized and sanitized records
│   └── featured_delivery_data.csv      <- Engineered features (Haversine distance, targets)
│
├── notebooks/
│   └── exploratory_analysis.ipynb      <- Exploratory Data Analysis & visual inspections
│
├── src/                                <- Modular Source Code
│   ├── __init__.py                     <- Package public interface
│   ├── data_loader.py                  <- Ingestion, schema validation & chronological split
│   ├── data_cleaning.py                <- Text normalization, GPS bounds validation
│   ├── feature_engineering.py          <- Haversine math, temporal & leakage-free features
│   ├── eda.py                          <- Automated visual EDA generation
│   ├── regression.py                   <- Continuous ETA regression pipelines
│   ├── classification.py               <- Imbalance-aware SLA breach classification
│   ├── clustering.py                   <- Spatial K-Means & courier behavioral segmentation
│   ├── model_evaluation.py             <- MAE, RMSE, R², ROC-AUC, PR-AUC, Confusion Matrix
│   ├── recommendations.py              <- Dynamic SLA buffering & fleet dispatch rules
│   └── prediction.py                   <- Unified inference service for single/batch orders
│
├── models/                             <- Lightweight Serialized Model Artifacts (< 3 MB total)
│   ├── best_eta_regressor.joblib       <- Fitted Preprocessor + Tuned XGBRegressor (1.82 MB)
│   ├── best_late_classifier.joblib     <- Fitted Preprocessor + Tuned XGBClassifier (628 KB)
│   ├── dark_store_kmeans.joblib        <- Spatial K-Means (K=5) + Centroid summaries (166 KB)
│   ├── rider_segmentation_kmeans.joblib<- Rider K-Means (K=3) + Scaler + Tier schemas (8.1 KB)
│   └── .gitkeep
│
├── reports/                            <- Empirical Reports & Visual Assets
│   ├── figures/                        <- High-resolution diagnostic figures & SHAP plots
│   ├── model_comparison.csv            <- Master benchmark ledger with verified metrics
│   ├── regression_report.md            <- Continuous ETA regression benchmark report
│   ├── classification_report.md        <- Late-order classification benchmark report
│   ├── clustering_report.md            <- Spatial demand zoning and courier segmentation report
│   ├── model_selection.md              <- Architecture selection rationale & trade-offs
│   ├── pre_training_audit.md           <- Technical code & pipeline audit
│   ├── data_validation_summary.md      <- Post-cleaning dataset validation summary
│   ├── final_model_audit.md            <- Reconciled metric consistency audit
│   ├── streamlit_validation_report.md  <- Streamlit dashboard integration validation report
│   └── final_prepublication_audit.md   <- Final pre-publication verification report
│
├── app/                                <- Streamlit Cyber-Navy Web Application
│   ├── __init__.py
│   ├── streamlit_app.py                <- Main entrypoint & sidebar router
│   ├── styles.py                       <- Custom Dark Navy theme, cyan/purple glowing accents
│   ├── data_service.py                 <- Cached data loading, inference & fallback services
│   └── views/                          <- 8 Dedicated Page Views
│       ├── page_executive.py           <- Page 1: Executive KPI Dashboard
│       ├── page_data_explorer.py       <- Page 2: Interactive Data Explorer
│       ├── page_eta_prediction.py      <- Page 3: Real-Time ETA Predictor
│       ├── page_late_risk.py           <- Page 4: Late-Order Risk Diagnostic
│       ├── page_demand_zones.py        <- Page 5: Dark Store Demand Zones (Folium Map)
│       ├── page_rider_segmentation.py  <- Page 6: Delivery Partner Behavioral Segmentation
│       ├── page_model_performance.py   <- Page 7: Model Benchmarks & SHAP Explainability
│       └── page_business_recommendations.py <- Page 8: Strategic Interventions
│
├── tests/                              <- Automated Test Suite (27 Unit Tests)
│   ├── test_pipeline.py                <- Data processing, leakage isolation & model tests
│   └── test_streamlit_views.py         <- Streamlit data services, inference & view tests
│
├── scripts/                            <- Automation & Verification Utilities
│   ├── train_and_evaluate.py           <- Training, hyperparameter tuning & SHAP generation
│   ├── generate_evaluation_curves.py   <- Residual, ROC, PR, and permutation importance curves
│   └── validate_streamlit_app.py       <- Streamlit smoke-testing script
│
├── requirements.txt                    <- Clean, unpinned dependency specification
├── config.py                           <- Centralized dynamic paths & parameters
├── main.py                             <- Unified CLI entrypoint
├── README.md                           <- Comprehensive technical documentation
├── .gitignore                          <- Version control exclusions
└── LICENSE                             <- MIT Open Source License
```

---

## 🛡️ Leakage Prevention & Prediction Timing

### Operational Lifecycle & Decision Point
In quick-commerce logistics, an order transitions through four stages:
1. **Stage 1 (Checkout):** Customer confirms cart and taps "Pay".
2. **Stage 2 (Courier Dispatch):** Dispatch algorithm matches order to an active delivery partner. **(Model Decision Point)**
3. **Stage 3 (Kitchen Picked):** Courier collects parcel from store (`Time_Order_picked`, `prep_time_min`).
4. **Stage 4 (Customer Dropoff):** Parcel delivered (`delivery_time_min`, `is_late`).

### Feature Availability Table

| Feature Name | Category | Stage 1 (Checkout) | Stage 2 (Dispatch) | Stage 3 (Picked) | Production Pipeline Status |
| :--- | :--- | :---: | :---: | :---: | :--- |
| `restaurant_latitude`, `longitude` | Spatial | Available | Available | Available | **Active Feature** |
| `delivery_location_lat`, `lon` | Spatial | Available | Available | Available | **Active Feature** |
| `distance_km` (Haversine) | Spatial | Available | Available | Available | **Active Feature** |
| `weather`, `road_traffic_density` | Environmental | Available | Available | Available | **Active Feature** |
| `order_hour`, `order_day_of_week` | Temporal | Available | Available | Available | **Active Feature** |
| `is_weekend`, `is_rush_hour` | Temporal | Available | Available | Available | **Active Feature** |
| `type_of_order`, `festival`, `city` | Context | Available | Available | Available | **Active Feature** |
| `delivery_person_age` | Courier Profile | Unavailable | **Available** | Available | **Active Feature (Dispatch)** |
| `delivery_person_ratings` | Courier Profile | Unavailable | **Available** | Available | **Active Feature (Dispatch)** |
| `type_of_vehicle`, `vehicle_condition`| Courier Profile | Unavailable | **Available** | Available | **Active Feature (Dispatch)** |
| `Time_Order_picked`, `prep_time_min` | Post-Order Latency | Unavailable | Unavailable | Available | **QUARANTINED (Leakage)** |
| `multiple_deliveries` | Route Outcome | Unavailable | Unavailable | Unavailable | **QUARANTINED (Leakage)** |
| `delivery_time_min`, `is_late` | Target Outcomes | Unavailable | Unavailable | Unavailable | **TARGET ONLY** |

*All models are strictly designated as **Stage 2: Courier Assignment / Dispatch-Time Prediction Models**. When invoked at Stage 1, courier features default to dark-store historical medians.*

---

## 📊 Verified Model Benchmark Results

All evaluations were executed on the untouched chronological holdout test set ($N=8,470$, orders placed on or after March 30, 2022) with preprocessing pipelines fitted strictly on training data ($N=29,775$).

### 1. Continuous ETA Regression Benchmark

| Model Architecture | Test MAE (min) | Test RMSE (min) | Test $R^2$ Score | Test MAPE (%) | Validation MAE | Validation RMSE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Mean Baseline** | $7.502$ | $9.312$ | $-0.0003$ | $34.34\%$ | $7.842\text{m}$ | $9.656\text{m}$ |
| **Ridge Regression** ($\alpha=50$) | $4.982$ | $6.280$ | $0.5451$ | $22.42\%$ | $5.059\text{m}$ | $6.340\text{m}$ |
| **Random Forest Regressor** | $3.330$ | $4.178$ | $0.7987$ | $14.59\%$ | $3.329\text{m}$ | $4.225\text{m}$ |
| **XGBoost Regressor** *(Selected)* | **$3.317$** | **$4.150$** | **$0.8013$** | **$14.56\%$** | **$3.291\text{m}$** | **$4.160\text{m}$** |

$$\Delta \text{MAE}_{\text{XGBoost vs Baseline}} = -4.185\text{ minutes (-55.8% error reduction)}$$

---

### 2. Late-Order SLA Breach Classification Benchmark ($\text{SLA} = 30.0\text{ min}$)

| Model Architecture | Optimal Cutoff ($T^*$) | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | $0.59$ | $0.8625$ | $0.7432$ | $0.7961$ | $0.7687$ | $0.9288$ | $0.8590$ |
| **Random Forest Classifier** | $0.57$ | $0.9242$ | **$0.8708$** | $0.8643$ | $0.8675$ | $0.9791$ | $0.9577$ |
| **XGBoost Classifier** *(Selected)* | **$0.59$** | **$0.9248$** | $0.8589$ | **$88.32\%$** | **$0.8709$** | **$0.9814$** | **$0.9614$** |

#### Reconciled Holdout Confusion Matrix ($N = 8,470$ at $T^* = 0.59$):
- **True Negatives (TN):** $5,685$ ($67.1\%$)
- **False Positives (FP):** $353$ ($4.2\%$)
- **False Negatives (FN):** $284$ ($3.4\%$) — *misses only 284 out of 2,432 actual late deliveries*
- **True Positives (TP):** $2,148$ ($25.4\%$) — *$88.32\%$ breach capture rate*

---

### 3. Spatial & Behavioral Clustering Results
- **Dark Store Demand Clusters ($K=5$):** Evaluated on $41,953$ valid GPS points ($3,640$ zero-coordinates quarantined). Peak Silhouette Score: **$0.7180$**.
  - Zone 1 (East Hub): $3,832$ orders ($9.1\%$), Lat $23.59^\circ$, Lon $85.43^\circ$
  - Zone 2 (West Hub): $14,782$ orders ($35.2\%$), Lat $20.64^\circ$, Lon $73.89^\circ$
  - Zone 3 (Central-South Hub): $3,790$ orders ($9.0\%$), Lat $17.19^\circ$, Lon $77.73^\circ$
  - Zone 4 (North Hub): $6,326$ orders ($15.1\%$), Lat $27.37^\circ$, Lon $76.84^\circ$
  - Zone 5 (South Hub): $13,223$ orders ($31.5\%$), Lat $12.27^\circ$, Lon $77.86^\circ$
- **Courier Behavioral Segments ($K=3$ on $1,320$ active couriers):**
  - **Tier 1 (Express, $30.0\%$):** $25.03\text{m}$ avg ETA, $4.68$ rating, $22.7\%$ late rate.
  - **Tier 2 (Standard Operations, $41.7\%$):** $25.91\text{m}$ avg ETA, $4.64$ rating, $28.1\%$ late rate, $41.98$ avg deliveries.
  - **Tier 3 (At-Risk / Needs Coaching, $28.3\%$):** $28.38\text{m}$ avg ETA, $4.60$ rating, $40.5\%$ late rate ($+17.8\%$ gap vs Tier 1).

---

## 💡 Strategic Business Recommendations

1. **Dynamic Weather & Traffic SLA Buffer System:** Replace rigid static 30m SLAs with dynamic buffers (+8m for Jam traffic, +6m for Fog/Stormy weather, +3m safety margin) displayed at checkout. Projected to reduce perceived SLA breaches by **$38.2\%$**, eliminating ~930 monthly complaints per 10k orders.
2. **Courier Tier-Based Algorithmic Dispatch Routing:** Automatically route high-risk orders ($P(\text{Late}) > 0.59$) exclusively to Tier 1 Express riders. Restrict Tier 3 riders to short-radius ($< 3\text{ km}$) off-peak batches with digital turn-by-turn route coaching.
3. **Strict Route Batching Quarantine During Adverse Conditions:** Completely ban secondary order stacking during High/Jam traffic or Fog/Stormy conditions. In empirical logs, orders with $\ge 2$ multi-deliveries suffered a **$100.00\%$ breach rate** ($2,346 / 2,346$ orders).

---

## ⚡ Quick Start & Installation

### 1. Prerequisites & Setup
Requires Python 3.10 to 3.12:
```powershell
# Clone the repository
git clone https://github.com/<your-username>/Quick-Commerce-Delivery-Intelligence.git
cd Quick-Commerce-Delivery-Intelligence

# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1  # Windows PowerShell
# source venv/bin/activate   # Linux / macOS

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Automated Test Suite (27 Tests)
```powershell
pytest -v
```

### 3. Launch the Interactive Streamlit Dashboard
```powershell
streamlit run app/streamlit_app.py
```
Open **`http://localhost:8501`** in any web browser to explore all 8 pages.

---

## 📄 Dataset Attribution & License
- **Dataset Attribution:** Built on the open-access Kaggle Delivery Time benchmark dataset (Indian urban delivery logistics). Strictly used for demonstration and research purposes.
- **Software License:** Licensed under the [MIT License](LICENSE).
