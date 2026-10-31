# Model Selection Report: Technical Rationale, Trade-Off Analysis & Production Architecture

**Project:** Quick-Commerce Delivery Intelligence  
**Phase:** Phase 3 – Model Comparison, Selection, and Production Readiness  
**Target Deployment:** Real-Time Order Dispatch & ETA Inference Engine  
**Generated Date:** 2026-10-31  

---

## 1. Master Model Comparison & Evaluation Ledger

The table below compiles evaluated test-set metrics and operational characteristics across all candidate architectures evaluated during Phase 3. All models were trained on the pre-March 23, 2022 split ($N=29,775$) and evaluated on the identical holdout test set ($N=8,470$, dates $\ge 2022-03-30$):

### Supervised Learning Models Benchmark (Chronological Test Set)

| Task Domain | Model Architecture | Core Algorithm | Primary Metric | Secondary Metric | Variance / Discrimination ($R^2$ or AUC) | Test Set Details | Training Latency | Serialized Artifact Size |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **ETA Regression** | Mean Baseline | `DummyRegressor` | $\text{MAE} = 7.502\text{m}$ | $\text{RMSE} = 9.312\text{m}$ | $R^2 = -0.0003$ | $\text{MAPE} = 34.34\%$ | $< 0.01\text{s}$ | $< 5\text{ KB}$ |
| **ETA Regression** | Ridge Regression | `Ridge` ($\alpha=50$) | $\text{MAE} = 4.982\text{m}$ | $\text{RMSE} = 6.280\text{m}$ | $R^2 = 0.5451$ | $\text{MAPE} = 22.42\%$ | $0.01\text{s}$ | $12\text{ KB}$ |
| **ETA Regression** | Random Forest Regressor | `RandomForestRegressor` | $\text{MAE} = 3.330\text{m}$ | $\text{RMSE} = 4.178\text{m}$ | $R^2 = 0.7987$ | $\text{MAPE} = 14.59\%$ | $0.76\text{s}$ | $34.2\text{ MB}$ |
| **ETA Regression** | **XGBoost Regressor** *(Selected)* | `XGBRegressor` | **$\text{MAE} = 3.317\text{m}$** | **$\text{RMSE} = 4.150\text{m}$** | **$R^2 = 0.8013$** | **$\text{MAPE} = 14.56\%$** | $1.43\text{s}$ | **$1.91\text{ MB}$** |
| **Late Classification** | Logistic Regression | `LogisticRegression` | $\text{F1} = 0.7687$ | $\text{Recall} = 79.61\%$ | $\text{ROC-AUC} = 0.9288$ | $\text{PR-AUC} = 0.8590$, $\text{Prec} = 74.32\%$ | $0.14\text{s}$ | $14\text{ KB}$ |
| **Late Classification** | Random Forest Classifier | `RandomForestClassifier` | $\text{F1} = 0.8675$ | $\text{Recall} = 86.43\%$ | $\text{ROC-AUC} = 0.9791$ | $\text{PR-AUC} = 0.9577$, $\text{Prec} = 87.08\%$ | $0.73\text{s}$ | $18.5\text{ MB}$ |
| **Late Classification** | **XGBoost Classifier** *(Selected)* | `XGBClassifier` | **$\text{F1} = 0.8709$** | **$\text{Recall} = 88.32\%$** | **$\text{ROC-AUC} = 0.9814$** | **$\text{PR-AUC} = 0.9614$**, $\text{Prec} = 85.89\%$ | $1.07\text{s}$ | **$643\text{ KB}$** |

---

## 2. Model Selection Rationale

### 2.1 Best ETA Regression Model: XGBoost Regressor
- **Winner:** `XGBoost Regressor` (`n_estimators=250`, `max_depth=7`, `learning_rate=0.03`, `subsample=0.85`).
- **Technical Justification:**
  1. **Top Predictive Accuracy:** Achieves the lowest error across all metrics on the chronological test set ($\text{MAE} = 3.317\text{ min}$, $\text{RMSE} = 4.150\text{ min}$, $R^2 = 0.8013$), delivering a $55.8\%$ error reduction over the mean baseline.
  2. **Superior Non-Linear Modeling:** Outperforms Ridge Regression by $+25.6$ percentage points in $R^2$ ($0.8013$ vs $0.5451$). Delivery times exhibit severe combinatorial interaction effects (e.g., severe weather compounding with peak traffic hours) that linear models cannot capture without manual higher-order polynomials.
  3. **Operational Footprint vs Random Forest:** While Random Forest achieves comparable test accuracy ($\text{MAE} = 3.330\text{ min}$, $R^2 = 0.7987$), the XGBoost artifact is **$18\times$ smaller** ($1.91\text{ MB}$ vs $34.2\text{ MB}$). In a production microservice processing thousands of checkout requests per second, tree traversal in memory-optimized XGBoost C++ runtimes delivers substantially lower latency and memory churn than traversing deep Scikit-Learn forest arrays.

---

### 2.2 Best Late-Order Classification Model: XGBoost Classifier
- **Winner:** `XGBoost Classifier` (`n_estimators=200`, `max_depth=6`, `learning_rate=0.05`, `scale_pos_weight=2.38`, `optimal_threshold=0.59`).
- **Technical Justification:**
  1. **Maximized SLA Breach Detection (Recall):** In quick commerce, a False Negative (customer promised on-time delivery who experiences an unexpected delay) causes direct customer churn and refund requests. XGBoost delivers the highest recall on the test set (**$88.32\%$** vs $86.43\%$ for Random Forest and $79.61\%$ for Logistic Regression), missing only $284$ out of $2,432$ late orders.
  2. **Exceptional Precision-Recall AUC:** Reaches **$\text{PR-AUC} = 0.9614$** and **$\text{ROC-AUC} = 0.9814$**, maintaining precision above $85\%$ even when detecting nearly $90\%$ of breaches.
  3. **Calibrated Thresholding:** The validation-tuned threshold ($T^* = 0.59$) successfully offsets the conservative bias of `scale_pos_weight`, striking an optimal operational balance between false alarm fatigue and breach risk mitigation.

---

### 2.3 Best Dark-Store Spatial Clustering Configuration
- **Winner:** `K-Means Spatial Clustering` with **$K = 5$ clusters** on strictly validated geographical coordinates ($N = 41,953$).
- **Technical Justification:**
  1. **Peak Silhouette Score:** $K=5$ yields the highest silhouette coefficient (**$0.7180$**), providing optimal cluster cohesion and separation across India's regional metropolitan clusters.
  2. **Geographical Meaningfulness:** The 5 centroids map cleanly to India's major commercial delivery hubs: Western Hub ($35.2\%$), Southern Hub ($31.5\%$), Northern Hub ($15.1\%$), Eastern Hub ($9.1\%$), and Central-South Hub ($9.0\%$).
  3. **Strict Zero-Coordinate Handling:** All $3,640$ unmapped $(0.0, 0.0)$ records were completely excluded rather than median-imputed, preventing severe centroid distortion.

---

### 2.4 Best Rider Segmentation Configuration
- **Winner:** `K-Means Behavioral Segmentation` with **$K = 3$ performance tiers** across active couriers ($\ge 3$ deliveries, $N = 1,320$).
- **Technical Justification:**
  1. **Actionable Operational Tiers:** Separates couriers into Tier 1: Express ($30.0\%$, late rate $22.7\%$), Tier 2: Standard ($41.7\%$, late rate $28.1\%$, volume leader), and Tier 3: At-Risk ($28.3\%$, late rate $40.5\%$).
  2. **Clear Dispatch Routing Utility:** Enables algorithmic dispatch routing where at-risk orders are routed to Tier 1 riders, while Tier 3 riders receive targeted coaching and lower-complexity delivery batches.

---

## 3. Production Deployment Architecture & Inference Pipeline

The selected models and preprocessors are packaged for production inference as illustrated below:

```
[Customer Checkout Event]
         │
         ▼
[Raw Order Payload: Age, Rating, Vehicle, Weather, Traffic, City, Lat/Lon]
         │
         ▼
[Feature Engineering: distance_km, order_hour, is_rush_hour, is_weekend, traffic_ordinal]
         │
         ▼ (Strict Leakage-Free Preprocessing Pipeline: models/best_eta_regressor.joblib)
         │  • Median Numerical Imputation & StandardScaler
         │  • Categorical Mode Imputation & OneHotEncoder (handle_unknown='ignore')
         │
         ├───► [XGBoost Regressor] ───► Predicted Continuous ETA (e.g., 24.6 min)
         │
         └───► [XGBoost Classifier] ──► Breach Probability (e.g., P = 0.72)
                                                 │
                                                 ▼ (Apply Calibrated Threshold T* = 0.59)
                                       Binary Alert: IS_LATE_RISK = TRUE
                                                 │
                                                 ▼
                                  [Dispatch Recommendation Engine]
                                  Route to Tier 1 Express Courier
```

---

## 4. Production Artifact Verification Summary

All selected models are saved in the project `models/` directory and verified through automated end-to-end unit tests:

| Artifact File | Size | Contents | Verification Status |
| :--- | :--- | :--- | :--- |
| [`models/best_eta_regressor.joblib`](file:///D:/Quick-Commerce-Delivery-Intelligence/models/best_eta_regressor.joblib) | $1.91\text{ MB}$ | Fitted Preprocessor + Tuned XGBRegressor + Column Schema | Tested & Verified |
| [`models/best_late_classifier.joblib`](file:///D:/Quick-Commerce-Delivery-Intelligence/models/best_late_classifier.joblib) | $643\text{ KB}$ | Fitted Preprocessor + Tuned XGBClassifier + Threshold $0.59$ | Tested & Verified |
| [`models/dark_store_kmeans.joblib`](file:///D:/Quick-Commerce-Delivery-Intelligence/models/dark_store_kmeans.joblib) | $170\text{ KB}$ | Spatial K-Means ($K=5$) + 5 Centroid Summaries | Tested & Verified |
| [`models/rider_segmentation_kmeans.joblib`](file:///D:/Quick-Commerce-Delivery-Intelligence/models/rider_segmentation_kmeans.joblib) | $8.25\text{ KB}$ | Rider K-Means ($K=3$) + Scaler + 3 Tier Profiles | Tested & Verified |

### Verification Passes:
- **Leakage Quarantine Check:** Verified that neither `Time_Order_picked`, `prep_time_min`, nor `multiple_deliveries` exist in the inference schemas.
- **Single-Record Inference:** Confirmed via `test_single_order_inference_without_picked_time` in `tests/test_pipeline.py`.
- **Deserialization Roundtrip:** All 4 joblib artifacts load cleanly without errors or warnings.

