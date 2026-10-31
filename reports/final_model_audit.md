# Phase 3 Final Model Consistency Audit & Production Verification Report

**Project:** Quick-Commerce Delivery Intelligence  
**Audit Scope:** Model Artifacts, Metric Consistency, Holdout Reconciliation, Prediction Timing, and Spatial Boundaries  
**Auditor:** Senior Machine Learning Engineer  
**Audit Date:** 2026-10-31  
**Status:** **PASSED — ALL METRICS RECONCILED & ARTIFACTS VERIFIED**  

---

## 1. Classification Consistency Audit & Holdout Reconciliation

### 1.1 Holdout Dataset Overview
The chronological holdout test set consists of **$8,470$ records** (orders placed on or after March 30, 2022). All models were evaluated on identical holdout records using preprocessing pipelines fitted strictly on training data ($N=29,775$).

### 1.2 Mathematical Reconciliation of the Holdout Late Rate
- **Actual Positives in Holdout ($y = 1$, Delivery Duration $> 30.0\text{ min}$):** **$2,432$ orders**
- **Actual Negatives in Holdout ($y = 0$, Delivery Duration $\le 30.0\text{ min}$):** **$6,038$ orders**
- **Total Holdout Orders:** $2,432 + 6,038 = \mathbf{8,470\text{ orders}}$
- **True Holdout Late Rate:**
  $$\text{Late Rate} = \frac{2,432}{8,470} = \mathbf{28.7131\%} \approx \mathbf{28.71\%}$$

> [!NOTE]
> **Audit Finding & Reconciliation of "30.59%":**
> In preliminary documentation in `reports/regression_report.md` (Table 2), a figure of $30.59\%$ was listed as an initial text entry for the holdout test set late rate. 
> 
> A mathematical check confirms that $30.59\%$ of $8,470$ equals $2,591$ orders, which contradicts the actual ground truth count of $2,432$ orders. The true, verified empirical late rate of the holdout test set is **$28.71\%$** ($2,432$ positive orders). 
> 
> `reports/regression_report.md` has been corrected to reflect the exact verified figure ($28.71\%$). `reports/classification_report.md` was already using the exact verified count of $2,432$ positive orders and baseline $28.71\%$.

---

### 1.3 XGBoost Classifier Confusion Matrix & Metric Verification
The production `XGBoost Classifier` was evaluated at the validation-calibrated decision probability threshold of **$T^* = 0.59$**:

```
                  ┌────────────────────────────────────────┐
                  │          ACTUAL GROUND TRUTH           │
                  │   On-Time (y=0)   │    Late (y=1)      │
┌───────────────┬─┴───────────────────┴────────────────────┤
│ PREDICTED     │                                          │
│ On-Time (ŷ=0) │  TN = 5,685 (67.1%) │  FN = 284  (3.4%)  │
│ Late    (ŷ=1) │  FP = 353   (4.2%)  │  TP = 2,148 (25.4%)│
└───────────────┴──────────────────────────────────────────┘
Total Test Records: 5,685 + 353 + 284 + 2,148 = 8,470
```

#### Detailed Element Verification:
1. **Total Actual Late Orders:** $\text{TP} + \text{FN} = 2,148 + 284 = \mathbf{2,432}$ ($28.71\%$)
2. **Total Actual On-Time Orders:** $\text{TN} + \text{FP} = 5,685 + 353 = \mathbf{6,038}$ ($71.29\%$)
3. **True Positives (TP):** **$2,148$** (Correctly flagged SLA breaches)
4. **True Negatives (TN):** **$5,685$** (Correctly identified on-time deliveries)
5. **False Positives (FP):** **$353$** (Predicted late, but arrived on time)
6. **False Negatives (FN):** **$284$** (Predicted on-time, but breached SLA)

#### Formulaic Verification of Performance Metrics:
- **Precision:**
  $$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}} = \frac{2,148}{2,148 + 353} = \frac{2,148}{2,501} = \mathbf{0.858856 \approx 0.8589\ (85.89\%)}$$
- **Recall:**
  $$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}} = \frac{2,148}{2,148 + 284} = \frac{2,148}{2,432} = \mathbf{0.883224 \approx 0.8832\ (88.32\%)}$$
- **F1-Score:**
  $$\text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} = 2 \times \frac{0.858856 \times 0.883224}{0.858856 + 0.883224} = \mathbf{0.87087 \approx 0.8709}$$
- **ROC-AUC Score:** **$0.9814$**
- **PR-AUC (Average Precision):** **$0.9614$**
- **Accuracy:** $\frac{5,685 + 2,148}{8,470} = \frac{7,833}{8,470} = \mathbf{0.92479 \approx 0.9248\ (92.48\%)}$

*Verification Confirmation:* All reported metrics exactly match the underlying model predictions evaluated at threshold $T = 0.59$ on the full holdout partition of $8,470$ records.

---

## 2. Operational Prediction Timing & Feature Availability

### 2.1 The Prediction Decision Point
A critical operational question for real-world deployment is: **When is the prediction generated?**

In quick-commerce fulfillment, an order undergoes four distinct operational stages:

```
[Stage 1: Checkout] ──► [Stage 2: Courier Dispatch] ──► [Stage 3: Kitchen Picked] ──► [Stage 4: Dropoff]
  Customer clicks "Pay"    Courier matched & assigned     Courier collects parcel       Doorstep handover
```

### 2.2 Feature Availability Matrix by Operational Stage

| Feature Name | Feature Category | Stage 1: Checkout (Order Placement) | Stage 2: Courier Dispatch (Matching) | Stage 3: Kitchen Picked (Pickup) | Stage 4: Dropoff (Outcome) | Production Usage Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| `order_date`, `order_hour` | Temporal | Available | Available | Available | Available | **Used in Model** |
| `order_day_of_week`, `is_weekend` | Temporal | Available | Available | Available | Available | **Used in Model** |
| `is_rush_hour` | Temporal | Available | Available | Available | Available | **Used in Model** |
| `restaurant_latitude`, `longitude` | Spatial | Available | Available | Available | Available | **Used in Model** |
| `delivery_location_lat`, `lon` | Spatial | Available | Available | Available | Available | **Used in Model** |
| `distance_km` | Spatial | Available | Available | Available | Available | **Used in Model** |
| `weather` | Environmental | Available | Available | Available | Available | **Used in Model** |
| `road_traffic_density` | Environmental | Available | Available | Available | Available | **Used in Model** |
| `traffic_density_ordinal` | Environmental | Available | Available | Available | Available | **Used in Model** |
| `type_of_order` | Basket Attribute | Available | Available | Available | Available | **Used in Model** |
| `festival`, `city` | Environmental/Context | Available | Available | Available | Available | **Used in Model** |
| `delivery_person_age` | Courier Attribute | **UNAVAILABLE** | **Available** | Available | Available | **Used in Model (Dispatch)** |
| `delivery_person_ratings` | Courier Attribute | **UNAVAILABLE** | **Available** | Available | Available | **Used in Model (Dispatch)** |
| `type_of_vehicle` | Courier Attribute | **UNAVAILABLE** | **Available** | Available | Available | **Used in Model (Dispatch)** |
| `vehicle_condition` | Courier Attribute | **UNAVAILABLE** | **Available** | Available | Available | **Used in Model (Dispatch)** |
| `Time_Order_picked` | Operational Latency | **UNAVAILABLE** | **UNAVAILABLE** | Available | Available | **STRICTLY QUARANTINED** |
| `prep_time_min` | Operational Latency | **UNAVAILABLE** | **UNAVAILABLE** | Available | Available | **STRICTLY QUARANTINED** |
| `multiple_deliveries` | Routing Outcome | **UNAVAILABLE** | **UNAVAILABLE** | **UNAVAILABLE** | Available | **STRICTLY QUARANTINED** |
| `delivery_time_min` | Outcome Target | **UNAVAILABLE** | **UNAVAILABLE** | **UNAVAILABLE** | Available | **TARGET ONLY** |
| `is_late` | Outcome Target | **UNAVAILABLE** | **UNAVAILABLE** | **UNAVAILABLE** | Available | **TARGET ONLY** |

### 2.3 Operational Conclusion on Prediction Timing
1. **Model Classification:** The current models are designated as **Courier Assignment / Dispatch-Time Models (Stage 2)**.
2. **Why Courier Attributes Are Valid at Dispatch:** When the dispatch engine matches an order to an available delivery partner, the courier's profile (`delivery_person_age`, `delivery_person_ratings`, `type_of_vehicle`, `vehicle_condition`) is fully known.
3. **Dispatch Actionability:** Predicting at Dispatch Time is highly valuable operationally because it allows the dispatch engine to evaluate whether the assigned courier's profile and route conditions create a high breach risk ($P > 0.59$). If risk is high, the system can trigger mitigation (e.g., re-assigning to a Tier 1 Express courier, suppressing order batching, or padding promised SLA).
4. **Checkout Fallback:** If predictions must be displayed to the customer at Stage 1 (Checkout), courier attributes are not yet assigned. The system must either:
   - Use fleet median imputation for courier attributes (e.g., median rating $4.7$, median age $29$, vehicle condition $1$), or
   - Maintain a dedicated Checkout-time sub-model that excludes courier attributes.

---

## 3. Serialized Model Artifact Validation

All 4 serialized model bundles in `models/` were loaded and inspected directly using `joblib`:

### 3.1 Best ETA Regressor: `models/best_eta_regressor.joblib`
- **File Size:** $1.91\text{ MB}$
- **Deserialization:** Successful.
- **Bundle Contents:**
  - `algorithm`: `'xgboost'`
  - `model`: Trained `XGBRegressor` (`n_estimators=250`, `max_depth=7`, `learning_rate=0.03`, `subsample=0.85`)
  - `preprocessor`: Fitted `ColumnTransformer` (9 numerical features median-imputed & scaled; 6 categorical features mode-imputed & one-hot encoded with `handle_unknown='ignore'`)
  - `feature_names`: Exactly 32 encoded dimensions matching input matrix schema.
  - `test_mae`: $3.317\text{ min}$, `test_rmse`: $4.150\text{ min}$, `test_r2`: $0.8013$
- **Inference Verification:** Executing `model.predict()` on transformed synthetic single-order inputs produces bounded, realistic continuous ETAs (e.g., $14.0\text{ min}$ under low traffic, $40.0\text{ min}$ under adverse fog/jam).

---

### 3.2 Best Late Classifier: `models/best_late_classifier.joblib`
- **File Size:** $643\text{ KB}$
- **Deserialization:** Successful.
- **Bundle Contents:**
  - `algorithm`: `'xgboost'`
  - `model`: Trained `XGBClassifier` (`n_estimators=200`, `max_depth=6`, `learning_rate=0.05`, `scale_pos_weight=2.38`)
  - `threshold`: **$0.59$** (calibrated validation cutoff)
  - `test_precision`: $0.8589$, `test_recall`: $0.8832$, `test_f1`: $0.8709$, `test_roc_auc`: $0.9814$, `test_pr_auc`: $0.9614$
- **Inference Verification:** Output probabilities strictly lie in $[0.0, 1.0]$. The calibrated threshold of $0.59$ correctly categorizes standard orders as on-time ($P=0.001$, flag $0$) and high-friction adverse orders as late-risk ($P=1.0$, flag $1$).

---

### 3.3 Dark Store Spatial K-Means: `models/dark_store_kmeans.joblib`
- **File Size:** $170\text{ KB}$
- **Deserialization:** Successful.
- **Bundle Contents:**
  - `n_clusters`: $5$
  - `kmeans_model`: Trained Scikit-Learn `KMeans` ($K=5$)
  - `centroids`: DataFrame ($5 \times 8$) containing centroid coordinates, demand volume, percentage share, average delivery duration, and SLA breach rate.
- **Inference Verification:** Calling `kmeans_model.predict()` on new latitude/longitude arrays correctly assigns cluster indices $0 - 4$. Coordinates outside India bounds or zero coordinates are successfully intercepted and flagged with cluster label $-1$.

---

### 3.4 Delivery Partner Behavioral Segmentation: `models/rider_segmentation_kmeans.joblib`
- **File Size:** $8.25\text{ KB}$
- **Deserialization:** Successful.
- **Bundle Contents:**
  - `kmeans_model`: Trained Scikit-Learn `KMeans` ($K=3$)
  - `scaler`: Fitted `StandardScaler` on 4 behavioral metrics:
    - Mean vector: $[26.35\text{m}, 4.64\text{ rating}, 29.99\%\text{ late}, 1.59\text{ daily vol}]$
    - Variance vector: $[3.75, 0.0046, 92.19, 0.0339]$
  - `segment_profiles`: DataFrame ($3 \times 7$) profiling Tier 1 (Express), Tier 2 (Standard Operations), and Tier 3 (At-Risk / Needs Coaching).
- **Inference Verification:** Partner feature vectors transformed via `scaler` cluster deterministically into their respective performance tiers.

---

## 4. Clustering Scope & Boundary Reporting

### 4.1 Geographic Demand Clusters: Macro Regions vs Micro Fulfillment
- **Valid GPS Points Clustered:** $41,953$ orders.
- **Excluded GPS Points:** $3,640$ orders ($7.98\%$) with $(0.0, 0.0)$ or invalid boundary coordinates.
- **Cluster Centroids & Regional Scope:**
  - **Zone 0 (East Hub):** Lat $23.59^\circ$, Lon $85.43^\circ$ — Eastern India (Kolkata / Ranchi corridor)
  - **Zone 1 (West Hub):** Lat $20.64^\circ$, Lon $73.89^\circ$ — Western India (Mumbai / Pune / Surat corridor)
  - **Zone 2 (Central-South Hub):** Lat $17.19^\circ$, Lon $77.73^\circ$ — Central-South India (Hyderabad / Telangana)
  - **Zone 3 (North Hub):** Lat $27.37^\circ$, Lon $76.84^\circ$ — Northern India (Delhi-NCR / Jaipur corridor)
  - **Zone 4 (South Hub):** Lat $12.27^\circ$, Lon $77.86^\circ$ — Southern India (Bangalore / Mysore / Chennai)

> [!WARNING]
> **Macro Regional Scope vs Micro Fulfillment:**
> The clusters computed here represent nationwide territorial clusters across major Indian metropolitan centers. They must **not** be presented as micro dark-store coordinates within an individual neighborhood. Intra-city dark-store site selection requires street-level network graph routing and neighborhood-level bounding, which should be clearly highlighted in the dashboard.

---

### 4.2 Courier Behavioral Segments
$1,320$ active couriers ($\ge 3$ deliveries) partition into three distinct behavioral tiers:
1. **Tier 1: High Efficiency / Express ($30.0\%$ fleet share, $396$ couriers):**
   - Mean ETA: $25.03\text{ min}$, Mean Rating: $4.68$, Late Rate: $22.72\%$.
2. **Tier 2: Standard Operations ($41.7\%$ fleet share, $550$ couriers):**
   - Mean ETA: $25.91\text{ min}$, Mean Rating: $4.64$, Late Rate: $28.10\%$, Volume: $41.98$ total orders.
3. **Tier 3: At-Risk / Needs Coaching ($28.3\%$ fleet share, $374$ couriers):**
   - Mean ETA: $28.38\text{ min}$, Mean Rating: $4.60$, Late Rate: $40.47\%$.

---

## 5. Explainability Verification & Non-Causal Interpretation

### 5.1 Observational vs Causal Associations
All model interpretations and SHAP summaries must adhere to non-causal language:

| Observed Feature Pattern | Improper Causal Phrasing (Avoid) | Proper Observational Phrasing (Approved) |
| :--- | :--- | :--- |
| `delivery_person_ratings < 4.5` | "Low ratings cause delivery delays." | "Low courier ratings **are strongly associated with** higher predicted delivery duration and elevated late risk." *(Both may be confounded by unfamiliarity with high-density delivery zones or difficult parking).* |
| `road_traffic_density == 'Jam'` | "Traffic jams make the courier late." | "Jam traffic conditions **correlate with** an average $+12\text{ to }+15\text{ minute}$ increase in predicted transit time." |
| `weather_Fog` | "Fog causes longer delivery times." | "Dense fog **is observed to accompany** significant speed reductions in historical transit records." |
| `festival == 'Yes'` | "Festivals cause orders to take 40+ minutes." | "In this historical Kaggle dataset, the festival indicator **empirically flags a specific subset** of orders where delivery times exceeded 38 minutes." |
| `vehicle_condition == 0` | "Poor vehicles cause breakdowns." | "Vehicle condition 0 **exhibits a negative predictive correlation** with on-time SLA achievement." |

---

## 6. Audit Summary & Readiness for Streamlit Integration

### Audit Verdict: **PASSED (Production-Ready)**
1. **Consistency Reconciled:** The holdout late rate is verified at **$28.71\%$** ($2,432$ positive orders out of $8,470$ total). All classification confusion matrix elements sum exactly to $8,470$.
2. **Prediction Timing Defined:** Models are designated as **Courier Assignment / Dispatch-Time Models** with clear operational rationale and feature availability documentation.
3. **Model Artifacts Validated:** All 4 `.joblib` bundles load cleanly, run inference successfully, and maintain consistent preprocessing schemas.
4. **Test Suite Status:** 18 automated unit tests pass in under 3 seconds.
5. **No Model Retraining Needed:** All model weights, hyperparameter selections, and calibrated cutoffs are verified to be mathematically sound and free from defects.

The project is cleared to begin **Streamlit Dashboard Integration** (`app/streamlit_app.py`).

