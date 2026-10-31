# Data Validation & Pipeline Verification Summary Report
**Project:** Quick-Commerce Delivery Intelligence – AI-Powered ETA Prediction, Late-Order Detection, and Delivery Optimization  
**Auditor:** Senior Machine Learning Engineer  
**Date:** October 31, 2026  
**Artifacts Validated:**
- `data/cleaned_delivery_data.csv`
- `data/featured_delivery_data.csv`
- `reports/figures/`
- `reports/eda_findings.md`

---

## 1. Row and Column Dimension Verification

Both data files were verified on disk with exact dimensional parity:

| File Name | File Size | Verified Row Count | Verified Column Count | Schema Status |
| :--- | :--- | :--- | :--- | :--- |
| **`data/train.xlsx` (Raw Source)** | $6.20\text{ MB}$ | 45,593 records | 20 columns | Verified intact (unchanged) |
| **`data/cleaned_delivery_data.csv`** | $5.36\text{ MB}$ | 45,593 records | 21 columns | Verified standardized |
| **`data/featured_delivery_data.csv`** | $7.78\text{ MB}$ | 45,593 records | 31 columns | Verified augmented |

### Column Inventory:
* **Cleaned Columns (21):** `id`, `delivery_person_id`, `delivery_person_age`, `delivery_person_ratings`, `restaurant_latitude`, `restaurant_longitude`, `delivery_location_latitude`, `delivery_location_longitude`, `order_date`, `weather`, `road_traffic_density`, `vehicle_condition`, `type_of_order`, `type_of_vehicle`, `multiple_deliveries`, `festival`, `city`, `delivery_time_min`, `is_valid_coordinates`, `time_ordered_min`, `time_picked_min`.
* **Featured Additional Columns (+10):** `distance_km`, `distance_imputed`, `prep_time_min`, `order_hour`, `order_day_of_week`, `is_weekend`, `is_rush_hour`, `city_code`, `traffic_density_ordinal`, `is_late`.

---

## 2. Missing Value Audit After Cleaning

Every single column across the 45,593 rows was audited for missing values (`np.nan`, `null`, `""`):

| Feature Name | Missing Count (Cleaned) | Missing Count (Featured) | Handling Rationale |
| :--- | :--- | :--- | :--- |
| `id` / `delivery_person_id` | **0 (0.00%)** | **0 (0.00%)** | Fully populated identifiers. |
| `delivery_person_age` | **0 (0.00%)** | **0 (0.00%)** | Imputed via training median ($30.0\text{ years}$). |
| `delivery_person_ratings` | **0 (0.00%)** | **0 (0.00%)** | Imputed via training median ($4.70$). |
| `weather` | **0 (0.00%)** | **0 (0.00%)** | Normalized (mode-imputed: Fog). |
| `road_traffic_density` | **0 (0.00%)** | **0 (0.00%)** | Normalized (mode-imputed: Low). |
| `vehicle_condition` | **0 (0.00%)** | **0 (0.00%)** | Discrete ordinal scale ($0, 1, 2, 3$). |
| `type_of_order` / `vehicle` | **0 (0.00%)** | **0 (0.00%)** | Fully populated categories. |
| `multiple_deliveries` | **0 (0.00%)** | **0 (0.00%)** | Median-imputed ($1.0$). |
| `festival` / `city` | **0 (0.00%)** | **0 (0.00%)** | Mode-imputed (`No` / `Metropolitian`). |
| `delivery_time_min` (Target) | **0 (0.00%)** | **0 (0.00%)** | 100% extracted numeric targets. |
| `distance_km` | **N/A** | **0 (0.00%)** | 100% populated (imputed via city median for missing GPS). |
| `restaurant_latitude` | **3,640 (7.98%)** | **3,640 (7.98%)** | **Intentionally retained as NaN** (invalid 0.0 GPS). |
| `restaurant_longitude` | **3,640 (7.98%)** | **3,640 (7.98%)** | **Intentionally retained as NaN** (invalid 0.0 GPS). |
| `delivery_location_latitude` | **3,640 (7.98%)** | **3,640 (7.98%)** | **Intentionally retained as NaN** (sensor noise). |
| `delivery_location_longitude`| **3,640 (7.98%)** | **3,640 (7.98%)** | **Intentionally retained as NaN** (sensor noise). |

> **Architectural Integrity Check:** The only columns containing `NaN` are the GPS coordinates for the 3,640 records where telemetry failed. They are intentionally **not** median-imputed to prevent artificial centroid distortion in spatial clustering.

---

## 3. Continuous Target Variable Distribution (`delivery_time_min`)

* **Count:** 45,593 records (100% complete)
* **Mean:** $26.29\text{ minutes}$
* **Standard Deviation:** $9.38\text{ minutes}$
* **Minimum:** $10.00\text{ minutes}$
* **25th Percentile (P25):** $19.00\text{ minutes}$
* **Median (50th Percentile):** $26.00\text{ minutes}$
* **75th Percentile (P75):** $32.00\text{ minutes}$
* **Maximum:** $54.00\text{ minutes}$
* **Distribution Shape:**
  - **Skewness:** $+0.486$ (Mild positive skew, well-behaved for tree models and linear models with standard scaling).
  - **Kurtosis:** $-0.311$ (Near-normal distribution, free of extreme outlier artifacts).

---

## 4. GPS Coordinate Audit & Boundary Verification

The geospatial coordinate pairs were verified against Indian geographical boundaries (`6.0°N` to `38.0°N` latitude, `68.0°E` to `98.0°E` longitude):

* **Total Order Coordinates Evaluated:** 45,593
* **Valid GPS Coordinate Pairs:** **41,953 records (92.02%)**
  - All 41,953 pairs verified to fall strictly within India's terrestrial bounds.
  - 431 negative `Restaurant_latitude` and 162 negative `Restaurant_longitude` records verified as sign-inversion telemetry errors and properly rectified to positive values matching their corresponding delivery points.
* **Invalid GPS Coordinate Pairs:** **3,640 records (7.98%)**
  - Characterized by `Restaurant_latitude == 0.0, Restaurant_longitude == 0.0` and delivery dropoff coordinates containing sensor noise (`0.01` to `0.11` degrees).
  - Explicitly flagged as `is_valid_coordinates = False` and retained as `np.nan`.

---

## 5. Haversine Distance Calculation Verification (`distance_km`)

* **Calculation Engine:** Haversine great-circle geodesic formula ($R = 6,371.0\text{ km}$).
* **Valid Coordinate Distance ($N = 41,953$):**
  - **Mean Distance:** $9.72\text{ km}$
  - **Median Distance:** $9.19\text{ km}$
  - **Minimum Distance:** $1.47\text{ km}$
  - **Maximum Distance:** $20.97\text{ km}$
  - **Standard Deviation:** $5.59\text{ km}$
* **Full Dataset Distance ($N = 45,593$, with City-Median Imputation):**
  - **Overall Mean Distance:** $9.66\text{ km}$
  - **Overall Median Distance:** $9.31\text{ km}$
  - **Imputed Count:** 3,640 records ($7.98\%$, tracked via `distance_imputed = 1`).
* **Sanity Check:** All computed distances fall cleanly within the realistic urban delivery range ($1.47\text{ km} - 20.97\text{ km}$), confirming that no distorted cross-continental distances exist.

---

## 6. Engineered Feature Distributions

Distributions for all features in the predictive training pipeline were verified:

| Feature Name | Type | Mean | Std | Min | Median | Max | Notes / Sanity Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `delivery_person_age` | Numeric | 29.58 | 5.77 | 15.0 | 30.0 | 50.0 | Realistic rider demographic spread. |
| `delivery_person_ratings` | Numeric | 4.64 | 0.32 | 1.0 | 4.70 | 5.0 | Left-skewed; typical platform ratings. |
| `multiple_deliveries` | Numeric | 0.75 | 0.57 | 0.0 | 1.00 | 3.0 | 0 orders: 30.9%, 1 order: 63.9%, 2+: 5.2%. |
| `vehicle_condition` | Ordinal | 1.02 | 0.84 | 0 | 1.0 | 3 | Evenly spread across tiers 0, 1, 2. |
| `traffic_density_ordinal` | Ordinal | 2.36 | 1.25 | 1 | 2.0 | 4 | Low (1): 35.3%, Med (2): 24.0%, Jam (4): 31.0%. |
| `order_hour` | Numeric | 17.48 | 4.77 | 0 | 19.0 | 23 | Concentrated around lunch/dinner peaks. |
| `is_rush_hour` | Binary | 0.50 | 0.50 | 0 | 0.0 | 1 | Exactly 50.0% orders during lunch/dinner rushes. |
| `is_weekend` | Binary | 0.00 | 0.00 | 0 | 0.0 | 0 | Deterministic weekday dataset profile. |

---

## 7. Strict Data Leakage Audit

The feature matrix extraction logic ([`FeatureEngineer.get_feature_matrices`](file:///d:/Quick-Commerce-Delivery-Intelligence/src/feature_engineering.py)) was audited directly in Python memory:

* **Generated Predictive Matrix Dimensions:**
  - $X_{\text{regression}}$: **45,593 rows $\times$ 16 features**
  - $X_{\text{classification}}$: **45,593 rows $\times$ 16 features**
* **Active Predictive Features ($X$):**
  1. `delivery_person_age`
  2. `delivery_person_ratings`
  3. `distance_km`
  4. `order_hour`
  5. `order_day_of_week`
  6. `is_weekend`
  7. `is_rush_hour`
  8. `multiple_deliveries`
  9. `vehicle_condition`
  10. `traffic_density_ordinal`
  11. `weather`
  12. `road_traffic_density`
  13. `type_of_order`
  14. `type_of_vehicle`
  15. `festival`
  16. `city`
* **Leakage Verification:**
  - `prep_time_min` present in $X$? $\rightarrow$ **NO (Strictly Excluded)**
  - `time_picked_min` present in $X$? $\rightarrow$ **NO (Strictly Excluded)**
  - `Time_Order_picked` present in $X$? $\rightarrow$ **NO (Strictly Excluded)**
  - `Time_taken(min)` / `delivery_time_min` present in $X$? $\rightarrow$ **NO (Strictly Excluded)**
  - `is_late` present in $X$? $\rightarrow$ **NO (Strictly Excluded)**
  - `speed_kmh` present in $X$? $\rightarrow$ **NO (Strictly Excluded)**

> **Result:** The predictive feature matrix $X$ contains **0 post-outcome features** and **0 target-derived columns**. Every feature is known strictly at order placement.

---

## 8. Late-Order Classification Target Distribution (`is_late`)

Evaluated at the demonstration SLA benchmark threshold ($\text{Cutoff} = 30.0\text{ minutes}$):

* **Class 0 (On-Time Delivery, $\le 30.0\text{ min}$):** **31,987 orders (70.16%)**
* **Class 1 (Late Delivery Breach, $> 30.0\text{ min}$):** **13,606 orders (29.84%)**
* **Class Imbalance Ratio (Negative / Positive):** $31,987 / 13,606 = \mathbf{2.351}$
* **Modeling Implication:** The balanced $70/30$ ratio provides a healthy positive event frequency without severe rarity problems. In [`src/classification.py`](file:///d:/Quick-Commerce-Delivery-Intelligence/src/classification.py), `scale_pos_weight = 2.35` perfectly counterbalances this ratio during XGBoost training.

---

## 9. EDA Visualizations Inspection

All five figures generated in [`reports/figures/`](file:///d:/Quick-Commerce-Delivery-Intelligence/reports/figures/) were verified:

1. **`delivery_time_distribution.png`:** Confirms the unimodal, slightly right-skewed distribution of delivery durations centered at $26.0\text{ min}$ with vertical markers for mean ($26.3\text{ min}$) and the $30\text{ min}$ SLA threshold.
2. **`eta_by_traffic_density.png`:** Visualizes the clear step-function increase in delivery time as road congestion escalates: Low ($21.5\text{ min}$), Medium ($26.7\text{ min}$), High ($27.2\text{ min}$), Jam ($31.2\text{ min}$).
3. **`eta_by_weather.png`:** Clearly distinguishes clear weather ($21.9\text{ min}$) from impaired visibility conditions like Fog ($28.7\text{ min}$) and Cloudy weather ($28.9\text{ min}$).
4. **`distance_vs_delivery_time.png`:** Scatter regression plot demonstrating that trip time increases steadily with Haversine distance, exhibiting higher variance beyond $10\text{ km}$.
5. **`feature_correlation_matrix.png`:** Heatmap confirming that `multiple_deliveries` ($r = +0.384$) and `traffic_density_ordinal` ($r = +0.366$) have the strongest positive correlations with delivery duration.

---

## 10. Top Three Delivery-Delay Patterns (Empirical Data Analysis)

Based on cross-tabulation and multivariate analysis across all 45,593 records, the three primary drivers of delivery delays are:

### Pattern 1: Batching Saturation Threshold (`multiple_deliveries >= 2`)
* **Data Evidence:**
  - Orders with $0$ secondary deliveries: Mean ETA $= 22.88\text{ min}$, Late Breach Rate $= \mathbf{18.12\%}$.
  - Orders with $1$ secondary delivery: Mean ETA $= 26.72\text{ min}$, Late Breach Rate $= \mathbf{29.86\%}$.
  - Orders with $2$ secondary deliveries ($N = 1,985$): Mean ETA $= \mathbf{40.45\text{ min}}$, Late Breach Rate $= \mathbf{100.00\%}$.
  - Orders with $3$ secondary deliveries ($N = 361$): Mean ETA $= \mathbf{47.82\text{ min}}$, Late Breach Rate $= \mathbf{100.00\%}$.
* **Operational Insight:** Quick-commerce delivery partners cannot handle 2 or more batched orders simultaneously without deterministically failing the 30-minute SLA. The dispatch recommendation engine must enforce an immediate hard ceiling of maximum 1 batch delivery.

### Pattern 2: Severe Traffic Congestion Compound Penalty (`Jam` + Adverse Weather)
* **Data Evidence:**
  - Free-flow traffic (`Low`): Mean ETA $= 21.46\text{ min}$, Late Breach Rate $= \mathbf{8.73\%}$.
  - Gridlock traffic (`Jam`): Mean ETA $= 31.18\text{ min}$, Late Breach Rate $= \mathbf{50.64\%}$.
  - Gridlock compounded with Fog (`Jam + Fog`): Mean ETA $= \mathbf{36.81\text{ min}}$, Late Breach Rate $= \mathbf{76.07\%}$.
  - Gridlock compounded with Overcast (`Jam + Cloudy`): Mean ETA $= \mathbf{36.69\text{ min}}$, Late Breach Rate $= \mathbf{74.84\%}$.
* **Operational Insight:** Severe road congestion increases average delivery duration by ~10 minutes, and when combined with impaired atmospheric conditions, breaches reach three out of four orders. Dynamic SLA buffering must add $+10.0$ minutes for Jam conditions and $+6.0$ minutes for Fog.

### Pattern 3: Festival Operational Strain & Vehicle Maintenance Degradation
* **Data Evidence:**
  - Non-festival days: Mean ETA $= 25.91\text{ min}$, Late Breach Rate $= \mathbf{28.44\%}$.
  - Festival days (`festival == 'Yes'`, $N = 896$): Mean ETA $= \mathbf{45.52\text{ min}}$, Late Breach Rate $= \mathbf{100.00\%}$.
  - Poor vehicle condition (`vehicle_condition == 0`, $N = 15,009$): Mean ETA $= 30.07\text{ min}$, Late Breach Rate $= \mathbf{41.68\%}$ vs well-maintained vehicles (`vehicle_condition == 1 or 2`), which average $24.36\text{ min}$ with a $23.88\%$ late rate.
* **Operational Insight:** During holiday demand surges, platform operations experience acute dispatch and kitchen bottlenecks. Additionally, unmaintained delivery vehicles cause a $+5.7$ minute average transit penalty, indicating that rider vehicle audits directly influence operational speed.

---

## 11. Final Readiness Declaration

| Task Type | Pipeline / Module | Readiness Status | Verification Basis |
| :--- | :--- | :--- | :--- |
| **Supervised Learning (Regression)** | `src/regression.py` (`ETARegressor`) | **READY** | Continuous target `delivery_time_min` verified; 0 missing values; 16 clean leakage-free features. |
| **Supervised Learning (Classification)** | `src/classification.py` (`LateOrderClassifier`) | **READY** | Binary target `is_late` verified (70/30 split); class imbalance ratio (2.35) mapped to `scale_pos_weight`. |
| **Unsupervised Learning (Clustering)** | `src/clustering.py` (`DeliveryOptimizerClustering`)| **READY** | 41,953 verified GPS points isolated; 3,640 invalid records quarantined with cluster label -1; feature scaling and deterministic tier ranking verified. |

### Final Conclusion:
The dataset and processing pipeline have been empirically validated and are **OFFICIALLY READY FOR MODEL TRAINING**.

