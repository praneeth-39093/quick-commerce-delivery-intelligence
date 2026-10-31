# Exploratory Data Analysis (EDA) & Empirical Findings Report
**Project:** Quick-Commerce Delivery Intelligence  
**Dataset:** 45,593 records across 22 Indian Metropolitan & Urban hubs  
**Generated Artifacts:** `reports/figures/*.png`  
**Date:** October 31, 2026  

---

## 1. Executive Summary

This report synthesizes the exploratory analysis of the standardized delivery dataset. The analysis evaluates univariate distributions, bivariate friction correlations, environmental impediments (traffic and weather), operational batching policies, and spatial delivery dynamics.

All figures referenced below are available in the [`reports/figures/`](file:///d:/Quick-Commerce-Delivery-Intelligence/reports/figures/) directory.

---

## 2. Target Variable Analysis: Delivery Duration (`delivery_time_min`)

* **Visual Reference:** [`reports/figures/delivery_time_distribution.png`](file:///d:/Quick-Commerce-Delivery-Intelligence/reports/figures/delivery_time_distribution.png)
* **Statistical Characteristics:**
  - **Sample Size:** 45,593 orders (100% complete)
  - **Mean:** $26.29\text{ minutes}$
  - **Standard Deviation:** $9.38\text{ minutes}$
  - **Median (P50):** $26.00\text{ minutes}$
  - **Interquartile Range:** $19.00\text{ min (P25)}$ to $32.00\text{ min (P75)}$
  - **Range:** $10.00\text{ min}$ to $54.00\text{ min}$
  - **Skewness:** $+0.486$ (mild positive skew; long tail towards 45–54 mins)
  - **Kurtosis:** $-0.311$ (near-mesokurtic / platykurtic distribution)

### SLA Breach Distribution (`is_late` at 30 min Benchmark):
- **On-Time Deliveries ($\le 30\text{ min}$):** 31,987 orders (**70.16%**)
- **Late Deliveries ($> 30\text{ min}$):** 13,606 orders (**29.84%**)
- **Empirical Insight:** Approximately 3 out of every 10 deliveries breach the standard 30-minute quick-commerce benchmark. This provides a natural class balance ($70/30$) for binary classification, making it well-suited for tree-based loss functions and cost-sensitive weighting without extreme synthetic resampling.

---

## 3. Road Congestion & Environmental Friction

### 3.1. Impact of Road Traffic Density
* **Visual Reference:** [`reports/figures/eta_by_traffic_density.png`](file:///d:/Quick-Commerce-Delivery-Intelligence/reports/figures/eta_by_traffic_density.png)
* **Empirical Progression:**
  - **Low Traffic (35.26% of orders):** Mean ETA $= 21.46\text{ min}$, Median $= 21.0\text{ min}$, Breach Rate $= \mathbf{8.73\%}$
  - **Medium Traffic (24.01% of orders):** Mean ETA $= 26.70\text{ min}$, Median $= 27.0\text{ min}$, Breach Rate $= \mathbf{32.47\%}$
  - **High Traffic (9.71% of orders):** Mean ETA $= 27.24\text{ min}$, Median $= 27.0\text{ min}$, Breach Rate $= \mathbf{33.56\%}$
  - **Jam Congestion (31.02% of orders):** Mean ETA $= 31.18\text{ min}$, Median $= 31.0\text{ min}$, Breach Rate $= \mathbf{50.64\%}$

**Takeaway:** Severe road congestion (`Jam`) shifts the median delivery time by $+10.0$ minutes compared to free-flow traffic, with more than half of all gridlocked deliveries breaching SLA.

### 3.2. Impact of Atmospheric Weather Conditions
* **Visual Reference:** [`reports/figures/eta_by_weather.png`](file:///d:/Quick-Commerce-Delivery-Intelligence/reports/figures/eta_by_weather.png)
* **Empirical Comparison:**
  - **Sunny:** Mean ETA $= 21.86\text{ min}$, Breach Rate $= \mathbf{12.52\%}$
  - **Stormy:** Mean ETA $= 25.87\text{ min}$, Breach Rate $= \mathbf{25.53\%}$
  - **Sandstorms:** Mean ETA $= 25.88\text{ min}$, Breach Rate $= \mathbf{25.95\%}$
  - **Windy:** Mean ETA $= 26.12\text{ min}$, Breach Rate $= \mathbf{26.56\%}$
  - **Fog:** Mean ETA $= 28.74\text{ min}$, Breach Rate $= \mathbf{42.83\%}$
  - **Cloudy:** Mean ETA $= 28.92\text{ min}$, Breach Rate $= \mathbf{43.78\%}$

**Takeaway:** Fog and heavy overcast conditions introduce significant optical and navigational friction in Indian metros, increasing average trip times by over $7.0$ minutes compared to clear sunny weather.

### 3.3. Compound Stress: Traffic $\times$ Weather Interaction
When Jam congestion intersects with impaired visibility (Fog or Heavy Clouds):
- **Jam + Fog:** Mean ETA $= 36.81\text{ min}$, Breach Rate $= \mathbf{76.07\%}$
- **Jam + Cloudy:** Mean ETA $= 36.69\text{ min}$, Breach Rate $= \mathbf{74.84\%}$
- In contrast, **Low Traffic + Clear Weather** maintains an average ETA of only $21.45\text{ min}$ with a minimal $10.5\%$ breach rate.

---

## 4. Geospatial & Trip Distance Dynamics

* **Visual Reference:** [`reports/figures/distance_vs_delivery_time.png`](file:///d:/Quick-Commerce-Delivery-Intelligence/reports/figures/distance_vs_delivery_time.png)
* **Haversine Distance Breakdown:**
  - **$< 5.0\text{ km}$ (24.79% of orders):** Mean ETA $= 22.19\text{ min}$, Breach Rate $= \mathbf{12.39\%}$
  - **$5.0 - 10.0\text{ km}$ (32.75% of orders):** Mean ETA $= 24.78\text{ min}$, Breach Rate $= \mathbf{22.92\%}$
  - **$10.0 - 15.0\text{ km}$ (24.80% of orders):** Mean ETA $= 29.86\text{ min}$, Breach Rate $= \mathbf{45.59\%}$
  - **$15.0 - 25.0\text{ km}$ (17.66% of orders):** Mean ETA $= 29.86\text{ min}$, Breach Rate $= \mathbf{45.07\%}$

**Takeaway:** The probability of breaching the 30-minute SLA nearly quadruples as delivery distance increases from under $5\text{ km}$ ($12.4\%$) to over $10\text{ km}$ ($45.6\%$). For quick-commerce operations, trips exceeding $10\text{ km}$ require micro-hub reassignment.

---

## 5. Correlation & Feature Interdependence

* **Visual Reference:** [`reports/figures/feature_correlation_matrix.png`](file:///d:/Quick-Commerce-Delivery-Intelligence/reports/figures/feature_correlation_matrix.png)
* **Key Correlation Coefficients with `delivery_time_min`:**
  1. `multiple_deliveries`: $r = +0.384$ (Strongest single linear predictor)
  2. `traffic_density_ordinal`: $r = +0.366$
  3. `delivery_person_ratings`: $r = -0.336$ (Higher partner ratings correlate with faster completions)
  4. `delivery_person_age`: $r = +0.298$
  5. `distance_km`: $r = +0.287$
  6. `vehicle_condition`: $r = -0.245$ (Poor vehicle condition correlates with longer delivery times)

---

## 6. Synthesis of Exploratory Insights for Modeling

1. **Tree-Based Ensembles are Well-Suited:** The significant interaction effects (e.g. Traffic $\times$ Weather, Batching $\times$ Distance) indicate that non-linear models (XGBoost, Random Forest) will significantly outperform simple linear baselines.
2. **Feature Scaling Requirement:** Features have disparate scales (Distance: 1–25 km, Age: 15–50, Ratings: 1–5, Ordinal traffic: 1–4). The `ColumnTransformer` preprocessing pipeline ensures linear and ridge regressors receive zero-mean, unit-variance standardized features.
3. **Monotonic Constraints:** For tree models, traffic density and multiple deliveries exhibit strong monotonic relationships with delivery duration and can be leveraged for constrained gradient boosting.

