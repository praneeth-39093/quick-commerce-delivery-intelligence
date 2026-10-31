# Unsupervised Learning Report: Spatial Demand Clustering & Courier Fleet Segmentation

**Project:** Quick-Commerce Delivery Intelligence  
**Tasks:**
1. Dark-Store Exploratory Demand Zone Spatial Clustering ($K\text{-Means}$)
2. Delivery Partner Behavioral Segmentation & Fleet Profiling ($K\text{-Means}$)  
**Generated Date:** 2026-10-31  

---

## 1. Executive Summary

Unsupervised learning in quick commerce serves two strategic operational functions:
1. **Network Infrastructure:** Identifying macro spatial order demand concentrations to understand geographic order distribution.
2. **Fleet Operations:** Segmenting delivery couriers based on multi-dimensional performance patterns (transit speed, customer ratings, SLA compliance, daily delivery throughput) to enable tiered dispatch and targeted coaching.

### Key Deliverables & Findings
- **Spatial Clustering:** 5 macro demand zones were identified across $41,953$ valid GPS orders. Crucially, these represent **broad regional demand centers across Indian metropolitan hubs**, not micro-fulfillment dark stores within an individual neighborhood.
- **Rider Segmentation:** $1,320$ active couriers ($\ge 3$ deliveries) were segmented into **3 distinct operational tiers**. Tier 1 ("High Efficiency / Express") achieves a $22.72\%$ late breach rate, whereas Tier 3 ("At-Risk / Needs Coaching") exhibits a $40.47\%$ late rate ($+17.75$ percentage points higher), despite handling comparable delivery volumes.

---

## 2. Part 1: Dark-Store Demand Zone Spatial Clustering

### 2.1 GPS Coordinate Cleansing & Boundary Validation
To prevent severe spatial distortion, coordinates were filtered using strict geographical boundaries for India ($6.0^\circ \le \text{lat} \le 37.0^\circ$, $68.0^\circ \le \text{lon} \le 98.0^\circ$):
- **Raw Records Ingested:** $45,593$
- **Invalid / Zero Coordinates Quarantined:** $3,640$ records ($7.98\%$) with $(0.0, 0.0)$ coordinates.
- **Valid Coordinates Clustered:** **$41,953$ records** ($92.02\%$).

Median imputation was intentionally forbidden for spatial clustering because imputing thousands of coordinates to a single geometric median creates an artificial high-density spike.

### 2.2 Optimal Cluster Selection: Elbow & Silhouette Analysis

![Elbow and Silhouette Curves](figures/dark_store_elbow_silhouette.png)

K-Means was evaluated across $K \in [2, 8]$:

| Number of Clusters ($K$) | Inertia (Sum of Squared Distances) | Silhouette Score | Regional Interpretation |
| :---: | :---: | :---: | :--- |
| $2$ | $1,614,820.4$ | $0.6841$ | Broad North-South partition |
| $3$ | $921,405.1$ | $0.6215$ | North, West, and South India |
| $4$ | $542,110.8$ | $0.6724$ | North, West, East, and South India |
| **$5$** | **$318,442.6$** | **$0.7180$** | **Optimal balance: 5 Major Regional Metropolitan Hubs** |
| $6$ | $239,810.2$ | $0.6429$ | Sub-cluster fragmentation |
| $7$ | $198,340.5$ | $0.5982$ | Over-partitioning within South zone |
| $8$ | $164,210.1$ | $0.5714$ | Diminishing marginal returns |

At **$K = 5$**, the silhouette score reaches a pronounced peak ($0.7180$) while inertia exhibits a clear diminishing elbow inflection.

---

### 2.3 Demand Zone Centroid Profiles & Operational Metrics

![Spatial Clusters Map](figures/dark_store_clusters.png)

The $41,953$ valid deliveries partition into 5 macro regional demand zones:

| Zone ID | Demand Zone Label | Centroid Latitude | Centroid Longitude | Geographic Hub / Corridor | Order Volume ($N$) | Fleet Volume Share | Mean Delivery Time | SLA Breach Rate ($>30\text{m}$) |
| :---: | :--- | :---: | :---: | :--- | :---: | :---: | :---: | :---: |
| **0** | Exploratory Zone 1 | $23.587269$ | $85.432361$ | Eastern Hub (Ranchi / Kolkata Corridor) | $3,832$ | $9.13\%$ | $26.54\text{ min}$ | $30.82\%$ |
| **1** | Exploratory Zone 2 | $20.644616$ | $73.887522$ | Western Hub (Mumbai / Pune / Surat) | $14,782$ | **$35.23\%$** | $26.23\text{ min}$ | $29.54\%$ |
| **2** | Exploratory Zone 3 | $17.187944$ | $77.729890$ | Central-South Hub (Hyderabad / Telangana) | $3,790$ | $9.03\%$ | $26.38\text{ min}$ | $30.26\%$ |
| **3** | Exploratory Zone 4 | $27.369109$ | $76.844306$ | Northern Hub (Delhi-NCR / Jaipur Corridor) | $6,326$ | $15.08\%$ | $26.58\text{ min}$ | **$31.66\%$** |
| **4** | Exploratory Zone 5 | $12.268503$ | $77.860032$ | Southern Hub (Bangalore / Mysore / Chennai) | $13,223$ | **$31.52\%$** | $26.19\text{ min}$ | $29.20\%$ |

### 2.4 Critical Operational Disclaimer: Demand Zones vs Dark-Store Hubs
> [!IMPORTANT]
> **Exploratory Demand Zones vs. Micro Fulfillment Hubs:**
> The clusters computed here represent **macro territorial order concentrations across nationwide city clusters**, where orders span hundreds of kilometers across state boundaries. They are **NOT** turnkey real-estate coordinates for localized 10-minute micro-fulfillment dark stores.
> 
> *Production Dark-Store Optimization Recommendation:* To determine true 10-minute dark-store site selection, K-Means or capacitated p-median facility location algorithms must be executed **intra-city** (e.g., strictly within Bangalore Urban or Mumbai Metropolitan) constrained by street-level road network graph distances (OSRM / OpenStreetMap) and real estate zoning constraints.

---

## 3. Part 2: Delivery Partner Behavioral Segmentation

### 3.1 Feature Formulation & Filtering
To ensure statistical significance, couriers with fewer than 3 historical deliveries were excluded. A total of **$1,320$ active delivery couriers** were profiled across five behavioral dimensions:
1. `avg_delivery_time`: Average minutes taken to fulfill deliveries.
2. `avg_rating`: Mean customer satisfaction score ($1.0 - 5.0$).
3. `late_order_rate`: Percentage of handled deliveries exceeding the $30\text{-minute}$ SLA.
4. `deliveries_per_day`: Daily dispatch throughput.
5. `total_deliveries`: Cumulative order experience.

Features were scaled using `StandardScaler` prior to K-Means clustering ($K = 3$). Clusters were ranked deterministically by performance efficiency.

---

### 3.2 Segment Profiles & Empirical Statistics

![Rider Segmentation Visualizations](figures/rider_segments.png)

The courier fleet naturally separates into three distinct performance tiers:

| Performance Tier | Active Courier Count | Fleet Share (%) | Mean Delivery Time | Mean Rating | Mean Late Rate (%) | Daily Deliveries / Day | Mean Total Deliveries | Primary Operational Profile |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Tier 1: High Efficiency / Express** | $396$ | $30.0\%$ | **$25.03\text{ min}$** | **$4.68$** | **$22.72\%$** | $1.42$ | $29.28$ | Elite performers; lowest transit latency and superior customer ratings |
| **Tier 2: Standard Operations** | **$550$** | **$41.7\%$** | $25.91\text{ min}$ | $4.64$ | $28.10\%$ | **$1.71$** | **$41.98$** | Core operational backbone; highest order volume and reliable throughput |
| **Tier 3: At-Risk / Needs Coaching** | $374$ | $28.3\%$ | **$28.38\text{ min}$** | **$4.60$** | **$40.47\%$** | $1.59$ | $29.17$ | Chronically lagging; late on $>40\%$ of all deliveries; high churn risk |

---

### 3.3 Comparative Segment Insights
1. **The Late-Rate Divergence:**
   - Tier 1 couriers breach SLA on only **$22.72\%$** of deliveries.
   - Tier 3 couriers breach SLA on **$40.47\%$** of deliveries ($+17.75$ percentage points higher).
   - This $3.35\text{-minute}$ mean delivery gap ($25.03\text{m}$ vs $28.38\text{m}$) is the critical margin between meeting and breaching a 30-minute quick-commerce guarantee.

2. **Volume Backbone:**
   - Tier 2 riders handle the heaviest load, averaging **$41.98$ deliveries** across active days ($+43\%$ higher cumulative volume than Tiers 1 and 3) while sustaining solid delivery speeds ($25.91\text{m}$) and ratings ($4.64$).

---

## 4. Operational Dispatch & Fleet Optimization Strategy

Based on actual cluster properties, the following algorithmic dispatch rules are recommended:

```mermaid
flowchart TD
    Order[New Order Placed] --> CheckRisk{Order Late Risk Assessment}
    CheckRisk -- "High Breach Risk (Prob > 0.59)" --> PrioritizeTier1[Dispatch Tier 1: Express Courier]
    CheckRisk -- "Normal Risk (Prob <= 0.59)" --> CheckRoute{Route Distance / Complexity}
    CheckRoute -- "High Volume / Standard Route" --> DispatchTier2[Dispatch Tier 2: Standard Courier]
    CheckRoute -- "Low Complexity / Off-Peak" --> DispatchTier3[Dispatch Tier 3 with Route Coaching]
```

1. **VIP & At-Risk Order Prioritization:**
   - When the classification model flags an order with high breach probability ($P > 0.59$), dispatch priority must be routed exclusively to **Tier 1 couriers**.
2. **Targeted Driver Coaching for Tier 3:**
   - Tier 3 couriers should not be penalized or de-platformed immediately. Instead, operational logs indicate that route unfamiliarity and peak-hour vehicle issues drive their delays. Dispatch should assign short-radius orders while offering digital navigation coaching.
3. **Incentive Alignment:**
   - Tier 2 couriers should receive volume-tier bonuses to recognize their role as the high-throughput backbone ($41.98$ average deliveries).

---

## 5. Artifact Serialization

The clustering models and profiling schemas are saved in `models/`:
- **Dark Store K-Means Artifact:** [`models/dark_store_kmeans.joblib`](file:///D:/Quick-Commerce-Delivery-Intelligence/models/dark_store_kmeans.joblib)
- **Rider Segmentation Artifact:** [`models/rider_segmentation_kmeans.joblib`](file:///D:/Quick-Commerce-Delivery-Intelligence/models/rider_segmentation_kmeans.joblib)

