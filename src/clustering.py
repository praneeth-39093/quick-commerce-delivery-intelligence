"""
Clustering and Geospatial Optimization Module.
Identifies delivery demand hotspots for Dark Store (micro-fulfillment hub) placement,
and segments delivery riders by speed and operational efficiency profiles.
"""

from typing import Dict, Any, Tuple, Optional, List
import logging
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

import config

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class DeliveryOptimizerClustering:
    """Clustering algorithms for network hub placement and partner operational profiling."""

    def __init__(self, n_clusters: int = config.CLUSTERING_CONFIG.n_clusters_zone):
        self.n_clusters = n_clusters
        self.kmeans_model: Optional[KMeans] = None
        self.centroids_: Optional[pd.DataFrame] = None
        self.scaler = StandardScaler()

    def evaluate_elbow_and_silhouette(
        self,
        df: pd.DataFrame,
        k_range: range = range(2, 9)
    ) -> Dict[str, List[float]]:
        """
        Calculates inertia and silhouette scores across cluster counts to guide optimal K selection.
        Filters strictly for verified in-bounds geographic coordinates.

        Args:
            df: DataFrame containing delivery coordinates.
            k_range: Range of K values to evaluate.

        Returns:
            Dict[str, List[float]]: inertias and silhouette scores.
        """
        lat_min, lat_max = config.LATITUDE_BOUNDS
        lon_min, lon_max = config.LONGITUDE_BOUNDS

        valid_mask = (
            df["delivery_location_latitude"].between(lat_min, lat_max) &
            df["delivery_location_longitude"].between(lon_min, lon_max)
        )
        coords = df.loc[valid_mask, ["delivery_location_latitude", "delivery_location_longitude"]].values

        if len(coords) < max(k_range):
            return {"k_values": list(k_range), "inertias": [], "silhouette_scores": []}

        sample_size = min(len(coords), 3000)
        sample_indices = np.random.choice(len(coords), sample_size, replace=False)
        sample_coords = coords[sample_indices]

        inertias = []
        silhouettes = []

        for k in k_range:
            km = KMeans(n_clusters=k, random_state=config.CLUSTERING_CONFIG.random_state, n_init=10)
            labels = km.fit_predict(sample_coords)
            inertias.append(float(km.inertia_))
            silhouettes.append(float(silhouette_score(sample_coords, labels)))

        return {
            "k_values": list(k_range),
            "inertias": inertias,
            "silhouette_scores": silhouettes
        }

    def fit_spatial_clusters(self, df: pd.DataFrame, n_clusters: Optional[int] = None) -> pd.DataFrame:
        """
        Groups delivery points into geographic zones and computes cluster centroids
        representing recommended dark store / micro-hub distribution points.
        Bypasses invalid coordinates (0.0 or out of bounds); assigns them cluster label -1.

        Args:
            df: DataFrame containing delivery coordinates.
            n_clusters: Optional cluster count override.

        Returns:
            pd.DataFrame: Original DataFrame augmented with 'delivery_zone_cluster' column.
        """
        k = n_clusters or self.n_clusters
        coord_cols = ["delivery_location_latitude", "delivery_location_longitude"]

        lat_min, lat_max = config.LATITUDE_BOUNDS
        lon_min, lon_max = config.LONGITUDE_BOUNDS

        valid_mask = (
            df[coord_cols[0]].between(lat_min, lat_max) &
            df[coord_cols[1]].between(lon_min, lon_max)
        )
        coords = df.loc[valid_mask, coord_cols].values

        if len(coords) < k:
            logger.warning("Insufficient valid coordinate rows for spatial clustering.")
            df_out = df.copy()
            df_out["delivery_zone_cluster"] = -1
            return df_out

        self.kmeans_model = KMeans(n_clusters=k, random_state=config.CLUSTERING_CONFIG.random_state, n_init=10)
        clusters = self.kmeans_model.fit_predict(coords)

        df = df.copy()
        df["delivery_zone_cluster"] = -1
        df.loc[valid_mask, "delivery_zone_cluster"] = clusters

        # Compute centroid statistics only from valid points
        centroids = []
        for idx, (lat, lon) in enumerate(self.kmeans_model.cluster_centers_):
            cluster_mask = clusters == idx
            cluster_size = int(np.sum(cluster_mask))

            # Retrieve order-level metrics for this cluster if available
            cluster_df = df.loc[valid_mask].iloc[cluster_mask]
            avg_eta = float(round(cluster_df["delivery_time_min"].mean(), 2)) if "delivery_time_min" in cluster_df.columns else np.nan
            late_pct = float(round(cluster_df["is_late"].mean() * 100, 2)) if "is_late" in cluster_df.columns else np.nan

            centroids.append({
                "cluster_id": idx,
                "demand_zone_name": f"Exploratory Zone {idx + 1}",
                "hub_latitude": float(round(lat, 6)),
                "hub_longitude": float(round(lon, 6)),
                "order_volume": cluster_size,
                "order_percentage": float(round((cluster_size / len(coords)) * 100, 2)),
                "avg_delivery_time_min": avg_eta,
                "late_order_percentage": late_pct
            })

        self.centroids_ = pd.DataFrame(centroids)
        logger.info(f"Spatial clustering generated {k} zone centroids from {len(coords)} valid GPS points.")
        return df

    def get_cluster_centroids(self) -> Optional[pd.DataFrame]:
        """Returns candidate dark store hub coordinates with aggregate volume metrics."""
        return self.centroids_

    def profile_delivery_partners(self, df: pd.DataFrame, n_clusters: int = 3) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Segments delivery partners by performance metrics (speed, ratings, volume, deliveries per day).
        Ranks clusters deterministically based on composite efficiency metrics.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (partner_level_df, segment_summary_profile_df)
        """
        req_cols = ["delivery_person_id", "delivery_time_min", "delivery_person_ratings"]
        if not all(col in df.columns for col in req_cols):
            logger.warning("Required columns for partner profiling missing.")
            return pd.DataFrame(), pd.DataFrame()

        # Compute experience and behavioral metrics per rider
        agg_dict = {
            "avg_delivery_time": ("delivery_time_min", "mean"),
            "avg_rating": ("delivery_person_ratings", "mean"),
            "total_deliveries": ("delivery_person_id", "count")
        }
        if "is_late" in df.columns:
            agg_dict["late_order_rate"] = ("is_late", lambda x: np.mean(x) * 100)

        if "order_date" in df.columns:
            agg_dict["active_days"] = ("order_date", "nunique")

        rider_summary = df.groupby("delivery_person_id").agg(**agg_dict).reset_index()

        if "active_days" in rider_summary.columns:
            rider_summary["deliveries_per_day"] = np.round(
                rider_summary["total_deliveries"] / np.maximum(rider_summary["active_days"], 1), 2
            )
        else:
            rider_summary["deliveries_per_day"] = 1.0

        if "late_order_rate" not in rider_summary.columns:
            rider_summary["late_order_rate"] = 0.0

        # Filter active partners with meaningful history (>= 3 deliveries)
        active = rider_summary[rider_summary["total_deliveries"] >= 3].copy()
        if len(active) < n_clusters:
            return rider_summary, pd.DataFrame()

        cluster_cols = ["avg_delivery_time", "avg_rating", "late_order_rate", "deliveries_per_day"]
        features = active[cluster_cols].values
        scaled_features = self.scaler.fit_transform(features)

        rider_kmeans = KMeans(n_clusters=n_clusters, random_state=config.MODEL_CONFIG.random_state, n_init=10)
        active["raw_cluster"] = rider_kmeans.fit_predict(scaled_features)

        # Meaningful Cluster Profiling: Rank clusters objectively
        # Lower delivery time, lower late rate, and higher rating = superior performance
        cluster_stats = active.groupby("raw_cluster").agg(
            mean_time=("avg_delivery_time", "mean"),
            mean_rating=("avg_rating", "mean"),
            mean_late_rate=("late_order_rate", "mean"),
            mean_daily_volume=("deliveries_per_day", "mean"),
            rider_count=("delivery_person_id", "count")
        ).reset_index()

        # Sort: ascending late rate & delivery time, descending rating
        cluster_stats = cluster_stats.sort_values(
            by=["mean_late_rate", "mean_time", "mean_rating"],
            ascending=[True, True, False]
        ).reset_index(drop=True)

        tier_mapping = {}
        tier_names = {
            0: "Tier 1: High Efficiency / Express",
            1: "Tier 2: Standard Operations",
            2: "Tier 3: At-Risk / Needs Coaching"
        }
        for rank, raw_id in enumerate(cluster_stats["raw_cluster"]):
            tier_mapping[raw_id] = rank

        active["efficiency_tier"] = active["raw_cluster"].map(tier_mapping)
        active["efficiency_tier_name"] = active["efficiency_tier"].map(tier_names)
        active = active.drop(columns=["raw_cluster"])

        # Create structured segment profile summary
        segment_profile = active.groupby("efficiency_tier_name").agg(
            rider_count=("delivery_person_id", "count"),
            avg_delivery_time_min=("avg_delivery_time", "mean"),
            avg_rating=("avg_rating", "mean"),
            avg_late_rate_pct=("late_order_rate", "mean"),
            avg_deliveries_per_day=("deliveries_per_day", "mean"),
            avg_total_deliveries=("total_deliveries", "mean")
        ).round(2).reset_index()

        return active, segment_profile

