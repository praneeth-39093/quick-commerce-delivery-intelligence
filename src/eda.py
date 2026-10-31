"""
Exploratory Data Analysis (EDA) Module.
Provides statistical profiling, distribution visualization, multivariate correlation analysis,
and automated figure generation for delivery performance insights.
"""

from pathlib import Path
from typing import Optional, Dict, Any
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import config

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class ExploratoryDataAnalysis:
    """Generates analytical insights and diagnostic plots for delivery intelligence."""

    def __init__(self, figures_dir: Optional[Path] = None):
        self.figures_dir = Path(figures_dir) if figures_dir else config.FIGURES_DIR
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        # Apply clean visual theme
        sns.set_theme(style="whitegrid", palette="muted")

    def generate_summary_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculates key statistical descriptors across numerical and categorical features."""
        num_summary = df.describe().to_dict()
        cat_cols = df.select_dtypes(include=["object", "category"]).columns
        cat_summary = {col: df[col].value_counts().to_dict() for col in cat_cols}

        return {
            "numerical_summary": num_summary,
            "categorical_summary": cat_summary,
            "missing_values": df.isnull().sum().to_dict()
        }

    def plot_target_distribution(self, df: pd.DataFrame, filename: str = "delivery_time_distribution.png") -> Path:
        """Plots ETA distribution with mean and SLA benchmark threshold lines."""
        target_col = config.CLEANED_TARGET_REGRESSION
        if target_col not in df.columns:
            logger.warning(f"Target column '{target_col}' not found for distribution plot.")
            return self.figures_dir / filename

        plt.figure(figsize=(10, 6))
        sns.histplot(df[target_col].dropna(), kde=True, bins=35, color="#1f77b4", edgecolor="black")
        plt.axvline(
            config.DEFAULT_SLA_MINUTES, color="#d62728", linestyle="--", linewidth=2,
            label=f"SLA Threshold ({config.DEFAULT_SLA_MINUTES:.0f} mins)"
        )
        mean_val = df[target_col].mean()
        plt.axvline(mean_val, color="#2ca02c", linestyle="-.", linewidth=2, label=f"Mean ETA ({mean_val:.1f} mins)")

        plt.title("Quick-Commerce Delivery Duration (ETA) Distribution", fontsize=14, fontweight="bold")
        plt.xlabel("Delivery Time (minutes)", fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        plt.legend(fontsize=11)
        plt.tight_layout()

        output_path = self.figures_dir / filename
        plt.savefig(output_path, dpi=300)
        plt.close()
        logger.info(f"Saved distribution plot to: {output_path}")
        return output_path

    def plot_delivery_time_by_traffic(self, df: pd.DataFrame, filename: str = "eta_by_traffic_density.png") -> Path:
        """Visualizes ETA variance across traffic density tiers."""
        traffic_col = "road_traffic_density"
        target_col = config.CLEANED_TARGET_REGRESSION

        if traffic_col not in df.columns or target_col not in df.columns:
            return self.figures_dir / filename

        plt.figure(figsize=(9, 6))
        order = ["Low", "Medium", "High", "Jam"]
        present_order = [o for o in order if o in df[traffic_col].unique()]

        sns.boxplot(
            data=df, x=traffic_col, y=target_col, order=present_order,
            hue=traffic_col, palette="Blues_d", legend=False, showmeans=True,
            meanprops={"marker": "o", "markerfacecolor": "red", "markeredgecolor": "red"}
        )
        plt.title("Impact of Road Traffic Density on Delivery Duration", fontsize=14, fontweight="bold")
        plt.xlabel("Traffic Density", fontsize=12)
        plt.ylabel("Delivery Time (minutes)", fontsize=12)
        plt.tight_layout()

        output_path = self.figures_dir / filename
        plt.savefig(output_path, dpi=300)
        plt.close()
        logger.info(f"Saved traffic analysis plot to: {output_path}")
        return output_path

    def plot_delivery_time_by_weather(self, df: pd.DataFrame, filename: str = "eta_by_weather.png") -> Path:
        """Visualizes ETA variance across atmospheric conditions."""
        weather_col = "weather"
        target_col = config.CLEANED_TARGET_REGRESSION

        if weather_col not in df.columns or target_col not in df.columns:
            return self.figures_dir / filename

        plt.figure(figsize=(10, 6))
        sns.barplot(
            data=df, x=weather_col, y=target_col,
            hue=weather_col, palette="viridis", legend=False, errorbar="se"
        )
        plt.title("Average Delivery Time by Weather Condition", fontsize=14, fontweight="bold")
        plt.xlabel("Weather Condition", fontsize=12)
        plt.ylabel("Mean Delivery Time (minutes)", fontsize=12)
        plt.xticks(rotation=25)
        plt.tight_layout()

        output_path = self.figures_dir / filename
        plt.savefig(output_path, dpi=300)
        plt.close()
        logger.info(f"Saved weather analysis plot to: {output_path}")
        return output_path

    def plot_distance_vs_time(self, df: pd.DataFrame, filename: str = "distance_vs_delivery_time.png") -> Path:
        """Scatter visualization of Haversine distance vs delivery duration."""
        dist_col = "distance_km"
        target_col = config.CLEANED_TARGET_REGRESSION

        if dist_col not in df.columns or target_col not in df.columns:
            return self.figures_dir / filename

        sample_df = df.sample(min(len(df), 2000), random_state=42) if len(df) > 2000 else df

        plt.figure(figsize=(10, 6))
        sns.regplot(
            data=sample_df, x=dist_col, y=target_col,
            scatter_kws={"alpha": 0.3, "color": "#2b5c8f"},
            line_kws={"color": "#d62728", "linewidth": 2}
        )
        plt.title("Geodesic Distance vs Delivery Duration", fontsize=14, fontweight="bold")
        plt.xlabel("Distance (km)", fontsize=12)
        plt.ylabel("Delivery Time (minutes)", fontsize=12)
        plt.tight_layout()

        output_path = self.figures_dir / filename
        plt.savefig(output_path, dpi=300)
        plt.close()
        logger.info(f"Saved distance vs time plot to: {output_path}")
        return output_path

    def plot_correlation_matrix(self, df: pd.DataFrame, filename: str = "feature_correlation_matrix.png") -> Path:
        """Plots heatmap of Pearson correlation coefficients across numerical indicators."""
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            return self.figures_dir / filename

        plt.figure(figsize=(11, 8))
        corr = numeric_df.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, vmin=-1, vmax=1)
        plt.title("Correlation Heatmap of Numerical Features", fontsize=14, fontweight="bold")
        plt.tight_layout()

        output_path = self.figures_dir / filename
        plt.savefig(output_path, dpi=300)
        plt.close()
        logger.info(f"Saved correlation matrix plot to: {output_path}")
        return output_path

    def run_full_eda(self, df: pd.DataFrame) -> Dict[str, Path]:
        """Executes all EDA plotting routines and saves artifact figures."""
        logger.info("Executing comprehensive EDA suite...")
        saved_figures = {
            "target_distribution": self.plot_target_distribution(df),
            "traffic_density": self.plot_delivery_time_by_traffic(df),
            "weather": self.plot_delivery_time_by_weather(df),
            "distance_vs_time": self.plot_distance_vs_time(df),
            "correlation_matrix": self.plot_correlation_matrix(df)
        }
        logger.info("Comprehensive EDA suite finished successfully.")
        return saved_figures

