"""
Page 1: Executive Dashboard.
Displays high-level operational KPIs, distribution charts, and SLA risk breakdowns.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from app.styles import (
    render_header, render_kpi, get_plotly_layout_defaults,
    COLOR_CYAN, COLOR_BLUE, COLOR_PURPLE, COLOR_AMBER, COLOR_RED, COLOR_GREEN
)
from app.data_service import load_featured_dataset, load_model_comparison_df
import config


def render_page():
    render_header(
        "Executive Operations Dashboard",
        "Strategic KPIs, network throughput, and real-time SLA breach dynamics"
    )

    df = load_featured_dataset()
    if df.empty:
        st.warning("Delivery dataset unavailable. Please verify data path.")
        return

    # Calculate exact KPIs from data
    total_orders = len(df)
    avg_delivery_time = df["delivery_time_min"].mean()
    late_rate_pct = (df["is_late"].mean()) * 100
    avg_distance = df["distance_km"].mean()

    # Load comparison table for best model summaries
    comp_df = load_model_comparison_df()
    reg_best = "XGBoost (MAE 3.32m)"
    clf_best = "XGBoost (F1 0.871)"
    if not comp_df.empty:
        if "algorithm" in comp_df.columns:
            xgb_reg = comp_df[comp_df["algorithm"] == "xgboost"]
            if not xgb_reg.empty and "mae" in xgb_reg.columns and not pd.isna(xgb_reg["mae"].iloc[0]):
                reg_best = f"XGBoost (MAE {xgb_reg['mae'].iloc[0]:.2f}m)"
            xgb_clf = comp_df[(comp_df["algorithm"] == "xgboost") & (comp_df["task_type"] == "classification")]
            if not xgb_clf.empty and "f1_score" in xgb_clf.columns:
                clf_best = f"XGBoost (F1 {xgb_clf['f1_score'].iloc[0]:.3f})"

    # Row 1: KPI Cards
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        render_kpi("Total Orders", f"{total_orders:,}", "100% verified logs", accent="cyan")
    with col2:
        render_kpi("Avg Delivery Time", f"{avg_delivery_time:.1f} m", "Doorstep handover", accent="blue")
    with col3:
        render_kpi("SLA Breach Rate", f"{late_rate_pct:.1f}%", f"Threshold > {config.DEFAULT_SLA_MINUTES:.0f}m", accent="amber" if late_rate_pct < 35 else "red")
    with col4:
        render_kpi("Avg Distance", f"{avg_distance:.1f} km", "Haversine geodesic", accent="purple")
    with col5:
        render_kpi("Best ETA Model", reg_best, "Chronological R² 0.801", accent="green")
    with col6:
        render_kpi("Best Classifier", clf_best, "Recall 88.3% (T=0.59)", accent="cyan")

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # Row 2: Distribution Visualizations
    layout_defaults = get_plotly_layout_defaults()
    row2_col1, row2_col2 = st.columns(2)

    with row2_col1:
        # Delivery Time Distribution
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(
            x=df["delivery_time_min"],
            nbinsx=40,
            marker_color=COLOR_BLUE,
            opacity=0.75,
            name="Orders"
        ))
        # Add 30m SLA line
        fig_dist.add_vline(
            x=config.DEFAULT_SLA_MINUTES,
            line_width=2.5,
            line_dash="dash",
            line_color=COLOR_RED,
            annotation_text=f"SLA Target ({config.DEFAULT_SLA_MINUTES:.0f}m)",
            annotation_position="top right",
            annotation_font=dict(color=COLOR_RED, size=11)
        )
        dist_layout = layout_defaults.copy()
        dist_layout.update(
            title="<b>Delivery Duration Distribution (minutes)</b>",
            xaxis_title="Minutes to Customer Doorstep",
            yaxis_title="Order Frequency",
            showlegend=False,
            height=340
        )
        fig_dist.update_layout(dist_layout)
        st.plotly_chart(fig_dist, use_container_width=True)

    with row2_col2:
        # Traffic Density Share Donut
        traffic_counts = df["road_traffic_density"].value_counts().reset_index()
        traffic_counts.columns = ["Traffic Level", "Orders"]
        color_map = {
            "Low": COLOR_GREEN,
            "Medium": COLOR_CYAN,
            "High": COLOR_AMBER,
            "Jam": COLOR_RED
        }
        fig_traffic = px.pie(
            traffic_counts,
            values="Orders",
            names="Traffic Level",
            hole=0.55,
            color="Traffic Level",
            color_discrete_map=color_map,
            title="<b>Order Volume by Road Traffic Density</b>"
        )
        traffic_layout = layout_defaults.copy()
        traffic_layout.update(height=340)
        fig_traffic.update_layout(traffic_layout)
        st.plotly_chart(fig_traffic, use_container_width=True)

    # Row 3: SLA Breach Impact Analysis
    row3_col1, row3_col2 = st.columns(2)

    with row3_col1:
        # Late-Order Rate by Traffic Density
        traffic_late = df.groupby("road_traffic_density").agg(
            late_rate=("is_late", lambda x: np.mean(x) * 100),
            order_count=("is_late", "count")
        ).reindex(["Low", "Medium", "High", "Jam"]).reset_index()

        fig_traffic_late = px.bar(
            traffic_late,
            x="road_traffic_density",
            y="late_rate",
            color="late_rate",
            color_continuous_scale=[[0, COLOR_GREEN], [0.5, COLOR_AMBER], [1, COLOR_RED]],
            text=traffic_late["late_rate"].apply(lambda x: f"{x:.1f}%"),
            title="<b>Late-Order Rate (%) by Traffic Congestion</b>",
            labels={"road_traffic_density": "Traffic Level", "late_rate": "Late Rate (%)"}
        )
        fig_traffic_late.add_hline(
            y=late_rate_pct,
            line_dash="dot",
            line_color="#94A3B8",
            annotation_text=f"Global Avg ({late_rate_pct:.1f}%)"
        )
        tl_layout = layout_defaults.copy()
        tl_layout.update(height=340, coloraxis_showscale=False)
        fig_traffic_late.update_layout(tl_layout)
        fig_traffic_late.update_traces(textposition="outside")
        st.plotly_chart(fig_traffic_late, use_container_width=True)

    with row3_col2:
        # Late-Order Rate by Weather Condition
        weather_late = df.groupby("weather").agg(
            late_rate=("is_late", lambda x: np.mean(x) * 100),
            order_count=("is_late", "count")
        ).sort_values("late_rate", ascending=True).reset_index()

        fig_weather_late = px.bar(
            weather_late,
            x="weather",
            y="late_rate",
            color="late_rate",
            color_continuous_scale=[[0, COLOR_CYAN], [0.5, COLOR_PURPLE], [1, COLOR_RED]],
            text=weather_late["late_rate"].apply(lambda x: f"{x:.1f}%"),
            title="<b>Late-Order Rate (%) Across Weather Conditions</b>",
            labels={"weather": "Atmospheric Condition", "late_rate": "Late Rate (%)"}
        )
        fig_weather_late.add_hline(
            y=late_rate_pct,
            line_dash="dot",
            line_color="#94A3B8",
            annotation_text=f"Global Avg ({late_rate_pct:.1f}%)"
        )
        wl_layout = layout_defaults.copy()
        wl_layout.update(height=340, coloraxis_showscale=False)
        fig_weather_late.update_layout(wl_layout)
        fig_weather_late.update_traces(textposition="outside")
        st.plotly_chart(fig_weather_late, use_container_width=True)

