"""
Page 6: Delivery Partner Behavioral Segmentation & Fleet Profiling.
Loads the saved K-Means rider segmentation model (K=3) to display operational tiers,
courier performance distributions, and algorithmic dispatch routing rules.
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
from app.data_service import load_rider_segmentation_bundle


def render_page():
    render_header(
        "Delivery Partner Behavioral Segmentation",
        "Multi-dimensional courier profiling across speed, customer ratings, SLA compliance, and throughput"
    )

    bundle = load_rider_segmentation_bundle()
    if bundle is None or "segment_profiles" not in bundle:
        st.error("Rider segmentation artifact not found. Please train models.")
        return

    profile_df = bundle["segment_profiles"]
    total_riders = profile_df["rider_count"].sum()

    # Top KPI Cards for the 3 Tiers
    col1, col2, col3 = st.columns(3)

    t1 = profile_df[profile_df["efficiency_tier_name"].str.contains("Tier 1")].iloc[0]
    t2 = profile_df[profile_df["efficiency_tier_name"].str.contains("Tier 2")].iloc[0]
    t3 = profile_df[profile_df["efficiency_tier_name"].str.contains("Tier 3")].iloc[0]

    with col1:
        render_kpi(
            "Tier 1: High Efficiency",
            f"{t1['rider_count']} Couriers ({t1['rider_count']/total_riders*100:.1f}%)",
            f"Avg ETA {t1['avg_delivery_time_min']:.1f}m | Late {t1['avg_late_rate_pct']:.1f}%",
            accent="green"
        )
    with col2:
        render_kpi(
            "Tier 2: Standard Operations",
            f"{t2['rider_count']} Couriers ({t2['rider_count']/total_riders*100:.1f}%)",
            f"Avg ETA {t2['avg_delivery_time_min']:.1f}m | Volume {t2['avg_total_deliveries']:.1f} orders",
            accent="blue"
        )
    with col3:
        render_kpi(
            "Tier 3: At-Risk / Needs Coaching",
            f"{t3['rider_count']} Couriers ({t3['rider_count']/total_riders*100:.1f}%)",
            f"Avg ETA {t3['avg_delivery_time_min']:.1f}m | Late {t3['avg_late_rate_pct']:.1f}%",
            accent="red"
        )

    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

    # Detailed Comparison Table
    st.markdown(r"### 📋 Segment Behavioral Profiles (Active Couriers $\ge 3$ Deliveries)")
    table_display = profile_df[[
        "efficiency_tier_name", "rider_count", "avg_delivery_time_min",
        "avg_rating", "avg_late_rate_pct", "avg_deliveries_per_day", "avg_total_deliveries"
    ]].copy()
    table_display.columns = [
        "Courier Performance Tier", "Couriers (N)", "Avg Delivery Time (min)",
        "Avg Customer Rating", "Late SLA Breach Rate (%)", "Daily Deliveries/Day", "Avg Cumulative Orders"
    ]
    st.dataframe(table_display, use_container_width=True)

    # Visualizations
    layout_defaults = get_plotly_layout_defaults()
    r_col1, r_col2 = st.columns(2)

    with r_col1:
        # Late-Order Rate vs Delivery Time by Tier
        fig_scatter = px.scatter(
            profile_df,
            x="avg_delivery_time_min",
            y="avg_late_rate_pct",
            size="rider_count",
            color="efficiency_tier_name",
            color_discrete_map={
                "Tier 1: High Efficiency / Express": COLOR_GREEN,
                "Tier 2: Standard Operations": COLOR_CYAN,
                "Tier 3: At-Risk / Needs Coaching": COLOR_RED
            },
            title="<b>SLA Breach Rate vs Delivery Speed by Tier</b>",
            labels={
                "avg_delivery_time_min": "Avg Delivery Time (minutes)",
                "avg_late_rate_pct": "Late Breach Rate (%)",
                "efficiency_tier_name": "Tier"
            }
        )
        s_layout = layout_defaults.copy()
        s_layout.update(height=340)
        fig_scatter.update_layout(s_layout)
        st.plotly_chart(fig_scatter, use_container_width=True)

    with r_col2:
        # Fleet Distribution Donut
        fig_donut = px.pie(
            profile_df,
            values="rider_count",
            names="efficiency_tier_name",
            hole=0.55,
            color="efficiency_tier_name",
            color_discrete_map={
                "Tier 1: High Efficiency / Express": COLOR_GREEN,
                "Tier 2: Standard Operations": COLOR_CYAN,
                "Tier 3: At-Risk / Needs Coaching": COLOR_RED
            },
            title="<b>Active Courier Fleet Distribution</b>"
        )
        d_layout = layout_defaults.copy()
        d_layout.update(height=340)
        fig_donut.update_layout(d_layout)
        st.plotly_chart(fig_donut, use_container_width=True)

    # Operational Dispatch & Support Guidance
    st.markdown("### 🎯 Operational Dispatch Policies & Coaching Rules")
    p1, p2, p3 = st.columns(3)

    with p1:
        st.markdown(f"""
        <div class="content-box" style="border-left: 4px solid {COLOR_GREEN};">
            <h4 style="color: {COLOR_GREEN}; margin-top:0;">Tier 1: Express Routing</h4>
            <p><b>Target Allocation:</b> High-risk SLA orders (P > 0.59), VIP customers, and multi-drop routes.</p>
            <p><b>Performance Driver:</b> Fastest delivery time (25.03m) and highest rating (4.68).</p>
            <p><b>Retention Action:</b> Priority incentive surge bonuses and early batch dispatch preference.</p>
        </div>
        """, unsafe_allow_html=True)

    with p2:
        st.markdown(f"""
        <div class="content-box" style="border-left: 4px solid {COLOR_CYAN};">
            <h4 style="color: {COLOR_CYAN}; margin-top:0;">Tier 2: Operational Backbone</h4>
            <p><b>Target Allocation:</b> Standard high-volume order flows and medium-distance suburban trips.</p>
            <p><b>Performance Driver:</b> Highest throughput partner group (averaging 41.98 lifetime orders).</p>
            <p><b>Retention Action:</b> Volume milestone bonuses and equipment refresh programs.</p>
        </div>
        """, unsafe_allow_html=True)

    with p3:
        st.markdown(f"""
        <div class="content-box" style="border-left: 4px solid {COLOR_RED};">
            <h4 style="color: {COLOR_RED}; margin-top:0;">Tier 3: Targeted Support</h4>
            <p><b>Target Allocation:</b> Restrict to low-complexity, short-radius (< 3 km) deliveries during off-peak.</p>
            <p><b>Performance Driver:</b> Breaches SLA on 40.5% of orders with 28.38m average transit latency.</p>
            <p><b>Retention Action:</b> Digital navigation coaching and vehicle inspection support.</p>
        </div>
        """, unsafe_allow_html=True)
