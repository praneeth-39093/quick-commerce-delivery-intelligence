"""
Page 5: Dark-Store Demand Zone Spatial Analysis.
Visualizes the 5 macro geographic demand zones derived from 41,953 valid GPS points
using Folium interactive mapping and Plotly cluster comparisons.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import folium
from streamlit_folium import st_folium

from app.styles import (
    render_header, render_kpi, get_plotly_layout_defaults,
    COLOR_CYAN, COLOR_BLUE, COLOR_PURPLE, COLOR_AMBER, COLOR_RED, COLOR_GREEN
)
from app.data_service import load_dark_store_bundle


def render_page():
    render_header(
        "Dark-Store Spatial Demand Zones",
        "Macro-level regional demand concentration clustering derived from 41,953 valid GPS deliveries"
    )

    # Mandatory operational disclaimer
    st.markdown("""
    > [!WARNING]
    > **Crucial Operational Scope Disclaimer:**  
    > These 5 clusters represent **macro-level nationwide demand regions across Indian metropolitan hubs**, where deliveries span multiple states and regional logistics corridors.  
    > They are **NOT** turnkey real-estate coordinates for localized 10-minute neighborhood micro-fulfillment dark stores. Intra-city dark-store placement requires street-level network graph distance modeling (e.g. OSRM) bounded strictly within individual city limits.
    """)

    bundle = load_dark_store_bundle()
    if bundle is None or "centroids" not in bundle:
        st.error("Dark store clustering model bundle not found. Please train models.")
        return

    centroids_df = bundle["centroids"]

    # Top KPI Strip across the 5 clusters
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        render_kpi("Spatial Clusters (K)", "5 Regions", "Peak Silhouette: 0.718", accent="cyan")
    with k2:
        render_kpi("Total Clustered Orders", f"{centroids_df['order_volume'].sum():,}", "92.02% valid GPS coords", accent="blue")
    with k3:
        render_kpi("Largest Hub Volume", f"{centroids_df['order_volume'].max():,} orders", f"Zone 2: {centroids_df['order_percentage'].max()}% share", accent="purple")
    with k4:
        render_kpi("Excluded GPS Points", "3,640 records", "7.98% zero-coords quarantined", accent="amber")

    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

    # Geographic Map & Summary Table
    col_map, col_table = st.columns([1.3, 1])

    with col_map:
        st.markdown("#### 🗺️ Nationwide Demand Hub Geolocation Map")
        # Center map on India
        m = folium.Map(
            location=[21.0, 78.5],
            zoom_start=5,
            tiles="CartoDB dark_matter"
        )

        zone_colors = ["#00F2FE", "#3B82F6", "#10B981", "#F59E0B", "#8B5CF6"]

        for idx, row in centroids_df.iterrows():
            lat = row["hub_latitude"]
            lon = row["hub_longitude"]
            zone_name = row["demand_zone_name"]
            vol = int(row["order_volume"])
            pct = row["order_percentage"]
            avg_eta = row["avg_delivery_time_min"]
            late_pct = row["late_order_percentage"]
            color = zone_colors[idx % len(zone_colors)]

            popup_html = f"""
            <div style="font-family: Arial; font-size: 12px; width: 200px;">
                <b style="color: {color}; font-size: 14px;">{zone_name}</b><br>
                <b>Centroid:</b> ({lat:.4f}, {lon:.4f})<br>
                <b>Order Volume:</b> {vol:,} ({pct}%)<br>
                <b>Avg Delivery Time:</b> {avg_eta} min<br>
                <b>Late Order Rate:</b> {late_pct}%<br>
            </div>
            """

            # Add marker
            folium.Marker(
                location=[lat, lon],
                popup=folium.Popup(popup_html, max_width=250),
                tooltip=f"{zone_name} ({vol:,} orders)",
                icon=folium.Icon(color="darkblue", icon="shopping-cart", prefix="fa")
            ).add_to(m)

            # Add coverage bubble proportional to order volume
            folium.Circle(
                location=[lat, lon],
                radius=float(np.sqrt(vol) * 3500),
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.25,
                weight=2
            ).add_to(m)

        st_folium(m, width="100%", height=420)

    with col_table:
        st.markdown("#### 📋 Demand Zone Centroid Statistics")
        display_df = centroids_df[[
            "demand_zone_name", "order_volume", "order_percentage",
            "avg_delivery_time_min", "late_order_percentage"
        ]].copy()
        display_df.columns = [
            "Regional Zone", "Orders (N)", "Volume Share (%)", "Avg ETA (min)", "Late Rate (%)"
        ]
        st.dataframe(display_df, use_container_width=True, height=220)

        st.markdown(r"""
        **Regional Corridor Breakdown:**
        - **Zone 1 (East):** Jharkhand / West Bengal corridor ($9.1\%$).
        - **Zone 2 (West):** Mumbai / Pune / Gujarat corridor ($35.2\%$).
        - **Zone 3 (Central-South):** Hyderabad / Telangana corridor ($9.0\%$).
        - **Zone 4 (North):** Delhi-NCR / Jaipur corridor ($15.1\%$).
        - **Zone 5 (South):** Bangalore / Tamil Nadu corridor ($31.5\%$).
        """)

    # Row 2: Cluster Comparison Charts
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    layout_defaults = get_plotly_layout_defaults()

    with c1:
        # Order Volume by Zone
        fig_vol = px.bar(
            centroids_df,
            x="demand_zone_name",
            y="order_volume",
            color="order_volume",
            color_continuous_scale=[[0, COLOR_CYAN], [1, COLOR_PURPLE]],
            text=centroids_df["order_volume"].apply(lambda x: f"{x:,}"),
            title="<b>Total Order Volume by Regional Hub</b>",
            labels={"demand_zone_name": "Regional Hub", "order_volume": "Order Count"}
        )
        v_layout = layout_defaults.copy()
        v_layout.update(height=320, coloraxis_showscale=False)
        fig_vol.update_layout(v_layout)
        fig_vol.update_traces(textposition="outside")
        st.plotly_chart(fig_vol, use_container_width=True)

    with c2:
        # Late-Order Rate by Zone
        fig_late = px.bar(
            centroids_df,
            x="demand_zone_name",
            y="late_order_percentage",
            color="late_order_percentage",
            color_continuous_scale=[[0, COLOR_GREEN], [0.5, COLOR_AMBER], [1, COLOR_RED]],
            text=centroids_df["late_order_percentage"].apply(lambda x: f"{x:.1f}%"),
            title="<b>SLA Breach Rate (%) Across Regional Hubs</b>",
            labels={"demand_zone_name": "Regional Hub", "late_order_percentage": "Late Rate (%)"}
        )
        l_layout = layout_defaults.copy()
        l_layout.update(height=320, coloraxis_showscale=False)
        fig_late.update_layout(l_layout)
        fig_late.update_traces(textposition="outside")
        st.plotly_chart(fig_late, use_container_width=True)
