"""
Main Streamlit Application Entrypoint.
Quick-Commerce Delivery Intelligence: AI-Powered ETA Prediction, Late-Order Detection,
and Delivery Optimization.
"""

import sys
from pathlib import Path

# Ensure project root is accessible on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
import config

from app.styles import apply_custom_theme
from app.views import (
    page_executive,
    page_data_explorer,
    page_eta_prediction,
    page_late_risk,
    page_demand_zones,
    page_rider_segmentation,
    page_model_performance,
    page_business_recommendations
)

# Page configuration
st.set_page_config(
    page_title="Quick-Commerce Delivery Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply unified dark navy styling
apply_custom_theme()

# Sidebar Brand Header
st.sidebar.markdown("""
<div style="padding: 10px 0 15px 0;">
    <h2 style="color: #00F2FE; margin: 0; font-size: 1.4rem; font-weight: 800; letter-spacing: -0.01em;">
        ⚡ QUICK-COMMERCE
    </h2>
    <div style="color: #94A3B8; font-size: 0.82rem; font-weight: 500; text-transform: uppercase; letter-spacing: 0.08em;">
        Delivery Intelligence Engine
    </div>
</div>
""", unsafe_allow_html=True)

# Navigation Menu
NAVIGATION_OPTIONS = {
    "📊 Executive Dashboard": page_executive.render_page,
    "🔍 Data Explorer": page_data_explorer.render_page,
    "⚡ ETA Prediction": page_eta_prediction.render_page,
    "🚨 Late-Order Risk": page_late_risk.render_page,
    "🗺️ Demand Zone Analysis": page_demand_zones.render_page,
    "🚴 Rider Segmentation": page_rider_segmentation.render_page,
    "🧪 Model Performance": page_model_performance.render_page,
    "💡 Business Recommendations": page_business_recommendations.render_page
}

selected_page = st.sidebar.radio(
    "Navigation Menu",
    options=list(NAVIGATION_OPTIONS.keys()),
    index=0
)

st.sidebar.markdown("---")

# Sidebar Operational Context
st.sidebar.markdown(f"""
<div style="font-size: 0.85rem; color: #94A3B8; line-height: 1.5;">
    <div style="color: #F8FAFC; font-weight: 600; margin-bottom: 6px;">System Parameters:</div>
    • Benchmark SLA: <b style="color: #00F2FE;">{config.DEFAULT_SLA_MINUTES:.0f} mins</b> (Configurable)<br>
    • Decision Point: <b style="color: #A855F7;">Courier Dispatch (T1)</b><br>
    • Late Threshold: <b style="color: #EF4444;">T* = 0.59</b><br>
    • Spatial Clustered: <b style="color: #3B82F6;">41,953 GPS Orders</b><br>
    • Active Couriers: <b style="color: #10B981;">1,320 Partners</b>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.caption("Quick-Commerce Delivery Intelligence v1.0.0 | Enterprise Edition")

# Execute selected page renderer
try:
    NAVIGATION_OPTIONS[selected_page]()
except Exception as e:
    st.error(f"Error loading page '{selected_page}': {e}")
    st.exception(e)
