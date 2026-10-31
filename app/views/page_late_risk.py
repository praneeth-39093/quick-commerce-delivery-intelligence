"""
Page 4: Late-Order SLA Breach Risk Assessment.
Loads the trained XGBoost Classifier and applies the calibrated threshold (T* = 0.59)
to evaluate delivery failure risk and trigger operational interventions.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from app.styles import (
    render_header, render_kpi, get_plotly_layout_defaults,
    COLOR_CYAN, COLOR_BLUE, COLOR_PURPLE, COLOR_AMBER, COLOR_RED, COLOR_GREEN
)
from app.data_service import execute_late_risk_inference


def render_page():
    render_header(
        "Late-Order SLA Breach Detection",
        "Probabilistic risk scoring and automated dispatch mitigation at the calibrated 0.59 threshold"
    )

    # Explanation banner on threshold calibration
    st.markdown(r"""
    > [!IMPORTANT]
    > **Threshold Engineering: Probability vs. Decision Cutoff**  
    > In standard classification, a default probability threshold of $0.50$ assumes equal class distribution and equal error costs. To account for positive class weighting ($2.38:1$), the threshold was tuned strictly on the validation set to **$T^* = 0.59$**.  
    > At $T = 0.59$, the model captures **$88.32\%$ of all SLA breaches (Recall)** while achieving **$85.89\%$ Precision**, minimizing false alarm fatigue.
    """)

    # Interactive Risk Evaluation Form
    with st.form("late_risk_form"):
        st.markdown("### 1. Order Route & Environmental Friction")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            distance_km = st.number_input("Delivery Distance (km)", min_value=0.5, max_value=30.0, value=6.5, step=0.5)
        with col2:
            traffic = st.selectbox("Road Traffic Density", ["Low", "Medium", "High", "Jam"], index=2)
        with col3:
            weather = st.selectbox("Weather Condition", ["Sunny", "Cloudy", "Windy", "Fog", "Stormy", "Sandstorms"], index=3)
        with col4:
            city_type = st.selectbox("City Tier", ["Metropolitian", "Urban", "Semi-Urban"], index=0)

        st.markdown("---")
        st.markdown("### 2. Courier Profile & Vehicle Status (At Dispatch)")
        col5, col6, col7, col8 = st.columns(4)
        with col5:
            rider_rating = st.slider("Courier Customer Rating", min_value=1.0, max_value=5.0, value=4.3, step=0.1)
        with col6:
            vehicle_cond = st.selectbox("Vehicle Condition Tier", [0, 1, 2], index=0, help="0: High mileage / prone to issues, 1: Normal, 2: Optimal")
        with col7:
            rider_age = st.slider("Courier Age", min_value=18, max_value=50, value=32, step=1)
        with col8:
            vehicle_type = st.selectbox("Vehicle Type", ["motorcycle", "scooter", "electric_bike", "bicycle"], index=0)

        st.markdown("---")
        st.markdown("### 3. Operational Context")
        col9, col10, col11 = st.columns(3)
        with col9:
            order_hour = st.slider("Order Placement Hour", min_value=0, max_value=23, value=19)
        with col10:
            order_type = st.selectbox("Order Basket Type", ["Snack", "Meal", "Drinks", "Buffet"], index=1)
        with col11:
            festival = st.selectbox("Festival / Major Event Day?", ["No", "Yes"], index=0)

        submit_risk = st.form_submit_button("🚨 Evaluate Late-Order Risk", use_container_width=True)

    if submit_risk:
        if distance_km <= 0:
            st.error("⚠️ Invalid Delivery Distance: Must be strictly greater than 0.0 km.")
            return

        if distance_km > 30.0:
            st.warning(f"⚠️ Long-distance delivery ({distance_km:.1f} km). Exceeds typical quick-commerce micro-fulfillment radius.")

        payload = {
            "distance_km": distance_km,
            "road_traffic_density": traffic,
            "weather": weather,
            "city": city_type,
            "delivery_person_ratings": rider_rating,
            "vehicle_condition": vehicle_cond,
            "delivery_person_age": rider_age,
            "type_of_vehicle": vehicle_type,
            "order_hour": order_hour,
            "order_day_of_week": 4,
            "is_weekend": 0,
            "type_of_order": order_type,
            "festival": festival,
            "order_date": "2022-04-01"
        }

        with st.spinner("Scoring order against XGBoost Classifier pipeline..."):
            result = execute_late_risk_inference(payload)

        if result.get("is_fallback", False):
            st.warning("⚠️ Trained XGBoost Classifier artifact is unavailable. Displaying baseline heuristic risk assessment.")

        late_prob = result["late_probability"]
        late_pct = result["late_probability_pct"]
        threshold = result["threshold_used"]
        decision = result["is_late_decision"]
        risk_cat = result["risk_category"]
        mitigation = result["mitigation_plan"]

        st.markdown("### 📊 Late-Order Risk Diagnostic")

        # Top metric strip
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            render_kpi("Breach Probability", f"{late_pct:.1f}%", f"Model score: {late_prob:.3f}", accent="red" if late_prob >= threshold else "green")
        with k2:
            render_kpi("Decision Threshold", f"T* = {threshold:.2f}", "Calibrated on Val", accent="purple")
        with k3:
            badge_color = "red" if decision == 1 else "green"
            verdict_text = "SLA BREACH ALERT" if decision == 1 else "ON-TIME EXPECTED"
            render_kpi("Prediction Verdict", verdict_text, f"Cutoff: P >= {threshold:.2f}", accent=badge_color)
        with k4:
            render_kpi("Risk Classification", risk_cat, f"Urgency: {mitigation['urgency_level']}", accent="amber" if risk_cat in ["MODERATE", "HIGH"] else ("red" if risk_cat == "CRITICAL" else "green"))

        st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

        # Visual Risk Gauge & Mitigation Plan
        col_g, col_p = st.columns([1, 1.2])

        with col_g:
            # Plotly Bullet Gauge
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=late_pct,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "<b>Late Delivery Risk Gauge (%)</b>", 'font': {'color': '#E2E8F0', 'size': 16}},
                number={'suffix': "%", 'font': {'color': '#EF4444' if decision == 1 else '#10B981', 'size': 32}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#94A3B8"},
                    'bar': {'color': COLOR_RED if decision == 1 else COLOR_GREEN, 'thickness': 0.3},
                    'bgcolor': "#1E293B",
                    'borderwidth': 1,
                    'bordercolor': "#334155",
                    'steps': [
                        {'range': [0, 30], 'color': "rgba(16, 185, 129, 0.25)"},
                        {'range': [30, 59], 'color': "rgba(245, 158, 11, 0.25)"},
                        {'range': [59, 100], 'color': "rgba(239, 68, 68, 0.25)"}
                    ],
                    'threshold': {
                        'line': {'color': "#FFFFFF", 'width': 4},
                        'thickness': 0.8,
                        'value': threshold * 100
                    }
                }
            ))
            g_layout = get_plotly_layout_defaults().copy()
            g_layout.update(height=320, margin=dict(l=30, r=30, t=50, b=20))
            fig_gauge.update_layout(g_layout)
            st.plotly_chart(fig_gauge, use_container_width=True)

        with col_p:
            st.markdown(f"""
            <div class="content-box">
                <h4 style="color: {'#EF4444' if decision == 1 else '#10B981'}; margin-top:0;">
                    Recommended Dispatch & Mitigation Protocol
                </h4>
                <p><b>Operational Urgency:</b> <span class="badge badge-{'red' if decision == 1 else 'green'}">{mitigation['urgency_level']}</span></p>
                <p style="font-weight: 600; margin-bottom: 6px;">Automated Intervention Checklist:</p>
                <ul>
            """, unsafe_allow_html=True)
            for action in mitigation["recommended_actions"]:
                st.markdown(f"<li>{action}</li>", unsafe_allow_html=True)
            st.markdown("""
                </ul>
                <div style="font-size: 0.85rem; color: #94A3B8; margin-top: 12px; border-top: 1px solid #1E293B; padding-top: 8px;">
                    Algorithm: <b>XGBoost Binary Classifier</b> | Features: <b>15 strictly leakage-free dispatch signals</b>
                </div>
            </div>
            """, unsafe_allow_html=True)
