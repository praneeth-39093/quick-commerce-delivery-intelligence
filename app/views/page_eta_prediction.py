"""
Page 3: Real-Time ETA Prediction.
Interactive inference form using the trained XGBoost Regressor pipeline,
strictly operating at the verified Courier Assignment / Dispatch decision point.
"""

from datetime import datetime, timedelta
import streamlit as st
import pandas as pd
import numpy as np

from app.styles import (
    render_header, render_kpi,
    COLOR_CYAN, COLOR_BLUE, COLOR_PURPLE, COLOR_AMBER, COLOR_RED, COLOR_GREEN
)
from app.data_service import execute_eta_inference


def render_page():
    render_header(
        "AI-Powered ETA Prediction Engine",
        "Continuous delivery duration estimation executed strictly at courier assignment & dispatch"
    )

    st.markdown("""
    > [!NOTE]
    > **Decision Point: Courier Assignment / Dispatch Time (Stage 2)**  
    > In accordance with the model audit, this form collects signals observable when an order is matched to an active delivery partner. Post-dispatch variables (`Time_Order_picked`, `prep_time_min`, and `multiple_deliveries`) are strictly excluded.
    """)

    # Prediction Input Form
    with st.form("eta_prediction_form"):
        st.markdown("### 1. Spatial & Geodesic Routing")
        col_geo1, col_geo2, col_geo3 = st.columns(3)
        with col_geo1:
            # Presets for popular hubs
            location_preset = st.selectbox(
                "City Hub Preset",
                ["Custom GPS Coordinates", "Bangalore Urban", "Mumbai Metropolitan", "Delhi-NCR", "Hyderabad Hub"]
            )

        # Coordinate Defaults
        if location_preset == "Bangalore Urban":
            def_lat1, def_lon1, def_lat2, def_lon2 = 12.9716, 77.5946, 12.9985, 77.6250
        elif location_preset == "Mumbai Metropolitan":
            def_lat1, def_lon1, def_lat2, def_lon2 = 19.0760, 72.8777, 19.1120, 72.9050
        elif location_preset == "Delhi-NCR":
            def_lat1, def_lon1, def_lat2, def_lon2 = 28.6139, 77.2090, 28.6450, 77.2410
        elif location_preset == "Hyderabad Hub":
            def_lat1, def_lon1, def_lat2, def_lon2 = 17.3850, 78.4867, 17.4200, 78.5200
        else:
            def_lat1, def_lon1, def_lat2, def_lon2 = 12.9716, 77.5946, 13.0116, 77.6346

        with col_geo2:
            store_lat = st.number_input("Dark Store / Restaurant Latitude", value=def_lat1, format="%.6f")
            store_lon = st.number_input("Dark Store / Restaurant Longitude", value=def_lon1, format="%.6f")
        with col_geo3:
            cust_lat = st.number_input("Delivery Destination Latitude", value=def_lat2, format="%.6f")
            cust_lon = st.number_input("Delivery Destination Longitude", value=def_lon2, format="%.6f")

        st.markdown("---")
        st.markdown("### 2. Environmental & Contextual Attributes")
        col_env1, col_env2, col_env3, col_env4 = st.columns(4)
        with col_env1:
            weather = st.selectbox("Atmospheric Weather", ["Sunny", "Cloudy", "Windy", "Fog", "Stormy", "Sandstorms"])
        with col_env2:
            traffic = st.selectbox("Road Traffic Density", ["Low", "Medium", "High", "Jam"], index=1)
        with col_env3:
            order_type = st.selectbox("Order Basket Type", ["Snack", "Meal", "Drinks", "Buffet"], index=1)
        with col_env4:
            city_type = st.selectbox("City Classification", ["Metropolitian", "Urban", "Semi-Urban"], index=0)

        st.markdown("---")
        st.markdown("### 3. Matched Courier Profile (Known at Dispatch)")
        col_r1, col_r2, col_r3, col_r4 = st.columns(4)
        with col_r1:
            rider_rating = st.slider("Courier Customer Rating", min_value=1.0, max_value=5.0, value=4.7, step=0.1)
        with col_r2:
            rider_age = st.slider("Courier Age (Years)", min_value=18, max_value=50, value=28, step=1)
        with col_r3:
            vehicle_type = st.selectbox("Delivery Vehicle Type", ["motorcycle", "scooter", "electric_bike", "bicycle"])
        with col_r4:
            vehicle_condition = st.selectbox("Vehicle Fleet Condition Tier", [0, 1, 2], index=2, help="0: Poor/Mechanical issues, 1: Standard, 2: Well-maintained/New")

        st.markdown("---")
        st.markdown("### 4. Temporal Dispatch Attributes")
        col_t1, col_t2, col_t3 = st.columns(3)
        with col_t1:
            current_clock = datetime.now()
            order_hour = st.slider("Order Placement Hour (0-23)", min_value=0, max_value=23, value=current_clock.hour)
        with col_t2:
            is_weekend_sel = st.selectbox("Weekend Order?", ["No (Weekday)", "Yes (Weekend)"], index=0)
            is_weekend = 1 if "Yes" in is_weekend_sel else 0
        with col_t3:
            festival_sel = st.selectbox("Festival / Major Holiday?", ["No", "Yes"], index=0)

        submitted = st.form_submit_button("⚡ Compute Real-Time ETA", use_container_width=True)

    if submitted:
        # Validate coordinate boundaries against actual geographical limits of India
        lat_min, lat_max = config.LATITUDE_BOUNDS
        lon_min, lon_max = config.LONGITUDE_BOUNDS

        if not (lat_min <= store_lat <= lat_max and lon_min <= store_lon <= lon_max):
            st.error(
                f"⚠️ Invalid Dark Store Coordinates ({store_lat:.4f}, {store_lon:.4f}): "
                f"Latitude must be between {lat_min}° and {lat_max}° N, and Longitude between {lon_min}° and {lon_max}° E (India boundary)."
            )
            return

        if not (lat_min <= cust_lat <= lat_max and lon_min <= cust_lon <= lon_max):
            st.error(
                f"⚠️ Invalid Delivery Destination Coordinates ({cust_lat:.4f}, {cust_lon:.4f}): "
                f"Latitude must be between {lat_min}° and {lat_max}° N, and Longitude between {lon_min}° and {lon_max}° E (India boundary)."
            )
            return

        payload = {
            "restaurant_latitude": store_lat,
            "restaurant_longitude": store_lon,
            "delivery_location_latitude": cust_lat,
            "delivery_location_longitude": cust_lon,
            "weather": weather,
            "road_traffic_density": traffic,
            "type_of_order": order_type,
            "city": city_type,
            "delivery_person_ratings": rider_rating,
            "delivery_person_age": rider_age,
            "type_of_vehicle": vehicle_type,
            "vehicle_condition": vehicle_condition,
            "order_hour": order_hour,
            "order_day_of_week": current_clock.weekday(),
            "is_weekend": is_weekend,
            "festival": festival_sel,
            "order_date": "2022-04-01"
        }

        with st.spinner("Executing pipeline preprocessing and XGBoost tree regression..."):
            result = execute_eta_inference(payload)

        pred_min = result["predicted_eta_min"]
        dist_km = result["distance_km"]
        model_name = result["model_name"]
        dynamic_sla = result["dynamic_sla"]
        fleet_rec = result["fleet_recommendation"]

        if result.get("is_fallback", False):
            st.warning("⚠️ Trained ML Model artifact is unavailable. Displaying baseline domain physics heuristic fallback.")

        if dist_km < 0.1:
            st.warning("⚠️ Delivery distance is under 100 meters. Please ensure dark store and delivery pins are distinct.")
        elif dist_km > 30.0:
            st.warning(f"⚠️ Long-distance delivery ({dist_km:.1f} km). Exceeds typical quick-commerce micro-fulfillment radius (≤ 15 km).")

        # Calculate estimated clock arrival
        now = datetime.now()
        arrival_clock = now + timedelta(minutes=pred_min)

        st.markdown("### 🎯 Predicted Delivery Intelligence Output")
        
        # Result KPI cards
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            render_kpi("Predicted Duration", f"{pred_min:.1f} mins", "Doorstep arrival", accent="cyan")
        with k2:
            render_kpi("Estimated Clock Time", arrival_clock.strftime("%I:%M %p"), f"Placed at {now.strftime('%I:%M %p')}", accent="blue")
        with k3:
            render_kpi("Promised SLA Window", f"{dynamic_sla['promised_sla_eta_min']:.0f} mins", f"Includes dynamic buffers", accent="purple")
        with k4:
            render_kpi("Calculated Distance", f"{dist_km:.2f} km", "Haversine geodesic", accent="green")

        # Operational details breakdown
        st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
        col_res1, col_res2 = st.columns(2)

        with col_res1:
            st.markdown("""
            <div class="content-box">
                <h4 style="color: #00F2FE; margin-top:0;">Dynamic SLA Buffer Breakdown</h4>
                <table style="width: 100%; border-collapse: collapse; font-size: 0.95rem;">
                    <tr><td style="padding: 6px 0;">Base ML Predicted ETA</td><td style="text-align: right; font-weight: bold;">""" + f"{pred_min:.1f} min" + """</td></tr>
                    <tr><td style="padding: 6px 0;">Weather Risk Buffer</td><td style="text-align: right; color: #F59E0B;">+""" + f"{dynamic_sla['weather_buffer_min']:.1f} min" + """</td></tr>
                    <tr><td style="padding: 6px 0;">Traffic Friction Buffer</td><td style="text-align: right; color: #F59E0B;">+""" + f"{dynamic_sla['traffic_buffer_min']:.1f} min" + """</td></tr>
                    <tr><td style="padding: 6px 0;">Safety Margin</td><td style="text-align: right; color: #10B981;">+""" + f"{dynamic_sla['safety_margin_min']:.1f} min" + """</td></tr>
                    <tr style="border-top: 1px solid #1E293B;"><td style="padding: 8px 0; font-weight: bold; color: #00F2FE;">Total Customer-Facing SLA Target</td><td style="text-align: right; font-weight: bold; color: #00F2FE;">""" + f"{dynamic_sla['promised_sla_eta_min']:.0f} min" + """</td></tr>
                </table>
            </div>
            """, unsafe_allow_html=True)

        with col_res2:
            st.markdown(f"""
            <div class="content-box">
                <h4 style="color: #8B5CF6; margin-top:0;">Fleet Dispatch Recommendation</h4>
                <p><b>Recommended Vehicle:</b> <span class="badge badge-purple">{fleet_rec['recommended_vehicle'].upper()}</span></p>
                <p><b>Routing Rationale:</b> {fleet_rec['rationale']}</p>
                <p><b>Batching Policy:</b> {fleet_rec['batching_policy']}</p>
                <p style="font-size: 0.85rem; color: #94A3B8; margin-top: 10px;">
                    Model Pipeline: <b>{model_name}</b>
                </p>
            </div>
            """, unsafe_allow_html=True)

