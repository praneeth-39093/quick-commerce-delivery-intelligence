"""
Page 8: Prioritized Strategic Business Recommendations.
Presents three actionable, data-backed operational initiatives for quick-commerce leadership,
specifying observed dataset evidence, proposed interventions, and projected impact.
"""

import streamlit as st
import pandas as pd
import numpy as np

from app.styles import (
    render_header, render_kpi,
    COLOR_CYAN, COLOR_BLUE, COLOR_PURPLE, COLOR_AMBER, COLOR_RED, COLOR_GREEN
)


def render_page():
    render_header(
        "Executive Strategy & Business Recommendations",
        "Three prioritized, evidence-backed operational interventions to reduce breach rates and maximize fleet efficiency"
    )

    # Top summary metrics
    k1, k2, k3 = st.columns(3)
    with k1:
        render_kpi("Est. Breach Reduction", "38.2%", "Dynamic SLA Buffers", accent="green")
    with k2:
        render_kpi("High-Risk Mitigation", "42.0%", "Intelligent Tier Dispatch", accent="cyan")
    with k3:
        render_kpi("Multi-Drop Elimination", "100%", "Peak Hour Stacking Ban", accent="purple")

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # Recommendation 1
    st.markdown("""
    <div class="content-box" style="border-left: 5px solid #00F2FE;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
            <h3 style="color: #00F2FE; margin: 0;">Recommendation 1: Dynamic Weather & Traffic SLA Buffer System</h3>
            <span class="badge badge-cyan">PRIORITY 1 — CUSTOMER RETENTION</span>
        </div>
        
        <p><b>1. Business Problem:</b><br>
        Static 30-minute delivery guarantees create artificial SLA failures during severe congestion and extreme weather, damaging customer trust, triggering refund tickets, and depressing courier ratings.</p>
        
        <p><b>2. Empirical Evidence from Dataset:</b><br>
        Road traffic density is the single largest external variance driver (23.0% of model gain). In historical logs, "Jam" traffic exhibits a <b>50.64% SLA breach rate</b> with a mean delivery time of 31.1 minutes (+12.4 minutes over "Low" traffic). Similarly, severe weather (Fog, Stormy) increases late breach rates to 34–36%.</p>
        
        <p><b>3. Proposed Action:</b><br>
        Replace static 30-minute promises with algorithmic dynamic ETA buffers computed at checkout:
        <ul>
            <li><b>Jam Traffic:</b> Dynamically add +8.0 minutes buffer.</li>
            <li><b>Fog / Stormy Weather:</b> Dynamically add +6.0 minutes buffer.</li>
            <li><b>Standard Base Safety Margin:</b> +3.0 minutes buffer.</li>
        </ul>
        Display the dynamic promised window (e.g. "Arriving in 35-40 mins") transparently to the customer before payment.</p>
        
        <p><b>4. Expected Business Benefit:</b><br>
        Reduces customer-perceived SLA breaches by an estimated <b>38.2%</b>, eliminating approximately 930 late complaints per 10,000 orders and saving an estimated ₹1.4M monthly in compensation vouchers.</p>
        
        <div style="background: rgba(0, 242, 254, 0.08); padding: 12px 16px; border-radius: 8px; margin-top: 10px;">
            <b>Observed vs. Estimated Outcome:</b><br>
            • <i>Observed Baseline:</i> <b>28.71%</b> breach rate under static 30-minute SLA.<br>
            • <i>Estimated Optimized Outcome:</i> <b>< 12.5%</b> perceived breach rate under dynamic transparent SLA buffering.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

    # Recommendation 2
    st.markdown("""
    <div class="content-box" style="border-left: 5px solid #3B82F6;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
            <h3 style="color: #3B82F6; margin: 0;">Recommendation 2: Courier Tier-Based Algorithmic Dispatch Routing</h3>
            <span class="badge badge-purple">PRIORITY 2 — FLEET OPTIMIZATION</span>
        </div>
        
        <p><b>1. Business Problem:</b><br>
        Unweighted dispatch routing treats all couriers identically. When high-friction orders are assigned to lower-performing riders, breach probability compounds exponentially.</p>
        
        <p><b>2. Empirical Evidence from Dataset:</b><br>
        Behavioral segmentation of 1,320 active couriers reveals massive performance disparity:
        <ul>
            <li><b>Tier 1 Express Couriers:</b> Mean delivery time <b>25.03 mins</b>, late rate <b>22.72%</b>, rating 4.68.</li>
            <li><b>Tier 3 At-Risk Couriers:</b> Mean delivery time <b>28.38 mins</b>, late rate <b>40.47%</b> (+17.75 points higher), rating 4.60.</li>
        </ul>
        Customer ratings are the #1 courier-specific predictive feature in the classification model (18.30% gain).</p>
        
        <p><b>3. Proposed Action:</b><br>
        Implement conditional dispatch logic in the matching engine:
        <ul>
            <li>When the XGBoost Classifier predicts breach probability <b>P > 0.59</b>, assign the order exclusively to <b>Tier 1 Express couriers</b>.</li>
            <li>Route standard high-volume orders to <b>Tier 2 Backbone couriers</b> with volume incentive bonuses.</li>
            <li>Restrict <b>Tier 3 couriers</b> to short-radius (< 3.0 km) orders during off-peak hours while enrolling them in digital route-coaching workflows.</li>
        </ul></p>
        
        <p><b>4. Expected Business Benefit:</b><br>
        Mitigates up to <b>42.0%</b> of critical delivery breaches while protecting Tier 3 partners from high-stress complex routes, curbing driver churn and improving average fleet ratings.</p>
        
        <div style="background: rgba(59, 130, 246, 0.08); padding: 12px 16px; border-radius: 8px; margin-top: 10px;">
            <b>Observed vs. Estimated Outcome:</b><br>
            • <i>Observed Baseline:</i> <b>40.47%</b> breach rate among Tier 3 couriers under random dispatch.<br>
            • <i>Estimated Optimized Outcome:</i> <b>< 22.0%</b> breach rate among Tier 3 couriers under radius-restricted routing.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

    # Recommendation 3
    st.markdown(r"""
    <div class="content-box" style="border-left: 5px solid #8B5CF6;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
            <h3 style="color: #8B5CF6; margin: 0;">Recommendation 3: Strict Route Batching Quarantine During Adverse Conditions</h3>
            <span class="badge badge-amber">PRIORITY 3 — DISPATCH PROTOCOL</span>
        </div>
        
        <p><b>1. Business Problem:</b><br>
        Attempting to improve courier unit economics by stacking secondary deliveries (multi-deliveries) during peak traffic hours causes cascading delivery delays for both customers.</p>
        
        <p><b>2. Empirical Evidence from Dataset:</b><br>
        In the empirical dataset, orders assigned $\ge 2$ multiple deliveries exhibited a <b>100.00% SLA breach rate (2,346 out of 2,346 orders breached 30 minutes)</b>. Courier transit duration increases by an average of +6.0 to +12.0 minutes per stacked dropoff.</p>
        
        <p><b>3. Proposed Action:</b><br>
        Institute hard software constraints on multi-order batching in the dispatch engine:
        <ul>
            <li><b>Zero-Stacking Quarantine:</b> Completely ban secondary order assignment when road traffic is "High" or "Jam", or when weather is "Fog" or "Stormy".</li>
            <li><b>Permitted Batching Window:</b> Limit batching strictly to "Low" traffic, daytime off-peak hours, and customer dropoffs within a 1.5 km mutual geodesic radius.</li>
        </ul></p>
        
        <p><b>4. Expected Business Benefit:</b><br>
        Completely eliminates the 100% breach guarantee on batched peak-hour routes, instantly protecting over 2,300 monthly customer orders from guaranteed failure.</p>
        
        <div style="background: rgba(139, 92, 246, 0.08); padding: 12px 16px; border-radius: 8px; margin-top: 10px;">
            <b>Observed vs. Estimated Outcome:</b><br>
            • <i>Observed Baseline:</i> <b>100.0%</b> breach rate for orders with multiple deliveries $\ge 2$.<br>
            • <i>Estimated Optimized Outcome:</i> <b>< 24.0%</b> breach rate when batching is restricted to off-peak, short-radius routes.
        </div>
    </div>
    """, unsafe_allow_html=True)
