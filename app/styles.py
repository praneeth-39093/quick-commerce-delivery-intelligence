"""
UI/UX Styling and Theming for Quick-Commerce Delivery Intelligence Dashboard.
Dark navy palette with cyan, blue, and purple modern cyber-executive accents.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.io as pio

# Color Palette Constants
COLOR_BG_DARK = "#0B0F19"
COLOR_CARD_BG = "#111827"
COLOR_CARD_BORDER = "#1E293B"
COLOR_SIDEBAR_BG = "#0F172A"

COLOR_CYAN = "#00F2FE"
COLOR_CYAN_DARK = "#06B6D4"
COLOR_BLUE = "#3B82F6"
COLOR_PURPLE = "#8B5CF6"
COLOR_PURPLE_LIGHT = "#A855F7"
COLOR_GREEN = "#10B981"
COLOR_RED = "#EF4444"
COLOR_AMBER = "#F59E0B"
COLOR_TEXT_MAIN = "#F8FAFC"
COLOR_TEXT_MUTED = "#94A3B8"

CUSTOM_CSS = """
<style>
    /* Global Base */
    .stApp {
        background-color: #0B0F19;
        color: #F8FAFC;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0F172A;
        border-right: 1px solid #1E293B;
    }
    
    /* Header Gradient */
    .main-title {
        background: linear-gradient(135deg, #00F2FE 0%, #3B82F6 50%, #8B5CF6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin-bottom: 0.2rem;
    }
    
    .sub-title {
        color: #94A3B8;
        font-size: 1.05rem;
        font-weight: 400;
        margin-bottom: 1.5rem;
    }
    
    /* KPI Metric Cards */
    .kpi-container {
        display: flex;
        flex-direction: column;
        background: #111827;
        border: 1px solid #1E293B;
        border-radius: 12px;
        padding: 18px 20px;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.4);
        position: relative;
        overflow: hidden;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .kpi-container:hover {
        border-color: #38BDF8;
        transform: translateY(-2px);
    }
    .kpi-title {
        color: #94A3B8;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 6px;
    }
    .kpi-value {
        color: #F8FAFC;
        font-size: 1.85rem;
        font-weight: 700;
        line-height: 1.2;
    }
    .kpi-delta {
        font-size: 0.82rem;
        font-weight: 500;
        margin-top: 6px;
    }
    .kpi-cyan { border-top: 3px solid #00F2FE; }
    .kpi-blue { border-top: 3px solid #3B82F6; }
    .kpi-purple { border-top: 3px solid #8B5CF6; }
    .kpi-green { border-top: 3px solid #10B981; }
    .kpi-amber { border-top: 3px solid #F59E0B; }
    .kpi-red { border-top: 3px solid #EF4444; }

    /* Custom Badges */
    .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.02em;
    }
    .badge-cyan { background: rgba(0, 242, 254, 0.15); color: #00F2FE; border: 1px solid rgba(0, 242, 254, 0.3); }
    .badge-purple { background: rgba(139, 92, 246, 0.15); color: #A855F7; border: 1px solid rgba(139, 92, 246, 0.3); }
    .badge-green { background: rgba(16, 185, 129, 0.15); color: #10B981; border: 1px solid rgba(16, 185, 129, 0.3); }
    .badge-red { background: rgba(239, 68, 68, 0.15); color: #EF4444; border: 1px solid rgba(239, 68, 68, 0.3); }
    .badge-amber { background: rgba(245, 158, 11, 0.15); color: #F59E0B; border: 1px solid rgba(245, 158, 11, 0.3); }
    
    /* Content Cards */
    .content-box {
        background: #111827;
        border: 1px solid #1E293B;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 20px;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid #1E293B;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        color: #94A3B8;
        border-radius: 6px 6px 0 0;
        padding: 8px 18px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        color: #00F2FE !important;
        border-bottom: 2px solid #00F2FE !important;
    }
</style>
"""

def apply_custom_theme():
    """Injects dark theme stylesheet into Streamlit app."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

def render_header(title: str, subtitle: str = ""):
    """Renders a stylish gradient title and subtitle."""
    st.markdown(f'<div class="main-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="sub-title">{subtitle}</div>', unsafe_allow_html=True)

def render_kpi(title: str, value: str, delta: str = "", accent: str = "cyan"):
    """
    Renders a styled KPI card.
    accent: 'cyan', 'blue', 'purple', 'green', 'amber', 'red'
    """
    delta_html = f'<div class="kpi-delta">{delta}</div>' if delta else ''
    st.markdown(f"""
    <div class="kpi-container kpi-{accent}">
        <div class="kpi-title">{title}</div>
        <div class="kpi-value">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)

def get_plotly_layout_defaults() -> dict:
    """Standardizes Plotly dark theme layout across all charts."""
    return dict(
        template="plotly_dark",
        paper_bgcolor="rgba(17, 24, 39, 0.5)",
        plot_bgcolor="rgba(17, 24, 39, 0.3)",
        font=dict(color="#E2E8F0", family="Inter, sans-serif", size=12),
        margin=dict(l=40, r=30, t=50, b=40),
        xaxis=dict(
            gridcolor="#1E293B",
            linecolor="#334155",
            zerolinecolor="#334155"
        ),
        yaxis=dict(
            gridcolor="#1E293B",
            linecolor="#334155",
            zerolinecolor="#334155"
        ),
        legend=dict(
            bgcolor="rgba(15, 23, 42, 0.7)",
            bordercolor="#1E293B",
            borderwidth=1
        )
    )

