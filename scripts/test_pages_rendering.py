"""
Page rendering simulation script.
Simulates rendering all 8 pages in Streamlit context and verifies no runtime errors.
"""

import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import tempfile
import os
os.environ["MPLCONFIGDIR"] = tempfile.gettempdir()

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

pages = [
    ("Page 1: Executive Dashboard", page_executive.render_page),
    ("Page 2: Data Explorer", page_data_explorer.render_page),
    ("Page 3: ETA Prediction", page_eta_prediction.render_page),
    ("Page 4: Late-Order Risk", page_late_risk.render_page),
    ("Page 5: Demand Zone Analysis", page_demand_zones.render_page),
    ("Page 6: Rider Segmentation", page_rider_segmentation.render_page),
    ("Page 7: Model Performance", page_model_performance.render_page),
    ("Page 8: Business Recommendations", page_business_recommendations.render_page)
]

def main():
    print("=" * 70)
    print("PAGE-BY-PAGE RENDERING VERIFICATION")
    print("=" * 70)
    for name, render_func in pages:
        print(f"Testing {name}...")
        try:
            render_func()
            print(f"  ✅ {name} rendered without exceptions.")
        except Exception as e:
            print(f"  ❌ {name} FAILED: {e}")
            raise e

    print("=" * 70)
    print("ALL 8 PAGES VERIFIED AND PASSED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    main()

