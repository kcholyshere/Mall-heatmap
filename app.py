import streamlit as st

st.set_page_config(page_title="Mall Heatmap", layout="wide")

st.title("Mall Heatmap")
st.write("Spatial occupancy heatmap pipeline for retail foot-traffic analysis.")

st.markdown("""
## How to use

**Core flow (no calibration required):**

| Step | Page | What it does |
|------|------|-------------|
| 1 | Upload & Detect | Upload a surveillance video and run YOLOv8 person detection |
| 3 | Heatmap | Camera-space occupancy heatmap overlaid on the emptiest frame |
| 4 | Time Analysis | Filter by time window to compare traffic patterns across periods |

**Optional — top-down floor plan view:**

| Step | Page | What it does |
|------|------|-------------|
| 2 | Calibrate | Pick reference point pairs to compute the camera → floor plan homography |

Complete Step 2 then open the *Top-down floor plan view* expander on the Heatmap page.
""")
