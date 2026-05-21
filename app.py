import streamlit as st

st.set_page_config(page_title="Mall Heatmap", layout="wide")

st.title("Mall Heatmap")
st.write("Spatial occupancy heatmap pipeline for retail foot-traffic analysis.")

st.markdown("""
## How to use

Work through the steps in order using the sidebar:

| Step | Page | What it does |
|------|------|-------------|
| 1 | Upload & Detect | Upload a surveillance video and run YOLOv8 person detection |
| 2 | Calibrate | Pick reference point pairs to compute the camera → floor plan homography |
| 3 | Heatmap | Adjust smoothing and colour map; preview and export the occupancy heatmap |
| 4 | Time Analysis | Filter by time window to compare traffic patterns across periods |

Each step carries its outputs forward — complete them in order.
""")
