import streamlit as st

st.set_page_config(page_title="Mall Heatmap", layout="wide")

st.title("Mall Heatmap")
st.write("Spatial occupancy heatmap pipeline for retail foot-traffic analysis.")

st.markdown("""
## How to use

| Step | Page | What it does |
|------|------|-------------|
| 1 | Upload & Detect | Upload a surveillance video and run YOLOv8 person detection |
| 2 | Heatmap | Camera-space occupancy heatmap overlaid on the emptiest frame |
| 3 | Time Analysis | Filter by time window to compare traffic patterns across periods |
| 4 | Tracking Insights | Unique footfall, dwell-time and person trajectories |

**Tracking (Step 4):** tick *“Enable person tracking”* in Step 1 to follow each person across
frames. This unlocks genuine footfall counts, dwell-time and trajectories - it needs dense
(~25-30 fps) video and is slower than the heatmap detection.

**Advanced (optional):** a top-down floor-plan view is available inside the Heatmap page -
tick *“Advanced - top-down floor-plan view”* to map the camera onto a floor plan. Not needed
for the core camera-space heatmap.
""")
