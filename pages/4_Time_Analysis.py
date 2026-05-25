from datetime import datetime, timedelta

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.heatmap import (CAM_H, CAM_W, FLOOR_H, FLOOR_W,
                         build_heatmap, project_points)

st.set_page_config(page_title="Time Analysis", layout="wide")
st.title("Step 4 — Time-Based Analysis")

if "detections_path" not in st.session_state:
    st.warning("Complete Step 1 first.")
    st.stop()

fps = st.session_state.get("fps", 1.0)
ref_date = st.session_state.get("ref_date", datetime.today().date())
ref_time = st.session_state.get("ref_time", datetime.strptime("11:00", "%H:%M").time())
ref_dt = datetime.combine(ref_date, ref_time)

st.caption(f"Reference: frame 1 = {ref_dt.strftime('%Y-%m-%d %H:%M')} at {fps} FPS — change in Step 1.")

sigma = st.slider("Smoothing (sigma)", 5, 80, 30)
colormap = st.selectbox("Colormap", ["hot", "jet", "plasma", "YlOrRd"])

col_start, col_end = st.columns(2)
with col_start:
    st.markdown("**Start**")
    start_date = st.date_input("Start date", value=ref_date, key="start_date")
    start_time = st.time_input("Start time", value=ref_time, step=60, key="start_time")
with col_end:
    st.markdown("**End**")
    end_date = st.date_input("End date", value=ref_date, key="end_date")
    end_time = st.time_input("End time", value=(ref_dt + timedelta(minutes=30)).time(), step=60, key="end_time")

start_dt = datetime.combine(start_date, start_time)
end_dt   = datetime.combine(end_date, end_time)

if start_dt >= end_dt:
    st.error("End must be after start.")
    st.stop()

start_frame = max(1, int((start_dt - ref_dt).total_seconds() * fps) + 1)
end_frame   = max(1, int((end_dt   - ref_dt).total_seconds() * fps) + 1)

df_raw = pd.read_csv(st.session_state["detections_path"])
window_raw = df_raw[(df_raw["frame_id"] >= start_frame) & (df_raw["frame_id"] <= end_frame)]

st.write(f"Frame range: **{start_frame}–{end_frame}** | Detections in window: **{len(window_raw):,}**")

if len(window_raw) == 0:
    st.warning("No detections in this window. Try a wider time range.")
    st.stop()

# --- Camera-space heatmap (default) ---
cam_heatmap = build_heatmap(window_raw, "x", "y", [CAM_W, CAM_H], [[0, CAM_W], [0, CAM_H]], sigma=sigma)

fig, ax = plt.subplots(figsize=(8, 6))
ax.imshow(cam_heatmap, cmap=colormap, alpha=0.8, vmin=0, vmax=1,
          extent=[0, CAM_W, CAM_H, 0])
ax.set_title(f"{start_dt.strftime('%Y-%m-%d %H:%M')} – {end_dt.strftime('%H:%M')}")
ax.axis("off")
st.pyplot(fig)
plt.close(fig)

# --- Top-down view (opt-in) ---
with st.expander("Top-down floor plan view (requires calibration)"):
    H = st.session_state.get("homography")
    if H is None:
        st.info("Complete Step 2 — Calibrate to enable the top-down view.")
    else:
        @st.cache_data
        def load_and_project(path, h_bytes, s_frame, e_frame):
            _H = np.frombuffer(h_bytes).reshape(3, 3)
            _df = pd.read_csv(path)
            _df = _df[(_df["frame_id"] >= s_frame) & (_df["frame_id"] <= e_frame)]
            proj = project_points(_df[["x", "y"]].values, _H)
            _df = _df.copy()
            _df["px"], _df["py"] = proj[:, 0], proj[:, 1]
            return _df[(_df["px"] >= 0) & (_df["px"] < FLOOR_W) &
                       (_df["py"] >= 0) & (_df["py"] < FLOOR_H)]

        df_proj = load_and_project(
            st.session_state["detections_path"], H.tobytes(), start_frame, end_frame
        )
        floor_img = st.session_state.get("floor_plan_img")
        heatmap_td = build_heatmap(df_proj, "px", "py",
                                   [FLOOR_W, FLOOR_H], [[0, FLOOR_W], [0, FLOOR_H]], sigma=sigma)
        fig, ax = plt.subplots(figsize=(4, 7))
        if floor_img is not None:
            ax.imshow(floor_img)
        ax.imshow(heatmap_td, cmap=colormap, alpha=0.6, vmin=0, vmax=1,
                  extent=[0, FLOOR_W, FLOOR_H, 0])
        ax.set_title(f"{start_dt.strftime('%Y-%m-%d %H:%M')} – {end_dt.strftime('%H:%M')}")
        ax.axis("off")
        st.pyplot(fig)
        plt.close(fig)
