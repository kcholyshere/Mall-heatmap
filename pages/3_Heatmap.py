from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.heatmap import (CAM_H, CAM_W, FLOOR_H, FLOOR_W,
                         backproject, build_heatmap, project_points)

st.set_page_config(page_title="Heatmap", layout="wide")
st.title("Step 3 — Heatmap")

if "detections_path" not in st.session_state:
    st.warning("Complete Step 1 first.")
    st.stop()


@st.cache_data
def load_detections(path):
    return pd.read_csv(path)


@st.cache_data
def get_background_frame(video_path, detections_path):
    """Return the frame with the fewest (but > 0) detections as an RGB array."""
    df = pd.read_csv(detections_path)
    counts = df.groupby("frame_id").size()
    positive = counts[counts > 0]
    target_frame = int(positive.idxmin()) if len(positive) > 0 else 1
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame - 1)
    ret, frame = cap.read()
    cap.release()
    if ret:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return None


df_raw = load_detections(st.session_state["detections_path"])
st.write(f"**{len(df_raw):,}** detections across **{df_raw['frame_id'].nunique():,}** sampled frames")

col_ctrl, col_view = st.columns([1, 2])

with col_ctrl:
    sigma = st.slider("Smoothing (sigma)", 5, 80, 30)
    colormap = st.selectbox("Colormap", ["hot", "jet", "plasma", "YlOrRd"])
    alpha = st.slider("Heatmap opacity", 0.1, 1.0, 0.6, step=0.05)

# --- Camera-space heatmap (default, no calibration needed) ---
cam_heatmap = build_heatmap(df_raw, "x", "y", [CAM_W, CAM_H], [[0, CAM_W], [0, CAM_H]], sigma=sigma)

video_path = st.session_state.get("video_path")
bg = get_background_frame(video_path, st.session_state["detections_path"]) if video_path else None

with col_view:
    fig, ax = plt.subplots(figsize=(8, 6))
    if bg is not None:
        ax.imshow(bg)
    ax.imshow(cam_heatmap, cmap=colormap, alpha=alpha, vmin=0, vmax=1,
              extent=[0, CAM_W, CAM_H, 0])
    ax.axis("off")
    ax.set_title("Camera-space occupancy heatmap")
    st.pyplot(fig)
    plt.close(fig)

# --- Top-down view (opt-in, requires calibration) ---
with st.expander("Top-down floor plan view (requires calibration)"):
    H = st.session_state.get("homography")
    if H is None:
        st.info("Complete Step 2 — Calibrate to enable the top-down view.")
    else:
        @st.cache_data
        def load_and_project(path, h_bytes):
            _H = np.frombuffer(h_bytes).reshape(3, 3)
            _df = pd.read_csv(path)
            proj = project_points(_df[["x", "y"]].values, _H)
            _df = _df.copy()
            _df["px"], _df["py"] = proj[:, 0], proj[:, 1]
            return _df[(_df["px"] >= 0) & (_df["px"] < FLOOR_W) &
                       (_df["py"] >= 0) & (_df["py"] < FLOOR_H)]

        df_proj = load_and_project(st.session_state["detections_path"], H.tobytes())
        floor_img = st.session_state.get("floor_plan_img")
        heatmap_td = build_heatmap(df_proj, "px", "py",
                                   [FLOOR_W, FLOOR_H], [[0, FLOOR_W], [0, FLOOR_H]], sigma=sigma)

        td_col1, td_col2 = st.columns(2)
        with td_col1:
            fig, ax = plt.subplots(figsize=(4, 7))
            if floor_img is not None:
                ax.imshow(floor_img)
            ax.imshow(heatmap_td, cmap=colormap, alpha=alpha, vmin=0, vmax=1,
                      extent=[0, FLOOR_W, FLOOR_H, 0])
            ax.axis("off")
            ax.set_title("Top-down floor plan")
            st.pyplot(fig)
            plt.close(fig)

        with td_col2:
            ref_frame = st.session_state.get("reference_frame")
            if ref_frame is not None:
                H_inv = np.linalg.inv(H)
                blended = backproject(heatmap_td, H_inv, ref_frame)
                st.image(blended, caption="Back-projected onto camera frame",
                         use_container_width=True)

# --- Export ---
out_dir = Path(__file__).resolve().parent.parent / "reports" / "figures"
out_dir.mkdir(parents=True, exist_ok=True)
if st.button("Export heatmap PNG"):
    fig, ax = plt.subplots(figsize=(8, 6))
    if bg is not None:
        ax.imshow(bg)
    ax.imshow(cam_heatmap, cmap=colormap, alpha=alpha, vmin=0, vmax=1,
              extent=[0, CAM_W, CAM_H, 0])
    ax.axis("off")
    out_path = out_dir / "heatmap_client.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    st.success(f"Saved to `{out_path}`")
