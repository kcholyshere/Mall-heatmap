import io

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from calibration_panel import render_calibration
from src.heatmap import (CAM_H, CAM_W, FLOOR_H, FLOOR_W,
                         backproject, build_heatmap, build_reliability_mask,
                         pick_background_frame, project_points, sampled_frame_count)

# Off-palette pink for the model-reliability overlay - deliberately unlike any heatmap colourmap.
RELIABILITY_COLOR = "#FF2BD6"


def overlay_reliability(ax, mask, frame_w, frame_h):
    """Mark the low-reliability zone with diagonal hatching (a texture, not a heat blob, so it
    can't be confused with the heatmap) plus a thin boundary, in an off-palette pink."""
    rows, cols = mask.shape
    xs = np.linspace(0, frame_w, cols)
    ys = np.linspace(0, frame_h, rows)
    with mpl.rc_context({"hatch.color": RELIABILITY_COLOR, "hatch.linewidth": 0.9}):
        ax.contourf(xs, ys, mask, levels=[0.5, 2.0], colors="none", hatches=["//"], zorder=5)
    ax.contour(xs, ys, mask, levels=[0.5], colors=RELIABILITY_COLOR, linewidths=1.2, zorder=6)
    ax.set_xlim(0, frame_w)
    ax.set_ylim(frame_h, 0)

st.set_page_config(page_title="Heatmap", layout="wide")
st.title("Step 2 - Heatmap")

if "detections_path" not in st.session_state:
    st.warning("Complete Step 1 first.")
    st.stop()


@st.cache_data
def load_detections(path):
    return pd.read_csv(path)


@st.cache_data
def get_background_frame(video_path, detections_path, fps, total_frames):
    return pick_background_frame(video_path, detections_path, fps, total_frames)


df_raw = load_detections(st.session_state["detections_path"])
st.write(f"**{len(df_raw):,}** detections across **{df_raw['frame_id'].nunique():,}** sampled frames")

col_ctrl, col_view = st.columns([1, 2])

with col_ctrl:
    sigma = st.slider("Smoothing (sigma)", 5, 80, 30)
    colormap = st.selectbox("Colormap", ["hot", "jet", "plasma", "YlOrRd"])
    alpha = st.slider("Heatmap opacity", 0.1, 1.0, 0.6, step=0.05)
    relative = st.checkbox(
        "Relative scale (0-1)", value=False,
        help="Off: average people present per sampled frame (honest occupancy). "
             "On: relative 0-1 (clean shape, comparable across clips).")
    conf_threshold = st.slider(
        "Confidence threshold", 0.1, 0.9, 0.5, step=0.05,
        help="Minimum YOLO confidence to count a detection. Raising it removes weak/false "
             "detections (e.g. a static object mistaken for a person) but also drops "
             "uncertain far-field people.")
    show_reliability = st.checkbox(
        "Show model-reliability overlay", value=False,
        help="Shades zones where YOLO under-detects (far field and weak-confidence "
             "cells). Absence of heat in a shaded zone does NOT mean the area is empty - "
             "the model likely cannot see people there. Driven by the confidence threshold.")

# --- Camera-space heatmap (default, no calibration needed) ---
fps = st.session_state.get("fps", 1.0)
total_frames = st.session_state.get("total_frames") or df_raw["frame_id"].max()
n_sampled = sampled_frame_count(total_frames, fps)

video_path = st.session_state.get("video_path")
bg = get_background_frame(video_path, st.session_state["detections_path"], fps, total_frames) if video_path else None
# Frame size = the displayed background's own dimensions (guarantees the heatmap aligns);
# fall back to session, then the Mall default.
if bg is not None:
    frame_h, frame_w = bg.shape[:2]
else:
    frame_w, frame_h = st.session_state.get("frame_size", (CAM_W, CAM_H))

# Only count detections the model is at least `conf_threshold` sure about. This drops weak
# false positives (e.g. a static object repeatedly mistaken for a person) from the heatmap;
# the reliability overlay below is still computed on the full set to show where we discarded.
df_plot = df_raw[df_raw["confidence"] >= conf_threshold] if "confidence" in df_raw.columns else df_raw
raw_heatmap = build_heatmap(df_plot, "x", "y", [frame_w, frame_h], [[0, frame_w], [0, frame_h]],
                            sigma=sigma, normalise=False)
if relative:
    cam_heatmap = raw_heatmap / raw_heatmap.max() if raw_heatmap.max() > 0 else raw_heatmap
    cbar_label = "relative occupancy"
    vmax = 1
else:
    cam_heatmap = raw_heatmap / n_sampled
    cbar_label = "occupancy"
    vmax = cam_heatmap.max() if cam_heatmap.max() > 0 else 1

avg_in_view = len(df_plot) / n_sampled
reliability_mask = (build_reliability_mask(df_raw, threshold=conf_threshold, frame_w=frame_w, frame_h=frame_h)
                    if show_reliability else None)

with col_view:
    if not relative:
        st.metric("Average people in view", f"{avg_in_view:.1f}")
    fig, ax = plt.subplots(figsize=(8, 6))
    if bg is not None:
        ax.imshow(bg)
    im = ax.imshow(cam_heatmap, cmap=colormap, alpha=alpha, vmin=0, vmax=vmax,
                   extent=[0, frame_w, frame_h, 0])
    cbar = plt.colorbar(im, ax=ax, shrink=0.6, label=cbar_label)
    if not relative:
        cbar.set_ticks([0, vmax])
        cbar.set_ticklabels(["low", "high"])
    if reliability_mask is not None:
        overlay_reliability(ax, reliability_mask, frame_w, frame_h)
    ax.axis("off")
    ax.set_title("Camera-space occupancy heatmap")
    st.pyplot(fig)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    st.download_button("Download heatmap PNG", buf.getvalue(),
                       file_name="heatmap.png", mime="image/png", key="dl_camera")
    if not relative:
        st.caption("Colour shows where people concentrate (low -> high); the "
                   "'Average people in view' figure above is the honest scene-wide magnitude. "
                   "For a genuine footfall *rate*, use the Tracking page (Step 3).")
    if show_reliability:
        st.caption("Pink hatched zones - low model reliability (far field / weak confidence). "
                   "Occupancy here is under-counted; an un-hatched-but-empty area is genuinely empty, "
                   "a hatched-but-empty area may simply be a model blind spot.")

# --- Advanced: top-down floor-plan view (optional) ---
# Gated by a checkbox (not an expander) so the calibration UI survives the per-click reruns.
st.divider()
if st.checkbox("Advanced - top-down floor-plan view (requires calibration)"):
    if st.session_state.get("homography") is None:
        st.caption("Map the camera view to a floor plan below. Optional - the camera-space "
                   "heatmap above is the main output.")
        render_calibration()

    H = st.session_state.get("homography")
    if H is not None:
        floor_w, floor_h = st.session_state.get("floor_size", (FLOOR_W, FLOOR_H))
        td_sigma = st.slider(
            "Top-down smoothing", 2, 40, 12, key="td_sigma",
            help="Separate from the camera-view smoothing - the floor plan is lower-resolution, "
                 "so it needs a smaller blur to avoid the over-smoothing the camera value causes.")

        @st.cache_data
        def load_and_project(path, h_bytes, fw, fh, conf):
            _H = np.frombuffer(h_bytes).reshape(3, 3)
            _df = pd.read_csv(path)
            if "confidence" in _df.columns:
                _df = _df[_df["confidence"] >= conf]
            proj = project_points(_df[["x", "y"]].values, _H)
            _df = _df.copy()
            _df["px"], _df["py"] = proj[:, 0], proj[:, 1]
            return _df[(_df["px"] >= 0) & (_df["px"] < fw) &
                       (_df["py"] >= 0) & (_df["py"] < fh)]

        df_proj = load_and_project(st.session_state["detections_path"], H.tobytes(),
                                   floor_w, floor_h, conf_threshold)
        floor_img = st.session_state.get("floor_plan_img")
        heatmap_td = build_heatmap(df_proj, "px", "py",
                                   [floor_w, floor_h], [[0, floor_w], [0, floor_h]], sigma=td_sigma)

        td_col1, td_col2 = st.columns(2)
        with td_col1:
            fig, ax = plt.subplots(figsize=(5, 5 * floor_h / floor_w))
            if floor_img is not None:
                ax.imshow(floor_img)
            ax.imshow(heatmap_td, cmap=colormap, alpha=alpha, vmin=0, vmax=1,
                      extent=[0, floor_w, floor_h, 0])
            ax.axis("off")
            ax.set_title("Top-down floor plan")
            st.pyplot(fig)
            td_buf = io.BytesIO()
            fig.savefig(td_buf, format="png", bbox_inches="tight", dpi=150)
            plt.close(fig)
            st.download_button("Download top-down PNG", td_buf.getvalue(),
                               file_name="heatmap_topdown.png", mime="image/png", key="dl_topdown")

        with td_col2:
            ref_frame = st.session_state.get("reference_frame")
            if ref_frame is not None:
                H_inv = np.linalg.inv(H)
                blended = backproject(heatmap_td, H_inv, ref_frame)
                st.image(blended, caption="Back-projected onto camera frame",
                         width="stretch")

        if st.button("Recalibrate"):
            for k in ("homography", "pairs", "pending_camera", "click_stage"):
                st.session_state.pop(k, None)
            st.session_state["force_recalibrate"] = True  # don't let the disk copy reload
            st.rerun()
