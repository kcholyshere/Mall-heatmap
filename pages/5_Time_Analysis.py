from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.heatmap import (CAM_H, CAM_W, FLOOR_H, FLOOR_W,
                         build_heatmap, compute_confidence_cutoff,
                         pick_background_frame, project_points, sampled_frame_count)
from src.tracking import compute_dwell, footfall_stats, moving_track_ids

st.set_page_config(page_title="Time Analysis", layout="wide")
st.title("Step 4 - Time-Based Analysis")


@st.cache_data
def get_background_frame(video_path, detections_path, fps, total_frames):
    return pick_background_frame(video_path, detections_path, fps, total_frames)

if "detections_path" not in st.session_state:
    st.warning("Complete Step 1 first.")
    st.stop()

fps = st.session_state.get("fps", 1.0)
ref_date = st.session_state.get("ref_date", datetime.today().date())
ref_time = st.session_state.get("ref_time", datetime.strptime("11:00", "%H:%M").time())
ref_dt = datetime.combine(ref_date, ref_time)

st.caption(f"Reference: frame 1 = {ref_dt.strftime('%Y-%m-%d %H:%M')} at {fps} FPS - change in Step 1.")

# End of footage in wall-clock time (+1s margin so the last sampled frame stays reachable).
total_frames = st.session_state.get("total_frames")
rec_end_dt = ref_dt + timedelta(seconds=total_frames / fps + 1) if total_frames else None
clamp_end = rec_end_dt is not None and (rec_end_dt - ref_dt) < timedelta(days=1)

sigma = st.slider("Smoothing (sigma)", 5, 80, 30)
colormap = st.selectbox("Colormap", ["hot", "jet", "plasma", "YlOrRd"])
relative = st.checkbox(
    "Relative scale (0-1)", value=False,
    help="Off: average people present per sampled frame (honest occupancy). "
         "On: relative 0-1 (clean shape, comparable across windows).")
conf_threshold = st.slider(
    "Confidence threshold", 0.1, 0.9, 0.5, step=0.05,
    help="Minimum YOLO confidence to count a detection. Raising it removes weak/false "
         "detections (e.g. a static object mistaken for a person) but also drops "
         "uncertain far-field people.")

col_start, col_end = st.columns(2)
with col_start:
    st.markdown("**Start**")
    start_date = st.date_input("Start date", value=ref_date, key="start_date")
    start_time = st.time_input("Start time", value=ref_time, step=60, key="start_time")
with col_end:
    st.markdown("**End**")
    default_end = min(ref_dt + timedelta(minutes=30), rec_end_dt) if rec_end_dt else ref_dt + timedelta(minutes=30)
    end_date = st.date_input("End date", value=default_end.date(), min_value=ref_date,
                             max_value=rec_end_dt.date() if clamp_end else None, key="end_date")
    end_time = st.time_input("End time", value=default_end.time(), step=60, key="end_time")

start_dt = datetime.combine(start_date, start_time)
end_dt   = datetime.combine(end_date, end_time)

# time_input has no max, so clamp the end to the recording's end (when the clip is < 1 day).
if clamp_end and end_dt > rec_end_dt:
    end_dt = rec_end_dt
    st.caption(f"End capped to the end of the recording ({rec_end_dt.strftime('%Y-%m-%d %H:%M')}).")

if start_dt >= end_dt:
    st.error("End must be after start.")
    st.stop()

start_frame = max(1, int((start_dt - ref_dt).total_seconds() * fps) + 1)
end_frame   = max(1, int((end_dt   - ref_dt).total_seconds() * fps) + 1)

df_raw = pd.read_csv(st.session_state["detections_path"])
window_raw = df_raw[(df_raw["frame_id"] >= start_frame) & (df_raw["frame_id"] <= end_frame)]

st.write(f"Frame range: **{start_frame}-{end_frame}** | Detections in window: **{len(window_raw):,}**")

if len(window_raw) == 0:
    st.warning("No detections in this window. Try a wider time range.")
    st.stop()

# --- Camera-space heatmap (default) ---
n_sampled = sampled_frame_count(
    st.session_state.get("total_frames") or int(df_raw["frame_id"].max()),
    fps, start_frame, end_frame)

video_path = st.session_state.get("video_path")
total_frames = st.session_state.get("total_frames") or int(df_raw["frame_id"].max())
bg = get_background_frame(video_path, st.session_state["detections_path"], fps, total_frames) if video_path else None
if bg is not None:
    frame_h, frame_w = bg.shape[:2]
else:
    frame_w, frame_h = st.session_state.get("frame_size", (CAM_W, CAM_H))

window_plot = window_raw[window_raw["confidence"] >= conf_threshold] if "confidence" in window_raw.columns else window_raw
raw_heatmap = build_heatmap(window_plot, "x", "y", [frame_w, frame_h], [[0, frame_w], [0, frame_h]],
                            sigma=sigma, normalise=False)
if relative:
    cam_heatmap = raw_heatmap / raw_heatmap.max() if raw_heatmap.max() > 0 else raw_heatmap
    cbar_label = "relative occupancy"
    vmax = 1
else:
    cam_heatmap = raw_heatmap / n_sampled
    cbar_label = "avg people present"
    vmax = cam_heatmap.max() if cam_heatmap.max() > 0 else 1

cutoff_y = compute_confidence_cutoff(window_raw, threshold=conf_threshold, frame_h=frame_h)

fig, ax = plt.subplots(figsize=(8, 6))
if bg is not None:
    ax.imshow(bg)
im = ax.imshow(cam_heatmap, cmap=colormap, alpha=0.6, vmin=0, vmax=vmax,
               extent=[0, frame_w, frame_h, 0])
plt.colorbar(im, ax=ax, shrink=0.6, label=cbar_label)
if cutoff_y is not None:
    ax.axhline(cutoff_y, color="white", linestyle="--", linewidth=1.5, alpha=0.8)
    ax.text(5, cutoff_y - 4, f"Reliability boundary (conf ≥ {conf_threshold:.2f})",
            color="white", fontsize=8, va="bottom")
ax.set_title(f"{start_dt.strftime('%Y-%m-%d %H:%M')} - {end_dt.strftime('%H:%M')}")
ax.axis("off")
st.pyplot(fig)
plt.close(fig)

# --- Tracking insights for this window (only when tracking was run in Step 1) ---
if "tracks_path" in st.session_state:
    st.divider()
    st.subheader("Tracking - this time window")
    tracks_all = pd.read_csv(st.session_state["tracks_path"])
    # Gate tracking on the clip's median confidence (as Step 3 does), not the heatmap's 0.5
    # slider - so the two pages report the same footfall, and low-confidence far-field people
    # aren't dropped. Static false positives are handled by the movement filter below.
    if "confidence" in tracks_all.columns:
        track_conf = round(min(0.9, max(0.1, tracks_all["confidence"].median())) / 0.05) * 0.05
        tracks_all = tracks_all[tracks_all["confidence"] >= track_conf]
    # Decide static-ness on the whole clip, not the window: a real person seen only briefly
    # in a short window could look static within it.
    keep = moving_track_ids(tracks_all)
    wt = tracks_all[(tracks_all["frame_id"] >= start_frame) &
                    (tracks_all["frame_id"] <= end_frame) &
                    (tracks_all["track_id"].isin(keep))]
    if wt.empty:
        st.info("No tracked people in this window.")
    else:
        window_frames = end_frame - start_frame + 1
        w_unique, w_throughput, w_durations = footfall_stats(wt, fps, window_frames)
        m1, m2, m3 = st.columns(3)
        m1.metric("Unique visitors", f"{w_unique:,}")
        m2.metric("Throughput", f"{w_throughput:.1f} people/min")
        m3.metric("Mean time in scene", f"{w_durations.mean():.1f} s")
        st.caption("De-duplicated via track IDs, with static objects removed - a genuine "
                   "footfall rate for the selected window.")
        grid_n = st.slider("Dwell grid resolution (cells across)", 8, 48, 24, step=4,
                           key="ta_dwell_grid")
        grid_rows = max(1, round(grid_n * frame_h / frame_w))
        dwell = compute_dwell(wt, fps, grid_n, grid_rows, frame_w, frame_h)
        fig, ax = plt.subplots(figsize=(8, 6))
        if bg is not None:
            ax.imshow(bg)
        im = ax.imshow(np.ma.masked_equal(dwell, 0), cmap="hot", alpha=0.65,
                       extent=[0, frame_w, frame_h, 0], vmin=0)
        plt.colorbar(im, ax=ax, shrink=0.6, label="mean dwell (s)")
        ax.axis("off")
        ax.set_title(f"Dwell-time {start_dt.strftime('%H:%M')} - {end_dt.strftime('%H:%M')}")
        st.pyplot(fig)
        plt.close(fig)

# --- Top-down view (opt-in) ---
with st.expander("Top-down floor plan view (requires calibration)"):
    H = st.session_state.get("homography")
    if H is None:
        st.info("Calibrate on the Heatmap page (tick 'Advanced - top-down floor-plan view') "
                "to enable the top-down view here.")
    else:
        floor_w, floor_h = st.session_state.get("floor_size", (FLOOR_W, FLOOR_H))
        td_sigma = st.slider(
            "Top-down smoothing", 2, 40, 12, key="td_sigma_time",
            help="Separate from the camera-view smoothing - the floor plan is lower-resolution, "
                 "so it needs a smaller blur to avoid over-smoothing.")

        @st.cache_data
        def load_and_project(path, h_bytes, s_frame, e_frame, fw, fh, conf):
            _H = np.frombuffer(h_bytes).reshape(3, 3)
            _df = pd.read_csv(path)
            _df = _df[(_df["frame_id"] >= s_frame) & (_df["frame_id"] <= e_frame)]
            if "confidence" in _df.columns:
                _df = _df[_df["confidence"] >= conf]
            proj = project_points(_df[["x", "y"]].values, _H)
            _df = _df.copy()
            _df["px"], _df["py"] = proj[:, 0], proj[:, 1]
            return _df[(_df["px"] >= 0) & (_df["px"] < fw) &
                       (_df["py"] >= 0) & (_df["py"] < fh)]

        df_proj = load_and_project(
            st.session_state["detections_path"], H.tobytes(), start_frame, end_frame,
            floor_w, floor_h, conf_threshold
        )
        floor_img = st.session_state.get("floor_plan_img")
        heatmap_td = build_heatmap(df_proj, "px", "py",
                                   [floor_w, floor_h], [[0, floor_w], [0, floor_h]], sigma=td_sigma)
        fig, ax = plt.subplots(figsize=(5, 5 * floor_h / floor_w))
        if floor_img is not None:
            ax.imshow(floor_img)
        ax.imshow(heatmap_td, cmap=colormap, alpha=0.6, vmin=0, vmax=1,
                  extent=[0, floor_w, floor_h, 0])
        ax.set_title(f"{start_dt.strftime('%Y-%m-%d %H:%M')} - {end_dt.strftime('%H:%M')}")
        ax.axis("off")
        st.pyplot(fig)
        plt.close(fig)
