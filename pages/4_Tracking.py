import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.collections import LineCollection

from src.heatmap import CAM_H, CAM_W, pick_background_frame
from src.tracking import (build_trajectories, compute_dwell, footfall_stats,
                          filter_static_tracks)

st.set_page_config(page_title="Tracking", layout="wide")
st.title("Step 3 - Tracking Insights")

if "tracks_path" not in st.session_state:
    st.warning("Run Step 1 with **person tracking enabled** first - the heatmap detection "
               "alone does not follow people across frames.")
    st.stop()


@st.cache_data
def load_tracks(path):
    return pd.read_csv(path)


@st.cache_data
def get_background_frame(video_path, detections_path, fps, total_frames):
    return pick_background_frame(video_path, detections_path, fps, total_frames)


fps = st.session_state.get("fps", 1.0)
df_all = load_tracks(st.session_state["tracks_path"])
if df_all.empty:
    st.error("No people were tracked in this video.")
    st.stop()
total_frames = st.session_state.get("total_frames") or int(df_all["frame_id"].max())

video_path = st.session_state.get("video_path")
bg = (get_background_frame(video_path, st.session_state["detections_path"], fps, total_frames)
      if video_path else None)
if bg is not None:
    frame_h, frame_w = bg.shape[:2]
else:
    frame_w, frame_h = st.session_state.get("frame_size", (CAM_W, CAM_H))

with st.expander("Advanced settings"):
    # Static objects (parking locks, signs, bins) get tagged as people at the low confidence
    # tracking needs - but they don't move, while people do. Drop tracks that barely move.
    hide_static = st.checkbox(
        "Hide static objects", value=True,
        help="Removes tracks that stay put for their whole life (furniture mistaken for a person). "
             "People walk; a parking lock or sign does not. Disable to see every track.")
    if hide_static:
        min_move = st.slider(
            "Minimum movement (body-lengths)", 0.1, 2.0, 0.5, step=0.1,
            help="A track is kept only if its foot point travels at least this many of its own "
                 "heights. Lower = stricter about calling something static.")
    # Default to this clip's median detection confidence: far-field subjects are detected at
    # lower confidence, so a fixed default would hide most of them on distant footage.
    default_conf = round(min(0.9, max(0.1, df_all["confidence"].median())) / 0.05) * 0.05
    conf_threshold = st.slider(
        "Detection sensitivity", 0.1, 0.9, round(default_conf, 2), step=0.05,
        help="Lower keeps more uncertain / far-away detections; higher keeps only strong ones. "
             "Defaults to this clip's median, since distant people are detected less confidently.")
df = df_all[df_all["confidence"] >= conf_threshold]
if df.empty:
    st.error("No tracked detections above this confidence threshold - lower it to see results.")
    st.stop()

if hide_static:
    n_before = df["track_id"].nunique()
    df = filter_static_tracks(df, min_disp_per_height=min_move)
    removed = n_before - df["track_id"].nunique()
    if "h" not in df_all.columns:
        st.info("These tracks were recorded before static filtering existed - re-run tracking "
                "in Step 1 to enable it.")
    elif removed:
        st.caption(f"Removed **{removed}** static object(s).")
    if df.empty:
        st.error("Every track looked static at this setting - raise the threshold or disable "
                 "the filter.")
        st.stop()

# --- A. Footfall / throughput (de-duplicated via track IDs) ---
st.header("Footfall")
unique, throughput, durations = footfall_stats(df, fps, total_frames)
c1, c2, c3 = st.columns(3)
c1.metric("Unique visitors", f"{unique:,}")
c2.metric("Throughput", f"{throughput:.1f} people/min")
c3.metric("Mean time in scene", f"{durations.mean():.1f} s")
st.caption("Each person is counted once via their track ID, so this is a genuine footfall "
           "rate - unlike the heatmap's scene-wide average occupancy.")

# --- B. Trajectories ---
st.divider()
st.header("Trajectories")
max_len_s = max(0.2, round(df.groupby("track_id").size().max() / fps, 1))
min_secs = st.slider("Minimum track length (seconds)", 0.0, max_len_s, min(0.3, max_len_s),
                     step=0.1, help="Hide very short tracks - usually flicker or false positives.")
trajectories = build_trajectories(df, min_len=max(1, int(min_secs * fps)))

fig, ax = plt.subplots(figsize=(8, 6))
if bg is not None:
    ax.imshow(bg)
cmap = plt.colormaps["tab20"]
segments = [path for _, path in trajectories if len(path) >= 2]
colors = [cmap(i % 20) for i in range(len(segments))]
ax.add_collection(LineCollection(segments, colors=colors, linewidths=1.5, alpha=0.8))
ax.set_xlim(0, frame_w)
ax.set_ylim(frame_h, 0)
ax.axis("off")
ax.set_title(f"{len(trajectories)} person paths")
st.pyplot(fig)
buf = io.BytesIO()
fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
plt.close(fig)
st.download_button("Download trajectories PNG", buf.getvalue(),
                   file_name="trajectories.png", mime="image/png", key="dl_traj")

# --- C. Dwell-time ---
st.divider()
st.header("Dwell-time")
st.caption("Average seconds a person spends in each cell. Hot cells are where people linger.")
grid_n = st.slider("Detail (grid cells across)", 8, 48, 24, step=4,
                   help="How finely the scene is divided for dwell-time. Higher = finer detail.")
grid_rows = max(1, round(grid_n * frame_h / frame_w))
dwell = compute_dwell(df, fps, grid_n, grid_rows, frame_w, frame_h)

fig, ax = plt.subplots(figsize=(8, 6))
if bg is not None:
    ax.imshow(bg)
im = ax.imshow(np.ma.masked_equal(dwell, 0), cmap="hot", alpha=0.65,
               extent=[0, frame_w, frame_h, 0], vmin=0)
plt.colorbar(im, ax=ax, shrink=0.6, label="mean dwell (s)")
ax.axis("off")
ax.set_title("Dwell-time per cell")
st.pyplot(fig)
buf = io.BytesIO()
fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
plt.close(fig)
st.download_button("Download dwell-time PNG", buf.getvalue(),
                   file_name="dwell_time.png", mime="image/png", key="dl_dwell")

# --- D. Time-in-scene distribution (least prominent) ---
st.divider()
st.subheader("Time-in-scene distribution")
fig, ax = plt.subplots(figsize=(5, 2))
ax.hist(durations.values, bins=20, color="#3b7dd8")
ax.set_xlabel("seconds in scene")
ax.set_ylabel("people")
st.pyplot(fig)
plt.close(fig)
