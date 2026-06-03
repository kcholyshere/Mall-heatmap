import hashlib
import tempfile
import time
from datetime import datetime
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from ultralytics import YOLO

from src.tracking import run_tracking

st.set_page_config(page_title="Upload & Detect", layout="wide")
st.title("Step 1 - Upload Video & Run Detection")


@st.cache_resource
def load_model():
    model_path = Path(__file__).resolve().parent.parent / "models" / "yolov8n.pt"
    return YOLO(str(model_path))


st.subheader("Recording start")
st.caption("When the footage begins - used to label time windows in Step 4.")
col1, col2 = st.columns(2)
with col1:
    ref_date = st.date_input("Recording date", value=datetime.today().date())
    st.session_state["ref_date"] = ref_date
with col2:
    ref_time = st.time_input("Start time of day", value=datetime.strptime("11:00", "%H:%M").time(), step=60)
    st.session_state["ref_time"] = ref_time

st.divider()

if "detections_path" in st.session_state:
    st.info(f"Detections already loaded from `{st.session_state['detections_path']}`. "
            "Upload a new video to rerun.")

uploaded = st.file_uploader(
    "Upload a surveillance video",
    type=["mp4", "avi", "mov", "mkv", "ts", "m4v", "wmv", "flv", "asf"])
local_path = st.text_input(
    "…or path to a video already on this machine",
    help="Best for large files - the uploader holds the file in memory (cap 5 GB), while this "
         "reads from disk. Accepts any format your system's ffmpeg can decode. Absolute or "
         "relative path.")
st.caption("Files over ~2 GB: use the local path above. Proprietary NVR formats (e.g. Dahua "
           ".dav, raw .264/.265) should be converted to .mp4 with ffmpeg first.")

video_path = None
if uploaded:
    suffix = Path(uploaded.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.read())
        video_path = tmp.name
    st.success(f"Video uploaded: **{uploaded.name}**")
elif local_path:
    if Path(local_path).exists():
        video_path = local_path
        st.success(f"Using local file: **{local_path}**")
    else:
        st.error(f"File not found: `{local_path}`")

if video_path:
    st.session_state["video_path"] = video_path

    # A new video means a new camera view - drop any prior calibration so a previous
    # camera's homography/floor plan can't bleed across (keyed on a stable file identity,
    # since the uploader writes a fresh temp path on every rerun).
    video_id = uploaded.name if uploaded else local_path
    if st.session_state.get("video_id") != video_id:
        st.session_state["video_id"] = video_id
        for k in ("homography", "reference_frame", "pairs", "pending_camera",
                  "floor_plan_img", "click_stage", "floor_size", "tracks_path"):
            st.session_state.pop(k, None)

    # Read frame rate and size straight from the file metadata - no manual entry needed.
    probe = cv2.VideoCapture(video_path)
    det_fps = probe.get(cv2.CAP_PROP_FPS)
    w = int(probe.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_meta = int(probe.get(cv2.CAP_PROP_FRAME_COUNT))
    probe.release()
    st.session_state["frame_size"] = (w, h)

    if det_fps and det_fps > 0:
        fps = det_fps
        st.caption(f"Detected from file: **{w}×{h}**, **{fps:.2f} fps**, {total_meta:,} frames "
                   f"(~{total_meta / fps / 60:.1f} min). Sampling 1 frame per second.")
    else:
        st.warning("Couldn't read the frame rate from this file - please enter it.")
        fps = st.number_input("Camera FPS", min_value=0.1, max_value=120.0, value=1.0, step=0.5)
    st.session_state["fps"] = fps

    # Each video gets its own artifact files (keyed on a stable id) so runs on different
    # videos don't overwrite one another, and a previously-processed video is reused as-is.
    interim = Path(__file__).resolve().parent.parent / "data" / "interim"
    vid_hash = hashlib.md5(video_id.encode()).hexdigest()[:8]
    det_path = interim / f"detections_{vid_hash}.csv"
    track_path = interim / f"tracks_{vid_hash}.csv"
    sample_interval = max(1, round(fps))

    enable_tracking = st.checkbox(
        "Enable person tracking (dense pass - slower, needs ~25-30 fps video)",
        help="Follows each person across frames (ByteTrack) to unlock unique footfall, "
             "dwell-time and trajectories in Step 3. Processes every frame, so it is much "
             "slower than the 1-frame-per-second detection used for the heatmap.")
    max_frames = None
    if enable_tracking:
        cap_s = st.number_input(
            "Limit to first N seconds (0 = whole video)", min_value=0, value=60, step=10,
            help="Caps a long clip so tracking stays quick for a demo.")
        max_frames = int(cap_s * fps) if cap_s else None
        st.warning("Keep this tab open until tracking finishes - navigating away cancels it "
                   "and you'll need to re-run.")

    if enable_tracking:
        if track_path.exists():
            st.session_state["tracks_path"] = str(track_path)
            st.session_state["detections_path"] = str(det_path)
            # Tracking may have been capped to the first N seconds, so the processed length
            # is the last tracked frame, not the full video.
            st.session_state["total_frames"] = int(pd.read_csv(track_path)["frame_id"].max())
            st.success("This video was already tracked - using cached tracks. "
                       "Open the Heatmap or Tracking page in the sidebar.")
            with st.expander("Re-process this video"):
                run_requested = st.button("Re-run tracking")
        else:
            run_requested = st.button("Run Tracking")
    elif det_path.exists():
        # Already processed - load automatically; don't make the user run detection again.
        st.session_state.pop("tracks_path", None)
        st.session_state["detections_path"] = str(det_path)
        st.session_state["total_frames"] = total_meta
        st.success("This video was already processed - using cached detections. "
                   "Open the Heatmap page in the sidebar.")
        with st.expander("Re-process this video"):
            run_requested = st.button("Re-run detection")
    else:
        st.session_state.pop("tracks_path", None)
        run_requested = st.button("Run Detection")

    if run_requested and enable_tracking:
        model = load_model()
        target = max_frames or total_meta or 0
        progress = st.progress(0, text="Tracking persons...")

        def _cb(fid):
            if target:
                progress.progress(min(fid / target, 1.0), text=f"Frame {fid} / {target}")

        start = time.time()
        tracks_df = run_tracking(video_path, model, max_frames=max_frames, progress_cb=_cb)
        elapsed = time.time() - start

        # Derive the 1-fps detections the Heatmap/Time pages expect from the dense tracks,
        # so those pages keep working unchanged (same schema, same people/min math).
        det_df = tracks_df[(tracks_df["frame_id"] - 1) % sample_interval == 0][
            ["frame_id", "x", "y", "confidence"]]
        interim.mkdir(parents=True, exist_ok=True)
        tracks_df.to_csv(track_path, index=False)
        det_df.to_csv(det_path, index=False)
        st.session_state["tracks_path"] = str(track_path)
        st.session_state["detections_path"] = str(det_path)
        processed = int(tracks_df["frame_id"].max()) if not tracks_df.empty else 0
        st.session_state["total_frames"] = processed

        progress.empty()
        st.success(
            f"Done - tracked **{tracks_df['track_id'].nunique():,}** unique people across "
            f"**{processed:,}** frames in **{elapsed:.1f}s**"
        )
        st.info("Open the Tracking page (Step 3) in the sidebar.")
        # Long runs let the user look away - chime + desktop notification on completion.
        # Both fire while the tab is backgrounded; the chime is the reliable part (some
        # browsers limit notification permission requests inside component iframes).
        components.html(
            """
            <script>
            try {
              const ctx = new (window.AudioContext || window.webkitAudioContext)();
              const o = ctx.createOscillator(), g = ctx.createGain();
              o.connect(g); g.connect(ctx.destination);
              o.frequency.value = 880; g.gain.value = 0.1;
              o.start(); o.stop(ctx.currentTime + 0.3);
            } catch (e) {}
            if ("Notification" in window) {
              const show = () => new Notification("Tracking complete",
                  {body: "Your video has been tracked - open Step 3."});
              if (Notification.permission === "granted") { show(); }
              else if (Notification.permission !== "denied") {
                Notification.requestPermission().then(p => { if (p === "granted") show(); });
              }
            }
            </script>
            """,
            height=0,
        )

    elif run_requested:
        model = load_model()
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # Sample 1 frame per second; frame_id still increments for every frame
        # so downstream time calculations remain correct.
        sample_interval = max(1, round(fps))
        progress = st.progress(0, text="Detecting persons...")
        rows = []
        frame_id = 0
        start = time.time()

        while cap.isOpened():
            ret = cap.grab()
            if not ret:
                break
            frame_id += 1
            if (frame_id - 1) % sample_interval != 0:
                if total > 0:
                    progress.progress(frame_id / total, text=f"Frame {frame_id} / {total}")
                continue
            ret, frame = cap.retrieve()
            if not ret:
                continue
            results = model(frame, classes=[0], verbose=False)
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                rows.append({
                    "frame_id": frame_id,
                    "x": (x1 + x2) / 2,
                    "y": y2,
                    "confidence": round(box.conf[0].item(), 4),
                })
            if total > 0:
                progress.progress(frame_id / total, text=f"Frame {frame_id} / {total}")

        cap.release()
        elapsed = time.time() - start

        df = pd.DataFrame(rows)
        det_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(det_path, index=False)
        st.session_state["detections_path"] = str(det_path)
        st.session_state["total_frames"] = frame_id

        progress.empty()
        sampled = total // sample_interval if total > 0 else len(df)
        st.success(
            f"Done - **{len(df):,}** detections from **{sampled:,}** sampled frames "
            f"(1 per {sample_interval}) in **{elapsed:.1f}s**"
        )
        st.info("Open the Heatmap page in the sidebar.")
