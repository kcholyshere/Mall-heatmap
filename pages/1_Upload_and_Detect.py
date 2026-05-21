import tempfile
import time
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st
from ultralytics import YOLO

st.set_page_config(page_title="Upload & Detect", layout="wide")
st.title("Step 1 — Upload Video & Run Detection")


@st.cache_resource
def load_model():
    model_path = Path(__file__).resolve().parent.parent / "models" / "yolov8n.pt"
    return YOLO(str(model_path))


if "detections_path" in st.session_state:
    st.info(f"Detections already loaded from `{st.session_state['detections_path']}`. "
            "Upload a new video to rerun.")

uploaded = st.file_uploader("Upload a surveillance video", type=["mp4", "avi", "mov", "mkv"])

if uploaded:
    suffix = Path(uploaded.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.read())
        video_path = tmp.name

    st.session_state["video_path"] = video_path
    st.success(f"Video saved: **{uploaded.name}**")

    if st.button("Run Detection"):
        model = load_model()
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        progress = st.progress(0, text="Detecting persons...")
        rows = []
        frame_id = 0
        start = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_id += 1
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
        out_path = Path(__file__).resolve().parent.parent / "data" / "interim" / "detections_client.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        st.session_state["detections_path"] = str(out_path)

        progress.empty()
        st.success(
            f"Done — **{len(df):,}** detections across **{frame_id}** frames in **{elapsed:.1f}s**"
        )
        st.dataframe(df.head(10))
        st.info("Proceed to **Step 2 — Calibrate** in the sidebar.")
