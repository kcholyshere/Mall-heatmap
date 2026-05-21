from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates

from src.calibration import compute_homography, extract_reference_frame, reprojection_errors
from src.heatmap import FLOOR_H, FLOOR_W, make_floor_canvas

st.set_page_config(page_title="Calibrate", layout="wide")
st.title("Step 2 — Calibrate Homography")

if "video_path" not in st.session_state:
    st.warning("Complete Step 1 first.")
    st.stop()

# Initialise session state
for key, default in [
    ("pairs", []),
    ("click_stage", "camera"),
    ("pending_camera", None),
    ("cam_nonce", 0),
    ("floor_nonce", 0),
    ("homography", None),
    ("floor_plan_img", None),
    ("reference_frame", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# --- Floor plan ---
st.subheader("Floor Plan")
floor_upload = st.file_uploader(
    "Upload a top-down architectural floor plan (leave blank to use the generated canvas)",
    type=["png", "jpg", "jpeg"],
)
if floor_upload:
    img = Image.open(floor_upload).convert("RGB").resize((FLOOR_W, FLOOR_H))
    st.session_state["floor_plan_img"] = np.array(img)
elif st.session_state["floor_plan_img"] is None:
    st.session_state["floor_plan_img"] = make_floor_canvas()

# --- Reference frame ---
if st.session_state["reference_frame"] is None:
    with st.spinner("Extracting reference frame from video..."):
        frame = extract_reference_frame(st.session_state["video_path"])
    if frame is None:
        st.error("Could not extract a frame from the video.")
        st.stop()
    st.session_state["reference_frame"] = frame


def _annotate(img, points, highlight=None):
    out = img.copy()
    for i, (x, y) in enumerate(points):
        cv2.circle(out, (int(x), int(y)), 8, (0, 200, 0), -1)
        cv2.putText(out, f"P{i + 1}", (int(x) + 10, int(y) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)
    if highlight is not None:
        cv2.circle(out, (int(highlight[0]), int(highlight[1])), 12, (255, 165, 0), 3)
    return out


cam_pts = [p["camera"] for p in st.session_state["pairs"]]
floor_pts = [p["floor"] for p in st.session_state["pairs"]]
cam_annotated = _annotate(st.session_state["reference_frame"], cam_pts,
                           highlight=st.session_state["pending_camera"])
floor_annotated = _annotate(st.session_state["floor_plan_img"], floor_pts)

# --- Click UI ---
st.subheader("Point Correspondence")
n = len(st.session_state["pairs"])
st.write(f"Pairs collected: **{n}** (minimum 4 required, more improves accuracy)")

stage = st.session_state["click_stage"]
if stage == "camera":
    st.info("Click a reference point on the **camera frame** (left)")
else:
    st.info("Now click the corresponding position on the **floor plan** (right)")

col1, col2 = st.columns(2)

with col1:
    st.caption("Camera frame")
    if stage == "camera":
        val = streamlit_image_coordinates(
            Image.fromarray(cam_annotated),
            key=f"cam_{st.session_state['cam_nonce']}",
        )
        if val:
            st.session_state["pending_camera"] = [val["x"], val["y"]]
            st.session_state["cam_nonce"] += 1
            st.session_state["click_stage"] = "floor"
            st.rerun()
    else:
        st.image(cam_annotated, use_container_width=True)

with col2:
    st.caption("Floor plan")
    if stage == "floor":
        val = streamlit_image_coordinates(
            Image.fromarray(floor_annotated),
            key=f"floor_{st.session_state['floor_nonce']}",
        )
        if val:
            st.session_state["pairs"].append({
                "camera": st.session_state["pending_camera"],
                "floor": [val["x"], val["y"]],
            })
            st.session_state["pending_camera"] = None
            st.session_state["floor_nonce"] += 1
            st.session_state["click_stage"] = "camera"
            st.rerun()
    else:
        st.image(floor_annotated, use_container_width=True)

# --- Pairs table + undo ---
if st.session_state["pairs"]:
    df = pd.DataFrame([
        {"#": i + 1,
         "Camera X": round(p["camera"][0]), "Camera Y": round(p["camera"][1]),
         "Floor X": round(p["floor"][0]), "Floor Y": round(p["floor"][1])}
        for i, p in enumerate(st.session_state["pairs"])
    ])
    st.dataframe(df, hide_index=True)

    if st.button("Undo last pair"):
        st.session_state["pairs"].pop()
        st.session_state["click_stage"] = "camera"
        st.session_state["pending_camera"] = None
        st.rerun()

# --- Compute homography ---
if len(st.session_state["pairs"]) >= 4:
    if st.button("Compute Homography"):
        src = np.float32([p["camera"] for p in st.session_state["pairs"]])
        dst = np.float32([p["floor"] for p in st.session_state["pairs"]])
        H, _ = compute_homography(src, dst)
        errors = reprojection_errors(src, dst, H)
        st.session_state["homography"] = H

        err_df = pd.DataFrame({
            "Point": [f"P{i + 1}" for i in range(len(errors))],
            "Reprojection error (px)": [f"{e:.1f}" for e in errors],
        })
        st.dataframe(err_df, hide_index=True)

        worst = errors.max()
        if worst > 20:
            st.warning(
                f"Max error {worst:.1f}px — consider adding more points "
                f"or repositioning P{int(errors.argmax()) + 1}."
            )
        else:
            st.success(f"Homography computed. Max reprojection error: **{worst:.1f}px**. "
                       "Proceed to Step 3.")

        out = Path(__file__).resolve().parent.parent / "data" / "processed" / "homography_session.npy"
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out, H)
