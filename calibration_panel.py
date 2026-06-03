"""Inline camera → floor-plan calibration UI.

Rendered as an advanced section on the Heatmap page (no longer a top-level sidebar page),
since the camera-space heatmap is the core deliverable and calibration is optional.
"""
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from streamlit_image_coordinates import streamlit_image_coordinates

from src.calibration import compute_homography, extract_reference_frame, reprojection_errors
from src.heatmap import CAM_H, CAM_W, make_blank_canvas

DISPLAY_W = 720  # width (px) the click images are shown at; aspect is preserved by the widget
PROCESSED = Path(__file__).resolve().parent / "data" / "processed"


def _annotate(img, points, highlight=None):
    out = img.copy()
    # Markers are drawn at full resolution but shown at DISPLAY_W, so scale them up to
    # stay visible after the downscale.
    s = max(1.0, img.shape[1] / DISPLAY_W)
    r, th, off = int(8 * s), max(1, int(s)), int(10 * s)
    for i, (x, y) in enumerate(points):
        cv2.circle(out, (int(x), int(y)), r, (0, 200, 0), -1)
        cv2.putText(out, f"P{i + 1}", (int(x) + off, int(y) - off // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5 * s, (0, 200, 0), th)
    if highlight is not None:
        cv2.circle(out, (int(highlight[0]), int(highlight[1])), int(12 * s), (255, 165, 0), max(2, int(3 * s)))
    return out


def render_calibration():
    """Render the point-correspondence calibration UI; sets st.session_state['homography']."""
    if "video_path" not in st.session_state:
        st.info("Run detection on a video first.")
        return

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

    # Reload a saved homography from disk only if it matches the current frame size AND the user
    # isn't explicitly recalibrating - otherwise a previous camera's homography would be applied,
    # or the Recalibrate button would have no effect.
    if st.session_state["homography"] is None and not st.session_state.get("force_recalibrate"):
        _saved = PROCESSED / "homography_session.npy"
        _meta = _saved.with_suffix(".json")
        if _saved.exists() and _meta.exists():
            meta = json.loads(_meta.read_text())
            saved_size = tuple(meta.get("frame_size", []))
            if saved_size and saved_size == tuple(st.session_state.get("frame_size", ())):
                st.session_state["homography"] = np.load(_saved)
                # Restore the floor size it was calibrated against so the top-down keeps the right
                # aspect ratio (otherwise it falls back to the Mall 820×1200 default).
                floor_size = meta.get("floor_size")
                if floor_size:
                    st.session_state["floor_size"] = tuple(floor_size)
                    if st.session_state.get("floor_plan_img") is None:
                        st.session_state["floor_plan_img"] = make_blank_canvas(*floor_size)
                st.info("Loaded a homography calibrated earlier on this camera.")
                return

    # --- Floor plan ---
    st.markdown("**Floor plan**")
    floor_upload = st.file_uploader(
        "Upload a top-down floor plan (optional - a blank grid is used otherwise)",
        type=["png", "jpg", "jpeg"], key="cal_floor_upload")
    if floor_upload:
        img = Image.open(floor_upload).convert("RGB")
        img.thumbnail((1200, 1200))  # cap size, preserve the plan's aspect ratio
        arr = np.array(img)
        st.session_state["floor_plan_img"] = arr
        st.session_state["floor_size"] = (arr.shape[1], arr.shape[0])
    elif st.session_state["floor_plan_img"] is None:
        cam_w, cam_h = st.session_state.get("frame_size", (CAM_W, CAM_H))
        canvas_h = round(DISPLAY_W * cam_h / cam_w)
        st.session_state["floor_plan_img"] = make_blank_canvas(DISPLAY_W, canvas_h)
        st.session_state["floor_size"] = (DISPLAY_W, canvas_h)

    # --- Reference frame ---
    if st.session_state["reference_frame"] is None:
        with st.spinner("Extracting reference frame from video..."):
            frame = extract_reference_frame(st.session_state["video_path"])
        if frame is None:
            st.error("Could not extract a frame from the video.")
            return
        st.session_state["reference_frame"] = frame

    cam_pts = [p["camera"] for p in st.session_state["pairs"]]
    floor_pts = [p["floor"] for p in st.session_state["pairs"]]
    cam_annotated = _annotate(st.session_state["reference_frame"], cam_pts,
                              highlight=st.session_state["pending_camera"])
    floor_annotated = _annotate(st.session_state["floor_plan_img"], floor_pts)

    # --- Click UI ---
    st.markdown("**Point correspondence**")
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
                Image.fromarray(cam_annotated), width=DISPLAY_W,
                key=f"cam_{st.session_state['cam_nonce']}")
            if val:
                nh, nw = cam_annotated.shape[:2]
                st.session_state["pending_camera"] = [val["x"] * nw / val["width"],
                                                      val["y"] * nh / val["height"]]
                st.session_state["cam_nonce"] += 1
                st.session_state["click_stage"] = "floor"
                st.rerun()
        else:
            st.image(cam_annotated, width=DISPLAY_W)

    with col2:
        st.caption("Floor plan")
        if stage == "floor":
            val = streamlit_image_coordinates(
                Image.fromarray(floor_annotated), width=DISPLAY_W,
                key=f"floor_{st.session_state['floor_nonce']}")
            if val:
                nh, nw = floor_annotated.shape[:2]
                st.session_state["pairs"].append({
                    "camera": st.session_state["pending_camera"],
                    "floor": [val["x"] * nw / val["width"], val["y"] * nh / val["height"]],
                })
                st.session_state["pending_camera"] = None
                st.session_state["floor_nonce"] += 1
                st.session_state["click_stage"] = "camera"
                st.rerun()
        else:
            st.image(floor_annotated, width=DISPLAY_W)

    # --- Undo ---
    if st.session_state["pairs"]:
        if st.button("Undo last pair"):
            st.session_state["pairs"].pop()
            st.session_state["click_stage"] = "camera"
            st.session_state["pending_camera"] = None
            st.rerun()

    # --- Compute homography ---
    if len(st.session_state["pairs"]) >= 4 and st.button("Compute Homography"):
        src = np.float32([p["camera"] for p in st.session_state["pairs"]])
        dst = np.float32([p["floor"] for p in st.session_state["pairs"]])
        H, _ = compute_homography(src, dst)
        errors = reprojection_errors(src, dst, H)
        st.session_state["homography"] = H
        st.session_state["force_recalibrate"] = False  # fresh calibration done; allow future reloads

        st.dataframe(pd.DataFrame({
            "Point": [f"P{i + 1}" for i in range(len(errors))],
            "Reprojection error (px)": [f"{e:.1f}" for e in errors],
        }), hide_index=True)

        worst = errors.max()
        if worst > 20:
            st.warning(f"Max error {worst:.1f}px - consider adding more points "
                       f"or repositioning P{int(errors.argmax()) + 1}.")
        else:
            st.success(f"Homography computed. Max reprojection error: **{worst:.1f}px**.")

        PROCESSED.mkdir(parents=True, exist_ok=True)
        out = PROCESSED / "homography_session.npy"
        np.save(out, H)
        # Tag with the frame + floor sizes it was calibrated on, so it is only auto-reloaded for
        # footage of the same dimensions and the top-down keeps the right aspect ratio.
        out.with_suffix(".json").write_text(json.dumps({
            "frame_size": list(st.session_state.get("frame_size", [])),
            "floor_size": list(st.session_state.get("floor_size", [])),
        }))
        st.rerun()
