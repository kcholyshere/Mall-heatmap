import cv2
import numpy as np


def compute_homography(src_pts, dst_pts):
    """Returns (H, mask) via RANSAC."""
    H, mask = cv2.findHomography(np.float32(src_pts), np.float32(dst_pts), cv2.RANSAC, 5.0)
    return H, mask


def reprojection_errors(src_pts, dst_pts, H):
    """Returns per-point reprojection error in pixels."""
    projected = cv2.perspectiveTransform(np.float32(src_pts).reshape(-1, 1, 2), H).reshape(-1, 2)
    return np.linalg.norm(projected - np.float32(dst_pts), axis=1)


def extract_reference_frame(video_path):
    """Returns first non-black frame as (H, W, 3) uint8 RGB array, or None."""
    cap = cv2.VideoCapture(str(video_path))
    frame = None
    while cap.isOpened():
        ret, f = cap.read()
        if not ret:
            break
        if f.mean() > 5:
            frame = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            break
    cap.release()
    return frame
