import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

FLOOR_W, FLOOR_H = 820, 1200
cx, cy, R = 520, 850, 130
CAM_W, CAM_H = 640, 480
SIGMA = 30

# Floor boundary from the 10-point manually selected homography
FLOOR_POLYGON = np.int32([
    [342, 681], [369, 552], [354, 409], [325, 206],
    [564, 202], [666, 403], [691, 552], [700, 699],
    [704, 860], [705, 1056],
])


def make_floor_canvas(polygon_pts=None):
    fp = np.ones((FLOOR_H, FLOOR_W, 3), dtype=np.uint8) * 245
    for x in range(0, FLOOR_W, 80):
        cv2.line(fp, (x, 0), (x, FLOOR_H), (215, 215, 215), 1)
    for y in range(0, FLOOR_H, 80):
        cv2.line(fp, (0, y), (FLOOR_W, y), (215, 215, 215), 1)
    cv2.ellipse(fp, (cx, cy), (R, R), 0, 0, 360, (170, 170, 170), 2)
    if polygon_pts is not None:
        cv2.polylines(fp, [polygon_pts.reshape(-1, 1, 2)], isClosed=False, color=(150, 150, 200), thickness=2)
    return fp


def make_blank_canvas(w, h):
    """Plain light-grey grid canvas of size (h, w) — no scene-specific decorations.

    Used as the default floor plan when the user has not uploaded one, sized to the
    camera's aspect ratio rather than the Mall-specific dimensions.
    """
    fp = np.ones((h, w, 3), dtype=np.uint8) * 245
    step = max(40, min(w, h) // 12)
    for x in range(0, w, step):
        cv2.line(fp, (x, 0), (x, h), (215, 215, 215), 1)
    for y in range(0, h, step):
        cv2.line(fp, (0, y), (w, y), (215, 215, 215), 1)
    return fp


def project_points(pts, H):
    """Returns (N, 2) array of projected floor-plan pixel coordinates."""
    pts_h = np.hstack([pts, np.ones((len(pts), 1))])
    proj = (H @ pts_h.T).T
    return proj[:, :2] / proj[:, 2:3]


def build_heatmap(df, xcol, ycol, bins, range_, sigma=SIGMA, normalise=True):
    """Returns a smoothed float array of shape (bins[1], bins[0]).

    normalise=True  → values in [0, 1] (relative occupancy).
    normalise=False → raw smoothed detection counts (divide by duration for per-minute values).
    """
    h, _, _ = np.histogram2d(df[xcol], df[ycol], bins=bins, range=range_)
    h = h.T
    h = gaussian_filter(h, sigma=sigma)
    if normalise:
        return h / h.max() if h.max() > 0 else h
    return h


def pick_background_frame(video_path, detections_path, fps, total_frames):
    """Return the first sampled frame with 0 detections, falling back to fewest detections.

    Returns an (H, W, 3) RGB uint8 array, or None if no frame could be read.
    """
    df = pd.read_csv(detections_path)
    detected_frames = set(df["frame_id"].unique())
    sample_interval = max(1, round(fps))
    sampled = range(1, total_frames + 1, sample_interval)
    zero_det = [f for f in sampled if f not in detected_frames]
    if zero_det:
        target_frame = zero_det[0]
    else:
        target_frame = int(df.groupby("frame_id").size().idxmin())
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame - 1)
    ret, frame = cap.read()
    cap.release()
    if ret:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return None


def compute_confidence_cutoff(df, threshold=0.5, n_bands=48, frame_h=CAM_H):
    """Return the y-pixel of the upper edge of the reliable detection zone.

    Bins detections into n_bands horizontal strips (top-to-bottom). Returns
    the y-pixel of the topmost band whose mean confidence meets the threshold.
    Returns None when confidence data is absent or all bands are above threshold.
    """
    if df.empty or "confidence" not in df.columns:
        return None
    band_h = frame_h / n_bands
    df = df.copy()
    df["_band"] = (df["y"] / band_h).astype(int).clip(0, n_bands - 1)
    means = df.groupby("_band")["confidence"].mean()
    reliable = means[means >= threshold]
    if reliable.empty:
        return None
    return int(reliable.index.min() * band_h)


def build_reliability_mask(df, threshold=0.5, grid_cols=32, grid_rows=24, min_count=3,
                           frame_w=CAM_W, frame_h=CAM_H):
    """Return a (grid_rows, grid_cols) float mask in [0, 1]; 1 = low model reliability.

    A 2D generalisation of `compute_confidence_cutoff`. Automated and case-agnostic
    (driven only by x, y, confidence). A cell is flagged unreliable when either:
      - it holds enough detections but their mean confidence is below `threshold`
        (weak detections — e.g. an occlusion pocket); or
      - it holds too few detections AND sits above the confidence frontier (the far
        field, where the model rarely fires at all — so absence of detections there
        signals a blind spot, not empty floor).
    Sparse cells below the frontier are left reliable (the model works there, so no
    detection plausibly means no person). Returns zeros when confidence is absent.
    """
    mask = np.zeros((grid_rows, grid_cols))
    if df.empty or "confidence" not in df.columns:
        return mask
    cw, ch = frame_w / grid_cols, frame_h / grid_rows
    cx = (df["x"].values / cw).astype(int).clip(0, grid_cols - 1)
    cy = (df["y"].values / ch).astype(int).clip(0, grid_rows - 1)
    flat = cy * grid_cols + cx
    counts = np.bincount(flat, minlength=grid_rows * grid_cols)
    conf_sum = np.bincount(flat, weights=df["confidence"].values, minlength=grid_rows * grid_cols)
    counts = counts.reshape(grid_rows, grid_cols)
    mean_conf = np.divide(conf_sum.reshape(grid_rows, grid_cols), counts,
                          out=np.zeros((grid_rows, grid_cols)), where=counts > 0)

    has_data = counts >= min_count
    mask[has_data & (mean_conf < threshold)] = 1.0

    cutoff_y = compute_confidence_cutoff(df, threshold=threshold, frame_h=frame_h)
    if cutoff_y is not None:
        row_bottom_y = (np.arange(grid_rows) + 1) * ch
        above_frontier = (row_bottom_y <= cutoff_y)[:, None]
        mask[(counts < min_count) & np.broadcast_to(above_frontier, mask.shape)] = 1.0
    return mask


def backproject(heatmap_norm, H_inv, background, alpha=0.7):
    """Returns an RGB uint8 image matching `background` with the heatmap blended on top."""
    h, w = background.shape[:2]
    hm_cam  = cv2.warpPerspective(heatmap_norm.astype(np.float32), H_inv, (w, h))
    hm_rgba = plt.colormaps['hot'](hm_cam)
    hm_rgb  = (hm_rgba[:, :, :3] * 255).astype(np.uint8)
    a       = (alpha * hm_cam)[:, :, np.newaxis]
    return np.clip((1 - a) * background + a * hm_rgb, 0, 255).astype(np.uint8)
