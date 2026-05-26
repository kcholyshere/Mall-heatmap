import cv2
import matplotlib.pyplot as plt
import numpy as np
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


def compute_confidence_cutoff(df, threshold=0.5, n_bands=48):
    """Return the y-pixel of the upper edge of the reliable detection zone.

    Bins detections into n_bands horizontal strips (top-to-bottom). Returns
    the y-pixel of the topmost band whose mean confidence meets the threshold.
    Returns None when confidence data is absent or all bands are above threshold.
    """
    if df.empty or "confidence" not in df.columns:
        return None
    band_h = CAM_H / n_bands
    df = df.copy()
    df["_band"] = (df["y"] / band_h).astype(int).clip(0, n_bands - 1)
    means = df.groupby("_band")["confidence"].mean()
    reliable = means[means >= threshold]
    if reliable.empty:
        return None
    return int(reliable.index.min() * band_h)


def backproject(heatmap_norm, H_inv, background, alpha=0.7):
    """Returns a (CAM_H, CAM_W, 3) uint8 image with the heatmap blended onto the background frame."""
    hm_cam  = cv2.warpPerspective(heatmap_norm.astype(np.float32), H_inv, (CAM_W, CAM_H))
    hm_rgba = plt.colormaps['hot'](hm_cam)
    hm_rgb  = (hm_rgba[:, :, :3] * 255).astype(np.uint8)
    a       = (alpha * hm_cam)[:, :, np.newaxis]
    return np.clip((1 - a) * background + a * hm_rgb, 0, 255).astype(np.uint8)
