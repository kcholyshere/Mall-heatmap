import numpy as np
import pandas as pd


def run_tracking(video_path, model, conf=0.25, max_frames=None, progress_cb=None):
    """Run a dense ByteTrack pass and return a DataFrame of tracked foot positions.

    Unlike the 1-fps detection loop, tracking needs consecutive frames so ByteTrack can
    associate each person across time, so this processes every frame. Foot point is the
    bottom-centre of the box ((x1+x2)/2, y2), matching the rest of the pipeline.

    Returns columns: frame_id, track_id, x, y, confidence (one row per person per frame).
    `max_frames` caps a long clip; `progress_cb(frame_id)` is called once per frame.
    """
    rows = []
    stream = model.track(source=video_path, stream=True, persist=True, classes=[0],
                         conf=conf, tracker="bytetrack.yaml", verbose=False)
    for frame_id, result in enumerate(stream, start=1):
        if max_frames is not None and frame_id > max_frames:
            break
        for box in result.boxes:
            if box.id is None:
                continue  # detection not yet assigned a track
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            rows.append({
                "frame_id": frame_id,
                "track_id": int(box.id[0].item()),
                "x": (x1 + x2) / 2,
                "y": y2,
                "confidence": round(box.conf[0].item(), 4),
            })
        if progress_cb is not None:
            progress_cb(frame_id)
    return pd.DataFrame(rows, columns=["frame_id", "track_id", "x", "y", "confidence"])


def footfall_stats(df, fps, total_frames):
    """Return de-duplicated footfall figures from tracked detections.

    Returns (unique_visitors, throughput_per_min, durations) where `durations` is a Series
    of seconds-in-scene per track (presence span: (last - first + 1) / fps).
    """
    if df.empty:
        return 0, 0.0, pd.Series(dtype=float)
    unique = int(df["track_id"].nunique())
    duration_min = max(total_frames / fps / 60, 1e-6)
    throughput = unique / duration_min
    spans = df.groupby("track_id")["frame_id"].agg(lambda s: s.max() - s.min() + 1)
    durations = spans / fps
    return unique, throughput, durations


def build_trajectories(df, min_len=1):
    """Return a list of (track_id, (N, 2) array) paths, sorted by frame_id, len >= min_len."""
    trajectories = []
    for tid, g in df.sort_values("frame_id").groupby("track_id"):
        if len(g) >= min_len:
            trajectories.append((int(tid), g[["x", "y"]].values))
    return trajectories


def compute_dwell(df, fps, grid_cols, grid_rows, frame_w, frame_h):
    """Return a (grid_rows, grid_cols) grid of mean dwell seconds per visitor.

    Each cell holds (person-frames in cell / fps) / (distinct tracks visiting cell), i.e. the
    average time a person spends in that cell. Cells no one visited are 0.
    """
    grid = np.zeros((grid_rows, grid_cols))
    if df.empty:
        return grid
    cw, ch = frame_w / grid_cols, frame_h / grid_rows
    cx = (df["x"].values / cw).astype(int).clip(0, grid_cols - 1)
    cy = (df["y"].values / ch).astype(int).clip(0, grid_rows - 1)
    cells = pd.DataFrame({"cell": cy * grid_cols + cx, "track_id": df["track_id"].values})
    person_frames = cells.groupby("cell").size()
    visitors = cells.groupby("cell")["track_id"].nunique()
    mean_dwell = (person_frames / fps) / visitors
    flat = grid.ravel()
    flat[mean_dwell.index.values] = mean_dwell.values
    return flat.reshape(grid_rows, grid_cols)
