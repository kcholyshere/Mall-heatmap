# Mall Heatmap

Spatial occupancy heatmap pipeline for retail foot-traffic analysis, built on the Mall dataset as a Data Science internship capstone.

**Business question:** Which areas of a physical retail space attract the most foot traffic?

## Pipeline

1. **Person detection** — YOLOv8 (pretrained on COCO) runs on each surveillance frame; bottom-centre of each bounding box is extracted as the foot position.
2. **Coordinate projection** — A homography matrix maps camera-space foot positions onto a 2D top-down floor plan.
3. **Heatmap generation** — A Gaussian-smoothed 2D histogram aggregates all projected positions into an occupancy heatmap, overlaid on the floor plan or back-projected onto a camera frame.
4. **Time-based analysis** — Detections are segmented by time window (frame index as proxy) to compare traffic patterns across periods.
5. **Ground truth validation** — GT head positions from `mall_gt.mat` are projected through the same homography for a YOLO vs. GT side-by-side comparison.

## Dataset

Mall dataset — 2,000 annotated frames (640×480) from a fixed surveillance camera in a shopping mall. Annotations provide per-frame pedestrian counts and head positions.

Place dataset files under `data/raw/`:

    data/raw/
      mall_gt.mat
      perspective_roi.mat
      frames/
        seq_000001.jpg
        ...
        seq_002000.jpg

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

## Notebooks

Run in order:

| Notebook | Purpose |
|----------|---------|
| `01_data_exploration.ipynb` | Inspect frames, counts, annotation format |
| `02_person_detection.ipynb` | Run YOLOv8 on all frames; save detections to `data/interim/` |
| `03_coordinate_projection.ipynb` | Compute homography; project detections to floor plan |
| `04_heatmap_generation.ipynb` | Generate and visualise occupancy heatmaps |
| `05_time_based_analysis.ipynb` | Compare heatmaps across time windows |
| `06_ground_truth_validation.ipynb` | Validate YOLO heatmap against GT annotations |
| `07_homography_improvement.ipynb` | 10-point homography with reprojection error analysis |

## Key outputs

| File | Description |
|------|-------------|
| `data/interim/detections.csv` | Per-frame foot positions (frame_id, x, y, confidence) |
| `data/processed/projected_detections.csv` | Foot positions projected to floor plan (4-point H) |
| `data/processed/projected_detections_10pt.csv` | Same, using 10-point homography |
| `data/processed/homography.npy` | 4-point homography matrix |
| `data/processed/homography_10pt.npy` | 10-point homography matrix |

## Project structure

    mall-heatmap/
      agent_docs/       Task tracking (TODOS.md)
      data/
        raw/            Original Mall dataset (read-only)
        interim/        Per-frame detection coordinates
        processed/      Projected coordinates, homography matrices
      models/           YOLOv8 weights
      notebooks/        Analysis pipeline (01–07)
      reports/figures/  Generated plots and heatmap visualisations
      src/              Shared utilities (heatmap.py, plots.py, pick_floor_pts.py)
      requirements.txt

## Stack

- Python 3.13, Jupyter notebooks
- YOLOv8 (ultralytics), OpenCV, NumPy, SciPy, Matplotlib, Seaborn
- Linter/formatter: Ruff
