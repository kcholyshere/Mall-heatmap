# Mall Heatmap

Spatial occupancy heatmap pipeline for retail foot-traffic analysis, built on the Mall dataset as a Data Science internship capstone.

**Business question:** Which areas of a physical retail space attract the most foot traffic?

It ships as a point-and-click **web app**: upload surveillance footage, and get back where people concentrate, how many unique visitors pass through, where they linger, and how that changes over time — no coding required.

## What it does

- **Occupancy heatmap** — where people concentrate in the camera view, overlaid on the emptiest frame of the footage.
- **Person tracking** (optional dense pass) — unique footfall / throughput, dwell-time per zone, and person trajectories, using ByteTrack to follow each person across frames.
- **Time-based analysis** — filter to any time window to compare traffic between periods (e.g. morning vs. afternoon).
- **Top-down floor-plan view** (optional) — calibrate four camera↔floor point pairs to project the heatmap onto an architectural floor plan.

### Who it's for

Retail operations, store-layout and merchandising teams, and security/facilities staff who want an evidence-based read on how a physical space is actually used — busiest zones, dead corners, queue build-up, and footfall by time of day.

## Running the app

Two ways to launch it. Both open the app in your browser at <http://localhost:8501>.

### Option A — double-click launcher (simplest)

Requires **Python 3.13.12** installed. The first launch creates a local environment and installs dependencies (slow, one-off); later launches are quick.

- **macOS:** double-click `run.command` (if macOS blocks it, right-click → Open the first time).
- **Windows:** double-click `run.bat`.

### Option B — Docker (no Python needed)

Requires **Docker Desktop**. Bundles all the heavy dependencies (PyTorch, Ultralytics) into a reproducible image.

```bash
docker compose up        # add --build the first time, or after code changes
```

Then open <http://localhost:8501>. Outputs are written to `./data` on the host (mounted into the container).

## Using the app

| Step | Page | What it does |
|------|------|-------------|
| 1 | Upload & Detect | Upload a video (or point to a local file); run YOLOv8 person detection. Tick **Enable person tracking** for the dense pass that unlocks Step 3. |
| 2 | Heatmap | Camera-space occupancy heatmap overlaid on the emptiest frame. Tuning knobs live under **Advanced settings**. |
| 3 | Tracking Insights | Unique footfall, dwell-time and person trajectories (requires tracking enabled in Step 1). |
| 4 | Time Analysis | Filter detections by time window to compare traffic across periods. |

Tracking needs dense (~25–30 fps) footage and is slower than the 1-frame-per-second detection used for the heatmap. Large files: use the *local file path* input in Step 1 rather than the uploader (which holds the whole file in memory).

## How it works

1. **Person detection** — YOLOv8 (pretrained on COCO) runs on surveillance frames; the bottom-centre of each bounding box is taken as the foot position.
2. **Coordinate projection** — A homography matrix maps camera-space foot positions onto a 2D top-down floor plan.
3. **Heatmap generation** — A Gaussian-smoothed 2D histogram aggregates positions into an occupancy heatmap, overlaid on the floor plan or back-projected onto a camera frame.
4. **Tracking** — ByteTrack assigns a persistent ID to each person across frames, giving genuine (de-duplicated) footfall, dwell-time per grid cell, and trajectories.
5. **Time-based analysis** — Detections are segmented by time window (frame index as proxy) to compare traffic patterns across periods.

## Dataset (research / notebooks)

Mall dataset — 2,000 annotated frames (640×480) from a fixed surveillance camera in a shopping mall. Annotations provide per-frame pedestrian counts and head positions.

Place dataset files under `data/raw/`:

    data/raw/
      mall_gt.mat
      perspective_roi.mat
      frames/
        seq_000001.jpg
        ...
        seq_002000.jpg

## Notebooks

The notebooks document the research pipeline behind the app. Run in order:

| Notebook | Purpose |
|----------|---------|
| `01_data_exploration.ipynb` | Inspect frames, counts, annotation format |
| `02_person_detection.ipynb` | Run YOLOv8 on all frames; save detections to `data/interim/` |
| `03_coordinate_projection.ipynb` | Compute homography; project detections to floor plan |
| `04_heatmap_generation.ipynb` | Generate and visualise occupancy heatmaps |
| `05_time_based_analysis.ipynb` | Compare heatmaps across time windows |
| `06_ground_truth_validation.ipynb` | Validate YOLO heatmap against GT annotations |
| `07_homography_improvement.ipynb` | 10-point homography with reprojection error analysis |

## Project structure

    mall-heatmap/
      app.py            Streamlit entry point (run via launcher or Docker)
      pages/            Web app steps (Upload & Detect, Heatmap, Tracking, Time Analysis)
      calibration_panel.py  Camera↔floor calibration UI
      agent_docs/       Task tracking (TODOS.md)
      data/
        raw/            Original Mall dataset (read-only)
        interim/        Per-video detection / track coordinates
        processed/      Projected coordinates, homography matrices
      models/           YOLOv8 weights (yolov8n.pt, bundled)
      notebooks/        Research pipeline (01–07)
      reports/figures/  Generated plots and heatmap visualisations
      src/              Shared logic (heatmap.py, tracking.py, plots.py)
      Dockerfile        CPU-only image
      docker-compose.yml
      run.command / run.bat   Double-click launchers (macOS / Windows)
      requirements.txt

## Requirements

- **Python 3.13.12** (for the local / launcher route; Docker bundles its own).
- Key libraries: Ultralytics (YOLOv8), OpenCV, PyTorch (CPU), NumPy, SciPy, Matplotlib, Seaborn, Streamlit.
- Linter/formatter: Ruff.

## Limitations

- Detection accuracy drops for far-field, occluded or edge-clipped people; the Heatmap page can shade low-reliability zones to make this explicit.
- The uploader holds the whole file in memory (5 GB cap) — steer large footage to the local-path input. Day-long single-file processing is out of scope (needs per-segment processing and ideally a GPU).
- Static objects can be mis-detected as people at low confidence; the Tracking page filters these out by movement.
