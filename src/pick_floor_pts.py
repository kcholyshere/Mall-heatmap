"""Interactive tool for picking floor plan correspondences.

Usage (from repo root):
    .venv/bin/python src/pick_floor_pts.py <frame_path> [pt1_u pt1_v pt2_u pt2_v ...]

Example:
    .venv/bin/python src/pick_floor_pts.py reports/figures/Empty_frame_85.jpg \
        95 214 121 152 118 105 114 39 369 25 499 106 536 143 577 204 608 296 638 415

For each camera point (highlighted in green), click the corresponding position
on the floor plan canvas. Prints src_pts / dst_pts ready to paste into a notebook.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use('MacOSX')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.heatmap import FLOOR_W, FLOOR_H, make_floor_canvas


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    frame_path = sys.argv[1]
    coords = list(map(float, sys.argv[2:]))
    if len(coords) % 2 != 0 or len(coords) == 0:
        print('Error: camera points must be pairs of u v coordinates.')
        sys.exit(1)

    camera_pts = np.float32(coords).reshape(-1, 2)
    frame = plt.imread(frame_path)
    fp = make_floor_canvas()
    xs = range(0, FLOOR_W + 1, 100)
    ys = range(0, FLOOR_H + 1, 100)
    floor_pts = []

    for i, (u, v) in enumerate(camera_pts):
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        fig.suptitle(
            f'Point P{i+1} of {len(camera_pts)} — click the corresponding position on the FLOOR PLAN (right)',
            fontsize=11, fontweight='bold'
        )

        ax1 = axes[0]
        ax1.imshow(frame)
        ax1.scatter(camera_pts[:, 0], camera_pts[:, 1], c='white', s=30, zorder=4)
        ax1.scatter([u], [v], c='lime', s=120, zorder=5, edgecolors='black', linewidths=1.5)
        for j, (pu, pv) in enumerate(camera_pts):
            ax1.text(pu + 5, pv - 5, f'P{j+1}', color='lime' if j == i else 'white',
                     fontsize=8, fontweight='bold')
        ax1.set_title('Camera frame (reference only)', fontsize=9)
        ax1.axis('off')

        ax2 = axes[1]
        ax2.imshow(fp, extent=[0, FLOOR_W, FLOOR_H, 0])
        for x in xs:
            ax2.axvline(x, color='c', lw=0.4, alpha=0.5)
        for y in ys:
            ax2.axhline(y, color='m', lw=0.4, alpha=0.5)
        for x in range(0, FLOOR_W + 1, 200):
            for y in range(0, FLOOR_H + 1, 200):
                ax2.text(x + 4, y + 16, f'({x},{y})', fontsize=6, color='gray')
        for j, fp_pt in enumerate(floor_pts):
            ax2.scatter([fp_pt[0]], [fp_pt[1]], c='lime', s=60, zorder=5)
            ax2.text(fp_pt[0] + 6, fp_pt[1] - 6, f'P{j+1}', color='lime', fontsize=7)
        ax2.set_xlim(0, FLOOR_W)
        ax2.set_ylim(FLOOR_H, 0)
        ax2.set_title(f'Click P{i+1} position here', fontsize=9)
        ax2.set_xlabel('Floor plan x')
        ax2.set_ylabel('Floor plan y')

        plt.tight_layout()
        clicks = plt.ginput(1, timeout=120)
        plt.close(fig)

        if not clicks:
            print(f'No click for P{i+1}, aborting.')
            sys.exit(1)

        fx, fy = clicks[0]
        floor_pts.append([fx, fy])
        print(f'P{i+1}: camera=({u:.0f},{v:.0f})  floor=({fx:.1f},{fy:.1f})')

    print('\n--- All pairs ---')
    print('src_pts = np.float32([')
    for i, ((u, v), (fx, fy)) in enumerate(zip(camera_pts, floor_pts)):
        print(f'    [{u:.0f}, {v:.0f}],  # P{i+1} -> floor ({fx:.0f}, {fy:.0f})')
    print('])')
    print('dst_pts = np.float32([')
    for i, (fx, fy) in enumerate(floor_pts):
        print(f'    [{fx:.0f}, {fy:.0f}],  # P{i+1}')
    print('])')


if __name__ == '__main__':
    main()
