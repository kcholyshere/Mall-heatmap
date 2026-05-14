import matplotlib.image as mpimg
import matplotlib.pyplot as plt


def view_detections(frame_indices, model, frames_dir, gt):
    """Compare YOLO person detections (top) with ground truth head positions (bottom).

    frame_indices : list[int]  1-indexed frame numbers, max 10
    model         : loaded ultralytics YOLO instance
    frames_dir    : pathlib.Path to the frames directory
    gt            : dict from scipy.io.loadmat('mall_gt.mat')
    """
    assert len(frame_indices) <= 10, "max 10 frames at a time"

    n = len(frame_indices)
    fig, axes = plt.subplots(2, n, figsize=(n * 4, 6))
    if n == 1:
        axes = axes.reshape(2, 1)

    for col, frame_id in enumerate(frame_indices):
        frame_path = frames_dir / f'seq_{frame_id:06d}.jpg'

        # YOLO detections
        results = model(frame_path, classes=[0], verbose=False)
        axes[0, col].imshow(results[0].plot()[..., ::-1])
        axes[0, col].set_title(f'Frame {frame_id}', fontsize=8)
        axes[0, col].axis('off')

        # Ground truth head positions
        heads = gt['frame'][0, frame_id - 1][0, 0][0]
        axes[1, col].imshow(mpimg.imread(str(frame_path)))
        axes[1, col].scatter(heads[:, 0], heads[:, 1], c='red', s=15, zorder=5)
        axes[1, col].axis('off')

    for ax, label in zip(axes[:, 0], ['YOLO', 'Ground truth']):
        ax.text(-0.05, 0.5, label, transform=ax.transAxes,
                va='center', ha='right', fontsize=9, fontweight='bold', rotation=90)

    plt.tight_layout()
    plt.show()
