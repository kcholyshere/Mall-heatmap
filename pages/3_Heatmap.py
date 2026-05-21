from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.heatmap import FLOOR_H, FLOOR_W, backproject, build_heatmap, project_points

st.set_page_config(page_title="Heatmap", layout="wide")
st.title("Step 3 — Heatmap")

if "detections_path" not in st.session_state or "homography" not in st.session_state:
    st.warning("Complete Steps 1 and 2 first.")
    st.stop()


@st.cache_data
def load_and_project(path, h_bytes):
    H = np.frombuffer(h_bytes).reshape(3, 3)
    df = pd.read_csv(path)
    proj = project_points(df[["x", "y"]].values, H)
    df = df.copy()
    df["px"], df["py"] = proj[:, 0], proj[:, 1]
    return df[(df["px"] >= 0) & (df["px"] < FLOOR_W) & (df["py"] >= 0) & (df["py"] < FLOOR_H)]


H = st.session_state["homography"]
df = load_and_project(st.session_state["detections_path"], H.tobytes())
floor_img = st.session_state.get("floor_plan_img")

st.write(f"**{len(df):,}** in-bounds detections across all frames")

col_ctrl, col_view = st.columns([1, 2])

with col_ctrl:
    sigma = st.slider("Smoothing (sigma)", 5, 80, 30)
    colormap = st.selectbox("Colormap", ["hot", "jet", "plasma", "YlOrRd"])
    alpha = st.slider("Heatmap opacity", 0.1, 1.0, 0.6, step=0.05)
    backproject_toggle = st.checkbox("Back-project onto camera frame")

heatmap = build_heatmap(df, sigma=sigma)

with col_view:
    if backproject_toggle and st.session_state.get("reference_frame") is not None:
        H_inv = np.linalg.inv(H)
        blended = backproject(heatmap, H_inv, st.session_state["reference_frame"])
        st.image(blended, caption="Heatmap back-projected onto camera frame",
                 use_container_width=True)
    else:
        fig, ax = plt.subplots(figsize=(4, 7))
        if floor_img is not None:
            ax.imshow(floor_img)
        ax.imshow(heatmap, cmap=colormap, alpha=alpha, vmin=0, vmax=1,
                  extent=[0, FLOOR_W, FLOOR_H, 0])
        ax.axis("off")
        st.pyplot(fig)
        plt.close(fig)

# Export
out_dir = Path(__file__).resolve().parent.parent / "reports" / "figures"
out_dir.mkdir(parents=True, exist_ok=True)
if st.button("Export heatmap PNG"):
    fig, ax = plt.subplots(figsize=(4, 7))
    if floor_img is not None:
        ax.imshow(floor_img)
    ax.imshow(heatmap, cmap=colormap, alpha=alpha, vmin=0, vmax=1,
              extent=[0, FLOOR_W, FLOOR_H, 0])
    ax.axis("off")
    out_path = out_dir / "heatmap_client.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    st.success(f"Saved to `{out_path}`")
