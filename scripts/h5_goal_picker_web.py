#!/usr/bin/env python3
"""
h5_goal_picker_web.py — Streamlit web visualizer for goal selection.

Works on headless servers (e.g. ASU Sol) via SSH port forwarding.

Usage:
  # Direct launch
  streamlit run scripts/h5_goal_picker_web.py -- --h5 path/to/file.h5

  # Or via the launcher (recommended)
  python scripts/run_goal_web_ui.py --h5 path/to/file.h5 --port 8501

Then on your local machine:
  ssh -L 8501:localhost:8501 user@sol
  open http://localhost:8501
"""

import argparse
import os
import sys
from pathlib import Path

import h5py
import numpy as np

# Allow running from repo root
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))


def main():
    import streamlit as st
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ── parse H5 path from CLI args ───────────────────────────────────────
    # Streamlit passes args after '--' to sys.argv
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", required=True)
    parser.add_argument("--sentinel", default=None,
                        help="Sentinel file to write when goal is set")
    # Filter out streamlit's own args
    known_args = []
    for arg in sys.argv[1:]:
        if arg.startswith("--h5") or arg.startswith("--sentinel"):
            known_args.append(arg)
        elif known_args and not known_args[-1].startswith("--"):
            pass  # skip
        elif known_args and known_args[-1] in ("--h5", "--sentinel"):
            known_args.append(arg)

    # Simple manual parse since streamlit interferes with argparse
    h5_path = None
    sentinel_path = None
    args = sys.argv[1:]
    for i, a in enumerate(args):
        if a == "--h5" and i + 1 < len(args):
            h5_path = args[i + 1]
        elif a == "--sentinel" and i + 1 < len(args):
            sentinel_path = args[i + 1]

    if h5_path is None:
        # Try env var fallback
        h5_path = os.environ.get("H5_GOAL_PICKER_PATH")

    if h5_path is None or not os.path.exists(h5_path):
        st.error(f"H5 file not found: {h5_path}")
        st.stop()

    if sentinel_path is None:
        sentinel_path = os.environ.get("H5_GOAL_PICKER_SENTINEL")

    # ── load H5 metadata ─────────────────────────────────────────────────
    st.set_page_config(page_title="H5 Goal Picker", layout="wide")
    st.title("H5 Goal Picker")
    st.caption(f"File: `{h5_path}`")

    f = h5py.File(h5_path, "r")
    n_steps = f["voxels"].shape[0]
    n_rgb = f["rgb_images"].shape[0]
    current_goal = int(f.attrs.get("goal_step", -1))

    # Build rgb lookup
    rgb_lookup = {}
    if "rgb_indices" in f:
        ri = f["rgb_indices"][:]
        for img_i, step_i in enumerate(ri):
            rgb_lookup[int(step_i)] = img_i

    has_actions_valid = "actions_valid" in f

    # ── sidebar ───────────────────────────────────────────────────────────
    st.sidebar.header("Navigation")
    st.sidebar.write(f"Steps: {n_steps}, RGB: {n_rgb}")
    st.sidebar.write(f"Current goal: {current_goal}")

    step_idx = st.sidebar.slider("Step", 0, n_steps - 1,
                                  value=n_steps - 1, key="step_slider")

    # Quick jump buttons
    col1, col2, col3 = st.sidebar.columns(3)
    with col1:
        if st.button("⏮ First"):
            st.session_state.step_slider = 0
            st.rerun()
    with col2:
        if st.button("⏭ Last"):
            st.session_state.step_slider = n_steps - 1
            st.rerun()
    with col3:
        if current_goal >= 0:
            if st.button(f"🎯 Goal ({current_goal})"):
                st.session_state.step_slider = current_goal
                st.rerun()

    # ── goal setting ──────────────────────────────────────────────────────
    st.sidebar.divider()
    st.sidebar.subheader("Set Goal")
    if st.sidebar.button(
        f"🎯 Set step {step_idx} as GOAL",
        type="primary",
        use_container_width=True,
    ):
        f.close()
        with h5py.File(h5_path, "r+") as fw:
            ts_val = int(fw["timestamps_ns"][step_idx])
            fw.attrs["goal_step"] = step_idx
            fw.attrs["goal_timestamp_ns"] = ts_val
            fw.attrs["goal_has_rgb"] = bool(fw["rgb_mask"][step_idx])
            if fw["rgb_mask"][step_idx] and "rgb_indices" in fw:
                ri = fw["rgb_indices"][:]
                matches = np.where(ri == step_idx)[0]
                if len(matches) > 0:
                    fw.attrs["goal_rgb_index"] = int(matches[0])

        st.sidebar.success(
            f"Goal saved! step={step_idx}, ts={ts_val}"
        )

        # Write sentinel to signal completion
        if sentinel_path:
            Path(sentinel_path).write_text(str(step_idx))
            st.sidebar.info("Sentinel written — launcher will auto-close.")

        # Reopen for continued viewing
        f = h5py.File(h5_path, "r")

    # ── main content ──────────────────────────────────────────────────────
    ts_ns = int(f["timestamps_ns"][step_idx])
    goal_tag = " **[GOAL]**" if step_idx == current_goal else ""
    st.subheader(f"Step {step_idx}/{n_steps - 1}  |  TS: {ts_ns} ns{goal_tag}")

    col_vox, col_rgb = st.columns(2)

    # ── Voxel panel ───────────────────────────────────────────────────────
    with col_vox:
        st.write("**Event Voxel** (sum of bins)")
        v = f["voxels"][step_idx]
        v_img = np.sum(v, axis=0)

        fig_vox, ax_vox = plt.subplots(figsize=(6, 3.5))
        ax_vox.imshow(v_img, cmap="viridis")
        ax_vox.axis("off")
        st.pyplot(fig_vox)
        plt.close(fig_vox)

    # ── RGB panel ─────────────────────────────────────────────────────────
    with col_rgb:
        if f["rgb_mask"][step_idx]:
            img_idx = rgb_lookup.get(step_idx)
            if img_idx is not None:
                img = f["rgb_images"][img_idx]
                st.write(f"**RGB** ({img.shape[1]}×{img.shape[0]})")
                st.image(img, use_container_width=True)
            else:
                st.warning("RGB mask True but no image index found")
        else:
            st.write("**RGB**")
            st.info("No RGB frame for this step")

    # ── Action trajectory panel ───────────────────────────────────────────
    st.write("**Action Trajectory** (dx, dy, dYaw) — relative waypoints")
    act = f["actions"][step_idx]
    xs = act[:, 0]
    ys = act[:, 1]

    fig_act, ax_act = plt.subplots(figsize=(8, 4))
    ax_act.plot(xs, ys, "b-o", linewidth=2, markersize=6,
                label="Waypoint trajectory")
    for i in range(len(act)):
        d_yaw = act[i, 2]
        ax_act.arrow(xs[i], ys[i],
                     0.05 * np.cos(d_yaw), 0.05 * np.sin(d_yaw),
                     head_width=0.02, color="green")
    ax_act.scatter(0, 0, c="red", marker="X", s=150, label="Robot (current)")

    valid_str = ""
    if has_actions_valid:
        valid_str = f"  |  valid={bool(f['actions_valid'][step_idx])}"
    ax_act.set_xlabel("dx (meters forward)")
    ax_act.set_ylabel("dy (meters left/right)")
    ax_act.set_title(f"8-step relative waypoints{valid_str}")
    ax_act.grid(True, linestyle="--", alpha=0.7)
    ax_act.axis("equal")
    ax_act.legend(loc="upper left")
    st.pyplot(fig_act)
    plt.close(fig_act)

    # ── step info ─────────────────────────────────────────────────────────
    with st.expander("Step details"):
        st.json({
            "step": step_idx,
            "timestamp_ns": ts_ns,
            "rgb_mask": bool(f["rgb_mask"][step_idx]),
            "actions_valid": bool(f["actions_valid"][step_idx]) if has_actions_valid else "N/A",
            "action_chunk": act.tolist(),
            "voxel_nonzero_frac": float((v != 0).sum() / v.size),
        })

    f.close()


if __name__ == "__main__":
    main()
