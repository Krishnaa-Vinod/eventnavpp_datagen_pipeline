#!/usr/bin/env python3
"""
h5_goal_picker_gui.py — Matplotlib GUI visualizer with goal selection.

Keybindings:
  Right / d  — next step
  Left  / a  — previous step
  g          — set current step as goal, write to H5, and close
  q          — quit without saving goal

Action plot interprets actions as (dx, dy, dYaw) relative waypoints
in the robot's local frame at the current step.
"""

import argparse
import os
import sys

import h5py
import numpy as np


def _check_display():
    """Check if a display is available for GUI."""
    display = os.environ.get("DISPLAY", "")
    if not display:
        print("ERROR: No DISPLAY set — cannot open GUI visualizer.")
        print("Use --visualizer web for headless environments (e.g. sol).")
        sys.exit(1)


def run_gui(h5_path: str) -> int:
    """Run the GUI goal picker.  Returns the selected goal step or -1."""
    _check_display()

    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt

    # Suppress Qt conflicts from OpenCV
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    import cv2  # noqa: F401
    os.environ.pop("QT_QPA_PLATFORM", None)

    f = h5py.File(h5_path, "r")
    voxels = f["voxels"]
    actions = f["actions"]
    actions_valid = f["actions_valid"] if "actions_valid" in f else None
    timestamps = f["timestamps_ns"]
    rgb_mask = f["rgb_mask"]
    rgb_images = f["rgb_images"]
    rgb_indices = f["rgb_indices"] if "rgb_indices" in f else None
    n_steps = voxels.shape[0]

    current_goal = int(f.attrs.get("goal_step", -1))

    # Build step → rgb_image index lookup
    rgb_lookup = {}
    if rgb_indices is not None:
        ri = rgb_indices[:]
        for img_i, step_i in enumerate(ri):
            rgb_lookup[int(step_i)] = img_i

    print(f"Steps: {n_steps}, RGB: {rgb_images.shape[0]}")
    print(f"Current goal_step: {current_goal}")
    print("Keys: Right/d=next, Left/a=prev, g=set goal & save, q=quit")

    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(2, 2)
    ax_vox = fig.add_subplot(gs[0, 0])
    ax_rgb = fig.add_subplot(gs[0, 1])
    ax_act = fig.add_subplot(gs[1, :])

    state = {"idx": n_steps - 1, "goal": -1}  # start at last step

    def update_plot():
        idx = state["idx"]
        goal_tag = ""
        if current_goal >= 0 and idx == current_goal:
            goal_tag = "  [EXISTING GOAL]"
        fig.suptitle(
            f"Step {idx}/{n_steps - 1}  |  "
            f"TS: {timestamps[idx]} ns{goal_tag}",
            fontsize=13,
        )

        # 1. Voxel
        ax_vox.clear()
        v = voxels[idx]
        v_img = np.sum(v, axis=0)
        ax_vox.imshow(v_img, cmap="viridis")
        ax_vox.set_title(f"Event Voxel (sum {v.shape[0]} bins)")
        ax_vox.axis("off")

        # 2. RGB
        ax_rgb.clear()
        if rgb_mask[idx]:
            img_idx = rgb_lookup.get(idx)
            if img_idx is not None:
                img = rgb_images[img_idx]
                ax_rgb.imshow(img)
                ax_rgb.set_title(f"RGB ({img.shape[1]}×{img.shape[0]})")
            else:
                ax_rgb.text(0.5, 0.5, "RGB lookup error", ha="center",
                            color="orange", fontsize=12)
        else:
            ax_rgb.text(0.5, 0.5, "NO RGB", ha="center", va="center",
                        color="red", fontsize=14)
        ax_rgb.axis("off")

        # 3. Action trajectory (dx, dy, dYaw)
        ax_act.clear()
        act = actions[idx]
        xs = act[:, 0]  # dx (forward)
        ys = act[:, 1]  # dy (left/right)

        ax_act.plot(xs, ys, "b-o", linewidth=2, markersize=6,
                    label="Waypoint trajectory")
        # Draw heading arrows using dYaw
        for i in range(len(act)):
            d_yaw = act[i, 2]
            ax_act.arrow(xs[i], ys[i],
                         0.05 * np.cos(d_yaw), 0.05 * np.sin(d_yaw),
                         head_width=0.02, color="green")

        ax_act.scatter(0, 0, c="red", marker="X", s=150,
                       label="Robot (current)")
        valid_str = ""
        if actions_valid is not None:
            valid_str = f"  valid={bool(actions_valid[idx])}"
        ax_act.set_xlabel("dx (meters forward)")
        ax_act.set_ylabel("dy (meters left/right)")
        ax_act.set_title(
            f"8-step relative waypoints (dx, dy, dYaw){valid_str}"
        )
        ax_act.grid(True, linestyle="--", alpha=0.7)
        ax_act.axis("equal")
        ax_act.legend(loc="upper left")

        plt.draw()

    def on_key(event):
        if event.key in ["right", "d"]:
            state["idx"] = min(state["idx"] + 1, n_steps - 1)
            update_plot()
        elif event.key in ["left", "a"]:
            state["idx"] = max(state["idx"] - 1, 0)
            update_plot()
        elif event.key == "g":
            state["goal"] = state["idx"]
            plt.close()
        elif event.key == "q":
            plt.close()

    fig.canvas.mpl_connect("key_press_event", on_key)
    update_plot()
    plt.tight_layout()
    plt.show()
    f.close()

    return state["goal"]


def write_goal(h5_path: str, goal_step: int):
    """Write goal step into the H5 file."""
    with h5py.File(h5_path, "r+") as f:
        ts = f["timestamps_ns"][goal_step]
        f.attrs["goal_step"] = goal_step
        f.attrs["goal_timestamp_ns"] = int(ts)

        # Optional: store goal pose if odom data available
        if "actions" in f:
            # The pose at goal_step can be reconstructed from odom,
            # but we don't have odom in the H5.  Store the timestamp only.
            pass

        # Check if goal has RGB
        has_rgb = bool(f["rgb_mask"][goal_step])
        f.attrs["goal_has_rgb"] = has_rgb

        # Find rgb index if available
        if has_rgb and "rgb_indices" in f:
            ri = f["rgb_indices"][:]
            matches = np.where(ri == goal_step)[0]
            if len(matches) > 0:
                f.attrs["goal_rgb_index"] = int(matches[0])

    print(f"Goal saved: step={goal_step}, timestamp_ns={ts}, has_rgb={has_rgb}")


def main():
    parser = argparse.ArgumentParser(
        description="GUI visualizer with goal selection for H5 dataset."
    )
    parser.add_argument("--h5", required=True, help="Path to H5 file")
    args = parser.parse_args()

    if not os.path.exists(args.h5):
        print(f"Error: {args.h5} not found")
        sys.exit(1)

    goal = run_gui(args.h5)

    if goal >= 0:
        write_goal(args.h5, goal)
    else:
        print("No goal selected — H5 unchanged.")


if __name__ == "__main__":
    main()
