#!/usr/bin/env python3
"""
h5_to_video.py  –  Generate MP4 visualization video from an HDF5 dataset.

Layout matches h5_visualizer.py:
  ┌──────────────┬──────────────┐
  │  Event Voxel │   RGB Frame  │
  ├──────────────┴──────────────┤
  │    Action Trajectory Plot   │
  └─────────────────────────────┘

Usage:
  python scripts/h5_to_video.py --h5 path/to/file.h5 [--out video.mp4] [--fps 4] [--swap-rb]
"""

import argparse
import os
import sys

import cv2
import h5py
import matplotlib
matplotlib.use("Agg")          # headless – no display needed
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg


def fig_to_array(fig):
    """Render a matplotlib figure to an RGB numpy array."""
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = canvas.buffer_rgba()
    arr = np.asarray(buf)
    return arr[:, :, :3].copy()        # drop alpha → (H, W, 3) RGB


def make_frame(idx, voxels, actions, timestamps,
               rgb_mask, rgb_images, rgb_lookup,
               n_steps, swap_rb, fig, axes):
    """Draw one frame into *fig* and return an RGB numpy array."""
    ax_vox, ax_rgb, ax_act = axes

    # ── title ──────────────────────────────────────────────────────────────
    fig.suptitle(f"Step {idx} / {n_steps - 1}  |  TS {timestamps[idx]} ns",
                 fontsize=13, fontweight="bold")

    # ── 1. event voxel ────────────────────────────────────────────────────
    ax_vox.clear()
    v = voxels[idx]
    v_img = np.sum(v, axis=0)                 # sum across bins
    ax_vox.imshow(v_img, cmap="inferno")
    ax_vox.set_title(f"Event Voxel ({v.shape[1]}×{v.shape[2]})\nSum of {v.shape[0]} bins",
                     fontsize=10)
    ax_vox.axis("off")

    # ── 2. RGB ─────────────────────────────────────────────────────────────
    ax_rgb.clear()
    if rgb_mask[idx]:
        img_idx = rgb_lookup.get(idx)
        if img_idx is not None:
            img = rgb_images[img_idx]
            if swap_rb:
                img = img[:, :, ::-1]         # BGR ↔ RGB
            ax_rgb.imshow(img)
            ax_rgb.set_title(f"RGB ({img.shape[1]}×{img.shape[0]})", fontsize=10)
        else:
            ax_rgb.text(0.5, 0.5, "INDEX ERROR", ha="center", va="center",
                        color="orange", fontsize=12)
    else:
        ax_rgb.text(0.5, 0.5, "NO RGB FRAME", ha="center", va="center",
                    color="red", fontsize=12)
    ax_rgb.axis("off")

    # ── 3. action trajectory ───────────────────────────────────────────────
    ax_act.clear()
    act = actions[idx]
    xs, ys = act[:, 0], act[:, 1]
    ax_act.plot(xs, ys, "b-o", linewidth=2, markersize=5, label="8-step action chunk")
    for i in range(len(act)):
        yaw = act[i, 2]
        ax_act.arrow(xs[i], ys[i],
                     0.05 * np.cos(yaw), 0.05 * np.sin(yaw),
                     head_width=0.02, color="green")
    ax_act.scatter(0, 0, c="red", marker="X", s=120, zorder=5,
                   label="Robot (current)")
    ax_act.set_xlabel("X  (meters forward)")
    ax_act.set_ylabel("Y  (meters left/right)")
    ax_act.set_title("Action Chunk (relative to current pose)", fontsize=10)
    ax_act.grid(True, linestyle="--", alpha=0.5)
    ax_act.axis("equal")
    ax_act.legend(loc="upper left", fontsize=8)

    return fig_to_array(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Generate MP4 visualization from HDF5 dataset.")
    parser.add_argument("--h5", required=True, help="Path to HDF5 file")
    parser.add_argument("--out", default=None,
                        help="Output MP4 path (default: <h5_stem>_viz.mp4)")
    parser.add_argument("--fps", type=int, default=4,
                        help="Frames per second (default 4)")
    parser.add_argument("--swap-rb", action="store_true",
                        help="Swap R↔B channels in stored RGB images "
                             "(fix for old Bayer demosaic bug)")
    parser.add_argument("--start", type=int, default=0,
                        help="First step to render (default 0)")
    parser.add_argument("--end", type=int, default=-1,
                        help="Last step to render, inclusive (-1 = last)")
    parser.add_argument("--dpi", type=int, default=120,
                        help="Figure DPI (higher = larger video)")
    args = parser.parse_args()

    if not os.path.isfile(args.h5):
        sys.exit(f"[error] File not found: {args.h5}")

    # ── default output path ────────────────────────────────────────────────
    if args.out is None:
        stem = os.path.splitext(os.path.basename(args.h5))[0]
        args.out = os.path.join(os.path.dirname(args.h5) or ".", f"{stem}_viz.mp4")

    # ── open HDF5 ──────────────────────────────────────────────────────────
    f = h5py.File(args.h5, "r")
    voxels     = f["voxels"]
    actions    = f["actions"]
    timestamps = f["timestamps_ns"]
    rgb_mask   = f["rgb_mask"]
    rgb_images = f["rgb_images"]
    rgb_indices = f["rgb_indices"] if "rgb_indices" in f else None

    n_steps = len(voxels)
    start = max(0, args.start)
    end = n_steps - 1 if args.end < 0 else min(args.end, n_steps - 1)
    total = end - start + 1

    print(f"H5        : {args.h5}")
    print(f"Steps     : {n_steps}  (rendering {start}..{end}, {total} frames)")
    print(f"RGB images: {len(rgb_images)}, swap R↔B: {args.swap_rb}")
    print(f"Output    : {args.out}  @ {args.fps} fps, {args.dpi} dpi")

    # ── build RGB lookup ───────────────────────────────────────────────────
    rgb_lookup = {}
    if rgb_indices is not None:
        ri = rgb_indices[:]
        for img_i, step_i in enumerate(ri):
            rgb_lookup[int(step_i)] = img_i

    # ── matplotlib figure (reused every frame) ─────────────────────────────
    fig = plt.figure(figsize=(14, 9), dpi=args.dpi)
    gs = fig.add_gridspec(2, 2)
    ax_vox = fig.add_subplot(gs[0, 0])
    ax_rgb = fig.add_subplot(gs[0, 1])
    ax_act = fig.add_subplot(gs[1, :])
    axes = (ax_vox, ax_rgb, ax_act)

    # render first frame to get video dimensions
    frame0 = make_frame(start, voxels, actions, timestamps,
                        rgb_mask, rgb_images, rgb_lookup,
                        n_steps, args.swap_rb, fig, axes)
    h, w = frame0.shape[:2]
    plt.tight_layout(rect=[0, 0, 1, 0.95])      # leave room for suptitle

    # ── open video writer ──────────────────────────────────────────────────
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.out, fourcc, args.fps, (w, h))
    if not writer.isOpened():
        sys.exit(f"[error] Cannot open video writer for {args.out}")

    # write first frame
    writer.write(cv2.cvtColor(frame0, cv2.COLOR_RGB2BGR))

    # ── render remaining frames ────────────────────────────────────────────
    for idx in range(start + 1, end + 1):
        frame = make_frame(idx, voxels, actions, timestamps,
                           rgb_mask, rgb_images, rgb_lookup,
                           n_steps, args.swap_rb, fig, axes)
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        if (idx - start) % 20 == 0 or idx == end:
            pct = (idx - start + 1) / total * 100
            print(f"  [{pct:5.1f}%] step {idx}/{end}", end="\r")

    writer.release()
    plt.close(fig)
    f.close()

    sz_mb = os.path.getsize(args.out) / 1e6
    print(f"\nDone. {total} frames → {args.out}  ({sz_mb:.1f} MB)")


if __name__ == "__main__":
    main()
