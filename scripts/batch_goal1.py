#!/usr/bin/env python3
"""
batch_goal1.py — Convert all Goal-1 rosbags, auto-set goal step, generate videos.

Pipeline per bag:
  1. Convert rosbag → H5  (reuses convert_all_rosbags.convert_bag)
  2. Find last non-blank voxel step → set as goal_step in H5 attrs
  3. Generate visualization video (RGB + events + actions + goal annotation)
     saved to  <out_dir>/media/<bag_name>_viz.mp4

Usage:
  python scripts/batch_goal1.py [--force] [--skip-video] [--jobs 1]
"""

import argparse
import gc
import json
import os
import sys
import time
import traceback

import cv2
import h5py
import numpy as np

# ── Add repo root to path so we can import the converter ──────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, SCRIPT_DIR)

from convert_all_rosbags import convert_bag

# ── Constants ─────────────────────────────────────────────────────────────────
BAGS_DIR = "/scratch/kvinod/bags"
OUT_DIR = "/scratch/kvinod/bags/ego_navi_goal_1"
MEDIA_DIR = os.path.join(OUT_DIR, "media")

# All Goal-1 bag names (Feb 19 2026, 165934 → 175704)
GOAL1_BAGS = [
    "data_collect_20260219_165934",
    "data_collect_20260219_170113",
    "data_collect_20260219_171045",
    "data_collect_20260219_171244",
    "data_collect_20260219_171435",
    "data_collect_20260219_171811",
    "data_collect_20260219_171940",
    "data_collect_20260219_172524",
    "data_collect_20260219_172858",
    "data_collect_20260219_173118",
    "data_collect_20260219_173408",
    "data_collect_20260219_173616",
    "data_collect_20260219_173849",
    "data_collect_20260219_174146",
    "data_collect_20260219_174441",
    "data_collect_20260219_174819",
    "data_collect_20260219_175156",
    "data_collect_20260219_175453",
    "data_collect_20260219_175704",
]


# ── Goal detection ────────────────────────────────────────────────────────────

def find_last_nonblank_voxel(h5_path):
    """Return the index of the last step whose voxel is not all-zero.

    Scans backward from the end for efficiency.
    Returns -1 if every voxel is blank.
    """
    with h5py.File(h5_path, "r") as f:
        voxels = f["voxels"]
        n = len(voxels)
        for i in range(n - 1, -1, -1):
            v = voxels[i]
            if np.any(v != 0):
                return i
    return -1


def set_goal_step(h5_path, goal_step):
    """Write goal_step into H5 attrs."""
    with h5py.File(h5_path, "r+") as f:
        f.attrs["goal_step"] = goal_step


# ── Video generation (inline, avoids subprocess overhead) ─────────────────────

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg


def _fig_to_array(fig):
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = canvas.buffer_rgba()
    arr = np.asarray(buf)
    return arr[:, :, :3].copy()


def _make_frame(idx, voxels, actions, timestamps, rgb_mask, rgb_images,
                rgb_lookup, n_steps, fig, axes, goal_step=-1):
    ax_vox, ax_rgb, ax_act = axes
    is_goal = (goal_step >= 0 and idx == goal_step)

    # title
    title = f"Step {idx} / {n_steps - 1}  |  TS {timestamps[idx]} ns"
    if is_goal:
        title += "  \u2605 GOAL STEP \u2605"
    if goal_step >= 0:
        title += f"  [goal={goal_step}]"
    fig.suptitle(title, fontsize=13, fontweight="bold",
                 color="green" if is_goal else "black")

    # 1. voxel
    ax_vox.clear()
    v = voxels[idx]
    v_img = np.sum(v, axis=0)
    ax_vox.imshow(v_img, cmap="inferno")
    vox_title = f"Event Voxel ({v.shape[1]}\u00d7{v.shape[2]})\nSum of {v.shape[0]} bins"
    if is_goal:
        vox_title = "\u2605 GOAL VOXEL \u2605\n" + vox_title
        for spine in ax_vox.spines.values():
            spine.set_visible(True)
            spine.set_color("lime")
            spine.set_linewidth(4)
    ax_vox.set_title(vox_title, fontsize=10,
                     color="lime" if is_goal else "black")
    ax_vox.axis("off")

    # 2. RGB (already stored in correct RGB order from fixed Bayer demosaic)
    ax_rgb.clear()
    if rgb_mask[idx]:
        img_idx = rgb_lookup.get(idx)
        if img_idx is not None:
            img = rgb_images[img_idx]
            ax_rgb.imshow(img)
            ax_rgb.set_title(f"RGB ({img.shape[1]}\u00d7{img.shape[0]})", fontsize=10)
        else:
            ax_rgb.text(0.5, 0.5, "INDEX ERROR", ha="center", va="center",
                        color="orange", fontsize=12)
    else:
        ax_rgb.text(0.5, 0.5, "NO RGB FRAME", ha="center", va="center",
                    color="red", fontsize=12)
    ax_rgb.axis("off")

    # 3. actions
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

    return _fig_to_array(fig)


def generate_video(h5_path, out_mp4, goal_step=-1, fps=4, dpi=100):
    """Generate visualization MP4 for one H5 file."""
    f = h5py.File(h5_path, "r")
    voxels = f["voxels"]
    actions = f["actions"]
    timestamps = f["timestamps_ns"]
    rgb_mask = f["rgb_mask"]
    rgb_images = f["rgb_images"]
    rgb_indices = f["rgb_indices"] if "rgb_indices" in f else None

    n_steps = len(voxels)

    # Build RGB lookup
    rgb_lookup = {}
    if rgb_indices is not None:
        ri = rgb_indices[:]
        for img_i, step_i in enumerate(ri):
            rgb_lookup[int(step_i)] = img_i

    fig = plt.figure(figsize=(14, 9), dpi=dpi)
    gs = fig.add_gridspec(2, 2)
    ax_vox = fig.add_subplot(gs[0, 0])
    ax_rgb = fig.add_subplot(gs[0, 1])
    ax_act = fig.add_subplot(gs[1, :])
    axes = (ax_vox, ax_rgb, ax_act)

    # First frame
    frame0 = _make_frame(0, voxels, actions, timestamps,
                         rgb_mask, rgb_images, rgb_lookup,
                         n_steps, fig, axes, goal_step)
    h, w = frame0.shape[:2]
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_mp4, fourcc, fps, (w, h))
    writer.write(cv2.cvtColor(frame0, cv2.COLOR_RGB2BGR))

    for idx in range(1, n_steps):
        frame = _make_frame(idx, voxels, actions, timestamps,
                            rgb_mask, rgb_images, rgb_lookup,
                            n_steps, fig, axes, goal_step)
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        if idx % 20 == 0 or idx == n_steps - 1:
            print(f"    video [{100*(idx+1)/n_steps:5.1f}%] step {idx}/{n_steps-1}",
                  end="\r")

    writer.release()
    plt.close(fig)
    f.close()

    sz_mb = os.path.getsize(out_mp4) / 1e6
    print(f"    video: {n_steps} frames → {out_mp4}  ({sz_mb:.1f} MB)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Batch: convert Goal-1 bags → H5 + goal + video")
    parser.add_argument("--force", action="store_true",
                        help="Re-convert even if H5 exists")
    parser.add_argument("--skip-video", action="store_true",
                        help="Skip video generation")
    parser.add_argument("--video-only", action="store_true",
                        help="Skip conversion, only generate videos from existing H5")
    parser.add_argument("--fps", type=int, default=4)
    parser.add_argument("--dpi", type=int, default=100)
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(MEDIA_DIR, exist_ok=True)

    results = []
    n = len(GOAL1_BAGS)

    for i, bag_name in enumerate(GOAL1_BAGS):
        bag_dir = os.path.join(BAGS_DIR, bag_name)
        h5_path = os.path.join(OUT_DIR, f"{bag_name}.h5")
        video_path = os.path.join(MEDIA_DIR, f"{bag_name}_viz.mp4")

        print(f"\n{'='*70}")
        print(f"[{i+1}/{n}] {bag_name}")
        print(f"{'='*70}")

        if not os.path.isdir(bag_dir):
            print(f"  SKIP: bag dir not found: {bag_dir}")
            results.append({"bag": bag_name, "status": "MISSING_BAG"})
            continue

        # ── Step 1: Convert ───────────────────────────────────────────
        if not args.video_only:
            if os.path.exists(h5_path) and not args.force:
                print(f"  H5 exists, skipping conversion (use --force to redo)")
            else:
                t0 = time.time()
                try:
                    r = convert_bag(
                        bag_dir, h5_path,
                        compression="lzf",
                        voxel_dtype="float32",
                        flush_every=10,
                    )
                    elapsed = time.time() - t0
                    print(f"  Converted: {r['steps']} steps, "
                          f"{r['rgb_stored']} RGB in {elapsed:.0f}s "
                          f"[{r['status']}]")
                    if r["status"] != "OK":
                        results.append({"bag": bag_name, "status": "CONVERT_FAIL",
                                        "error": r.get("error")})
                        continue
                except Exception as e:
                    print(f"  CONVERT ERROR: {e}")
                    traceback.print_exc()
                    results.append({"bag": bag_name, "status": "CONVERT_ERROR",
                                    "error": str(e)})
                    continue
                finally:
                    gc.collect()

        if not os.path.isfile(h5_path):
            print(f"  SKIP: H5 not found: {h5_path}")
            results.append({"bag": bag_name, "status": "NO_H5"})
            continue

        # ── Step 2: Find goal & set in H5 ────────────────────────────
        print(f"  Finding last non-blank voxel...")
        goal_step = find_last_nonblank_voxel(h5_path)
        if goal_step < 0:
            print(f"  WARNING: All voxels are blank! Setting goal_step=-1")
        else:
            print(f"  Goal step = {goal_step} (last non-blank voxel)")
        set_goal_step(h5_path, goal_step)

        # Quick summary
        with h5py.File(h5_path, "r") as f:
            n_steps = len(f["voxels"])
            n_rgb = len(f["rgb_images"])
            gs = int(f.attrs.get("goal_step", -1))
            print(f"  H5 summary: {n_steps} steps, {n_rgb} RGB, goal_step={gs}")

        # ── Step 3: Generate video ────────────────────────────────────
        if args.skip_video:
            print(f"  Skipping video (--skip-video)")
        else:
            print(f"  Generating video...")
            try:
                generate_video(h5_path, video_path,
                               goal_step=goal_step,
                               fps=args.fps, dpi=args.dpi)
            except Exception as e:
                print(f"  VIDEO ERROR: {e}")
                traceback.print_exc()
                results.append({"bag": bag_name, "status": "VIDEO_FAIL",
                                "error": str(e)})
                continue
            finally:
                gc.collect()

        results.append({
            "bag": bag_name, "status": "OK",
            "h5": h5_path, "video": video_path,
            "steps": n_steps, "rgb": n_rgb, "goal_step": goal_step,
        })

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"BATCH COMPLETE")
    print(f"{'='*70}")
    ok = sum(1 for r in results if r["status"] == "OK")
    fail = sum(1 for r in results if r["status"] != "OK")
    print(f"  OK: {ok}  |  Errors: {fail}  |  Total: {len(results)}")
    print()
    for r in results:
        status = r["status"]
        sym = "\u2713" if status == "OK" else "\u2717"
        extra = ""
        if status == "OK":
            extra = f"  steps={r['steps']}, rgb={r['rgb']}, goal={r['goal_step']}"
        elif "error" in r and r["error"]:
            extra = f"  {r['error']}"
        print(f"  {sym} {r['bag']}: {status}{extra}")

    # Save report
    report_path = os.path.join(OUT_DIR, "goal1_report.json")
    with open(report_path, "w") as fh:
        json.dump(results, fh, indent=2, default=str)
    print(f"\nReport: {report_path}")


if __name__ == "__main__":
    main()
