#!/usr/bin/env python3
"""
convert_all_rosbags.py — Batch ROSbag → HDF5 converter

Converts GOOD bags (from bag_report.jsonl) to HDF5 files.
Uses pure-Python bag reader (handles truncated MCAP) and EVT3 decoder.

Replicates the logic of bag_to_h5.py but without ROS2 dependencies.

Output directory: --out-dir (default: /scratch/kvinod/bags/eGo_navi_.h5/)
  One .h5 per bag, named <bag_name>.h5

H5 schema:
  voxels       (N, 5, 720, 1280)   float32  gzip
  actions      (N, 8, 3)           float32
  timestamps_ns(N,)                int64
  rgb_mask     (N,)                bool
  rgb_images   (M, 1024, 1280, 3)  uint8    gzip
  rgb_indices  (M,)                int32

Usage
-----
  python scripts/convert_all_rosbags.py --bags-dir /scratch/kvinod/bags
  python scripts/convert_all_rosbags.py --bags-dir /scratch/kvinod/bags --bag-name data_collect_20260206_165620
"""

import argparse
import bisect
import json
import os
import sys
import time
import traceback
from pathlib import Path

import cv2
import h5py
import numpy as np
from scipy.spatial.transform import Rotation as R

# Add repo root to path
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from bag_reader import read_bag, deserialize, get_topics, get_ts
from evt3_decoder import decode_evt3

# ── constants (match bag_to_h5.py) ─────────────────────────────────────────────
VOXEL_WINDOW_US = 250_000   # 250 ms windows
NUM_BINS        = 5
HEIGHT, WIDTH   = 720, 1280
RGB_HEIGHT, RGB_WIDTH = 1024, 1280
ACTION_CHUNK    = 8
MIN_TRAJ_LEN    = 10

# Topic aliases
EVENT_ALIASES = ["/event_camera/events"]
RGB_ALIASES   = ["/cam_sync/cam0/image_raw", "/flir_camera/image_raw"]
ODOM_ALIASES  = ["/odom"]


# ── helpers ────────────────────────────────────────────────────────────────────
def resolve_topic(available_topics, aliases):
    for alias in aliases:
        if alias in available_topics:
            return alias
    return None


def get_header_stamp_ns(msg):
    if hasattr(msg, "header"):
        return msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
    return None


def events_to_voxel_fast(events, num_bins, height, width, t_start, t_end):
    """Optimised voxeliser using bincount (from bag_to_h5.py)."""
    voxel = np.zeros(num_bins * height * width, dtype=np.float32)
    if events is None or len(events) == 0:
        return voxel.reshape(num_bins, height, width)

    t = events[:, 2].astype(np.float64)
    dt = float(t_end - t_start)
    if dt <= 0:
        return voxel.reshape(num_bins, height, width)

    bin_norm = (t - t_start) / dt * (num_bins - 1)
    bin_low  = np.floor(bin_norm).astype(np.int32)
    bin_high = bin_low + 1
    w_high   = (bin_norm - bin_low).astype(np.float32)
    w_low    = (1.0 - w_high).astype(np.float32)

    x, y = events[:, 0].astype(np.int32), events[:, 1].astype(np.int32)
    pol = np.where(events[:, 3] > 0, 1.0, -1.0).astype(np.float32)

    base_idx = y * width + x
    idx_low  = bin_low * (height * width) + base_idx
    idx_high = bin_high * (height * width) + base_idx

    mask_l = (
        (bin_low >= 0) & (bin_low < num_bins)
        & (x >= 0) & (x < width) & (y >= 0) & (y < height)
    )
    mask_h = (
        (bin_high >= 0) & (bin_high < num_bins)
        & (x >= 0) & (x < width) & (y >= 0) & (y < height)
    )

    voxel += np.bincount(
        idx_low[mask_l], weights=(pol * w_low)[mask_l], minlength=len(voxel)
    )
    voxel += np.bincount(
        idx_high[mask_h], weights=(pol * w_high)[mask_h], minlength=len(voxel)
    )

    voxel = voxel.reshape(num_bins, height, width)
    for b in range(num_bins):
        v_max = np.abs(voxel[b]).max()
        if v_max > 0:
            voxel[b] /= v_max
    return voxel


def interpolate_pose(odom_ts, poses, query_ts):
    idx = bisect.bisect_left(odom_ts, query_ts)
    if idx == 0:
        return poses[0]
    if idx >= len(odom_ts):
        return poses[-1]
    t0, t1 = odom_ts[idx - 1], odom_ts[idx]
    alpha = (query_ts - t0) / (t1 - t0) if t1 != t0 else 0.0
    p0, p1 = poses[idx - 1], poses[idx]
    x = p0[0] + alpha * (p1[0] - p0[0])
    y = p0[1] + alpha * (p1[1] - p0[1])
    dyaw = np.arctan2(np.sin(p1[2] - p0[2]), np.cos(p1[2] - p0[2]))
    return np.array([x, y, p0[2] + alpha * dyaw])


def compute_action_chunk(odom_ts, poses, current_ts_ns):
    ref = interpolate_pose(odom_ts, poses, current_ts_ns)
    chunk = []
    win_ns = VOXEL_WINDOW_US * 1000
    for k in range(1, ACTION_CHUNK + 1):
        fut = interpolate_pose(odom_ts, poses, current_ts_ns + k * win_ns)
        dx, dy = fut[0] - ref[0], fut[1] - ref[1]
        dx_l = dx * np.cos(ref[2]) + dy * np.sin(ref[2])
        dy_l = -dx * np.sin(ref[2]) + dy * np.cos(ref[2])
        chunk.append([
            dx_l, dy_l,
            np.arctan2(np.sin(fut[2] - ref[2]), np.cos(fut[2] - ref[2])),
        ])
    return np.array(chunk, dtype=np.float32)


def decode_rgb(msg):
    """Decode an image message to RGB (NOT BGR) numpy array."""
    h, w = msg.height, msg.width
    data = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    enc = msg.encoding.lower()

    if "bayer_rggb" in enc:
        img = data.reshape((h, w))
        img = cv2.cvtColor(img, cv2.COLOR_BayerRG2RGB)
    elif "bayer" in enc:
        img = data.reshape((h, w))
        img = cv2.cvtColor(img, cv2.COLOR_BayerBG2RGB)
    elif "bgr" in enc:
        img = data.reshape((h, w, 3))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif "rgb" in enc:
        img = data.reshape((h, w, 3))
    elif "mono" in enc:
        img = data.reshape((h, w))
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        # Default: try 3-channel
        try:
            img = data.reshape((h, w, 3))
        except ValueError:
            img = data.reshape((h, w))
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    # Horizontal flip to match original bag_to_h5.py behaviour
    img = cv2.flip(img, 1)
    return img


# ── single bag converter ──────────────────────────────────────────────────────
def convert_bag(bag_dir: str, h5_path: str, dry_run: bool = False):
    """Convert one bag to HDF5. Returns dict with stats."""
    bag_name = os.path.basename(bag_dir)
    result = {
        "bag_name": bag_name,
        "h5_path": h5_path,
        "status": "UNKNOWN",
        "steps": 0,
        "rgb_stored": 0,
        "error": None,
    }

    # Resolve topics from metadata
    available = get_topics(bag_dir)
    event_topic = resolve_topic(available, EVENT_ALIASES)
    rgb_topic = resolve_topic(available, RGB_ALIASES)
    odom_topic = resolve_topic(available, ODOM_ALIASES)

    if not all([event_topic, rgb_topic, odom_topic]):
        result["status"] = "SKIP"
        result["error"] = f"Missing topics: ev={event_topic} rgb={rgb_topic} odom={odom_topic}"
        return result

    if dry_run:
        result["status"] = "DRY_RUN"
        return result

    print(f"  Pass 1: Pre-scanning odom + rgb timestamps ...")

    # ── Pass 1: collect odom, rgb timestamps, find event sync offset ──────
    odom_ts = []
    odom_poses = []
    rgb_ts = []
    rgb_raw = []  # (bag_ts, raw_data, msgtype)
    bag_to_sensor_offset_ns = None

    for msg in read_bag(bag_dir):
        if msg.topic == odom_topic:
            obj = deserialize(msg)
            ts = get_header_stamp_ns(obj)
            if ts is None:
                continue
            p, q = obj.pose.pose.position, obj.pose.pose.orientation
            yaw = R.from_quat([q.x, q.y, q.z, q.w]).as_euler("xyz")[2]
            odom_ts.append(ts)
            odom_poses.append(np.array([p.x, p.y, yaw]))

        elif msg.topic == rgb_topic:
            obj = deserialize(msg)
            ts = get_header_stamp_ns(obj)
            if ts:
                rgb_ts.append(ts)
                rgb_raw.append((ts, msg.data, msg.msgtype))

        elif msg.topic == event_topic and bag_to_sensor_offset_ns is None:
            obj = deserialize(msg)
            hdr_ts = get_header_stamp_ns(obj)
            if hdr_ts is not None:
                # Decode first events to get actual sensor timestamp
                raw_ev = bytes(obj.events)
                if len(raw_ev) >= 4:
                    first_evs = decode_evt3(raw_ev, obj.width, obj.height)
                    if first_evs is not None and len(first_evs) > 0:
                        first_sensor_us = int(first_evs[0, 2])
                        bag_to_sensor_offset_ns = hdr_ts - first_sensor_us * 1000
                if bag_to_sensor_offset_ns is None:
                    # Fallback: use time_base field (may be 0 → wrong)
                    bag_to_sensor_offset_ns = hdr_ts - (obj.time_base * 1000)

    if not odom_ts:
        result["status"] = "FAIL"
        result["error"] = "No odom messages"
        return result
    if bag_to_sensor_offset_ns is None:
        result["status"] = "FAIL"
        result["error"] = "No event messages for sync"
        return result

    odom_poses = [p for _, p in sorted(zip(odom_ts, odom_poses))]
    odom_ts.sort()
    rgb_ts_sorted = sorted(rgb_ts)

    print(f"  Odom: {len(odom_ts)} msgs, RGB: {len(rgb_ts)} frames")
    print(f"  Sensor offset: {bag_to_sensor_offset_ns / 1e6:.2f} ms")

    # ── Pass 2: stream events, build voxels ───────────────────────────────
    print(f"  Pass 2: Streaming events → voxels ...")

    win_ns = VOXEL_WINDOW_US * 1000
    curr_win_start = max(odom_ts[0], rgb_ts_sorted[0] if rgb_ts_sorted else odom_ts[0], bag_to_sensor_offset_ns)
    last_safe = odom_ts[-1] - ACTION_CHUNK * win_ns

    if curr_win_start + win_ns > last_safe:
        result["status"] = "FAIL"
        result["error"] = "Trajectory too short for action chunks"
        return result

    event_buffer = []
    step_idx = 0
    rgb_stored_count = 0

    # Pre-index rgb_raw by header timestamp for quick lookup
    rgb_data_map = {}
    for ts_val, raw_data, msgtype in rgb_raw:
        rgb_data_map[ts_val] = (raw_data, msgtype)

    os.makedirs(os.path.dirname(h5_path), exist_ok=True)

    with h5py.File(h5_path, "w") as f:
        d_vox = f.create_dataset(
            "voxels", (0, NUM_BINS, HEIGHT, WIDTH),
            maxshape=(None, NUM_BINS, HEIGHT, WIDTH),
            dtype="float32", compression="gzip",
            chunks=(1, NUM_BINS, HEIGHT, WIDTH),
        )
        d_act = f.create_dataset(
            "actions", (0, ACTION_CHUNK, 3),
            maxshape=(None, ACTION_CHUNK, 3), dtype="float32",
        )
        d_ts = f.create_dataset(
            "timestamps_ns", (0,), maxshape=(None,), dtype="int64",
        )
        d_msk = f.create_dataset(
            "rgb_mask", (0,), maxshape=(None,), dtype="bool",
        )
        d_rgb = f.create_dataset(
            "rgb_images", (0, RGB_HEIGHT, RGB_WIDTH, 3),
            maxshape=(None, RGB_HEIGHT, RGB_WIDTH, 3),
            dtype="uint8", compression="gzip",
            chunks=(1, RGB_HEIGHT, RGB_WIDTH, 3),
        )
        d_rid = f.create_dataset(
            "rgb_indices", (0,), maxshape=(None,), dtype="int32",
        )

        for bag_msg in read_bag(bag_dir):
            if bag_msg.topic != event_topic:
                continue
            if curr_win_start + win_ns > last_safe:
                break

            # Decode event packet
            obj = deserialize(bag_msg)
            raw_events = bytes(obj.events)
            if len(raw_events) < 4:
                continue

            evs = decode_evt3(raw_events, obj.width, obj.height)
            if evs is None or len(evs) == 0:
                continue

            event_buffer.append(evs)

            # Check if we have enough events to fill the current window
            t_sensor_query = (curr_win_start + win_ns - bag_to_sensor_offset_ns) // 1000
            if evs[-1, 2] < t_sensor_query:
                continue

            # Concatenate and process windows
            all_evs = np.concatenate(event_buffer)

            while len(all_evs) > 0 and all_evs[-1, 2] >= t_sensor_query:
                t_sensor_start = (curr_win_start - bag_to_sensor_offset_ns) // 1000
                split = bisect.bisect_left(all_evs[:, 2], t_sensor_query)

                voxel = events_to_voxel_fast(
                    all_evs[:split], NUM_BINS, HEIGHT, WIDTH,
                    t_sensor_start, t_sensor_query,
                )

                # RGB matching
                win_center = curr_win_start + win_ns // 2
                r_idx = bisect.bisect_left(rgb_ts_sorted, win_center)
                has_rgb = False

                if r_idx < len(rgb_ts_sorted):
                    nearest_ts = rgb_ts_sorted[r_idx]
                    if abs(nearest_ts - win_center) < win_ns // 2:
                        if nearest_ts in rgb_data_map:
                            raw_data, msgtype = rgb_data_map[nearest_ts]
                            try:
                                typestore = get_ts()
                                img_msg = typestore.deserialize_cdr(raw_data, msgtype)
                                img = decode_rgb(img_msg)
                                d_rgb.resize(rgb_stored_count + 1, axis=0)
                                d_rid.resize(rgb_stored_count + 1, axis=0)
                                d_rgb[rgb_stored_count] = img
                                d_rid[rgb_stored_count] = step_idx
                                rgb_stored_count += 1
                                has_rgb = True
                            except Exception:
                                pass

                # Write step
                for d in [d_vox, d_act, d_ts, d_msk]:
                    d.resize(step_idx + 1, axis=0)
                d_vox[step_idx] = voxel
                d_act[step_idx] = compute_action_chunk(
                    odom_ts, odom_poses, curr_win_start
                )
                d_ts[step_idx] = curr_win_start
                d_msk[step_idx] = has_rgb

                step_idx += 1
                curr_win_start += win_ns
                t_sensor_query = (
                    curr_win_start + win_ns - bag_to_sensor_offset_ns
                ) // 1000

                if step_idx % 20 == 0:
                    print(
                        f"\r  Steps: {step_idx} | RGB: {rgb_stored_count}",
                        end="", flush=True,
                    )

                all_evs = all_evs[split:]
                if curr_win_start + win_ns > last_safe:
                    break

            event_buffer = [all_evs] if len(all_evs) > 0 else []

        # Store metadata
        f.attrs["bag_name"] = bag_name
        f.attrs["bag_dir"] = bag_dir
        f.attrs["event_topic"] = event_topic
        f.attrs["rgb_topic"] = rgb_topic
        f.attrs["odom_topic"] = odom_topic
        f.attrs["voxel_window_us"] = VOXEL_WINDOW_US
        f.attrs["num_bins"] = NUM_BINS
        f.attrs["action_chunk"] = ACTION_CHUNK

    print(f"\n  Done: {step_idx} steps, {rgb_stored_count} RGB frames → {h5_path}")

    result["status"] = "OK"
    result["steps"] = step_idx
    result["rgb_stored"] = rgb_stored_count
    return result


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Convert GOOD ROS2 bags to HDF5 dataset."
    )
    parser.add_argument("--bags-dir", required=True)
    parser.add_argument(
        "--out-dir",
        default="/scratch/kvinod/bags/eGo_navi_.h5",
        help="Output directory for .h5 files.",
    )
    parser.add_argument(
        "--report",
        default=None,
        help="Path to bag_report.jsonl (default: <bags-dir>/bag_report.jsonl).",
    )
    parser.add_argument(
        "--bag-name",
        default=None,
        help="Convert only this bag (skip report check).",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true", help="Re-convert even if .h5 exists.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Determine which bags to convert
    if args.bag_name:
        bag_dirs = [os.path.join(args.bags_dir, args.bag_name)]
        if not os.path.isdir(bag_dirs[0]):
            print(f"[ERROR] {bag_dirs[0]} not found")
            sys.exit(1)
    else:
        # Load report
        report_path = args.report or os.path.join(args.bags_dir, "bag_report.jsonl")
        if not os.path.exists(report_path):
            print(f"[ERROR] No report at {report_path}. Run check_rosbags.py first.")
            sys.exit(1)

        good_bags = []
        with open(report_path) as f:
            for line in f:
                r = json.loads(line)
                if r["status"] == "GOOD":
                    good_bags.append(r["bag_dir"])

        if not good_bags:
            print("[ERROR] No GOOD bags in report.")
            sys.exit(1)

        bag_dirs = good_bags
        print(f"Found {len(bag_dirs)} GOOD bag(s) to convert.")

    # Warm up typestore
    get_ts()

    results = []
    for i, bag_dir in enumerate(bag_dirs):
        bag_name = os.path.basename(bag_dir)
        h5_path = os.path.join(args.out_dir, f"{bag_name}.h5")

        print(f"\n[{i+1}/{len(bag_dirs)}] {bag_name}")

        if os.path.exists(h5_path) and not args.force:
            print(f"  SKIP (exists): {h5_path}")
            results.append({
                "bag_name": bag_name, "status": "SKIP_EXISTS",
                "h5_path": h5_path, "steps": 0, "rgb_stored": 0, "error": None,
            })
            continue

        t0 = time.time()
        try:
            r = convert_bag(bag_dir, h5_path, dry_run=args.dry_run)
        except Exception as e:
            r = {
                "bag_name": bag_name, "status": "FAIL",
                "h5_path": h5_path, "steps": 0, "rgb_stored": 0,
                "error": str(e),
                "traceback": traceback.format_exc(),
            }
        r["elapsed_sec"] = round(time.time() - t0, 1)
        results.append(r)

        if r["status"] == "FAIL":
            print(f"  FAIL: {r['error']}")

    # Save conversion report
    report_out = os.path.join(args.out_dir, "convert_report.jsonl")
    with open(report_out, "w") as f:
        for r in results:
            f.write(json.dumps(r, default=str) + "\n")

    # Summary
    ok = sum(1 for r in results if r["status"] == "OK")
    fail = sum(1 for r in results if r["status"] == "FAIL")
    skip = sum(1 for r in results if "SKIP" in r["status"])
    total_steps = sum(r.get("steps", 0) for r in results)
    total_rgb = sum(r.get("rgb_stored", 0) for r in results)

    print(f"\n{'='*60}")
    print(f"  OK: {ok}  FAIL: {fail}  SKIP: {skip}  TOTAL: {len(results)}")
    print(f"  Total steps: {total_steps}  Total RGB: {total_rgb}")
    print(f"  Report: {report_out}")
    print(f"{'='*60}")

    if fail > 0:
        print("\nFailed bags:")
        for r in results:
            if r["status"] == "FAIL":
                print(f"  {r['bag_name']}: {r['error']}")


if __name__ == "__main__":
    main()
