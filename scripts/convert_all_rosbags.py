#!/usr/bin/env python3
"""
convert_all_rosbags.py  — v3: Odom-driven ROSbag → HDF5 converter

Architecture (v3)
~~~~~~~~~~~~~~~~~
* **Odom-driven stepping**: Steps are defined by odom time range, NOT by
  event coverage or RGB availability.  This ensures the full trajectory
  is captured including the goal at the end.
* **Actions as relative waypoints**: (dx, dy, dYaw) in robot local frame,
  exactly matching NoMaD/ViNT/GNM-style mid-level actions.
* **RGB nearest-neighbor matching**: For each step, pick the nearest RGB
  frame by timestamp (checking both neighbors around bisect point).
  rgb_mask indicates whether RGB is present for each step.
* **Goal labeling support**: goal_step attribute (default -1) can be set
  post-conversion by the goal picker visualizer.
* **Crash safety**: Writes to ``<path>.tmp``, atomically renames on success.
* **LZF compression** by default.

H5 schema (v3)
~~~~~~~~~~~~~~
  voxels          (N, 5, 720, 1280)   float32   [lzf]
  actions         (N, 8, 3)           float32     — (dx, dy, dYaw) relative waypoints
  actions_valid   (N,)                bool        — True if full horizon inside odom range
  timestamps_ns   (N,)                int64
  rgb_mask        (N,)                bool
  rgb_images      (M, 1024, 1280, 3)  uint8     [lzf]
  rgb_indices     (M,)                int32

  Attributes:
    bag_name, bag_dir, event_topic, rgb_topic, odom_topic
    voxel_window_us, num_bins, action_chunk
    compression, voxel_dtype
    actions_space='relative_waypoints'
    actions_repr='dx_dy_dyaw'
    actions_frame='base_link_at_step'
    dyaw_wrapped='true'
    goal_step=-1  (updated by goal picker)

Usage
-----
  # Fast default (lzf, CPU, sequential)
  python scripts/convert_all_rosbags.py --bags-dir /scratch/kvinod/bags

  # Single bag
  python scripts/convert_all_rosbags.py --bags-dir /scratch/kvinod/bags \\
      --bag-name data_collect_20260207_150734 --force
"""

import argparse
import bisect
import gc
import json
import multiprocessing
import os
import sys
import time
import traceback
from pathlib import Path

import cv2
import h5py
import numpy as np
from scipy.spatial.transform import Rotation as R

# ── repo imports ───────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from bag_reader import read_bag, deserialize, get_topics, get_ts
from evt3_decoder import decode_evt3, EVT3StreamDecoder
from resource_utils import get_rss_mb, detect_resources, compute_safe_jobs

# ── constants ──────────────────────────────────────────────────────────────────
VOXEL_WINDOW_US = 250_000   # 250 ms windows
NUM_BINS        = 5
HEIGHT, WIDTH   = 720, 1280
RGB_HEIGHT, RGB_WIDTH = 1024, 1280
ACTION_CHUNK    = 8
MIN_TRAJ_LEN    = 10

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


def wrap_to_pi(angle):
    """Wrap angle to [-pi, pi]."""
    return np.arctan2(np.sin(angle), np.cos(angle))


# ── voxelisers ─────────────────────────────────────────────────────────────────

def events_to_voxel_fast(events, num_bins, height, width, t_start, t_end):
    """CPU voxeliser using numpy.bincount."""
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


def events_to_voxel_gpu(events, num_bins, height, width, t_start, t_end, device):
    """GPU voxeliser using torch.scatter_add_."""
    import torch

    n_flat = num_bins * height * width
    voxel = torch.zeros(n_flat, dtype=torch.float32, device=device)
    if events is None or len(events) == 0:
        return voxel.reshape(num_bins, height, width).cpu().numpy()

    dt = float(t_end - t_start)
    if dt <= 0:
        return voxel.reshape(num_bins, height, width).cpu().numpy()

    ev_t = torch.from_numpy(events[:, 2].astype(np.float64)).to(device)
    ev_x = torch.from_numpy(events[:, 0].astype(np.int64)).to(device)
    ev_y = torch.from_numpy(events[:, 1].astype(np.int64)).to(device)
    ev_p = torch.from_numpy(events[:, 3].astype(np.float32)).to(device)
    pol = torch.where(ev_p > 0, 1.0, -1.0)

    bin_norm = ((ev_t - t_start) / dt * (num_bins - 1)).float()
    bin_low = bin_norm.floor().long()
    bin_high = bin_low + 1
    w_high = (bin_norm - bin_low.float())
    w_low = 1.0 - w_high

    base_idx = ev_y * width + ev_x
    idx_low = bin_low * (height * width) + base_idx
    idx_high = bin_high * (height * width) + base_idx

    valid = (ev_x >= 0) & (ev_x < width) & (ev_y >= 0) & (ev_y < height)
    valid_l = valid & (bin_low >= 0) & (bin_low < num_bins)
    valid_h = valid & (bin_high >= 0) & (bin_high < num_bins)

    voxel.scatter_add_(0, idx_low[valid_l], (pol * w_low)[valid_l])
    voxel.scatter_add_(0, idx_high[valid_h], (pol * w_high)[valid_h])

    voxel = voxel.reshape(num_bins, height, width)
    for b in range(num_bins):
        v_max = voxel[b].abs().max()
        if v_max > 0:
            voxel[b] /= v_max

    return voxel.cpu().numpy()


# ── pose / action helpers ──────────────────────────────────────────────────────

def interpolate_pose(odom_ts, poses, query_ts):
    """Interpolate odom pose at an arbitrary timestamp."""
    idx = bisect.bisect_left(odom_ts, query_ts)
    if idx == 0:
        return poses[0].copy()
    if idx >= len(odom_ts):
        return poses[-1].copy()
    t0, t1 = odom_ts[idx - 1], odom_ts[idx]
    alpha = (query_ts - t0) / (t1 - t0) if t1 != t0 else 0.0
    p0, p1 = poses[idx - 1], poses[idx]
    x = p0[0] + alpha * (p1[0] - p0[0])
    y = p0[1] + alpha * (p1[1] - p0[1])
    dyaw = wrap_to_pi(p1[2] - p0[2])
    return np.array([x, y, p0[2] + alpha * dyaw])


def compute_action_chunk(odom_ts, poses, current_ts_ns, odom_end_ns):
    """Compute (dx, dy, dYaw) relative waypoint actions.

    Returns:
        actions: (ACTION_CHUNK, 3) float32 — (dx, dy, dYaw) per horizon step
        valid: bool — True if full horizon is within odom range
    """
    ref = interpolate_pose(odom_ts, poses, current_ts_ns)
    chunk = []
    win_ns = VOXEL_WINDOW_US * 1000
    cos_yaw = np.cos(ref[2])
    sin_yaw = np.sin(ref[2])

    horizon_end = current_ts_ns + ACTION_CHUNK * win_ns
    valid = horizon_end <= odom_end_ns

    for k in range(1, ACTION_CHUNK + 1):
        fut_ts = current_ts_ns + k * win_ns
        fut = interpolate_pose(odom_ts, poses, fut_ts)

        # World-frame delta
        dx_w = fut[0] - ref[0]
        dy_w = fut[1] - ref[1]

        # Rotate into robot local frame at current step (R(-yaw))
        dx_l =  dx_w * cos_yaw + dy_w * sin_yaw
        dy_l = -dx_w * sin_yaw + dy_w * cos_yaw

        # Relative yaw change
        d_yaw = wrap_to_pi(fut[2] - ref[2])

        chunk.append([dx_l, dy_l, d_yaw])

    return np.array(chunk, dtype=np.float32), valid


# ── image helpers ──────────────────────────────────────────────────────────────

def decode_rgb(msg):
    """Decode an image message to RGB numpy array."""
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
        try:
            img = data.reshape((h, w, 3))
        except ValueError:
            img = data.reshape((h, w))
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    img = cv2.flip(img, 1)
    return img


# ── compression helper ─────────────────────────────────────────────────────────

def _comp_kwargs(compression, gzip_level):
    """Return dict of compression keyword args for h5py.create_dataset."""
    if compression == "lzf":
        return {"compression": "lzf"}
    elif compression == "gzip":
        return {"compression": "gzip", "compression_opts": gzip_level}
    return {}


# ── nearest RGB matching ──────────────────────────────────────────────────────

def find_nearest_rgb(rgb_ts_sorted, win_center, tolerance_ns):
    """Find the nearest RGB timestamp to win_center within tolerance.

    Checks both neighbors around the bisect insertion point.

    Returns:
        (rgb_ts, True) if found within tolerance
        (None, False) if no match
    """
    if not rgb_ts_sorted:
        return None, False

    r_idx = bisect.bisect_left(rgb_ts_sorted, win_center)

    best_ts = None
    best_dist = tolerance_ns + 1  # sentinel

    # Check r_idx (right neighbor)
    if r_idx < len(rgb_ts_sorted):
        dist = abs(rgb_ts_sorted[r_idx] - win_center)
        if dist < best_dist:
            best_dist = dist
            best_ts = rgb_ts_sorted[r_idx]

    # Check r_idx - 1 (left neighbor)
    if r_idx > 0:
        dist = abs(rgb_ts_sorted[r_idx - 1] - win_center)
        if dist < best_dist:
            best_dist = dist
            best_ts = rgb_ts_sorted[r_idx - 1]

    if best_ts is not None and best_dist <= tolerance_ns:
        return best_ts, True
    return None, False


# ── single-bag converter ──────────────────────────────────────────────────────

def convert_bag(
    bag_dir: str,
    h5_path: str,
    *,
    compression: str = "lzf",
    gzip_level: int = 1,
    voxel_dtype: str = "float32",
    use_gpu: bool = False,
    flush_every: int = 10,
    log_memory: bool = False,
    dry_run: bool = False,
    rgb_tolerance_factor: float = 0.5,
):
    """Convert one bag to HDF5 with odom-driven stepping.

    Architecture:
      Pass 1 — lightweight scan: collect odom poses, RGB timestamps,
               and event sync offset.  **No RGB data stored in RAM.**
      Pass 2 — Pre-compute ALL step timestamps from odom timeline.
               Stream events + RGB; for each step, produce voxel (may be
               zero if no events), match nearest RGB, compute relative
               waypoint actions.  Write each step to extendable H5.
    """
    bag_name = os.path.basename(bag_dir)
    result = {
        "bag_name": bag_name,
        "h5_path": h5_path,
        "status": "UNKNOWN",
        "steps": 0,
        "rgb_stored": 0,
        "error": None,
    }

    # ── resolve topics ────────────────────────────────────────────────────
    available = get_topics(bag_dir)
    event_topic = resolve_topic(available, EVENT_ALIASES)
    rgb_topic = resolve_topic(available, RGB_ALIASES)
    odom_topic = resolve_topic(available, ODOM_ALIASES)

    if not event_topic:
        result["status"] = "FAIL"
        result["error"] = f"Missing event topic (tried {EVENT_ALIASES})"
        return result
    if not odom_topic:
        result["status"] = "FAIL"
        result["error"] = f"Missing odom topic (tried {ODOM_ALIASES})"
        return result
    # RGB is optional — no failure if missing
    if not rgb_topic:
        print(f"  WARNING: No RGB topic found — proceeding without RGB")

    if dry_run:
        result["status"] = "DRY_RUN"
        return result

    # ── Pass 1: lightweight scan (no RGB data stored) ─────────────────────
    print(f"  Pass 1: scanning odom + rgb timestamps …")
    odom_ts = []
    odom_poses = []
    rgb_ts = []
    bag_to_sensor_offset_ns = None

    p1_topics = {odom_topic, event_topic}
    if rgb_topic:
        p1_topics.add(rgb_topic)

    for msg in read_bag(bag_dir, topics=p1_topics):
        if msg.topic == odom_topic:
            obj = deserialize(msg)
            ts = get_header_stamp_ns(obj)
            if ts is None:
                continue
            p, q = obj.pose.pose.position, obj.pose.pose.orientation
            yaw = R.from_quat([q.x, q.y, q.z, q.w]).as_euler("xyz")[2]
            odom_ts.append(ts)
            odom_poses.append(np.array([p.x, p.y, yaw]))

        elif rgb_topic and msg.topic == rgb_topic:
            # Quick CDR timestamp extraction (skip full deserialization)
            if len(msg.data) >= 12:
                _sec = int.from_bytes(msg.data[4:8], "little")
                _nsec = int.from_bytes(msg.data[8:12], "little")
                ts = _sec * 1_000_000_000 + _nsec
                if ts:
                    rgb_ts.append(ts)

        elif msg.topic == event_topic and bag_to_sensor_offset_ns is None:
            obj = deserialize(msg)
            hdr_ts = get_header_stamp_ns(obj)
            if hdr_ts is not None:
                raw_ev = bytes(obj.events)
                if len(raw_ev) >= 4:
                    first_evs, _state = decode_evt3(raw_ev, obj.width, obj.height)
                    if first_evs is not None and len(first_evs) > 0:
                        first_sensor_us = int(first_evs[0, 2])
                        bag_to_sensor_offset_ns = hdr_ts - first_sensor_us * 1000
                if bag_to_sensor_offset_ns is None:
                    bag_to_sensor_offset_ns = hdr_ts - (obj.time_base * 1000)

    if not odom_ts:
        result["status"] = "FAIL"
        result["error"] = "No odom messages"
        return result
    if bag_to_sensor_offset_ns is None:
        result["status"] = "FAIL"
        result["error"] = "No event messages for sync"
        return result

    # Sort odom by timestamp
    odom_poses = [p for _, p in sorted(zip(odom_ts, odom_poses))]
    odom_ts.sort()
    rgb_ts_sorted = sorted(rgb_ts)

    print(f"  Odom: {len(odom_ts)} msgs, RGB: {len(rgb_ts)} frames")
    print(f"  Sensor offset: {bag_to_sensor_offset_ns / 1e6:.2f} ms")
    if log_memory:
        print(f"  RSS after Pass 1: {get_rss_mb():.0f} MB")

    # ── compute step timestamps (ODOM-DRIVEN) ─────────────────────────────
    win_ns = VOXEL_WINDOW_US * 1000
    odom_start = odom_ts[0]
    odom_end = odom_ts[-1]

    # Steps span the full odom range.  Generate steps until the step
    # window fits: step_start + win_ns <= odom_end.
    step_timestamps = []
    t = odom_start
    while t + win_ns <= odom_end:
        step_timestamps.append(t)
        t += win_ns

    n_total_steps = len(step_timestamps)
    if n_total_steps < 1:
        result["status"] = "FAIL"
        result["error"] = "Trajectory too short for even 1 step"
        return result

    print(f"  Odom-driven steps: {n_total_steps} "
          f"(odom span: {(odom_end - odom_start)/1e9:.2f}s)")

    # RGB tolerance: max distance from step center for RGB match
    rgb_tolerance_ns = int(win_ns * rgb_tolerance_factor)

    # ── precompute which RGB timestamps are needed ────────────────────────
    needed_rgb_ts = set()
    for step_ts in step_timestamps:
        center = step_ts + win_ns // 2
        rgb_ts_match, found = find_nearest_rgb(
            rgb_ts_sorted, center, rgb_tolerance_ns
        )
        if found:
            needed_rgb_ts.add(rgb_ts_match)

    print(f"  Will buffer ≤{len(needed_rgb_ts)} needed RGB frames on-the-fly")

    # ── GPU setup ─────────────────────────────────────────────────────────
    gpu_device = None
    voxelizer = events_to_voxel_fast
    if use_gpu:
        try:
            import torch
            if torch.cuda.is_available():
                gpu_device = torch.device("cuda:0")
                def _gpu_vox(e, nb, h, w, ts, te):
                    return events_to_voxel_gpu(e, nb, h, w, ts, te, gpu_device)
                voxelizer = _gpu_vox
                print(f"  GPU: {torch.cuda.get_device_name(0)}")
            else:
                print("  GPU requested but CUDA unavailable; using CPU")
        except ImportError:
            print("  GPU requested but torch not installed; using CPU")

    # ── dtype / compression ───────────────────────────────────────────────
    np_voxel_dtype = np.float16 if voxel_dtype == "float16" else np.float32
    comp = _comp_kwargs(compression, gzip_level)
    print(
        f"  compression={compression}, voxel_dtype={voxel_dtype}, "
        f"flush_every={flush_every}"
    )

    # ── Pass 2: streaming events + on-demand RGB ─────────────────────────
    # Memory-efficient: events are streamed message-by-message and buffered
    # only for the active step window.  This avoids loading all events
    # (~16 GB+) into a single array, keeping peak RSS under the SLURM
    # memory limit.
    print(f"  Pass 2: streaming events + RGB …")

    tmp_path = h5_path + ".tmp"
    os.makedirs(os.path.dirname(tmp_path) or ".", exist_ok=True)

    # Pre-compute sensor-time window boundaries for every step
    step_sensor_start = np.array(
        [(t - bag_to_sensor_offset_ns) / 1000.0 for t in step_timestamps],
        dtype=np.float64,
    )
    step_sensor_end = step_sensor_start + VOXEL_WINDOW_US  # µs

    # ── helper: finalize a step (voxel + action + RGB) ────────────────
    def _finalize_step(
        step_idx, event_buf, f, d_vox, d_act, d_act_valid,
        d_ts, d_msk, d_rgb, d_rid, rgb_stored_count_ref,
    ):
        """Produce voxel from buffered events, compute action, match RGB,
        and write the step to HDF5.  Returns updated rgb_stored_count."""
        s_start = step_sensor_start[step_idx]
        s_end = step_sensor_end[step_idx]
        step_ts = step_timestamps[step_idx]
        rgb_count = rgb_stored_count_ref[0]

        # ── Voxel: extract events in this window ──────────────────────
        if event_buf:
            buf = np.concatenate(event_buf)
            mask = (buf[:, 2] >= s_start) & (buf[:, 2] < s_end)
            win_events = buf[mask] if mask.any() else None
            del buf, mask
        else:
            win_events = None

        voxel = voxelizer(
            win_events, NUM_BINS, HEIGHT, WIDTH, s_start, s_end,
        )
        del win_events
        if np_voxel_dtype == np.float16:
            voxel = voxel.astype(np.float16)

        # ── Actions ───────────────────────────────────────────────────
        action_chunk, action_valid = compute_action_chunk(
            odom_ts, odom_poses, step_ts, odom_end,
        )

        # ── RGB matching ──────────────────────────────────────────────
        win_center = step_ts + win_ns // 2
        rgb_match_ts, has_rgb = find_nearest_rgb(
            rgb_ts_sorted, win_center, rgb_tolerance_ns,
        )
        if has_rgb and rgb_match_ts in rgb_cache:
            img = rgb_cache[rgb_match_ts]
            d_rgb.resize(rgb_count + 1, axis=0)
            d_rid.resize(rgb_count + 1, axis=0)
            d_rgb[rgb_count] = img
            d_rid[rgb_count] = step_idx
            rgb_count += 1
        else:
            has_rgb = False

        # ── Write step ────────────────────────────────────────────────
        d_vox[step_idx] = voxel
        d_act[step_idx] = action_chunk
        d_act_valid[step_idx] = action_valid
        d_ts[step_idx] = step_ts
        d_msk[step_idx] = has_rgb

        rgb_stored_count_ref[0] = rgb_count
        return rgb_count

    # ── streaming state ───────────────────────────────────────────────
    step_cursor = 0          # next step to finalize
    event_buffer = []        # list of small np arrays (rolling window)
    rgb_cache = {}
    evt3_decoder = EVT3StreamDecoder(width=WIDTH, height=HEIGHT)
    total_event_count = 0
    total_event_msgs = 0
    rgb_stored_count = [0]   # mutable ref for nested fn

    p2_topics = {event_topic}
    if rgb_topic:
        p2_topics.add(rgb_topic)

    with h5py.File(tmp_path, "w") as f:
        d_vox = f.create_dataset(
            "voxels", (n_total_steps, NUM_BINS, HEIGHT, WIDTH),
            dtype=np_voxel_dtype,
            chunks=(1, NUM_BINS, HEIGHT, WIDTH),
            **comp,
        )
        d_act = f.create_dataset(
            "actions", (n_total_steps, ACTION_CHUNK, 3),
            dtype="float32",
        )
        d_act_valid = f.create_dataset(
            "actions_valid", (n_total_steps,),
            dtype="bool",
        )
        d_ts = f.create_dataset(
            "timestamps_ns", (n_total_steps,),
            dtype="int64",
        )
        d_msk = f.create_dataset(
            "rgb_mask", (n_total_steps,),
            dtype="bool",
        )
        d_rgb = f.create_dataset(
            "rgb_images", (0, RGB_HEIGHT, RGB_WIDTH, 3),
            maxshape=(None, RGB_HEIGHT, RGB_WIDTH, 3),
            dtype="uint8",
            chunks=(1, RGB_HEIGHT, RGB_WIDTH, 3),
            **comp,
        )
        d_rid = f.create_dataset(
            "rgb_indices", (0,),
            maxshape=(None,),
            dtype="int32",
        )

        # ── stream bag messages ───────────────────────────────────────
        for bag_msg in read_bag(bag_dir, topics=p2_topics):
            # --- RGB: cache only needed frames ---
            if rgb_topic and bag_msg.topic == rgb_topic:
                if len(bag_msg.data) >= 12:
                    _sec = int.from_bytes(bag_msg.data[4:8], "little")
                    _nsec = int.from_bytes(bag_msg.data[8:12], "little")
                    ts = _sec * 1_000_000_000 + _nsec
                    if ts in needed_rgb_ts:
                        try:
                            obj = deserialize(bag_msg)
                            rgb_cache[ts] = decode_rgb(obj)
                        except Exception:
                            pass
                continue

            if bag_msg.topic != event_topic:
                continue

            # --- Events: decode and buffer ---
            obj = deserialize(bag_msg)
            raw_events = bytes(obj.events)
            if len(raw_events) < 4:
                continue

            evs = evt3_decoder.decode(raw_events)
            if evs is None or len(evs) == 0:
                continue

            # Discard events before current step's window to bound memory.
            # Without this, millions of early events accumulate in the
            # buffer before the first step can be finalized.
            if step_cursor < n_total_steps:
                min_needed = step_sensor_start[step_cursor]
                keep = evs[:, 2] >= min_needed
                if not keep.any():
                    continue
                if not keep.all():
                    evs = evs[keep]

            event_buffer.append(evs)
            total_event_count += len(evs)
            total_event_msgs += 1

            # --- Finalize completed steps ---
            latest_sensor_t = evs[:, 2].max()
            while (step_cursor < n_total_steps
                   and latest_sensor_t >= step_sensor_end[step_cursor]):
                _finalize_step(
                    step_cursor, event_buffer,
                    f, d_vox, d_act, d_act_valid,
                    d_ts, d_msk, d_rgb, d_rid, rgb_stored_count,
                )

                # periodic flush + progress
                if (step_cursor + 1) % flush_every == 0:
                    f.flush()
                if (step_cursor + 1) % 20 == 0:
                    extra = ""
                    if log_memory:
                        extra = f" | RSS: {get_rss_mb():.0f} MB"
                    print(
                        f"\r  Steps: {step_cursor + 1}/{n_total_steps} | "
                        f"RGB: {rgb_stored_count[0]} | "
                        f"Events: {total_event_count:,}{extra}",
                        end="", flush=True,
                    )

                step_cursor += 1

                # Trim buffer: drop blocks entirely before current step
                if step_cursor < n_total_steps:
                    trim_t = step_sensor_start[step_cursor]
                    event_buffer = [
                        b for b in event_buffer
                        if b[:, 2].max() >= trim_t
                    ]

        # ── finalize remaining steps after all messages ───────────────
        while step_cursor < n_total_steps:
            _finalize_step(
                step_cursor, event_buffer,
                f, d_vox, d_act, d_act_valid,
                d_ts, d_msk, d_rgb, d_rid, rgb_stored_count,
            )

            if (step_cursor + 1) % flush_every == 0:
                f.flush()
            if (step_cursor + 1) % 20 == 0:
                extra = ""
                if log_memory:
                    extra = f" | RSS: {get_rss_mb():.0f} MB"
                print(
                    f"\r  Steps: {step_cursor + 1}/{n_total_steps} | "
                    f"RGB: {rgb_stored_count[0]} | "
                    f"Events: {total_event_count:,}{extra}",
                    end="", flush=True,
                )

            step_cursor += 1

            if step_cursor < n_total_steps:
                trim_t = step_sensor_start[step_cursor]
                event_buffer = [
                    b for b in event_buffer
                    if b[:, 2].max() >= trim_t
                ]

        print(f"\n  Events: {total_event_count:,} total from "
              f"{total_event_msgs} msgs (streamed)")

        # ── metadata attributes ───────────────────────────────────────
        f.attrs["bag_name"] = bag_name
        f.attrs["bag_dir"] = bag_dir
        f.attrs["event_topic"] = event_topic
        f.attrs["rgb_topic"] = rgb_topic if rgb_topic else ""
        f.attrs["odom_topic"] = odom_topic
        f.attrs["voxel_window_us"] = VOXEL_WINDOW_US
        f.attrs["num_bins"] = NUM_BINS
        f.attrs["action_chunk"] = ACTION_CHUNK
        f.attrs["compression"] = compression
        f.attrs["voxel_dtype"] = voxel_dtype

        # Action semantics metadata
        f.attrs["actions_space"] = "relative_waypoints"
        f.attrs["actions_repr"] = "dx_dy_dyaw"
        f.attrs["actions_frame"] = "base_link_at_step"
        f.attrs["dyaw_wrapped"] = "true"

        # Goal (default: not set)
        f.attrs["goal_step"] = -1

    # ── atomic rename ─────────────────────────────────────────────────────
    os.replace(tmp_path, h5_path)

    # Clean up
    del event_buffer
    rgb_cache.clear()

    rss = get_rss_mb()
    print(
        f"\n  Done: {n_total_steps} steps, {rgb_stored_count[0]} RGB → {h5_path}"
        + (f"  (RSS {rss:.0f} MB)" if log_memory else "")
    )

    result["status"] = "OK"
    result["steps"] = n_total_steps
    result["rgb_stored"] = rgb_stored_count[0]
    result["peak_rss_mb"] = round(rss, 1) if rss > 0 else None
    return result


# ── multiprocessing worker ─────────────────────────────────────────────────────

def _convert_worker(args_tuple):
    """Worker function for multiprocessing."""
    bag_dir, h5_path, kwargs = args_tuple
    bag_name = os.path.basename(bag_dir)
    print(f"[Worker] {bag_name}")
    t0 = time.time()
    try:
        r = convert_bag(bag_dir, h5_path, **kwargs)
    except Exception as e:
        r = {
            "bag_name": bag_name,
            "status": "FAIL",
            "h5_path": h5_path,
            "steps": 0,
            "rgb_stored": 0,
            "error": str(e),
            "traceback": traceback.format_exc(),
        }
    r["elapsed_sec"] = round(time.time() - t0, 1)
    return r


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Convert ROS2 bags to HDF5 dataset (v3, odom-driven)."
    )
    parser.add_argument("--bags-dir", required=True)
    parser.add_argument(
        "--out-dir",
        default="/scratch/kvinod/bags/eGo_navi_.h5",
        help="Output directory for .h5 files.",
    )
    parser.add_argument("--report", default=None)
    parser.add_argument("--bag-name", default=None, help="Convert only this bag.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--force", action="store_true", help="Re-convert even if .h5 exists."
    )

    parser.add_argument(
        "--compression",
        choices=["lzf", "gzip", "none"],
        default="lzf",
        help="HDF5 compression (default: lzf — fastest).",
    )
    parser.add_argument(
        "--gzip-level",
        type=int,
        default=1,
        choices=range(1, 10),
        metavar="1-9",
        help="gzip level (only when --compression gzip, default: 1).",
    )
    parser.add_argument(
        "--voxel-dtype",
        choices=["float32", "float16"],
        default="float32",
        help="Voxel storage dtype (default: float32).",
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Use GPU (torch CUDA) for voxel accumulation.",
    )
    parser.add_argument(
        "--jobs",
        default="1",
        help="Parallel bag conversions: integer or 'auto' (default: 1).",
    )
    parser.add_argument(
        "--flush-every",
        type=int,
        default=10,
        help="Flush H5 every N steps (default: 10).",
    )
    parser.add_argument(
        "--log-memory",
        action="store_true",
        help="Log RSS memory every 20 steps.",
    )
    args = parser.parse_args()

    # ── resource detection ────────────────────────────────────────────────
    resources = detect_resources()
    gpu_str = (
        f"Yes ({resources.get('gpu_name', '?')}, "
        f"{resources.get('gpu_mem_gb', '?')} GB)"
        if resources.get("gpu_available")
        else "No"
    )
    print(
        f"Hardware: {resources['cpu_count']} CPUs, "
        f"{resources.get('ram_available_gb', 0):.0f}/"
        f"{resources.get('ram_total_gb', 0):.0f} GB RAM, "
        f"GPU: {gpu_str}"
    )

    # ── parse jobs ────────────────────────────────────────────────────────
    if args.jobs == "auto":
        jobs = compute_safe_jobs(
            resources.get("ram_available_gb", 4), resources["cpu_count"]
        )
        print(f"Auto jobs: {jobs}")
    else:
        jobs = max(1, int(args.jobs))

    print(
        f"Settings: compression={args.compression}, "
        f"voxel_dtype={args.voxel_dtype}, "
        f"gpu={args.use_gpu}, jobs={jobs}, "
        f"flush_every={args.flush_every}"
    )

    os.makedirs(args.out_dir, exist_ok=True)

    # ── determine bags to convert ─────────────────────────────────────────
    if args.bag_name:
        bag_dirs = [os.path.join(args.bags_dir, args.bag_name)]
        if not os.path.isdir(bag_dirs[0]):
            print(f"[ERROR] {bag_dirs[0]} not found")
            sys.exit(1)
    else:
        report_path = args.report or os.path.join(
            args.bags_dir, "bag_report.jsonl"
        )
        if not os.path.exists(report_path):
            print(
                f"[ERROR] No report at {report_path}. "
                f"Run check_rosbags.py first."
            )
            sys.exit(1)

        good_bags = []
        with open(report_path) as fh:
            for line in fh:
                r = json.loads(line)
                if r["status"] == "GOOD":
                    good_bags.append(r["bag_dir"])

        if not good_bags:
            print("[ERROR] No GOOD bags in report.")
            sys.exit(1)

        bag_dirs = good_bags
        print(f"Found {len(bag_dirs)} GOOD bag(s) to convert.")

    # ── clean stale .tmp files ────────────────────────────────────────────
    for bag_dir in bag_dirs:
        bn = os.path.basename(bag_dir)
        tmp = os.path.join(args.out_dir, f"{bn}.h5.tmp")
        if os.path.exists(tmp):
            print(f"  Removing stale temp: {tmp}")
            os.remove(tmp)

    # ── warm up typestore ─────────────────────────────────────────────────
    get_ts()

    # ── shared settings dict ──────────────────────────────────────────────
    settings = dict(
        compression=args.compression,
        gzip_level=args.gzip_level,
        voxel_dtype=args.voxel_dtype,
        use_gpu=args.use_gpu,
        flush_every=args.flush_every,
        log_memory=args.log_memory,
        dry_run=args.dry_run,
    )

    # ── run conversion ────────────────────────────────────────────────────
    results = []

    work_items = []
    for bag_dir in bag_dirs:
        bn = os.path.basename(bag_dir)
        h5_path = os.path.join(args.out_dir, f"{bn}.h5")
        if os.path.exists(h5_path) and not args.force:
            results.append({
                "bag_name": bn, "status": "SKIP_EXISTS",
                "h5_path": h5_path, "steps": 0, "rgb_stored": 0,
                "error": None,
            })
            continue
        work_items.append((bag_dir, h5_path))

    if jobs > 1 and len(work_items) > 1:
        print(f"\nParallel mode: {jobs} workers for {len(work_items)} bags")
        mp_items = [(bd, hp, settings) for bd, hp in work_items]
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(jobs) as pool:
            for r in pool.imap_unordered(_convert_worker, mp_items):
                r.setdefault("elapsed_sec", 0)
                results.append(r)
                status = r.get("status", "?")
                print(
                    f"  Finished {r['bag_name']}: {status} "
                    f"({r.get('steps', 0)} steps, "
                    f"{r.get('elapsed_sec', 0):.0f}s)"
                )
    else:
        for i, (bag_dir, h5_path) in enumerate(work_items):
            bn = os.path.basename(bag_dir)
            print(f"\n[{i + 1}/{len(work_items)}] {bn}")
            t0 = time.time()
            try:
                r = convert_bag(bag_dir, h5_path, **settings)
            except Exception as e:
                r = {
                    "bag_name": bn, "status": "FAIL",
                    "h5_path": h5_path, "steps": 0, "rgb_stored": 0,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                }
            r["elapsed_sec"] = round(time.time() - t0, 1)
            results.append(r)
            gc.collect()

            if r["status"] == "FAIL":
                print(f"  FAIL: {r['error']}")

    # ── write conversion report ───────────────────────────────────────────
    report_out = os.path.join(args.out_dir, "convert_report.jsonl")
    with open(report_out, "w") as fh:
        for r in results:
            fh.write(json.dumps(r, default=str) + "\n")

    # ── summary ───────────────────────────────────────────────────────────
    ok = sum(1 for r in results if r.get("status") == "OK")
    fail = sum(1 for r in results if r.get("status") == "FAIL")
    skip = sum(1 for r in results if "SKIP" in r.get("status", ""))
    total_steps = sum(r.get("steps", 0) for r in results)
    total_rgb = sum(r.get("rgb_stored", 0) for r in results)

    print(f"\n{'=' * 60}")
    print(f"  OK: {ok}  FAIL: {fail}  SKIP: {skip}  TOTAL: {len(results)}")
    print(f"  Total steps: {total_steps}  Total RGB: {total_rgb}")
    print(f"  Report: {report_out}")
    print(f"{'=' * 60}")

    if fail > 0:
        print("\nFailed bags:")
        for r in results:
            if r.get("status") == "FAIL":
                print(f"  {r['bag_name']}: {r.get('error', '?')}")


if __name__ == "__main__":
    main()
