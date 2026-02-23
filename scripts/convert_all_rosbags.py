#!/usr/bin/env python3
"""
convert_all_rosbags.py  — v2: Memory-safe batch ROSbag → HDF5 converter

Changes from v1
~~~~~~~~~~~~~~~~
* **OOM fix**: Pass 1 no longer stores raw RGB CDR bytes in RAM.
  RGB frames are decoded on-the-fly in Pass 2 and written directly to H5.
* **Crash safety**: Writes to ``<path>.tmp``, atomically renames on success.
  Stale ``.tmp`` files from previous kills are cleaned up on restart.
* **LZF compression** by default (≈5–10× faster writes than gzip).
* **Optional GPU voxelisation** via ``--use-gpu`` (torch scatter_add).
* **Resource-aware parallelism** via ``--jobs auto``.
* **Memory logging** via ``--log-memory`` (RSS every 20 steps).
* **Periodic H5 flush** via ``--flush-every`` (default 10 steps).

H5 schema (unchanged)
~~~~~~~~~~~~~~~~~~~~~
  voxels        (N, 5, 720, 1280)   float32   [lzf]
  actions       (N, 8, 3)           float32
  timestamps_ns (N,)                int64
  rgb_mask      (N,)                bool
  rgb_images    (M, 1024, 1280, 3)  uint8     [lzf]
  rgb_indices   (M,)                int32

Usage
-----
  # Fast default (lzf, CPU, sequential)
  python scripts/convert_all_rosbags.py --bags-dir /scratch/kvinod/bags

  # Maximise speed
  python scripts/convert_all_rosbags.py --bags-dir /scratch/kvinod/bags \\
      --compression lzf --use-gpu --jobs auto --log-memory

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
from evt3_decoder import decode_evt3
from resource_utils import get_rss_mb, detect_resources, compute_safe_jobs

# ── constants (match bag_to_h5.py) ─────────────────────────────────────────────
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
):
    """Convert one bag to HDF5 with bounded memory.

    Pass 1 — lightweight scan: collect odom poses, RGB timestamps,
             and event sync offset.  **No RGB data stored in RAM.**
    Pass 2 — streaming: read events + RGB together; decode RGB on-demand;
             write each step immediately to an extendable H5 dataset.
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

    if not all([event_topic, rgb_topic, odom_topic]):
        result["status"] = "SKIP"
        result["error"] = (
            f"Missing topics: ev={event_topic} rgb={rgb_topic} odom={odom_topic}"
        )
        return result

    if dry_run:
        result["status"] = "DRY_RUN"
        return result

    # ── Pass 1: lightweight scan (no RGB data stored) ─────────────────────
    print(f"  Pass 1: scanning odom + rgb timestamps …")
    odom_ts = []
    odom_poses = []
    rgb_ts = []                         # header timestamps only
    bag_to_sensor_offset_ns = None

    p1_topics = {odom_topic, rgb_topic, event_topic}

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

        elif msg.topic == rgb_topic:
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
                    first_evs = decode_evt3(raw_ev, obj.width, obj.height)
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

    # ── compute window range ──────────────────────────────────────────────
    win_ns = VOXEL_WINDOW_US * 1000
    curr_win_start = max(
        odom_ts[0],
        rgb_ts_sorted[0] if rgb_ts_sorted else odom_ts[0],
        bag_to_sensor_offset_ns,
    )
    last_safe = odom_ts[-1] - ACTION_CHUNK * win_ns

    if curr_win_start + win_ns > last_safe:
        result["status"] = "FAIL"
        result["error"] = "Trajectory too short for action chunks"
        return result

    # ── precompute which RGB timestamps are needed ────────────────────────
    needed_rgb_ts = set()
    tmp_start = curr_win_start
    while tmp_start + win_ns <= last_safe:
        center = tmp_start + win_ns // 2
        r_idx = bisect.bisect_left(rgb_ts_sorted, center)
        if r_idx < len(rgb_ts_sorted):
            if abs(rgb_ts_sorted[r_idx] - center) < win_ns // 2:
                needed_rgb_ts.add(rgb_ts_sorted[r_idx])
        # also check r_idx-1 (could be nearer)
        if r_idx > 0:
            if abs(rgb_ts_sorted[r_idx - 1] - center) < win_ns // 2:
                needed_rgb_ts.add(rgb_ts_sorted[r_idx - 1])
        tmp_start += win_ns

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
    print(f"  Pass 2: streaming events → voxels …")

    tmp_path = h5_path + ".tmp"
    os.makedirs(os.path.dirname(tmp_path) or ".", exist_ok=True)

    event_buffer = []
    step_idx = 0
    rgb_stored_count = 0
    rgb_cache = {}          # header_ts → decoded numpy image
    curr_win = curr_win_start
    ev_msg_since_step = 0   # stall detection counter
    stalled = False

    p2_topics = {event_topic, rgb_topic}

    with h5py.File(tmp_path, "w") as f:
        d_vox = f.create_dataset(
            "voxels", (0, NUM_BINS, HEIGHT, WIDTH),
            maxshape=(None, NUM_BINS, HEIGHT, WIDTH),
            dtype=np_voxel_dtype,
            chunks=(1, NUM_BINS, HEIGHT, WIDTH),
            **comp,
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
            dtype="uint8",
            chunks=(1, RGB_HEIGHT, RGB_WIDTH, 3),
            **comp,
        )
        d_rid = f.create_dataset(
            "rgb_indices", (0,), maxshape=(None,), dtype="int32",
        )

        for bag_msg in read_bag(bag_dir, topics=p2_topics):
            if curr_win + win_ns > last_safe:
                break
            if stalled:
                break

            # ── on-demand RGB caching ─────────────────────────────────
            if bag_msg.topic == rgb_topic:
                # Quick timestamp from CDR header (avoid full deserialization
                # for non-needed frames — saves ~75% of RGB processing)
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

            # ── event processing ──────────────────────────────────────
            obj = deserialize(bag_msg)
            raw_events = bytes(obj.events)
            if len(raw_events) < 4:
                continue

            evs = decode_evt3(raw_events, obj.width, obj.height)
            if evs is None or len(evs) == 0:
                continue

            event_buffer.append(evs)

            # ── buffer safety: trim old events periodically ───────────
            if len(event_buffer) > 200:
                all_tmp = np.concatenate(event_buffer)
                t_keep = (curr_win - bag_to_sensor_offset_ns) // 1000
                mask = all_tmp[:, 2] >= t_keep
                event_buffer = [all_tmp[mask]] if mask.any() else []
                del all_tmp

            t_sensor_query = (curr_win + win_ns - bag_to_sensor_offset_ns) // 1000
            if evs[-1, 2] < t_sensor_query:
                ev_msg_since_step += 1
                # stall detection: if 10000 event messages without a new step
                if ev_msg_since_step > 10000:
                    print(
                        f"\n  WARNING: stall at step {step_idx} "
                        f"(evs[-1,2]={evs[-1,2]:.0f} < query={t_sensor_query:.0f}), "
                        f"skipping rest of bag"
                    )
                    stalled = True
                    break
                continue

            all_evs = np.concatenate(event_buffer)

            while len(all_evs) > 0 and all_evs[-1, 2] >= t_sensor_query:
                t_sensor_start = (curr_win - bag_to_sensor_offset_ns) // 1000
                split = bisect.bisect_left(all_evs[:, 2], t_sensor_query)

                voxel = voxelizer(
                    all_evs[:split], NUM_BINS, HEIGHT, WIDTH,
                    t_sensor_start, t_sensor_query,
                )
                if np_voxel_dtype == np.float16:
                    voxel = voxel.astype(np.float16)

                # ── RGB matching (same logic as v1) ───────────────────
                win_center = curr_win + win_ns // 2
                r_idx = bisect.bisect_left(rgb_ts_sorted, win_center)
                has_rgb = False

                if r_idx < len(rgb_ts_sorted):
                    nearest_ts = rgb_ts_sorted[r_idx]
                    if abs(nearest_ts - win_center) < win_ns // 2:
                        if nearest_ts in rgb_cache:
                            img = rgb_cache.pop(nearest_ts)
                            d_rgb.resize(rgb_stored_count + 1, axis=0)
                            d_rid.resize(rgb_stored_count + 1, axis=0)
                            d_rgb[rgb_stored_count] = img
                            d_rid[rgb_stored_count] = step_idx
                            rgb_stored_count += 1
                            has_rgb = True

                # ── write step ────────────────────────────────────────
                for d in [d_vox, d_act, d_ts, d_msk]:
                    d.resize(step_idx + 1, axis=0)
                d_vox[step_idx] = voxel
                d_act[step_idx] = compute_action_chunk(
                    odom_ts, odom_poses, curr_win
                )
                d_ts[step_idx] = curr_win
                d_msk[step_idx] = has_rgb

                step_idx += 1
                ev_msg_since_step = 0   # reset stall counter
                curr_win += win_ns
                t_sensor_query = (
                    curr_win + win_ns - bag_to_sensor_offset_ns
                ) // 1000

                # ── periodic flush + progress ─────────────────────────
                if step_idx % flush_every == 0:
                    f.flush()

                if step_idx % 20 == 0:
                    extra = ""
                    if log_memory:
                        extra = f" | RSS: {get_rss_mb():.0f} MB"
                    print(
                        f"\r  Steps: {step_idx} | RGB: {rgb_stored_count}"
                        f"{extra}",
                        end="", flush=True,
                    )

                # ── trim event buffer + clean old RGB cache ───────────
                all_evs = all_evs[split:]
                if curr_win + win_ns > last_safe:
                    break

                old_keys = [k for k in rgb_cache if k < curr_win - win_ns]
                for k in old_keys:
                    del rgb_cache[k]

            event_buffer = [all_evs] if len(all_evs) > 0 else []

        # ── metadata attributes ───────────────────────────────────────
        f.attrs["bag_name"] = bag_name
        f.attrs["bag_dir"] = bag_dir
        f.attrs["event_topic"] = event_topic
        f.attrs["rgb_topic"] = rgb_topic
        f.attrs["odom_topic"] = odom_topic
        f.attrs["voxel_window_us"] = VOXEL_WINDOW_US
        f.attrs["num_bins"] = NUM_BINS
        f.attrs["action_chunk"] = ACTION_CHUNK
        f.attrs["compression"] = compression
        f.attrs["voxel_dtype"] = voxel_dtype

    # ── atomic rename ─────────────────────────────────────────────────────
    os.replace(tmp_path, h5_path)

    rss = get_rss_mb()
    print(
        f"\n  Done: {step_idx} steps, {rgb_stored_count} RGB → {h5_path}"
        + (f"  (RSS {rss:.0f} MB)" if log_memory else "")
    )

    result["status"] = "OK"
    result["steps"] = step_idx
    result["rgb_stored"] = rgb_stored_count
    result["peak_rss_mb"] = round(rss, 1) if rss > 0 else None
    return result


# ── multiprocessing worker ─────────────────────────────────────────────────────

def _convert_worker(args_tuple):
    """Worker function for multiprocessing.  Receives (bag_dir, h5_path, kwargs)."""
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
        description="Convert GOOD ROS2 bags to HDF5 dataset (v2)."
    )
    # ── existing flags ────────────────────────────────────────────────────
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

    # ── new performance flags ─────────────────────────────────────────────
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

    # Pre-filter skips
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
        # ── parallel mode ─────────────────────────────────────────────
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
        # ── sequential mode ───────────────────────────────────────────
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
            gc.collect()   # release memory between bags

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
