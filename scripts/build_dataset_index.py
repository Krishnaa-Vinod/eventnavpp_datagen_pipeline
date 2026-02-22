#!/usr/bin/env python3
"""
build_dataset_index.py — Build a unified metadata index over all HDF5 files

Creates a JSON index (`dataset_index.json`) summarising each converted H5:
  - bag name, file path, file size
  - number of steps, rgb frames, duration
  - voxel fill statistics, action range
  - sensor topics used

Also creates `dataset_index.csv` for quick spreadsheet viewing.

Usage
-----
  python scripts/build_dataset_index.py --h5-dir /scratch/kvinod/bags/eGo_navi_.h5
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np


def index_h5(h5_path: str) -> dict:
    """Extract metadata from a single HDF5 file."""
    name = os.path.basename(h5_path)
    bag_name = name.replace(".h5", "")

    rec = {
        "bag_name": bag_name,
        "h5_path": h5_path,
        "file_size_mb": round(os.path.getsize(h5_path) / 1e6, 1),
    }

    try:
        with h5py.File(h5_path, "r") as f:
            # Basic counts
            n_steps = f["voxels"].shape[0]
            n_rgb = f["rgb_images"].shape[0] if "rgb_images" in f else 0

            rec["n_steps"] = n_steps
            rec["n_rgb"] = n_rgb

            # Shapes
            rec["voxel_shape"] = list(f["voxels"].shape)
            rec["action_shape"] = list(f["actions"].shape)
            if n_rgb > 0:
                rec["rgb_shape"] = list(f["rgb_images"].shape)

            # Duration
            if n_steps > 0:
                ts = f["timestamps_ns"][:]
                rec["duration_s"] = round(float((ts[-1] - ts[0]) / 1e9), 2)
                rec["ts_start_ns"] = int(ts[0])
                rec["ts_end_ns"] = int(ts[-1])
            else:
                rec["duration_s"] = 0.0

            # Voxel fill (sample-based)
            sample_n = min(n_steps, 20)
            if sample_n > 0:
                idx = np.linspace(0, n_steps - 1, sample_n, dtype=int)
                v_sample = f["voxels"][list(idx)]
                rec["voxel_nonzero_frac"] = round(
                    float((v_sample != 0).sum()) / v_sample.size, 6
                )
                rec["voxel_min"] = round(float(v_sample.min()), 4)
                rec["voxel_max"] = round(float(v_sample.max()), 4)

            # Action range
            if n_steps > 0:
                a = f["actions"][:]
                rec["action_min"] = round(float(a.min()), 4)
                rec["action_max"] = round(float(a.max()), 4)
                rec["action_absmax"] = round(float(np.abs(a).max()), 4)

            # RGB mask
            if "rgb_mask" in f:
                rec["rgb_mask_true_count"] = int(f["rgb_mask"][:].sum())

            # Metadata attributes
            for attr in ["bag_name", "event_topic", "rgb_topic", "odom_topic",
                         "voxel_window_us", "num_bins", "action_chunk"]:
                if attr in f.attrs:
                    val = f.attrs[attr]
                    rec[f"attr_{attr}"] = (
                        val.decode() if isinstance(val, bytes) else
                        str(val) if isinstance(val, np.generic) else val
                    )

    except Exception as e:
        rec["error"] = str(e)

    return rec


def main():
    parser = argparse.ArgumentParser(
        description="Build metadata index over HDF5 dataset files."
    )
    parser.add_argument("--h5-dir", required=True,
                        help="Directory containing .h5 files.")
    args = parser.parse_args()

    h5_dir = args.h5_dir
    h5_files = sorted(
        [os.path.join(h5_dir, f) for f in os.listdir(h5_dir) if f.endswith(".h5")]
    )

    if not h5_files:
        print(f"No .h5 files found in {h5_dir}")
        sys.exit(1)

    print(f"Indexing {len(h5_files)} file(s) ...\n")

    records = []
    total_steps = 0
    total_rgb = 0
    total_dur = 0.0
    total_size = 0.0

    for h5_path in h5_files:
        rec = index_h5(h5_path)
        records.append(rec)

        total_steps += rec.get("n_steps", 0)
        total_rgb += rec.get("n_rgb", 0)
        total_dur += rec.get("duration_s", 0.0)
        total_size += rec.get("file_size_mb", 0.0)

        print(f"  {rec['bag_name']}: {rec.get('n_steps', '?')} steps, "
              f"{rec.get('n_rgb', '?')} rgb, "
              f"{rec.get('duration_s', '?')}s, "
              f"{rec.get('file_size_mb', '?')} MB")

    # Build index
    index = {
        "h5_dir": h5_dir,
        "n_files": len(records),
        "total_steps": total_steps,
        "total_rgb_frames": total_rgb,
        "total_duration_s": round(total_dur, 2),
        "total_size_mb": round(total_size, 1),
        "files": records,
    }

    # Write JSON index
    json_path = os.path.join(h5_dir, "dataset_index.json")
    with open(json_path, "w") as f:
        json.dump(index, f, indent=2, default=str)

    # Write CSV
    csv_path = os.path.join(h5_dir, "dataset_index.csv")
    csv_fields = [
        "bag_name", "n_steps", "n_rgb", "duration_s", "file_size_mb",
        "voxel_nonzero_frac", "voxel_min", "voxel_max",
        "action_min", "action_max", "action_absmax",
        "rgb_mask_true_count",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        for rec in records:
            writer.writerow(rec)

    print(f"\n{'='*60}")
    print(f"  Files indexed: {len(records)}")
    print(f"  Total steps:   {total_steps}")
    print(f"  Total RGB:     {total_rgb}")
    print(f"  Total duration: {total_dur:.1f}s ({total_dur/60:.1f} min)")
    print(f"  Total size:    {total_size:.1f} MB ({total_size/1024:.1f} GB)")
    print(f"  Index: {json_path}")
    print(f"  CSV:   {csv_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
