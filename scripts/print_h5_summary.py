#!/usr/bin/env python3
"""
print_h5_summary.py — Print a quick summary of an H5 dataset file.

Shows: steps, rgb_count, duration, goal_step, action semantics,
       first action vector, and new odom-driven attributes.
"""

import argparse
import os
import sys

import h5py
import numpy as np


def summarize(h5_path: str):
    """Print summary of one H5 file."""
    name = os.path.basename(h5_path)
    print(f"{'=' * 60}")
    print(f"  File: {name}")
    print(f"  Path: {h5_path}")
    print(f"{'=' * 60}")

    with h5py.File(h5_path, "r") as f:
        # Datasets
        for ds_name in ["voxels", "actions", "actions_valid", "timestamps_ns",
                         "rgb_mask", "rgb_images", "rgb_indices"]:
            if ds_name in f:
                ds = f[ds_name]
                print(f"  {ds_name:20s}  shape={ds.shape}  dtype={ds.dtype}")
            else:
                print(f"  {ds_name:20s}  MISSING")

        n_steps = f["voxels"].shape[0] if "voxels" in f else 0
        n_rgb = f["rgb_images"].shape[0] if "rgb_images" in f else 0

        print()
        print(f"  Steps:      {n_steps}")
        print(f"  RGB:        {n_rgb}")
        print(f"  RGB mask:   {int(f['rgb_mask'][:].sum())} True / {n_steps} total"
              if "rgb_mask" in f else "")

        if "timestamps_ns" in f and n_steps > 0:
            ts = f["timestamps_ns"]
            dur = (ts[-1] - ts[0]) / 1e9
            print(f"  Duration:   {dur:.2f}s")
            print(f"  First TS:   {ts[0]}")
            print(f"  Last TS:    {ts[-1]}")

        # Actions valid
        if "actions_valid" in f:
            av = f["actions_valid"][:]
            n_valid = int(av.sum())
            n_invalid = n_steps - n_valid
            print(f"  Actions valid: {n_valid} / {n_steps} "
                  f"({n_invalid} near-end steps without full horizon)")

        # Goal
        goal_step = int(f.attrs.get("goal_step", -1))
        print()
        if goal_step >= 0:
            print(f"  Goal step:  {goal_step}")
            if "goal_timestamp_ns" in f.attrs:
                print(f"  Goal TS:    {f.attrs['goal_timestamp_ns']}")
            if "goal_has_rgb" in f.attrs:
                print(f"  Goal RGB:   {f.attrs['goal_has_rgb']}")
        else:
            print(f"  Goal step:  NOT SET (default: -1)")

        # Attributes
        print()
        print("  Attributes:")
        for key in sorted(f.attrs.keys()):
            val = f.attrs[key]
            print(f"    {key}: {val}")

        # First action
        if "actions" in f and n_steps > 0:
            print()
            act0 = f["actions"][0]
            print(f"  First action chunk (step 0):")
            print(f"    Shape: {act0.shape}")
            for k in range(min(act0.shape[0], 8)):
                dx, dy, dyaw = act0[k]
                print(f"    k={k+1}: dx={dx:+.4f}m  dy={dy:+.4f}m  "
                      f"dYaw={dyaw:+.4f}rad ({np.degrees(dyaw):+.1f}°)")

        # Voxel stats
        if "voxels" in f and n_steps > 0:
            print()
            sample_idx = min(n_steps - 1, n_steps // 2)
            v = f["voxels"][sample_idx]
            nz = (v != 0).sum() / v.size
            print(f"  Voxel stats (step {sample_idx}):")
            print(f"    range: [{v.min():.3f}, {v.max():.3f}]")
            print(f"    nonzero: {nz:.4%}")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Print summary of H5 dataset file(s)."
    )
    parser.add_argument("--h5", required=True,
                        help="Path to H5 file or directory of H5 files")
    args = parser.parse_args()

    if os.path.isdir(args.h5):
        files = sorted([
            os.path.join(args.h5, f)
            for f in os.listdir(args.h5) if f.endswith(".h5")
        ])
        for fp in files:
            try:
                summarize(fp)
            except Exception as e:
                print(f"  ERROR: {e}\n")
    else:
        summarize(args.h5)


if __name__ == "__main__":
    main()
