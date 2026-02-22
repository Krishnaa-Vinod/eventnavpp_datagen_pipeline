#!/usr/bin/env python3
"""
validate_h5.py — Validate converted HDF5 dataset files

Checks each .h5 file in the output directory for:
  1. Required datasets present with correct dtypes and shapes
  2. No NaN / Inf values in voxels or actions
  3. Voxel range within [-1, 1] (normalised)
  4. Voxel non-zero fraction above threshold
  5. Action values within reasonable bounds
  6. Timestamp monotonicity
  7. RGB mask ↔ image count consistency
  8. RGB pixel statistics (not all-black, not clipped)
  9. Metadata attributes present

Outputs:
  - validate_report.jsonl  (one JSON object per file)
  - validate_report.csv    (summary table)

Usage
-----
  python scripts/validate_h5.py --h5-dir /scratch/kvinod/bags/eGo_navi_.h5
  python scripts/validate_h5.py --h5-dir /scratch/kvinod/bags/eGo_navi_.h5 --file data_collect_20260206_165620.h5
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np

# ── expected schema ────────────────────────────────────────────────────────────
EXPECTED_DATASETS = {
    "voxels":        {"ndim": 4, "dtype_kind": "f"},  # (N, 5, 720, 1280)
    "actions":       {"ndim": 3, "dtype_kind": "f"},  # (N, 8, 3)
    "timestamps_ns": {"ndim": 1, "dtype_kind": "i"},  # (N,)
    "rgb_mask":      {"ndim": 1, "dtype_kind": "b"},  # (N,)
    "rgb_images":    {"ndim": 4, "dtype_kind": "u"},  # (M, 1024, 1280, 3)
    "rgb_indices":   {"ndim": 1, "dtype_kind": "i"},  # (M,)
}

# Thresholds
MIN_STEPS = 5
MIN_VOXEL_NONZERO_FRAC = 0.001   # at least 0.1% non-zero across all steps
MAX_ACTION_MAGNITUDE = 50.0       # generous bound for local actions
ACTION_NAN_TOL = 0               # no NaN allowed


def validate_h5(h5_path: str) -> dict:
    """Validate a single HDF5 file. Returns result dict."""
    name = os.path.basename(h5_path)
    result = {
        "filename": name,
        "h5_path": h5_path,
        "status": "UNKNOWN",
        "steps": 0,
        "rgb_count": 0,
        "warnings": [],
        "errors": [],
    }

    if not os.path.exists(h5_path):
        result["status"] = "MISSING"
        result["errors"].append("File does not exist")
        return result

    try:
        with h5py.File(h5_path, "r") as f:
            # 1. Schema check
            for ds_name, spec in EXPECTED_DATASETS.items():
                if ds_name not in f:
                    result["errors"].append(f"Missing dataset: {ds_name}")
                    continue
                ds = f[ds_name]
                if ds.ndim != spec["ndim"]:
                    result["errors"].append(
                        f"{ds_name}: expected ndim={spec['ndim']}, got {ds.ndim}"
                    )
                if ds.dtype.kind != spec["dtype_kind"]:
                    result["errors"].append(
                        f"{ds_name}: expected dtype.kind='{spec['dtype_kind']}', "
                        f"got '{ds.dtype.kind}'"
                    )

            if result["errors"]:
                result["status"] = "FAIL"
                return result

            n_steps = f["voxels"].shape[0]
            n_rgb = f["rgb_images"].shape[0]
            result["steps"] = n_steps
            result["rgb_count"] = n_rgb

            # Check step count
            if n_steps < MIN_STEPS:
                result["errors"].append(
                    f"Too few steps: {n_steps} < {MIN_STEPS}"
                )

            # Shape consistency
            for ds_name in ["actions", "timestamps_ns", "rgb_mask"]:
                if f[ds_name].shape[0] != n_steps:
                    result["errors"].append(
                        f"{ds_name} length {f[ds_name].shape[0]} != voxels length {n_steps}"
                    )

            if f["rgb_indices"].shape[0] != n_rgb:
                result["errors"].append(
                    f"rgb_indices length {f['rgb_indices'].shape[0]} != "
                    f"rgb_images length {n_rgb}"
                )

            # Dimension checks
            vox_shape = f["voxels"].shape
            if vox_shape[1:] != (5, 720, 1280):
                result["warnings"].append(
                    f"Unexpected voxel shape: {vox_shape}"
                )

            act_shape = f["actions"].shape
            if act_shape[1:] != (8, 3):
                result["warnings"].append(
                    f"Unexpected action shape: {act_shape}"
                )

            rgb_shape = f["rgb_images"].shape
            if n_rgb > 0 and rgb_shape[1:] != (1024, 1280, 3):
                result["warnings"].append(
                    f"Unexpected RGB shape: {rgb_shape}"
                )

            # 2. NaN / Inf checks (sample-based for large files)
            sample_size = min(n_steps, 50)
            sample_idx = np.linspace(0, n_steps - 1, sample_size, dtype=int)

            vox_sample = f["voxels"][list(sample_idx)]
            if np.any(np.isnan(vox_sample)):
                result["errors"].append("NaN found in voxels")
            if np.any(np.isinf(vox_sample)):
                result["errors"].append("Inf found in voxels")

            act_all = f["actions"][:]
            if np.any(np.isnan(act_all)):
                result["errors"].append("NaN found in actions")
            if np.any(np.isinf(act_all)):
                result["errors"].append("Inf found in actions")

            # 3. Voxel range
            vox_min, vox_max = float(vox_sample.min()), float(vox_sample.max())
            result["voxel_min"] = vox_min
            result["voxel_max"] = vox_max
            if vox_min < -1.01 or vox_max > 1.01:
                result["warnings"].append(
                    f"Voxel values outside [-1,1]: [{vox_min:.4f}, {vox_max:.4f}]"
                )

            # 4. Voxel non-zero fraction
            nonzero_frac = float((vox_sample != 0).sum()) / vox_sample.size
            result["voxel_nonzero_frac"] = round(nonzero_frac, 6)
            if nonzero_frac < MIN_VOXEL_NONZERO_FRAC:
                result["warnings"].append(
                    f"Low voxel fill: {nonzero_frac:.6f} < {MIN_VOXEL_NONZERO_FRAC}"
                )

            # 5. Action bounds
            act_absmax = float(np.abs(act_all).max())
            result["action_absmax"] = round(act_absmax, 4)
            if act_absmax > MAX_ACTION_MAGNITUDE:
                result["warnings"].append(
                    f"Large action magnitude: {act_absmax:.4f} > {MAX_ACTION_MAGNITUDE}"
                )

            # 6. Timestamp monotonicity
            ts = f["timestamps_ns"][:]
            ts_diff = np.diff(ts)
            non_mono = int((ts_diff <= 0).sum())
            result["ts_non_monotonic"] = non_mono
            if non_mono > 0:
                result["errors"].append(
                    f"Timestamps not monotonic: {non_mono} violations"
                )

            # 7. RGB mask consistency
            rgb_mask = f["rgb_mask"][:]
            n_mask_true = int(rgb_mask.sum())
            result["rgb_mask_true"] = n_mask_true

            # Number of rgb_images should match number of True values in mask
            # (each True in mask should have a corresponding rgb stored)
            if n_rgb != n_mask_true:
                # This is expected if some rgb frames failed to decode
                result["warnings"].append(
                    f"rgb_images count ({n_rgb}) != rgb_mask True count "
                    f"({n_mask_true})"
                )

            # rgb_indices should be valid step indices
            if n_rgb > 0:
                ri = f["rgb_indices"][:]
                if ri.min() < 0 or ri.max() >= n_steps:
                    result["errors"].append(
                        f"rgb_indices out of range: [{ri.min()}, {ri.max()}] "
                        f"for {n_steps} steps"
                    )
                # Should be monotonically increasing
                ri_diff = np.diff(ri)
                if np.any(ri_diff < 0):
                    result["warnings"].append(
                        "rgb_indices not monotonically increasing"
                    )

            # 8. RGB pixel statistics
            if n_rgb > 0:
                # Sample a few images
                rgb_sample_n = min(n_rgb, 5)
                rgb_idx = np.linspace(0, n_rgb - 1, rgb_sample_n, dtype=int)
                all_black = True
                for idx in rgb_idx:
                    img = f["rgb_images"][int(idx)]
                    if img.max() > 0:
                        all_black = False
                        break
                if all_black:
                    result["errors"].append("All sampled RGB images are black")

                # Check for reasonable mean pixel values
                img0 = f["rgb_images"][0]
                mean_val = float(img0.mean())
                result["rgb_mean_px"] = round(mean_val, 1)
                if mean_val < 5:
                    result["warnings"].append(
                        f"Very dark RGB: mean={mean_val:.1f}"
                    )
                elif mean_val > 250:
                    result["warnings"].append(
                        f"Very bright RGB: mean={mean_val:.1f}"
                    )

            # 9. Metadata attributes
            expected_attrs = [
                "bag_name", "voxel_window_us", "num_bins", "action_chunk"
            ]
            missing_attrs = [a for a in expected_attrs if a not in f.attrs]
            if missing_attrs:
                result["warnings"].append(
                    f"Missing metadata attrs: {missing_attrs}"
                )

            # Duration
            duration_s = float((ts[-1] - ts[0]) / 1e9)
            result["duration_s"] = round(duration_s, 2)

    except Exception as e:
        result["errors"].append(f"Exception: {e}")

    # Final status
    if result["errors"]:
        result["status"] = "FAIL"
    elif result["warnings"]:
        result["status"] = "WARN"
    else:
        result["status"] = "PASS"

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Validate converted HDF5 dataset files."
    )
    parser.add_argument("--h5-dir", required=True,
                        help="Directory containing .h5 files.")
    parser.add_argument("--file", default=None,
                        help="Validate only this .h5 file (name or path).")
    args = parser.parse_args()

    h5_dir = args.h5_dir

    if args.file:
        if os.path.isabs(args.file):
            h5_files = [args.file]
        else:
            h5_files = [os.path.join(h5_dir, args.file)]
    else:
        h5_files = sorted(
            [os.path.join(h5_dir, f) for f in os.listdir(h5_dir)
             if f.endswith(".h5")]
        )

    if not h5_files:
        print(f"No .h5 files found in {h5_dir}")
        sys.exit(1)

    print(f"Validating {len(h5_files)} file(s) in {h5_dir}\n")

    results = []
    for i, h5_path in enumerate(h5_files):
        name = os.path.basename(h5_path)
        r = validate_h5(h5_path)
        results.append(r)

        status_icon = {"PASS": "✓", "WARN": "⚠", "FAIL": "✗", "MISSING": "?"}.get(
            r["status"], "?"
        )
        extra = ""
        if r["steps"] > 0:
            extra = f"  steps={r['steps']} rgb={r['rgb_count']}"
            if "duration_s" in r:
                extra += f" dur={r['duration_s']}s"
        print(f"[{i+1}/{len(h5_files)}] {status_icon} {r['status']:5s}  {name}{extra}")

        if r["errors"]:
            for e in r["errors"]:
                print(f"    ERROR: {e}")
        if r["warnings"]:
            for w in r["warnings"]:
                print(f"    WARN:  {w}")

    # Write reports
    jsonl_path = os.path.join(h5_dir, "validate_report.jsonl")
    with open(jsonl_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, default=str) + "\n")

    csv_path = os.path.join(h5_dir, "validate_report.csv")
    csv_fields = [
        "filename", "status", "steps", "rgb_count", "duration_s",
        "voxel_nonzero_frac", "voxel_min", "voxel_max",
        "action_absmax", "ts_non_monotonic", "rgb_mask_true", "rgb_mean_px",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    # Summary
    n_pass = sum(1 for r in results if r["status"] == "PASS")
    n_warn = sum(1 for r in results if r["status"] == "WARN")
    n_fail = sum(1 for r in results if r["status"] == "FAIL")
    total_steps = sum(r.get("steps", 0) for r in results)

    print(f"\n{'='*60}")
    print(f"  PASS: {n_pass}  WARN: {n_warn}  FAIL: {n_fail}  TOTAL: {len(results)}")
    print(f"  Total steps across all files: {total_steps}")
    print(f"  Reports: {jsonl_path}")
    print(f"           {csv_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
