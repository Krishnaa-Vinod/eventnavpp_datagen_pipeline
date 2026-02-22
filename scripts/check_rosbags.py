#!/usr/bin/env python3
"""
check_rosbags.py — Bag Quality Checker

Scans a directory tree for ROS2 bag directories (containing metadata.yaml),
runs quality checks against configs/qa.yaml thresholds, and produces:
  - bag_report.jsonl   (one JSON object per bag)
  - bag_report.csv     (summary table)

Bags that fail any REQUIRED check are marked BAD with reasons.
Handles truncated MCAP files via mcap StreamReader fallback.

Usage
-----
    python scripts/check_rosbags.py --bags-dir /scratch/kvinod/bags
    python scripts/check_rosbags.py --bags-dir /scratch/kvinod/bags --config configs/qa.yaml
"""

import argparse
import csv
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import yaml

# Add repo root to path for bag_reader
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from bag_reader import read_bag, deserialize, get_topics, get_ts

DEFAULT_CONFIG = REPO_ROOT / "configs" / "qa.yaml"


def find_bag_dirs(root: str):
    """Discover all ROS2 bag directories under *root* (contain metadata.yaml)."""
    bag_dirs = []
    for dirpath, dirnames, filenames in os.walk(root):
        if "metadata.yaml" in filenames:
            bag_dirs.append(dirpath)
            dirnames.clear()
    bag_dirs.sort()
    return bag_dirs


def get_header_stamp_ns(msg):
    """Extract header stamp as nanoseconds, or None."""
    if hasattr(msg, "header"):
        return msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
    return None


def resolve_topic(available_topics, aliases):
    """Return the first alias found in available_topics, or None."""
    for alias in aliases:
        if alias in available_topics:
            return alias
    return None


def check_bag(bag_dir: str, cfg: dict):
    """Run all QA checks on one bag. Returns a report dict."""
    report = {
        "bag_dir": bag_dir,
        "bag_name": os.path.basename(bag_dir),
        "status": "UNKNOWN",
        "reader": "unknown",
        "reasons": [],
        "warnings": [],
        "topic_map": {},
        "msg_counts": {},
        "duration_sec": None,
    }

    # ── 1. Get topics from metadata ────────────────────────────────────────
    available_topics = get_topics(bag_dir)
    if not available_topics:
        report["status"] = "BAD"
        report["reasons"].append("No metadata.yaml or no topics found")
        return report

    # ── 2. Required topics ─────────────────────────────────────────────────
    for category, aliases in cfg["required_topics"].items():
        actual = resolve_topic(available_topics, aliases)
        if actual is None:
            report["reasons"].append(
                f"Missing required topic '{category}': none of {aliases} found"
            )
        else:
            report["topic_map"][category] = actual

    # ── 3. Optional topics ─────────────────────────────────────────────────
    for category, aliases in cfg.get("optional_topics", {}).items():
        actual = resolve_topic(available_topics, aliases)
        if actual:
            report["topic_map"][category] = actual
        else:
            report["warnings"].append(f"Optional topic '{category}' not found")

    # If required topics are missing, mark BAD early (still try to read)
    topic_fail = bool(report["reasons"])

    # ── 4. Read messages for deeper checks ─────────────────────────────────
    cat_timestamps = {}  # category -> list of (bag_ts, header_ts)
    msg_counts = {}
    reader_type = "unknown"

    try:
        for msg in read_bag(bag_dir):
            # Determine reader type from first message
            if reader_type == "unknown":
                reader_type = "rosbags+mcap_fallback"

            # Map topic to category
            cat = None
            for c, t in report["topic_map"].items():
                if t == msg.topic:
                    cat = c
                    break
            if cat is None:
                # Count uncategorised topics too
                msg_counts[msg.topic] = msg_counts.get(msg.topic, 0) + 1
                continue

            msg_counts[cat] = msg_counts.get(cat, 0) + 1

            # Extract header stamp (sample: every N-th message to save time)
            header_ts = None
            if msg_counts[cat] <= 50 or msg_counts[cat] % 100 == 0:
                try:
                    obj = deserialize(msg)
                    header_ts = get_header_stamp_ns(obj)
                except Exception:
                    pass

            if cat not in cat_timestamps:
                cat_timestamps[cat] = []
            cat_timestamps[cat].append((msg.timestamp, header_ts))

        report["reader"] = "bag_reader"
    except Exception as e:
        report["status"] = "BAD"
        report["reasons"].append(f"Cannot read bag: {e}")
        report["traceback"] = traceback.format_exc()
        return report

    report["msg_counts"] = msg_counts

    # ── 5. Duration ────────────────────────────────────────────────────────
    all_bag_ts = []
    for entries in cat_timestamps.values():
        all_bag_ts.extend([e[0] for e in entries])
    if all_bag_ts:
        duration_ns = max(all_bag_ts) - min(all_bag_ts)
        report["duration_sec"] = round(duration_ns / 1e9, 2)
    else:
        report["duration_sec"] = 0.0

    min_dur = cfg.get("min_duration_sec", 2.0)
    if report["duration_sec"] < min_dur:
        report["reasons"].append(
            f"Duration {report['duration_sec']:.1f}s < {min_dur}s"
        )

    # ── 6. Message count minimums ──────────────────────────────────────────
    for cat, min_key in [
        ("events", "min_event_msgs"),
        ("rgb", "min_rgb_frames"),
        ("odom", "min_odom_msgs"),
    ]:
        min_val = cfg.get(min_key, 10)
        actual_count = msg_counts.get(cat, 0)
        if actual_count < min_val:
            report["reasons"].append(
                f"'{cat}' has {actual_count} msgs < min {min_val}"
            )

    # ── 7. Timestamp monotonicity ──────────────────────────────────────────
    tol = cfg.get("monotonic_tolerance_count", 5)
    for cat, entries in cat_timestamps.items():
        bag_ts_arr = [e[0] for e in entries]
        violations = sum(
            1 for j in range(1, len(bag_ts_arr)) if bag_ts_arr[j] < bag_ts_arr[j - 1]
        )
        if violations > tol:
            report["reasons"].append(
                f"'{cat}' has {violations} out-of-order timestamps (tol={tol})"
            )

    # ── 8. Header vs bag-time drift ────────────────────────────────────────
    max_drift_ms = cfg.get("max_header_vs_bagtime_drift_ms", 5000.0)
    for cat, entries in cat_timestamps.items():
        drifts = [
            abs(bag_ts - hdr_ts) / 1e6
            for bag_ts, hdr_ts in entries
            if hdr_ts is not None
        ]
        if drifts:
            p95 = float(np.percentile(drifts, 95))
            if p95 > max_drift_ms:
                report["reasons"].append(
                    f"'{cat}' header drift p95={p95:.0f}ms > {max_drift_ms}ms"
                )

    # ── 9. Temporal overlap ────────────────────────────────────────────────
    min_overlap = cfg.get("min_overlap_fraction", 0.5)
    required_cats = list(cfg["required_topics"].keys())
    cat_ranges = {}
    for cat in required_cats:
        if cat in cat_timestamps and cat_timestamps[cat]:
            ts_list = [e[0] for e in cat_timestamps[cat]]
            cat_ranges[cat] = (min(ts_list), max(ts_list))

    if len(cat_ranges) >= 2 and report["duration_sec"] > 0:
        overlap_start = max(r[0] for r in cat_ranges.values())
        overlap_end = min(r[1] for r in cat_ranges.values())
        overlap_dur = max(0, overlap_end - overlap_start)
        total_dur = max(r[1] for r in cat_ranges.values()) - min(
            r[0] for r in cat_ranges.values()
        )
        overlap_frac = overlap_dur / total_dur if total_dur > 0 else 0.0
        report["overlap_fraction"] = round(overlap_frac, 4)
        if overlap_frac < min_overlap:
            report["reasons"].append(
                f"Temporal overlap {overlap_frac:.2%} < {min_overlap:.0%}"
            )

    # ── Final verdict ──────────────────────────────────────────────────────
    report["status"] = "BAD" if report["reasons"] else "GOOD"
    return report


def main():
    parser = argparse.ArgumentParser(
        description="Check ROS2 bag quality against QA thresholds."
    )
    parser.add_argument("--bags-dir", required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    bag_dirs = find_bag_dirs(args.bags_dir)
    if not bag_dirs:
        print(f"[ERROR] No bag directories found under {args.bags_dir}")
        sys.exit(1)
    print(f"Found {len(bag_dirs)} bag(s) under {args.bags_dir}")

    get_ts()  # warm up typestore

    out_dir = args.out_dir or args.bags_dir
    os.makedirs(out_dir, exist_ok=True)
    jsonl_path = os.path.join(out_dir, "bag_report.jsonl")
    csv_path = os.path.join(out_dir, "bag_report.csv")

    reports = []
    good_count = bad_count = 0

    for i, bag_dir in enumerate(bag_dirs):
        bag_name = os.path.basename(bag_dir)
        print(
            f"\n[{i+1}/{len(bag_dirs)}] Checking {bag_name} ...",
            end=" ",
            flush=True,
        )
        t0 = time.time()
        report = check_bag(bag_dir, cfg)
        elapsed = time.time() - t0
        report["check_time_sec"] = round(elapsed, 1)
        reports.append(report)

        if report["status"] == "GOOD":
            good_count += 1
            mc = report["msg_counts"]
            print(
                f"GOOD  ({elapsed:.1f}s)  "
                f"ev={mc.get('events',0)} rgb={mc.get('rgb',0)} odom={mc.get('odom',0)}"
            )
        else:
            bad_count += 1
            reasons = "; ".join(report["reasons"][:2])
            print(f"BAD   ({elapsed:.1f}s) — {reasons}")

    # ── Write JSONL ────────────────────────────────────────────────────────
    with open(jsonl_path, "w") as f:
        for r in reports:
            f.write(json.dumps(r, default=str) + "\n")
    print(f"\nJSONL report: {jsonl_path}")

    # ── Write CSV ──────────────────────────────────────────────────────────
    csv_fields = [
        "bag_name", "status", "duration_sec",
        "events_msgs", "rgb_msgs", "odom_msgs",
        "rgb_topic", "overlap", "reasons",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for r in reports:
            writer.writerow({
                "bag_name": r["bag_name"],
                "status": r["status"],
                "duration_sec": f'{r["duration_sec"]:.1f}' if r["duration_sec"] else "",
                "events_msgs": r["msg_counts"].get("events", 0),
                "rgb_msgs": r["msg_counts"].get("rgb", 0),
                "odom_msgs": r["msg_counts"].get("odom", 0),
                "rgb_topic": r["topic_map"].get("rgb", ""),
                "overlap": r.get("overlap_fraction", ""),
                "reasons": "; ".join(r["reasons"]),
            })
    print(f"CSV report:   {csv_path}")

    # ── Summary ────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  GOOD: {good_count}   BAD: {bad_count}   TOTAL: {len(reports)}")
    print(f"{'='*60}")
    if bad_count:
        print("\nBAD bags:")
        for r in reports:
            if r["status"] == "BAD":
                print(f"  {r['bag_name']}: {'; '.join(r['reasons'][:2])}")


if __name__ == "__main__":
    main()
