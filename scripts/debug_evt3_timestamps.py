#!/usr/bin/env python3
"""
debug_evt3_timestamps.py — Diagnose EVT3 timestamp monotonicity issues.

Reads event messages from a bag and reports per-message timestamp ranges,
backward jumps (wraps), and relationship to step windows.

Usage:
    python scripts/debug_evt3_timestamps.py --bag /path/to/bag --nmsgs 200
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from bag_reader import read_bag, deserialize, get_topics, get_ts
from evt3_decoder import decode_evt3

EVENT_ALIASES = ["/event_camera/events"]


def resolve_topic(available, aliases):
    for a in aliases:
        if a in available:
            return a
    return None


def main():
    parser = argparse.ArgumentParser(description="EVT3 timestamp diagnostics")
    parser.add_argument("--bag", required=True, help="Path to bag directory")
    parser.add_argument("--nmsgs", type=int, default=200,
                        help="Number of event messages to analyze")
    parser.add_argument("--use-unwrap", action="store_true",
                        help="Use EVT3StreamDecoder (stateful unwrap) instead of raw")
    args = parser.parse_args()

    available = get_topics(args.bag)
    event_topic = resolve_topic(available, EVENT_ALIASES)
    if not event_topic:
        print(f"ERROR: No event topic found in {args.bag}")
        sys.exit(1)

    # Optionally use stateful decoder
    decoder = None
    if args.use_unwrap:
        try:
            from evt3_decoder import EVT3StreamDecoder
            decoder = EVT3StreamDecoder()
            print("Using EVT3StreamDecoder (stateful unwrap)")
        except ImportError:
            print("WARN: EVT3StreamDecoder not available, using raw decode_evt3")

    get_ts()  # warm up typestore

    msg_idx = 0
    prev_max_t = None
    first_hdr_ts = None
    first_sensor_t = None
    total_backward_jumps = 0
    max_backward_jump = 0
    wrap_events = []

    print(f"\n{'idx':>5} | {'hdr_ts':>20} | {'t_min':>12} | {'t_max':>12} | "
          f"{'n_events':>10} | {'jump':>12} | {'note'}")
    print("-" * 100)

    for bag_msg in read_bag(args.bag, topics={event_topic}):
        if msg_idx >= args.nmsgs:
            break

        obj = deserialize(bag_msg)
        hdr_ts = None
        if hasattr(obj, 'header'):
            hdr_ts = obj.header.stamp.sec * 1_000_000_000 + obj.header.stamp.nanosec
        if first_hdr_ts is None and hdr_ts is not None:
            first_hdr_ts = hdr_ts

        raw = bytes(obj.events)
        if len(raw) < 4:
            msg_idx += 1
            continue

        if decoder is not None:
            evs = decoder.decode(raw, obj.width, obj.height)
        else:
            evs, _state = decode_evt3(raw, obj.width, obj.height)

        if evs is None or len(evs) == 0:
            msg_idx += 1
            continue

        t_col = evs[:, 2]
        t_min = int(t_col.min())
        t_max = int(t_col.max())

        if first_sensor_t is None:
            first_sensor_t = t_min

        # Check intra-message monotonicity
        intra_diffs = np.diff(t_col)
        intra_backward = int((intra_diffs < 0).sum())

        # Check inter-message jump
        note = ""
        jump = 0
        if prev_max_t is not None:
            jump = t_min - prev_max_t
            if jump < -1000:  # >1ms backward
                total_backward_jumps += 1
                if abs(jump) > abs(max_backward_jump):
                    max_backward_jump = jump
                note = f"*** BACKWARD {jump:+d} us"
                wrap_events.append((msg_idx, jump, t_min, t_max, prev_max_t))
            elif jump < 0:
                note = f"minor backward {jump:+d} us"

        if intra_backward > 0:
            note += f" [intra-backward: {intra_backward}]"

        print(f"{msg_idx:5d} | {hdr_ts or 0:20d} | {t_min:12d} | {t_max:12d} | "
              f"{len(evs):10d} | {jump:+12d} | {note}")

        prev_max_t = t_max
        msg_idx += 1

    # Summary
    print(f"\n{'='*80}")
    print(f"SUMMARY ({msg_idx} messages analyzed)")
    print(f"  First header timestamp:  {first_hdr_ts}")
    print(f"  First sensor timestamp:  {first_sensor_t} us")
    if prev_max_t and first_sensor_t:
        span = prev_max_t - first_sensor_t
        print(f"  Last sensor timestamp:   {prev_max_t} us")
        print(f"  Sensor time span:        {span} us = {span/1e6:.3f} s")
    print(f"  Total backward jumps >1ms:  {total_backward_jumps}")
    print(f"  Max backward jump:          {max_backward_jump} us = {max_backward_jump/1e6:.3f} s")

    if wrap_events:
        print(f"\n  Backward jump details:")
        WRAP_24 = 2**24  # 16777216 us
        for idx, jump, tmin, tmax, prev in wrap_events:
            near_wrap = abs(abs(jump) - WRAP_24) < 1_000_000
            wrap_note = " <-- near 2^24 wrap!" if near_wrap else ""
            print(f"    msg {idx}: jump={jump:+d} us  "
                  f"(prev_max={prev}, this_min={tmin}){wrap_note}")

        print(f"\n  2^24 = {WRAP_24} us = {WRAP_24/1e6:.3f} s")
        print(f"  If jumps are near -{WRAP_24}, the EVT3 24-bit timestamp is wrapping.")
    else:
        print(f"\n  No significant backward jumps detected — timestamps appear monotonic.")


if __name__ == "__main__":
    main()
