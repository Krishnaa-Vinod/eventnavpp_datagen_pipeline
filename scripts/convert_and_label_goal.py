#!/usr/bin/env python3
"""
convert_and_label_goal.py — Convert one bag to H5, then launch visualizer
for goal labeling.

Workflow:
  1) Convert bag → H5 (reuses convert_bag() from convert_all_rosbags.py)
  2) Launch visualizer automatically:
     - If DISPLAY exists → GUI (matplotlib)
     - Otherwise → web (streamlit)
  3) After goal is set, writes goal into H5 and exits.
  4) If user quits without setting goal, H5 keeps goal_step=-1.

Usage:
  # Auto-detect visualizer
  python scripts/convert_and_label_goal.py \\
      --bag /scratch/kvinod/bags/data_collect_20260207_150734

  # Force web mode (for sol)
  python scripts/convert_and_label_goal.py \\
      --bag /scratch/kvinod/bags/data_collect_20260207_150734 \\
      --visualizer web --port 8501

  # Force GUI mode
  python scripts/convert_and_label_goal.py \\
      --bag /scratch/kvinod/bags/data_collect_20260207_150734 \\
      --visualizer gui
"""

import argparse
import os
import sys
from pathlib import Path

# Ensure repo imports work
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))


def main():
    parser = argparse.ArgumentParser(
        description="Convert one bag to H5 and label goal interactively."
    )
    parser.add_argument("--bag", required=True,
                        help="Path to bag directory")
    parser.add_argument("--out-dir",
                        default="/scratch/kvinod/bags/ego_navi_h5_v2",
                        help="Output directory for H5 files")
    parser.add_argument("--compression",
                        choices=["lzf", "gzip", "none"], default="lzf")
    parser.add_argument("--visualizer",
                        choices=["auto", "gui", "web"], default="auto",
                        help="Visualizer mode (default: auto)")
    parser.add_argument("--port", type=int, default=8501,
                        help="Port for web visualizer (default: 8501)")
    parser.add_argument("--force", action="store_true",
                        help="Re-convert even if H5 exists")
    parser.add_argument("--log-memory", action="store_true")
    parser.add_argument("--skip-visualizer", action="store_true",
                        help="Skip visualizer (just convert)")
    args = parser.parse_args()

    bag_dir = os.path.abspath(args.bag)
    if not os.path.isdir(bag_dir):
        print(f"Error: {bag_dir} is not a directory")
        sys.exit(1)

    bag_name = os.path.basename(bag_dir)
    h5_path = os.path.join(args.out_dir, f"{bag_name}.h5")

    # ── Step 1: Convert ──────────────────────────────────────────────────
    if os.path.exists(h5_path) and not args.force:
        print(f"H5 exists: {h5_path}")
        print("Use --force to re-convert, or proceeding to visualizer.")
    else:
        print(f"Converting: {bag_dir}")
        print(f"Output: {h5_path}")
        print()

        from convert_all_rosbags import convert_bag
        from bag_reader import get_ts
        get_ts()  # warm up typestore

        result = convert_bag(
            bag_dir, h5_path,
            compression=args.compression,
            log_memory=args.log_memory,
        )

        if result["status"] != "OK":
            print(f"\nConversion FAILED: {result.get('error', 'unknown')}")
            sys.exit(1)

        print(f"\nConversion OK: {result['steps']} steps, "
              f"{result['rgb_stored']} RGB")

    if args.skip_visualizer:
        print(f"\nH5 saved: {h5_path}")
        return

    # ── Step 2: Launch visualizer ────────────────────────────────────────
    mode = args.visualizer
    if mode == "auto":
        if os.environ.get("DISPLAY"):
            mode = "gui"
        else:
            mode = "web"

    print(f"\nLaunching {mode} visualizer …")

    if mode == "gui":
        from h5_goal_picker_gui import run_gui, write_goal

        goal = run_gui(h5_path)
        if goal >= 0:
            write_goal(h5_path, goal)
        else:
            print("No goal selected — H5 unchanged.")

    elif mode == "web":
        import subprocess
        launcher = SCRIPT_DIR / "run_goal_web_ui.py"
        cmd = [
            sys.executable, str(launcher),
            "--h5", h5_path,
            "--port", str(args.port),
        ]
        proc = subprocess.run(cmd)
        if proc.returncode != 0:
            print(f"Web UI exited with code {proc.returncode}")

    print(f"\nDone. H5: {h5_path}")


if __name__ == "__main__":
    main()
