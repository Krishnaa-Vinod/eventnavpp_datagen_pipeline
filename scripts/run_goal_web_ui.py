#!/usr/bin/env python3
"""
run_goal_web_ui.py — Launch Streamlit goal picker and wait for completion.

Starts streamlit as a subprocess, prints SSH port-forward instructions,
waits for a sentinel file indicating goal was set, then stops streamlit.

Usage:
  python scripts/run_goal_web_ui.py --h5 /path/to/file.h5 --port 8501
"""

import argparse
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(
        description="Launch Streamlit goal picker for H5 files."
    )
    parser.add_argument("--h5", required=True, help="Path to H5 file")
    parser.add_argument("--port", type=int, default=8501,
                        help="Port for Streamlit (default: 8501)")
    parser.add_argument("--timeout", type=int, default=3600,
                        help="Max seconds to wait (default: 3600)")
    args = parser.parse_args()

    if not os.path.exists(args.h5):
        print(f"Error: {args.h5} not found")
        sys.exit(1)

    # Create sentinel file
    sentinel = tempfile.NamedTemporaryFile(
        prefix="goal_sentinel_", suffix=".txt", delete=False
    )
    sentinel_path = sentinel.name
    sentinel.close()
    os.remove(sentinel_path)  # Remove — streamlit will create it when goal is set

    # Resolve streamlit script path
    web_script = SCRIPT_DIR / "h5_goal_picker_web.py"
    if not web_script.exists():
        print(f"Error: {web_script} not found")
        sys.exit(1)

    hostname = os.uname().nodename

    print("=" * 60)
    print("  H5 Goal Picker — Web UI")
    print("=" * 60)
    print(f"  H5 file:  {args.h5}")
    print(f"  Port:     {args.port}")
    print()
    print("  To access from your local machine, run:")
    print(f"    ssh -L {args.port}:localhost:{args.port} $USER@sol")
    print(f"  Then open: http://localhost:{args.port}")
    print()
    print(f"  (Server hostname: {hostname})")
    print("  Press Ctrl+C to abort without saving goal.")
    print("=" * 60)
    print()

    # Set env vars for the streamlit app
    env = os.environ.copy()
    env["H5_GOAL_PICKER_PATH"] = os.path.abspath(args.h5)
    env["H5_GOAL_PICKER_SENTINEL"] = sentinel_path

    # Launch streamlit
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(web_script),
        "--server.port", str(args.port),
        "--server.headless", "true",
        "--server.address", "0.0.0.0",
        "--browser.gatherUsageStats", "false",
        "--",
        "--h5", os.path.abspath(args.h5),
        "--sentinel", sentinel_path,
    ]

    proc = subprocess.Popen(cmd, env=env)

    try:
        elapsed = 0
        poll_interval = 2
        while elapsed < args.timeout:
            # Check if sentinel file was created
            if os.path.exists(sentinel_path):
                goal_step = Path(sentinel_path).read_text().strip()
                print(f"\nGoal set to step {goal_step}! Closing web UI...")
                time.sleep(1)
                break

            # Check if streamlit process died
            if proc.poll() is not None:
                print(f"\nStreamlit exited with code {proc.returncode}")
                break

            time.sleep(poll_interval)
            elapsed += poll_interval
        else:
            print(f"\nTimeout after {args.timeout}s — closing.")
    except KeyboardInterrupt:
        print("\nInterrupted — closing without saving goal.")
    finally:
        # Kill streamlit
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()

        # Clean up sentinel
        if os.path.exists(sentinel_path):
            os.remove(sentinel_path)

    print("Done.")


if __name__ == "__main__":
    main()
