# eventnavpp_datagen_pipeline

End-to-end pipeline for converting ROS2 bag recordings (event camera + RGB + odometry) into HDF5 datasets for event-based visual navigation training.

## Repository layout

```
├── bag_to_h5.py                        # Original converter (requires ROS2 deps)
├── h5_visualizer.py                    # Interactive HDF5 viewer (matplotlib/TkAgg)
├── bag_reader.py                       # Pure-Python ROS2 bag reader (handles truncated MCAP)
├── evt3_decoder.py                     # Pure-Python Prophesee EVT3 event decoder
├── configs/
│   └── qa.yaml                         # QA thresholds for bag quality checks
├── scripts/
│   ├── check_rosbags.py                # Step B – bag quality checker
│   ├── convert_all_rosbags.py          # Step C – v3 odom-driven bag → H5 converter
│   ├── validate_h5.py                  # Step D – H5 validation (v3 schema aware)
│   ├── print_h5_summary.py             # Quick H5 inspection
│   ├── h5_goal_picker_gui.py           # Goal labeling (matplotlib GUI)
│   ├── h5_goal_picker_web.py           # Goal labeling (Streamlit / headless)
│   ├── run_goal_web_ui.py              # Streamlit launcher with SSH instructions
│   ├── convert_and_label_goal.py       # Convert one bag + label goal (single CLI)
│   ├── resource_utils.py               # CPU/RAM/GPU resource detection
│   └── build_dataset_index.py          # Dataset metadata index
├── docs/
│   └── PIPELINE_OVERVIEW.md            # Full pipeline documentation
└── event_slam_ws/                      # ROS2 workspace (event SLAM)
```

## Quick start

```bash
conda activate nomad-eventvoxels

# 1. Check bag quality
python scripts/check_rosbags.py --bags-dir /scratch/kvinod/bags

# 2. Convert GOOD bags to HDF5 (v3 – odom-driven stepping)
python scripts/convert_all_rosbags.py \
    --bags-dir /scratch/kvinod/bags \
    --out-dir /scratch/kvinod/bags/ego_navi_h5_v2

# 2b. Convert a single bag
python scripts/convert_all_rosbags.py \
    --bags-dir /scratch/kvinod/bags \
    --bag-name data_collect_20260207_150734 --force --log-memory

# 3. Validate converted files
python scripts/validate_h5.py --h5-dir /scratch/kvinod/bags/ego_navi_h5_v2

# 4. Quick H5 inspection
python scripts/print_h5_summary.py --h5 <path_to.h5>

# 5. Visualize & label goal
python scripts/h5_goal_picker_gui.py --h5 <path_to.h5>
```

## Convert + label goal (single command)

```bash
# GUI mode (if DISPLAY is set)
python scripts/convert_and_label_goal.py \
    --bag /scratch/kvinod/bags/data_collect_20260207_150734 \
    --out-dir /scratch/kvinod/bags/ego_navi_h5_v2

# Headless (Streamlit web UI — e.g. on Sol)
python scripts/convert_and_label_goal.py \
    --bag /scratch/kvinod/bags/data_collect_20260207_150734 \
    --out-dir /scratch/kvinod/bags/ego_navi_h5_v2 \
    --visualizer web --port 8501

# SSH port forwarding for Sol
ssh -L 8501:localhost:8501 <user>@sol.asu.edu
```

## HDF5 schema (v3)

| Dataset | Shape | Dtype | Description |
|---------|-------|-------|-------------|
| `voxels` | `(N, 5, 720, 1280)` | float32 | Normalised event voxel grid (5 temporal bins) |
| `actions` | `(N, 8, 3)` | float32 | 8-step relative waypoints `(dx, dy, dYaw)` in robot local frame |
| `actions_valid` | `(N,)` | bool | True if full 8-step horizon is within odom range |
| `timestamps_ns` | `(N,)` | int64 | ROS nanosecond timestamp per step |
| `rgb_mask` | `(N,)` | bool | Whether an RGB frame is available for this step |
| `rgb_images` | `(M, 1024, 1280, 3)` | uint8 | RGB images (stored in **RGB** channel order) |
| `rgb_indices` | `(M,)` | int32 | Maps each RGB image to its corresponding step index |

**Attributes:**

| Attribute | Example | Description |
|-----------|---------|-------------|
| `actions_space` | `relative_waypoints` | Action representation type |
| `actions_repr` | `dx_dy_dyaw` | Per-waypoint format |
| `actions_frame` | `base_link_at_step` | Coordinate frame for actions |
| `dyaw_wrapped` | `true` | dYaw wrapped to [-π, π] |
| `goal_step` | `-1` (unset) or `≥0` | Labeled goal step index |
| `goal_timestamp_ns` | | Timestamp of the goal step |

## Key parameters

- **Stepping**: Odom-driven (full trajectory from first to last odom timestamp)
- **Voxel window**: 250 ms
- **Temporal bins**: 5
- **Event resolution**: 720 × 1280 (Prophesee EVT3)
- **RGB resolution**: 1024 × 1280 (Bayer RGGB demosaiced)
- **Action horizon**: 8 steps (2 seconds ahead)
- **Action semantics**: `(dx, dy, dYaw)` in robot local frame at each step, matching NoMaD/ViNT/GNM

## v3 changes from v2

- **Odom-driven stepping**: Steps span full odom range instead of being gated by event/RGB coverage
- **Streaming events**: Memory-efficient processing (~1 GB peak vs 16+ GB for bulk loading)
- **`actions_valid` flag**: Marks steps near trajectory end where action horizon extends past odom data
- **Goal labeling**: `goal_step` attribute (set via GUI/web goal picker)
- **Both-neighbor RGB matching**: Checks left and right timestamps around bisect point

## Converter CLI flags (v3)

| Flag | Default | Description |
|------|---------|-------------|
| `--bags-dir` | *(required)* | Root directory containing bag folders |
| `--out-dir` | `/scratch/kvinod/bags/eGo_navi_.h5` | Output directory for `.h5` files |
| `--bag-name` | `None` | Convert only this single bag |
| `--force` | `False` | Re-convert even if `.h5` already exists |
| `--compression` | `lzf` | HDF5 compression: `lzf` (fastest), `gzip`, `none` |
| `--gzip-level` | `1` | gzip level 1–9 (only with `--compression gzip`) |
| `--voxel-dtype` | `float32` | Voxel storage dtype: `float32`, `float16` |
| `--use-gpu` | `False` | GPU voxelisation via torch CUDA `scatter_add_` |
| `--jobs` | `1` | Parallel bag workers: integer or `auto` (RAM-safe) |
| `--flush-every` | `10` | Flush H5 file every N steps |
| `--log-memory` | `False` | Log RSS memory usage every 20 steps |
| `--report` | `None` | Path to `bag_report.jsonl` (auto-detected) |
| `--dry-run` | `False` | Print plan without converting |

### Crash safety

- Each bag writes to a temporary file (`<name>.h5.tmp`), then atomically
  renames to `<name>.h5` on success.
- On restart, stale `.tmp` files from previous kills are removed automatically.

### Recommended settings

```bash
# Safe and fast (single-threaded, LZF, memory logging)
python scripts/convert_all_rosbags.py --bags-dir /scratch/kvinod/bags \
    --compression lzf --log-memory

# Maximum throughput on a multi-core node
python scripts/convert_all_rosbags.py --bags-dir /scratch/kvinod/bags \
    --compression lzf --jobs auto --use-gpu --log-memory
```

## Dependencies

```
python ≥ 3.11
h5py, numpy, opencv-python, scipy, matplotlib
rosbags ≥ 0.11      # pure-Python ROS2 bag reader
mcap ≥ 1.3           # MCAP streaming (truncated bag fallback)
mcap-ros2-support    # CDR deserialization
pyyaml, rich, tqdm   # utilities
```

## Notes

- **Truncated MCAP files**: 44/46 bags have truncated MCAP end magic. The pipeline handles this via `mcap.stream_reader.StreamReader` fallback.
- **Color fix**: The original `bag_to_h5.py` stored BGR; the new pipeline stores **RGB** directly. The visualizer has been updated to match.
- **EVT3 decoding**: Uses a custom pure-Python decoder (no `event_camera_py` dependency). Handles EVT_ADDR_Y, EVT_ADDR_X, VECT_BASE_X, VECT_12, VECT_8, TIME_LOW, TIME_HIGH.
- **Polarity**: This sensor produces ~99.9% OFF events (pol=0 → -1 in voxels).

See [docs/PIPELINE_OVERVIEW.md](docs/PIPELINE_OVERVIEW.md) for full technical details.
