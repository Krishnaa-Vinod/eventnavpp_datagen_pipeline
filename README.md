# eventnavpp_datagen_pipeline

End-to-end pipeline for converting ROS2 bag recordings (event camera + RGB + odometry) into HDF5 datasets for event-based visual navigation training.

## Repository layout

```
├── bag_to_h5.py              # Original converter (requires ROS2 deps)
├── h5_visualizer.py           # Interactive HDF5 viewer (matplotlib/TkAgg)
├── bag_reader.py              # Pure-Python ROS2 bag reader (handles truncated MCAP)
├── evt3_decoder.py            # Pure-Python Prophesee EVT3 event decoder
├── configs/
│   └── qa.yaml                # QA thresholds for bag quality checks
├── scripts/
│   ├── check_rosbags.py       # Step B – bag quality checker
│   ├── convert_all_rosbags.py # Step C – batch bag → H5 converter (v2: OOM-safe, LZF)
│   ├── resource_utils.py      # CPU/RAM/GPU resource detection
│   ├── validate_h5.py         # Step D – H5 validation
│   └── build_dataset_index.py # Step D – dataset metadata index
├── docs/
│   └── PIPELINE_OVERVIEW.md   # Full pipeline documentation
└── event_slam_ws/             # ROS2 workspace (event SLAM)
```

## Quick start

```bash
# Activate environment
conda activate nomad-eventvoxels

# 1. Check bag quality (marks BAD bags)
python scripts/check_rosbags.py --bags-dir /scratch/kvinod/bags

# 2. Convert GOOD bags to HDF5
python scripts/convert_all_rosbags.py \
    --bags-dir /scratch/kvinod/bags \
    --out-dir /scratch/kvinod/bags/eGo_navi_.h5

# 2b. Convert with speed/safety options (recommended)
python scripts/convert_all_rosbags.py \
    --bags-dir /scratch/kvinod/bags \
    --compression lzf --log-memory --flush-every 10

# 2c. Maximise throughput (parallel, GPU voxelisation)
python scripts/convert_all_rosbags.py \
    --bags-dir /scratch/kvinod/bags \
    --compression lzf --use-gpu --jobs auto --log-memory

# 2d. Re-convert a single bag
python scripts/convert_all_rosbags.py \
    --bags-dir /scratch/kvinod/bags \
    --bag-name data_collect_20260207_150734 --force

# 3. Validate converted files
python scripts/validate_h5.py --h5-dir /scratch/kvinod/bags/eGo_navi_.h5

# 4. Build dataset index
python scripts/build_dataset_index.py --h5-dir /scratch/kvinod/bags/eGo_navi_.h5

# 5. Visualize a file
python h5_visualizer.py --h5 /scratch/kvinod/bags/eGo_navi_.h5/data_collect_20260219_171045.h5
```

## HDF5 schema

| Dataset | Shape | Dtype | Description |
|---------|-------|-------|-------------|
| `voxels` | `(N, 5, 720, 1280)` | float32 | Normalised event voxel grid (5 temporal bins) |
| `actions` | `(N, 8, 3)` | float32 | 8-step action chunks `(dx, dy, dyaw)` in local frame |
| `timestamps_ns` | `(N,)` | int64 | ROS nanosecond timestamp per step |
| `rgb_mask` | `(N,)` | bool | Whether an RGB frame is available for this step |
| `rgb_images` | `(M, 1024, 1280, 3)` | uint8 | RGB images (stored in **RGB** channel order) |
| `rgb_indices` | `(M,)` | int32 | Maps each RGB image to its corresponding step index |

## Key parameters

- **Voxel window**: 250 ms
- **Temporal bins**: 5
- **Event resolution**: 720 × 1280 (Prophesee EVT3)
- **RGB resolution**: 1024 × 1280 (Bayer RGGB demosaiced)
- **Action horizon**: 8 steps (2 seconds ahead)

## Converter CLI flags (v2)

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
