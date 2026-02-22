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
│   ├── convert_all_rosbags.py # Step C – batch bag → H5 converter
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
