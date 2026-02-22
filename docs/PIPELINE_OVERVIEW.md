# Pipeline Overview

Full documentation for the ROSbag → HDF5 dataset generation pipeline.

## Architecture

```
                      ┌────────────────────┐
   ROS2 bags (.mcap)  │  check_rosbags.py  │  → bag_report.jsonl
   /scratch/kvinod/   │  (Step B)          │    GOOD / BAD per bag
   bags/              └────────┬───────────┘
                               │
                      ┌────────▼───────────┐
                      │ convert_all_       │  → .h5 per GOOD bag
                      │ rosbags.py         │    + convert_report.jsonl
                      │ (Step C)           │
                      └────────┬───────────┘
                               │
                      ┌────────▼───────────┐
                      │  validate_h5.py    │  → validate_report.jsonl
                      │  (Step D)          │    PASS / WARN / FAIL
                      └────────┬───────────┘
                               │
                      ┌────────▼───────────┐
                      │ build_dataset_     │  → dataset_index.json
                      │ index.py (Step D)  │    + dataset_index.csv
                      └────────┬───────────┘
                               │
                      ┌────────▼───────────┐
                      │  h5_visualizer.py  │   Interactive viewer
                      │  (Step E)          │   (matplotlib / TkAgg)
                      └────────────────────┘
```

## 1. Bag Reader (`bag_reader.py`)

Pure-Python ROS2 bag reader with two backends:

1. **Primary**: `rosbags.rosbag2.Reader` — requires valid MCAP end magic
2. **Fallback**: `mcap.stream_reader.StreamReader` — handles truncated MCAP files (catches `EndOfFile`)

### TypeStore registration

Custom message types (e.g. `event_camera_msgs/msg/EventPacket`) are registered with `rosbags.typesys`:

```
EventPacket:
  std_msgs/Header header
  uint32 height       # 720
  uint32 width        # 1280
  uint64 seq
  uint64 time_base    # sensor epoch in µs (0 for this dataset)
  string encoding     # "evt3"
  bool is_bigendian
  uint8[] events      # raw EVT3 binary payload
```

### API

- `read_bag(bag_dir)` → iterator of `BagMessage(topic, timestamp, data, msgtype)`
- `deserialize(msg)` → deserialized Python object
- `get_topics(bag_dir)` → set of topic names (from metadata.yaml)
- `get_ts()` → shared TypeStore instance

## 2. EVT3 Decoder (`evt3_decoder.py`)

Pure-Python implementation of the Prophesee EVT3 binary event encoding.

### Format

Each "word" is 16-bit little-endian. The top 4 bits encode the type:

| Code | Name | Payload | Description |
|------|------|---------|-------------|
| 0x0 | EVT_ADDR_Y | `[11]=pol, [10:0]=y` | Sets current y-coordinate and polarity |
| 0x2 | EVT_ADDR_X | `[11]=t_inc, [10:0]=x` | Emits a single event at (x, y, pol) |
| 0x3 | VECT_BASE_X | `[11:0]=base_x` | Sets base x for vectorised events |
| 0x4 | VECT_12 | 12-bit bitmap | Emits events at base_x + set-bit positions |
| 0x5 | VECT_8 | `[11:8]=offset, [7:0]=bitmap` | 8-bit bitmap with offset |
| 0x6 | TIME_LOW | `[11:0]=low12` | Sets low 12 bits of timestamp |
| 0x8 | TIME_HIGH | `[11:0]=high12` | Sets high 12 bits (bits 12–23) |
| 0xF | VECTOR_12_Y | same as EVT_ADDR_Y | Alternate y-address for vectors |
| 0xE | CONTINUED_12 | — | **Ignored** (sensor sync markers that cause spurious ts jumps) |

### Timestamp handling

The sensor timestamps are in microseconds. The decoder maintains `ts_low` (12 bits) and `ts_high` (12 bits) to form 24-bit timestamps. The CONTINUED_12 (0xE) codes with payload 0xFF followed by TIME_HIGH are sync/calibration markers that would create ~4.3 billion µs jumps — these are filtered out.

### Polarity distribution

This specific sensor produces ~99.9% OFF events (polarity=0). In the voxel grid, OFF maps to -1 and ON maps to +1.

### API

```python
decode_evt3(raw_bytes, width, height) → np.ndarray  # shape (N, 4): x, y, t_us, pol
```

## 3. Bag Quality Checker (`scripts/check_rosbags.py`)

Validates each bag directory against thresholds in `configs/qa.yaml`:

- **Readability**: Can the bag be opened (handles truncated MCAP via fallback)
- **Required topics**: At least one alias per category (events, rgb, odom)
- **Message counts**: Minimum 10 msgs per required topic
- **Duration**: At least 2 seconds of data
- **Monotonicity**: No more than 5 out-of-order timestamps per topic
- **Header drift**: Header timestamps within 5 seconds of bag time
- **Temporal overlap**: ≥50% of duration has all required topics present

### Output

```
bag_report.jsonl  — one JSON per bag: {bag_name, bag_dir, status, ...}
bag_report.csv    — summary table
```

### Results on current dataset

- **45 GOOD**, **1 BAD** (`data_collect_20260207_152052` — corrupt zstd-compressed MCAP)

## 4. Batch Converter (`scripts/convert_all_rosbags.py`)

Two-pass conversion for each GOOD bag:

### Pass 1: Pre-scan

Reads the entire bag once to collect:
- Odometry messages → `odom_ts[]`, `odom_poses[]` (x, y, yaw from quaternion)
- RGB messages → `rgb_ts[]`, raw CDR data stored in memory
- First event message → `bag_to_sensor_offset_ns` (ROS ns ↔ sensor µs mapping)

**Offset calculation** (critical for correctness):
```python
# Decode first event packet to get actual sensor timestamp
first_sensor_us = decode_evt3(first_event_msg.events)[0, 2]
bag_to_sensor_offset_ns = header_ts_ns - first_sensor_us * 1000
```

This correctly maps between ROS wall-clock nanoseconds and the sensor's internal microsecond clock, regardless of the `time_base` field value.

### Pass 2: Event streaming

Iterates event messages in chronological order, accumulating into 250ms voxel windows:

1. Decode EVT3 events from each message
2. Buffer events until the latest timestamp exceeds the current window end
3. Split the buffer at the window boundary
4. Build voxel grid via `events_to_voxel_fast()` (bilinear temporal interpolation into 5 bins, per-bin normalisation)
5. Match nearest RGB frame to window center (within ±125ms)
6. Compute 8-step action chunk from interpolated odometry
7. Write step to H5 and advance window

### Voxelisation

Each event contributes to two adjacent temporal bins with bilinear weights:

```
bin_norm = (t - t_start) / (t_end - t_start) * (num_bins - 1)
bin_low, bin_high = floor(bin_norm), floor(bin_norm) + 1
w_low = 1 - (bin_norm - bin_low)
voxel[bin_low, y, x] += polarity * w_low
voxel[bin_high, y, x] += polarity * w_high
```

Each bin is normalised by its maximum absolute value.

### Colour handling

The Bayer RGGB sensor data is demosaiced to **RGB** (not BGR):
```python
cv2.cvtColor(raw, cv2.COLOR_BayerRG2RGB)
```

Images are horizontally flipped to match the original `bag_to_h5.py` convention, then stored as RGB uint8 in H5. This fixes the previous "orange appears as blue" colour bug.

### CLI

```bash
# Convert all GOOD bags
python scripts/convert_all_rosbags.py --bags-dir /scratch/kvinod/bags

# Single bag
python scripts/convert_all_rosbags.py --bags-dir /scratch/kvinod/bags \
    --bag-name data_collect_20260219_171045

# Force re-convert
python scripts/convert_all_rosbags.py --bags-dir /scratch/kvinod/bags --force

# Dry run
python scripts/convert_all_rosbags.py --bags-dir /scratch/kvinod/bags --dry-run
```

## 5. H5 Validator (`scripts/validate_h5.py`)

Checks each HDF5 file for:

| Check | Threshold |
|-------|-----------|
| Required datasets present | All 6 datasets with correct dtype |
| No NaN/Inf | Zero tolerance for voxels and actions |
| Voxel range | [-1, 1] (normalised) |
| Voxel non-zero fraction | ≥ 0.1% |
| Action magnitude | ≤ 50 |
| Timestamp monotonicity | Strictly increasing |
| RGB/mask consistency | rgb_images count ≈ rgb_mask True count |
| RGB not all-black | Mean pixel value > 5 |
| Metadata attributes | bag_name, voxel_window_us, etc. |

Outputs: `validate_report.jsonl`, `validate_report.csv`

## 6. Dataset Index (`scripts/build_dataset_index.py`)

Aggregates metadata from all H5 files into:
- `dataset_index.json` — complete index with per-file statistics
- `dataset_index.csv` — spreadsheet-friendly summary

## 7. H5 Visualizer (`h5_visualizer.py`)

Interactive matplotlib viewer with three panels:
- **Event voxel** — sum across temporal bins (viridis colormap)
- **RGB frame** — nearest RGB goal image (displayed directly as RGB)
- **Action trajectory** — 8-step predicted path with robot heading arrows

Controls: `d`/→ = next step, `a`/← = previous, `q` = quit

### Color handling

Stores and displays **RGB** directly — no BGR↔RGB conversion needed. The O(N) `np.where()` lookup for rgb_indices has been replaced with a precomputed dict for O(1) access.

## Dataset statistics

| Metric | Value |
|--------|-------|
| Total bags | 46 |
| GOOD bags | 45 |
| BAD bags | 1 |
| Input format | ROS2 bag (MCAP) |
| Truncated bags | 44 (handled via StreamReader fallback) |
| Event encoding | Prophesee EVT3 |
| Image encoding | Bayer RGGB 8-bit |

## Environment

```
Platform: ASU Sol HPC
Python: 3.11.14
Conda env: nomad-eventvoxels
Key packages: rosbags 0.11.0, mcap 1.3.1, h5py, opencv-python, scipy
```
