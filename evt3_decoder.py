#!/usr/bin/env python3
"""
Pure-Python EVT3 decoder for Prophesee / Metavision event camera data.

Decodes the compact binary encoding used by event_camera_msgs/msg/EventPacket
when encoding == "evt3".

EVT3 format (little-endian 16-bit words):
  Bits [15:12] = type code, bits [11:0] = payload

  0x0  EVT_ADDR_Y — sets current y + polarity for subsequent events
         payload[11]   = polarity  (0 = OFF, 1 = ON)
         payload[10:0] = y  (11 bits)

  0x2  EVT_ADDR_X — emits one CD event
         payload[11]   = time_increment (adds +1 to timestamp before emit)
         payload[10:0] = x  (11 bits)

  0x3  VECT_BASE_X — sets base x for vectorised events
         payload[11:0] = base_x  (12 bits)

  0x4  VECT_12 — emits up to 12 events from bitmap
         payload[11:0] = 12-bit bitmap (bit i → event at base_x + i)

  0x5  VECT_8 — emits up to 8 events from bitmap
         payload[7:0]  = 8-bit bitmap
         payload[11:8] = offset  (events at base_x + offset*8 + i)

  0x6  TIME_LOW — sets low 12 bits of timestamp
  0x8  TIME_HIGH — sets high bits of timestamp

  0xF  VECTOR_12_Y — alias for EVT_ADDR_Y (sets y + polarity)
  0x7  CONTINUED_4, 0xA  EXT_TRIGGER, 0xC  OTHERS, 0xE  CONTINUED_12 — skipped
"""

import numpy as np


def _ffill(positions, values, n):
    """Forward-fill: propagate *values* at sorted *positions* to all later indices.

    Returns an int64 array of length *n* where positions before the first set
    position are 0, and every position after uses the most recent value.
    """
    if len(positions) == 0:
        return np.zeros(n, dtype=np.int64)
    # Build segment lengths and use np.repeat (single C call, no arange)
    vals = np.empty(len(positions) + 1, dtype=np.int64)
    vals[0] = 0
    vals[1:] = values
    bounds = np.empty(len(positions) + 2, dtype=np.int64)
    bounds[0] = 0
    bounds[1:-1] = positions
    bounds[-1] = n
    lengths = np.diff(bounds)
    return np.repeat(vals, lengths)


def decode_evt3(raw_bytes: bytes, width: int = 1280, height: int = 720):
    """Decode EVT3 binary data into structured event arrays (vectorised).

    Parameters
    ----------
    raw_bytes : bytes
        Raw event payload from EventPacket.events.
    width, height : int
        Sensor dimensions for bounds clipping.

    Returns
    -------
    events : np.ndarray, shape (N, 4), dtype int64
        Columns: [x, y, timestamp_us, polarity].
        Empty (0, 4) array if no valid events.
    """
    if len(raw_bytes) < 2:
        return np.empty((0, 4), dtype=np.int64)

    words = np.frombuffer(raw_bytes, dtype=np.uint16)
    n = len(words)
    if n == 0:
        return np.empty((0, 4), dtype=np.int64)

    codes = (words >> 12).astype(np.uint8)
    payloads = (words & 0xFFF).astype(np.int64)

    # ── timestamps ─────────────────────────────────────────────────────
    # TIME_HIGH (0x8): sets bits 12+
    th_pos = np.where(codes == 0x8)[0]
    time_high = _ffill(th_pos, payloads[th_pos] << 12, n)

    # TIME_LOW (0x6): sets bits 0-11
    tl_pos = np.where(codes == 0x6)[0]
    time_low_base = _ffill(tl_pos, payloads[tl_pos], n)

    # EVT_ADDR_X (0x2) bit 11 → cumulative +1 increment, resets at TIME_LOW
    is_evt_x = codes == 0x2
    has_inc = is_evt_x & ((payloads >> 11) & 1).astype(bool)
    inc = np.zeros(n, dtype=np.int64)
    inc[has_inc] = 1
    cum_inc = np.cumsum(inc)

    cum_at_reset = _ffill(tl_pos, cum_inc[tl_pos], n) if len(tl_pos) > 0 else np.zeros(n, dtype=np.int64)

    timestamp = time_high + time_low_base + (cum_inc - cum_at_reset)

    # ── y + polarity (EVT_ADDR_Y=0x0, VECTOR_12_Y=0xF) ───────────────
    y_mask = (codes == 0x0) | (codes == 0xF)
    y_pos = np.where(y_mask)[0]
    y_ff = _ffill(y_pos, payloads[y_pos] & 0x7FF, n)
    pol_ff = _ffill(y_pos, (payloads[y_pos] >> 11) & 1, n)

    # ── base_x (VECT_BASE_X=0x3) ──────────────────────────────────────
    bx_pos = np.where(codes == 0x3)[0]
    bx_ff = _ffill(bx_pos, payloads[bx_pos], n)

    # ── collect events ─────────────────────────────────────────────────
    #  EVT_ADDR_X (0x2): one event per word
    evt_x_pos = np.where(is_evt_x)[0]
    n_ex = len(evt_x_pos)

    #  VECT_12 (0x4): bitmap → up to 12 events per word
    v12_pos = np.where(codes == 0x4)[0]
    #  VECT_8 (0x5): bitmap → up to 8 events per word
    v8_pos = np.where(codes == 0x5)[0]

    # --- EVT_ADDR_X events ---
    if n_ex > 0:
        ex_x = payloads[evt_x_pos] & 0x7FF
        ex_y = y_ff[evt_x_pos]
        ex_t = timestamp[evt_x_pos]
        ex_p = pol_ff[evt_x_pos]
    else:
        ex_x = ex_y = ex_t = ex_p = np.empty(0, dtype=np.int64)

    # --- VECT_12 events (bitmap expansion) ---
    if len(v12_pos) > 0:
        v12_pay = payloads[v12_pos]
        bits_12 = np.arange(12, dtype=np.int64)
        v12_set = ((v12_pay[:, None] >> bits_12[None, :]) & 1).astype(bool)
        v12_flat = v12_set.ravel()
        v12_src = np.repeat(v12_pos, 12)[v12_flat]
        v12_bit = np.tile(bits_12, len(v12_pos))[v12_flat]
        v12_x = bx_ff[v12_src] + v12_bit
        v12_y = y_ff[v12_src]
        v12_t = timestamp[v12_src]
        v12_p = pol_ff[v12_src]
    else:
        v12_x = v12_y = v12_t = v12_p = np.empty(0, dtype=np.int64)

    # --- VECT_8 events (bitmap + offset expansion) ---
    if len(v8_pos) > 0:
        v8_pay = payloads[v8_pos]
        v8_bitmap = v8_pay & 0xFF
        v8_offset = (v8_pay >> 8) & 0xF
        bits_8 = np.arange(8, dtype=np.int64)
        v8_set = ((v8_bitmap[:, None] >> bits_8[None, :]) & 1).astype(bool)
        v8_flat = v8_set.ravel()
        v8_src = np.repeat(v8_pos, 8)[v8_flat]
        v8_bit = np.tile(bits_8, len(v8_pos))[v8_flat]
        v8_x = bx_ff[v8_src] + np.repeat(v8_offset, 8)[v8_flat] * 8 + v8_bit
        v8_y = y_ff[v8_src]
        v8_t = timestamp[v8_src]
        v8_p = pol_ff[v8_src]
    else:
        v8_x = v8_y = v8_t = v8_p = np.empty(0, dtype=np.int64)

    # ── combine + bounds filter ────────────────────────────────────────
    all_x = np.concatenate([ex_x, v12_x, v8_x])
    all_y = np.concatenate([ex_y, v12_y, v8_y])
    all_t = np.concatenate([ex_t, v12_t, v8_t])
    all_p = np.concatenate([ex_p, v12_p, v8_p])

    valid = (all_x >= 0) & (all_x < width) & (all_y >= 0) & (all_y < height)
    if not valid.any():
        return np.empty((0, 4), dtype=np.int64)

    return np.column_stack([all_x[valid], all_y[valid], all_t[valid], all_p[valid]])
