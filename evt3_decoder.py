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


def _ffill(positions, values, n, initial=0):
    """Forward-fill: propagate *values* at sorted *positions* to all later indices.

    Returns an int64 array of length *n* where positions before the first set
    position use *initial*, and every position after uses the most recent value.
    """
    if len(positions) == 0:
        return np.full(n, initial, dtype=np.int64)
    # Build segment lengths and use np.repeat (single C call, no arange)
    vals = np.empty(len(positions) + 1, dtype=np.int64)
    vals[0] = initial
    vals[1:] = values
    bounds = np.empty(len(positions) + 2, dtype=np.int64)
    bounds[0] = 0
    bounds[1:-1] = positions
    bounds[-1] = n
    lengths = np.diff(bounds)
    return np.repeat(vals, lengths)


def decode_evt3(raw_bytes: bytes, width: int = 1280, height: int = 720,
                initial_time_high: int = 0, initial_time_low: int = 0,
                initial_y: int = 0, initial_pol: int = 0,
                initial_base_x: int = 0, time_high_epoch: int = 0):
    """Decode EVT3 binary data into structured event arrays (vectorised).

    Parameters
    ----------
    raw_bytes : bytes
        Raw event payload from EventPacket.events.
    width, height : int
        Sensor dimensions for bounds clipping.
    initial_time_high : int
        Carry-over TIME_HIGH value (already shifted <<12) from previous packet.
    initial_time_low : int
        Carry-over TIME_LOW value from previous packet.
    initial_y, initial_pol, initial_base_x : int
        Carry-over spatial state from previous packet.
    time_high_epoch : int
        Number of 2^24 timestamp wraps already seen in earlier packets.

    Returns
    -------
    events : np.ndarray, shape (N, 4), dtype int64
        Columns: [x, y, timestamp_us, polarity].
        Empty (0, 4) array if no valid events.
    final_state : dict
        State to pass as initial_* to the next call for continuity.
        Includes ``time_high_epoch`` for 2^24 wrap tracking.
    """
    empty_state = {
        "time_high": initial_time_high, "time_low": initial_time_low,
        "y": initial_y, "pol": initial_pol, "base_x": initial_base_x,
        "time_high_epoch": time_high_epoch,
    }

    if len(raw_bytes) < 2:
        return np.empty((0, 4), dtype=np.int64), empty_state

    words = np.frombuffer(raw_bytes, dtype=np.uint16)
    n = len(words)
    if n == 0:
        return np.empty((0, 4), dtype=np.int64), empty_state

    codes = (words >> 12).astype(np.uint8)
    payloads = (words & 0xFFF).astype(np.int64)

    # ── timestamps ─────────────────────────────────────────────────────
    # TIME_HIGH (0x8): sets bits 12+ — carry over from previous packet
    th_pos = np.where(codes == 0x8)[0]
    time_high_raw = _ffill(th_pos, payloads[th_pos] << 12, n,
                        initial=initial_time_high)

    # ── Detect and unwrap 2^24 timestamp wraps ────────────────────────
    # TIME_HIGH is 12 bits; combined with TIME_LOW it gives a 24-bit
    # timestamp (max 2^24 - 1 = 16,777,215 µs ≈ 16.777 s).  After that
    # the TIME_HIGH counter wraps from 4095 back to 0.
    WRAP_24 = 1 << 24           # 16_777_216 µs
    HALF_WRAP_TH = 1 << 23      # detection threshold (half range in shifted TH)

    if len(th_pos) > 0:
        # Compare each TIME_HIGH set-point against its predecessor
        th_changes = np.empty(len(th_pos) + 1, dtype=np.int64)
        th_changes[0] = initial_time_high      # carry-over from previous packet
        th_changes[1:] = payloads[th_pos] << 12

        diffs = np.diff(th_changes)
        wrap_flags = diffs < -HALF_WRAP_TH     # large negative jump → wrap

        cum_wraps = np.cumsum(wrap_flags) + time_high_epoch

        # Forward-fill the epoch offset to every word position
        epoch_arr = _ffill(th_pos, cum_wraps * WRAP_24, n,
                           initial=time_high_epoch * WRAP_24)
        final_epoch = int(cum_wraps[-1])
    else:
        epoch_arr = np.full(n, time_high_epoch * WRAP_24, dtype=np.int64)
        final_epoch = time_high_epoch

    time_high = time_high_raw + epoch_arr

    # TIME_LOW (0x6): sets bits 0-11 — carry over from previous packet
    tl_pos = np.where(codes == 0x6)[0]
    time_low_base = _ffill(tl_pos, payloads[tl_pos], n,
                            initial=initial_time_low)

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
    y_ff = _ffill(y_pos, payloads[y_pos] & 0x7FF, n, initial=initial_y)
    pol_ff = _ffill(y_pos, (payloads[y_pos] >> 11) & 1, n, initial=initial_pol)

    # ── base_x (VECT_BASE_X=0x3) ──────────────────────────────────────
    bx_pos = np.where(codes == 0x3)[0]
    bx_ff = _ffill(bx_pos, payloads[bx_pos], n, initial=initial_base_x)

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

    # ── build final state for carry-over ───────────────────────────────
    final_state = {
        "time_high": int(time_high_raw[-1]),  # raw (without epoch offset)
        "time_low": int(time_low_base[-1]),
        "y": int(y_ff[-1]),
        "pol": int(pol_ff[-1]),
        "base_x": int(bx_ff[-1]),
        "time_high_epoch": final_epoch,
    }

    if not valid.any():
        return np.empty((0, 4), dtype=np.int64), final_state

    return np.column_stack([all_x[valid], all_y[valid],
                            all_t[valid], all_p[valid]]), final_state


# ── Stateful stream decoder ───────────────────────────────────────────────────

class EVT3StreamDecoder:
    """Stateful EVT3 decoder that carries TIME_HIGH/TIME_LOW across packets.

    Root cause of "black voxels" bug:
        Prophesee EVT3 packets can begin with event words (EVT_ADDR_X,
        VECT_12, etc.) *before* any TIME_HIGH or TIME_LOW word.  The raw
        ``decode_evt3()`` forward-fills with 0 for those early events,
        producing timestamps near 0 instead of the correct ~12 million µs.
        The converter's ``min_needed`` filter then discards all those events
        (since the step window expects timestamps around 12M µs), resulting
        in empty voxels → black frames.

    Fix:
        This class carries the last TIME_HIGH and TIME_LOW values from the
        previous packet into the next call, so events before the first
        TIME_HIGH/TIME_LOW in a packet inherit the correct timestamp state.

    Usage::

        decoder = EVT3StreamDecoder(width=1280, height=720)
        for msg in bag_messages:
            events = decoder.decode(msg.raw_events)
            # events have correct, continuous timestamps
    """

    def __init__(self, width: int = 1280, height: int = 720, verbose: bool = False):
        self.width = width
        self.height = height
        self.verbose = verbose

        # Carry-over state
        self._time_high = 0
        self._time_low = 0
        self._y = 0
        self._pol = 0
        self._base_x = 0
        self._time_high_epoch = 0

        # Diagnostics
        self.packets_decoded = 0
        self.total_events = 0

    def decode(self, raw_bytes: bytes, width: int = None, height: int = None):
        """Decode one EVT3 packet with state carry-over.

        Parameters
        ----------
        raw_bytes : bytes
            Raw event payload (EventPacket.events).
        width, height : int, optional
            Override sensor dimensions.

        Returns
        -------
        events : np.ndarray, shape (N, 4), dtype int64
            [x, y, timestamp_us, polarity] with correct continuous timestamps.
        """
        w = width or self.width
        h = height or self.height

        events, state = decode_evt3(
            raw_bytes, w, h,
            initial_time_high=self._time_high,
            initial_time_low=self._time_low,
            initial_y=self._y,
            initial_pol=self._pol,
            initial_base_x=self._base_x,
            time_high_epoch=self._time_high_epoch,
        )

        # Update carry-over state
        self._time_high = state["time_high"]
        self._time_low = state["time_low"]
        self._y = state["y"]
        self._pol = state["pol"]
        self._base_x = state["base_x"]
        self._time_high_epoch = state["time_high_epoch"]

        self.packets_decoded += 1
        self.total_events += len(events)

        if self.verbose and len(events) > 0:
            t = events[:, 2]
            print(f"  [EVT3Stream] pkt {self.packets_decoded}: "
                  f"{len(events)} evts, t=[{t.min()}, {t.max()}] µs, "
                  f"carry TH={self._time_high}")

        return events

    def reset(self):
        """Reset all carry-over state."""
        self._time_high = 0
        self._time_low = 0
        self._y = 0
        self._pol = 0
        self._base_x = 0
        self._time_high_epoch = 0
        self.packets_decoded = 0
        self.total_events = 0
