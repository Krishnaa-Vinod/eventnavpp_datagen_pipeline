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


def decode_evt3(raw_bytes: bytes, width: int = 1280, height: int = 720):
    """Decode EVT3 binary data into structured event arrays.

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

    # Pre-allocate (worst case: every word is an event)
    xs = np.empty(n, dtype=np.int64)
    ys = np.empty(n, dtype=np.int64)
    ts = np.empty(n, dtype=np.int64)
    ps = np.empty(n, dtype=np.int64)

    current_time = np.int64(0)
    current_y = np.int64(0)
    current_pol = np.int64(0)
    base_x = np.int64(0)
    count = 0

    for i in range(n):
        w = int(words[i])
        code = (w >> 12) & 0xF
        payload = w & 0xFFF

        if code == 0x0 or code == 0xF:
            # EVT_ADDR_Y / VECTOR_12_Y: set y + polarity
            current_pol = np.int64((payload >> 11) & 1)
            current_y = np.int64(payload & 0x7FF)

        elif code == 0x2:
            # EVT_ADDR_X: emit one event
            if payload & 0x800:
                current_time += 1
            x = np.int64(payload & 0x7FF)
            if 0 <= x < width and 0 <= current_y < height:
                xs[count] = x
                ys[count] = current_y
                ts[count] = current_time
                ps[count] = current_pol
                count += 1

        elif code == 0x3:
            # VECT_BASE_X
            base_x = np.int64(payload)

        elif code == 0x4:
            # VECT_12: 12-bit bitmap → up to 12 events
            for bit in range(12):
                if payload & (1 << bit):
                    x = base_x + np.int64(bit)
                    if 0 <= x < width and 0 <= current_y < height:
                        xs[count] = x
                        ys[count] = current_y
                        ts[count] = current_time
                        ps[count] = current_pol
                        count += 1

        elif code == 0x5:
            # VECT_8: 8-bit bitmap with offset
            bitmap = payload & 0xFF
            offset = (payload >> 8) & 0xF
            for bit in range(8):
                if bitmap & (1 << bit):
                    x = base_x + np.int64(offset * 8 + bit)
                    if 0 <= x < width and 0 <= current_y < height:
                        xs[count] = x
                        ys[count] = current_y
                        ts[count] = current_time
                        ps[count] = current_pol
                        count += 1

        elif code == 0x6:
            # TIME_LOW
            current_time = (current_time & ~np.int64(0xFFF)) | np.int64(payload)

        elif code == 0x8:
            # TIME_HIGH (standalone, ignore CONTINUED_12 combinations)
            new_high = np.int64(payload) << 12
            current_time = new_high | (current_time & np.int64(0xFFF))

        # Skip: 0x7 (CONTINUED_4), 0xA (EXT_TRIGGER),
        #        0xC (OTHERS), 0xE (CONTINUED_12)

    if count == 0:
        return np.empty((0, 4), dtype=np.int64)

    return np.column_stack([xs[:count], ys[:count], ts[:count], ps[:count]])
