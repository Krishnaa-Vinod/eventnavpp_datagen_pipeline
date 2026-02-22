#!/usr/bin/env python3
"""
bag_reader.py — Unified ROS2 bag reader with truncation fallback.

Primary: rosbags.rosbag2.Reader  (handles bag directories, multiple files)
Fallback: mcap.stream_reader.StreamReader  (handles truncated .mcap files)

Both paths yield (topic, bag_timestamp_ns, raw_cdr_bytes, msgtype_str) tuples,
and deserialization uses the rosbags typestore.
"""

import glob
import os
from dataclasses import dataclass
from typing import Iterator, Optional, Tuple

import yaml
from rosbags.typesys import Stores, get_typestore, get_types_from_msg

# ── type store (module-level singleton) ────────────────────────────────────────
_typestore = None


def get_ts():
    """Return (and lazily init) the shared typestore with EventPacket registered."""
    global _typestore
    if _typestore is not None:
        return _typestore
    _typestore = get_typestore(Stores.ROS2_HUMBLE)
    event_msg_def = """
std_msgs/Header header
uint32 height
uint32 width
uint64 seq
uint64 time_base
string encoding
bool is_bigendian
uint8[] events
"""
    _typestore.register(
        get_types_from_msg(event_msg_def, "event_camera_msgs/msg/EventPacket")
    )
    return _typestore


@dataclass
class BagMessage:
    """One message from a bag."""
    topic: str
    timestamp: int          # bag/log time in nanoseconds
    data: bytes             # raw CDR payload
    msgtype: str            # e.g. 'sensor_msgs/msg/Image'


def deserialize(msg: BagMessage):
    """Deserialize a BagMessage using the shared typestore."""
    return get_ts().deserialize_cdr(msg.data, msg.msgtype)


# ── reader implementations ────────────────────────────────────────────────────

def _read_via_rosbags(bag_dir: str) -> Iterator[BagMessage]:
    """Read a bag directory using rosbags (requires valid MCAP end magic)."""
    from rosbags.rosbag2 import Reader
    with Reader(bag_dir) as reader:
        for conn, ts, rawdata in reader.messages():
            yield BagMessage(
                topic=conn.topic,
                timestamp=ts,
                data=rawdata,
                msgtype=conn.msgtype,
            )


def _read_via_mcap_stream(bag_dir: str) -> Iterator[BagMessage]:
    """Read a bag directory using mcap StreamReader (handles truncated files)."""
    from mcap.stream_reader import StreamReader

    # Find all mcap files, sorted by name (they're numbered sequentially)
    mcap_files = sorted(glob.glob(os.path.join(bag_dir, "*.mcap*")))
    if not mcap_files:
        raise FileNotFoundError(f"No .mcap files found in {bag_dir}")

    for mcap_path in mcap_files:
        channels = {}
        schemas = {}
        with open(mcap_path, "rb") as f:
            reader = StreamReader(f, skip_magic=False)
            for record in reader.records:
                rtype = type(record).__name__
                if rtype == "Schema":
                    schemas[record.id] = record
                elif rtype == "Channel":
                    channels[record.id] = record
                elif rtype == "Message":
                    ch = channels.get(record.channel_id)
                    if ch is None:
                        continue
                    yield BagMessage(
                        topic=ch.topic,
                        timestamp=record.log_time,
                        data=record.data,
                        msgtype=ch.schema_id
                        and schemas.get(ch.schema_id)
                        and schemas[ch.schema_id].name
                        or ch.topic,
                    )


def _fix_msgtype(name: str) -> str:
    """Normalise ROS2 message type string.

    MCAP schemas often store types as 'sensor_msgs/msg/Image' already,
    but sometimes with different separators.
    """
    # rosbags expects 'pkg/msg/Type' format
    if name and "/" not in name:
        return name  # give up normalising
    return name


def _read_via_mcap_stream_fixed(bag_dir: str) -> Iterator[BagMessage]:
    """Read bag directory via mcap StreamReader with proper msgtype resolution.

    Handles truncated MCAP files gracefully (catches EndOfFile).
    """
    from mcap.stream_reader import StreamReader

    mcap_files = sorted(glob.glob(os.path.join(bag_dir, "*.mcap*")))
    if not mcap_files:
        raise FileNotFoundError(f"No .mcap files found in {bag_dir}")

    for mcap_path in mcap_files:
        channels = {}
        schemas = {}
        with open(mcap_path, "rb") as f:
            reader = StreamReader(f, skip_magic=False)
            try:
                for record in reader.records:
                    rtype = type(record).__name__
                    if rtype == "Schema":
                        schemas[record.id] = record
                    elif rtype == "Channel":
                        channels[record.id] = record
                    elif rtype == "Message":
                        ch = channels.get(record.channel_id)
                        if ch is None:
                            continue
                        # Resolve message type from schema
                        schema = (
                            schemas.get(ch.schema_id) if ch.schema_id else None
                        )
                        if schema:
                            msgtype = _fix_msgtype(schema.name)
                        else:
                            msgtype = ch.topic  # last resort
                        yield BagMessage(
                            topic=ch.topic,
                            timestamp=record.log_time,
                            data=record.data,
                            msgtype=msgtype,
                        )
            except Exception:
                # EndOfFile (truncated MCAP) or other stream errors —
                # we've yielded everything we could from this file.
                pass


# ── public API ─────────────────────────────────────────────────────────────────

def get_topics(bag_dir: str) -> dict:
    """Return {topic_name: msgtype_str} from metadata.yaml or by scanning."""
    meta_path = os.path.join(bag_dir, "metadata.yaml")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = yaml.safe_load(f)
        info = meta.get("rosbag2_bagfile_information", {})
        topics = {}
        for entry in info.get("topics_with_message_count", []):
            tm = entry.get("topic_metadata", {})
            topics[tm.get("name", "")] = tm.get("type", "")
        return topics
    return {}


def read_bag(bag_dir: str, fallback: bool = True) -> Iterator[BagMessage]:
    """Iterate messages from a ROS2 bag directory.

    Tries rosbags first; if that fails (truncated MCAP), falls back to
    mcap StreamReader.
    """
    try:
        yield from _read_via_rosbags(bag_dir)
        return
    except Exception:
        if not fallback:
            raise

    # Fallback: mcap StreamReader
    yield from _read_via_mcap_stream_fixed(bag_dir)
