#!/usr/bin/env python3
"""
mock_pi.py
----------
Simulates two FrostByte Pi devices publishing detections to shared Redis.
Each device publishes 4 detections clustered around its own base location.

Usage:
    python scripts/mock_pi.py
    python scripts/mock_pi.py --lat 42.348555 --lon -71.116347
    python scripts/mock_pi.py --lat 42.348555 --lon -71.116347 --count 4 --interval 10
"""
from __future__ import annotations

import argparse
import json
import os
import random
import struct
import time
import zlib
from datetime import datetime, timezone
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

import redis

DETECTION_KEY_PREFIX = "detection:"
DETECTION_TTL_SECONDS = 3600
MAX_DETECTIONS_PER_DEVICE = 10
SEPARATOR = b"\n---MASK---\n"

DEFAULT_LAT = 42.348555
DEFAULT_LON = -71.116347
OFFSET = 0.0005  # ~50m radius per device cluster

# Second device is offset ~200m from the first
DEVICE_OFFSET = 0.002


def _make_test_mask(width: int = 64, height: int = 64) -> bytes:
    def make_chunk(chunk_type: bytes, data: bytes) -> bytes:
        chunk = chunk_type + data
        return struct.pack(">I", len(data)) + chunk + struct.pack(">I", zlib.crc32(chunk) & 0xFFFFFFFF)
    signature = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    raw_data = b"".join(b"\x00" + bytes([128] * width) for _ in range(height))
    idat = zlib.compress(raw_data)
    return signature + make_chunk(b"IHDR", ihdr) + make_chunk(b"IDAT", idat) + make_chunk(b"IEND", b"")


def publish_detection(client: redis.Redis, device_id: str, latitude: float, longitude: float, confidence: float) -> str:
    key = f"{DETECTION_KEY_PREFIX}{device_id}"
    entry = {
        "device_id": device_id,
        "geotag": f"Mock {device_id}",
        "latitude": latitude,
        "longitude": longitude,
        "confidence": confidence,
        "capture_id": None,
        "session_id": None,
        "metadata": {},
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mask": None,
    }
    payload = json.dumps(entry).encode("utf-8") + SEPARATOR + _make_test_mask()
    pipe = client.pipeline()
    pipe.lpush(key, payload)
    pipe.ltrim(key, 0, MAX_DETECTIONS_PER_DEVICE - 1)
    pipe.expire(key, DETECTION_TTL_SECONDS)
    pipe.execute()
    return key


def main() -> None:
    parser = argparse.ArgumentParser(description="Mock Pi — simulates 2 devices publishing to shared Redis")
    parser.add_argument("--lat",      type=float, default=DEFAULT_LAT)
    parser.add_argument("--lon",      type=float, default=DEFAULT_LON)
    parser.add_argument("--count",    type=int,   default=4,  help="Detections per device (default: 4, total: 8)")
    parser.add_argument("--interval", type=int,   default=20, help="Seconds between detections")
    parser.add_argument("--fixed",    action="store_true", help="No random offset within each device cluster")
    args = parser.parse_args()

    password = os.environ.get("REDIS_PASSWORD")
    if not password:
        raise SystemExit("Error: REDIS_PASSWORD not set.")

    client = redis.Redis(host="localhost", port=6380, password=password, decode_responses=False)
    client.ping()

    # Device base locations — device 2 is ~200m away from device 1
    devices = [
        {"id": "mock-pi-1", "base_lat": args.lat,               "base_lon": args.lon},
        {"id": "mock-pi-2", "base_lat": args.lat + DEVICE_OFFSET, "base_lon": args.lon + DEVICE_OFFSET},
    ]

    total = args.count * len(devices)
    print(f"Mock Pi starting — {len(devices)} devices × {args.count} detections = {total} total")
    print(f"Device 1 (mock-pi-1): ({devices[0]['base_lat']:.6f}, {devices[0]['base_lon']:.6f})")
    print(f"Device 2 (mock-pi-2): ({devices[1]['base_lat']:.6f}, {devices[1]['base_lon']:.6f})")
    print(f"Interval: {args.interval}s between detections")
    print("-" * 60)

    # Interleave detections from both devices
    for i in range(args.count):
        for device in devices:
            if args.fixed:
                lat = device["base_lat"]
                lon = device["base_lon"]
            else:
                lat = device["base_lat"] + random.uniform(-OFFSET, OFFSET)
                lon = device["base_lon"] + random.uniform(-OFFSET, OFFSET)

            confidence = round(random.uniform(0.6, 0.95), 2)
            key = publish_detection(client, device["id"], lat, lon, confidence)

            print(f"[{i+1}/{args.count}] {device['id']}")
            print(f"         key={key}")
            print(f"         lat={lat:.6f}  lon={lon:.6f}  conf={confidence}")
            print(f"         time={datetime.now().strftime('%H:%M:%S')}")

        if i < args.count - 1:
            print(f"         waiting {args.interval}s...")
            time.sleep(args.interval)

    print("-" * 60)
    print(f"Done. {total} detections published across {len(devices)} devices.")


if __name__ == "__main__":
    main()
