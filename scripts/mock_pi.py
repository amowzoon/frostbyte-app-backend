#!/usr/bin/env python3
"""
mock_pi.py
----------
Simulates a FrostByte Pi device publishing detections to shared Redis.
Publishes 3 detections, 20 seconds apart, near a fixed base location
with a small random offset each time.

Usage:
    python scripts/mock_pi.py
    python scripts/mock_pi.py --lat 42.348555 --lon -71.116347
    python scripts/mock_pi.py --lat 42.348555 --lon -71.116347 --count 5 --interval 10
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

# Default base location — override with --lat/--lon
DEFAULT_LAT = 42.348555
DEFAULT_LON = -71.116347
OFFSET = 0.0005  # ~50m radius


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
        "geotag": "Mock Pi",
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
    parser = argparse.ArgumentParser(description="Mock Pi — publishes detections to shared Redis")
    parser.add_argument("--lat",      type=float, default=DEFAULT_LAT)
    parser.add_argument("--lon",      type=float, default=DEFAULT_LON)
    parser.add_argument("--count",    type=int,   default=3,  help="Number of detections to publish")
    parser.add_argument("--interval", type=int,   default=20, help="Seconds between detections")
    parser.add_argument("--device-id", default="mock-pi")
    parser.add_argument("--fixed",    action="store_true", help="Use exact lat/lon with no random offset")
    args = parser.parse_args()

    password = os.environ.get("REDIS_PASSWORD")
    if not password:
        raise SystemExit("Error: REDIS_PASSWORD not set.")

    client = redis.Redis(host="localhost", port=6380, password=password, decode_responses=False)
    client.ping()

    print(f"Mock Pi starting — {args.count} detections, {args.interval}s apart")
    print(f"Base location: ({args.lat}, {args.lon})")
    print(f"Offset: {'none (fixed)' if args.fixed else f'±{OFFSET} degrees (~50m)'}")
    print("-" * 50)

    for i in range(args.count):
        if args.fixed:
            lat = args.lat
            lon = args.lon
        else:
            lat = args.lat + random.uniform(-OFFSET, OFFSET)
            lon = args.lon + random.uniform(-OFFSET, OFFSET)

        confidence = round(random.uniform(0.6, 0.95), 2)
        key = publish_detection(client, args.device_id, lat, lon, confidence)

        print(f"[{i+1}/{args.count}] Published detection")
        print(f"         key={key}")
        print(f"         lat={lat:.6f}  lon={lon:.6f}  conf={confidence}")
        print(f"         time={datetime.now().strftime('%H:%M:%S')}")

        if i < args.count - 1:
            print(f"         waiting {args.interval}s...")
            time.sleep(args.interval)

    print("-" * 50)
    print(f"Done. {args.count} detections published.")


if __name__ == "__main__":
    main()