#!/usr/bin/env python3
"""Simulate a frostbyte-backend detection publish for local dev testing.

Pushes a synthetic detection entry into the shared Redis, triggering a
keyspace notification exactly as the real backend would.

Requirements:
    pip install redis python-dotenv

Usage:
    python scripts/publish_test_detection.py
    python scripts/publish_test_detection.py --device-id my-pi --geotag "My Roof" --lat 42.35 --lon -71.06
    python scripts/publish_test_detection.py --count 3   # publish 3 times to test LTRIM
"""
from __future__ import annotations

import argparse
import json
import os
import struct
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
MAX_DETECTIONS_PER_DEVICE = 2
SEPARATOR = b"\n---MASK---\n"


def _make_test_mask(width: int = 64, height: int = 64) -> bytes:
    def make_chunk(chunk_type: bytes, data: bytes) -> bytes:
        chunk = chunk_type + data
        return struct.pack(">I", len(data)) + chunk + struct.pack(">I", zlib.crc32(chunk) & 0xFFFFFFFF)
    signature = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    raw_data = b"".join(b"\x00" + bytes([128] * width) for _ in range(height))
    idat = zlib.compress(raw_data)
    return signature + make_chunk(b"IHDR", ihdr) + make_chunk(b"IDAT", idat) + make_chunk(b"IEND", b"")


def publish_detection(client: redis.Redis, device_id: str, geotag: str, latitude: float, longitude: float) -> str:
    key = f"{DETECTION_KEY_PREFIX}{device_id}"
    entry = {
        "device_id": device_id,
        "geotag": geotag,
        "latitude": latitude,
        "longitude": longitude,
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


def parse_detection(raw: bytes) -> tuple[dict, bytes]:
    parts = raw.split(SEPARATOR, 1)
    if len(parts) != 2:
        raise ValueError("Invalid detection entry format")
    return json.loads(parts[0].decode("utf-8")), parts[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish test detections to local dev Redis")
    parser.add_argument("--device-id", default="test-pi")
    parser.add_argument("--geotag", default="Test Location")
    parser.add_argument("--lat", type=float, default=42.35)
    parser.add_argument("--lon", type=float, default=-71.06)
    parser.add_argument("--count", type=int, default=1)
    args = parser.parse_args()

    password = os.environ.get("REDIS_PASSWORD")
    if not password:
        raise SystemExit("Error: REDIS_PASSWORD not set. Copy .env.template to .env and set a password.")

    client = redis.Redis(host="localhost", port=6380, password=password, decode_responses=False)
    client.ping()

    for i in range(args.count):
        key = publish_detection(client, args.device_id, args.geotag, args.lat, args.lon)
        stored = client.lrange(key, 0, 0)[0]
        meta, mask = parse_detection(stored)
        list_len = client.llen(key)
        print(f"[{i+1}/{args.count}] key={key}  list_len={list_len}  mask={len(mask)}B  ts={meta['timestamp']}")

    if args.count >= MAX_DETECTIONS_PER_DEVICE:
        print(f"\nLTRIM check: list has {client.llen(key)} entries (expect {MAX_DETECTIONS_PER_DEVICE})")


if __name__ == "__main__":
    main()